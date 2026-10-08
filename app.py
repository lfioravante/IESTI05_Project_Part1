import os
import time
import subprocess
import threading
import io
import numpy as np
from PIL import Image
from flask import Flask, render_template, Response, jsonify
from ai_edge_litert.interpreter import Interpreter

app = Flask(__name__)

MODEL_PATH = "model/qModel_TrashNET.tflite"
LABELS_PATH = "model/labels.txt"

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines() if line.strip()]

latest_result = {
    "label": "Aguardando...",
    "confidence": 0.0,
    "inference_ms": 0.0,
    "active": True
}
result_lock = threading.Lock()

class CameraStream:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        comando = [
            "rpicam-vid",
            "-t", "0",
            "--width", "640",
            "--height", "480",
            "--framerate", "15",
            "--codec", "mjpeg",
            "--quality", "50",
            "--inline",
            "-o", "-"
        ]
        
        processo = subprocess.Popen(comando, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        buffer = b''
        
        while True:
            data = processo.stdout.read(4096)
            if not data:
                break
            
            buffer += data
            a = buffer.find(b'\xff\xd8')
            b = buffer.find(b'\xff\xd9')
            
            if a != -1 and b != -1:
                jpg = buffer[a:b+2]
                buffer = buffer[b+2:]
                
                with self.lock:
                    self.frame = jpg

    def get_frame(self):
        with self.lock:
            return self.frame

camera_stream = CameraStream()

def inference_worker():
    try:
        interpreter = Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"Erro ao alocar modelo: {e}")
        return
        
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    _, height, width, _ = input_details[0]['shape']
    in_scale, in_zero_point = input_details[0]['quantization']
    out_scale, out_zero_point = output_details[0]['quantization']
    
    while True:
        frame_bytes = camera_stream.get_frame()
        if frame_bytes and latest_result["active"]:
            try:
                start_time = time.time()
                
                imagem_pil = Image.open(io.BytesIO(frame_bytes)).convert('RGB').resize((width, height))
                matriz_imagem = np.array(imagem_pil)
                
                img_norm = matriz_imagem.astype(np.float32) / 255.0
                img_quant = np.clip(np.round(img_norm / in_scale) + in_zero_point, -128, 127).astype(np.int8)
                dados_entrada = np.expand_dims(img_quant, axis=0)
                
                interpreter.set_tensor(input_details[0]['index'], dados_entrada)
                interpreter.invoke()
                
                output_quant = interpreter.get_tensor(output_details[0]['index'])[0]
                output_real = (output_quant.astype(np.float32) - out_zero_point) * out_scale
                
                class_idx = int(np.argmax(output_real))
                confidence = float(output_real[class_idx])
                
                latency_ms = (time.time() - start_time) * 1000
                
                with result_lock:
                    latest_result["label"] = labels[class_idx]
                    latest_result["confidence"] = confidence
                    latest_result["inference_ms"] = latency_ms
                    
            except Exception as e:
                pass
                
        time.sleep(0.05)

threading.Thread(target=inference_worker, daemon=True).start()

def generate_frames():
    while True:
        frame = camera_stream.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.05)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    with result_lock:
        return jsonify(latest_result)

@app.route('/control/<action>', methods=['POST'])
def control_action(action):
    global latest_result
    with result_lock:
        if action == 'start':
            latest_result["active"] = True
        elif action == 'stop':
            latest_result["active"] = False
    return jsonify({"status": "success"})

if __name__ == '__main__':
    time.sleep(2)
    app.run(host='0.0.0.0', port=5000, debug=False)
