import unittest
import os
import glob
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

class TestInferenceEngine(unittest.TestCase):
    def setUp(self):
        # Definição dos caminhos mapeados
        self.model_path = "model/qModel_TrashNET.tflite"
        self.labels_path = "model/labels.txt"
        self.dataset_path = "dataset/TrashNET"

    def test_interpreter_allocation(self):
        """Assegura a leitura da arquitetura quantizada e formatos de entrada/saída"""
        try:
            interpreter = Interpreter(model_path=self.model_path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            
            self.assertEqual(input_details[0]['shape'].tolist(), [1, 224, 224, 3])
            self.assertEqual(input_details[0]['dtype'], np.int8)
        except Exception as e:
            self.fail(f"A alocação do modelo falhou: {e}")

    def test_labels_exist(self):
        """Verifica a presença e leitura do arquivo de rótulos"""
        self.assertTrue(os.path.exists(self.labels_path), "Arquivo labels.txt não encontrado")
        with open(self.labels_path, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f.readlines() if line.strip()]
        self.assertGreater(len(labels), 0, "A lista de rótulos está vazia")

    def test_pipeline_with_real_dataset(self):
        """Carrega uma imagem real do dataset para validar o fluxo de predição completo"""
        if not os.path.exists(self.dataset_path):
            self.skipTest(f"Diretório do dataset não encontrado em {self.dataset_path}")
            
        # Busca a primeira imagem JPG disponível em qualquer subdiretório do TrashNET
        search_pattern = os.path.join(self.dataset_path, "**", "*.jpg")
        images_found = glob.glob(search_pattern, recursive=True)
        
        if not images_found:
            self.skipTest("Nenhuma imagem JPG foi encontrada dentro do diretório do dataset.")
            
        test_image_path = images_found[0]
        
        try:
            # Setup do interpretador
            interpreter = Interpreter(model_path=self.model_path)
            interpreter.allocate_tensors()
            in_details = interpreter.get_input_details()[0]
            out_details = interpreter.get_output_details()[0]
            
            _, height, width, _ = in_details['shape']
            in_scale, in_zero_point = in_details['quantization']
            
            # Pré-processamento
            imagem_pil = Image.open(test_image_path).convert('RGB').resize((width, height))
            matriz_imagem = np.array(imagem_pil)
            img_norm = matriz_imagem.astype(np.float32) / 255.0
            img_quant = np.clip(np.round(img_norm / in_scale) + in_zero_point, -128, 127).astype(np.int8)
            dados_entrada = np.expand_dims(img_quant, axis=0)
            
            # Inferência
            interpreter.set_tensor(in_details['index'], dados_entrada)
            interpreter.invoke()
            output_quant = interpreter.get_tensor(out_details['index'])[0]
            
            # Validações de conformidade
            self.assertEqual(len(output_quant), 6, "A saída não possui o número correto de classes (6)")
            
        except Exception as e:
            self.fail(f"Falha na execução do pipeline com imagem real: {e}")

if __name__ == "__main__":
    unittest.main()
