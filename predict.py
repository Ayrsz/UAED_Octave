import os
import time
import argparse
import cv2 as cv
import numpy as np
import torch
from torch import nn
from torchvision.utils import save_image
import importlib

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
MEAN_SUBTRACTION = np.array((104.00698793, 116.66876762, 122.67891434))
FONT_SETTINGS = {
    "fontFace": cv.FONT_HERSHEY_SIMPLEX,
    "fontScale": 1,
    "color": (1.0, 0, 0),
    "thickness": 3
}

def preprocess_image(image_path, method='standard', resize_factor=0.5):
    """Centraliza a lógica de tratamento de imagem."""
    img = cv.imread(image_path).astype(np.float32)
    
    if method == 'standard':
        img -= MEAN_SUBTRACTION
        if resize_factor != 1.0:
            img = cv.resize(img, (0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv.INTER_CUBIC)
        img = np.transpose(img, (2, 0, 1))
        tensor = torch.from_numpy(np.expand_dims(img, 0))
    else:
        # Método para UAED/Uncertainty
        img = np.transpose(img, (2, 0, 1))
        tensor = torch.from_numpy(np.expand_dims(img, 0)) / 255.0
        
    return tensor

def calculate_fps(model, dummy_input, seconds=5):
    """Calcula a performance do modelo."""
    print(f"Iniciando teste de FPS por {seconds}s...")
    start_time = time.time()
    images_processed = 0
    while (time.time() - start_time) < seconds:
        _ = model(dummy_input)
        images_processed += 1
    
    total_time = time.time() - start_time
    fps = images_processed / total_time
    return fps

class EdgePredictor:
    def __init__(self, net, device, output_dir='./preds/'):
        self.net = net
        self.device = device
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _save_result(self, tensor_list, original_path):
        filename = 'border_' + os.path.basename(original_path)
        save_path = os.path.join(self.output_dir, filename)
        save_image(tensor_list, save_path)
        print(f"Salvo: {save_path}")


    @torch.no_grad()
    def run_uncertainty(self, image_paths):
        self.net.eval()
        for path in image_paths:
            tensor = preprocess_image(path, method='uaed').to(self.device)
            mean, std = self.net(tensor)
            
            dist = torch.distributions.Normal(loc=mean, scale=std + 0.001)
            outputs = torch.sigmoid(dist.rsample())
            
            # Stack: [Sample, Mean, Std]
            combined = torch.cat([outputs, mean, std], dim=0) 
            self._save_result(combined, path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default='model.sigma_logit_unetpp')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input_folder', type=str, default='./ValidateImages/')
    parser.add_argument('--test_fps', action='store_true')
    parser.add_argument('--distribution', default='gs', type=str, help='the output distribution')
    
    args = parser.parse_args()


    MODEL_NAME = args.model_file
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    Model = importlib.import_module(MODEL_NAME)
    model = Model.Mymodel(args).to(torch.device(device))
    model = nn.DataParallel(model)
    
    checkpoint = torch.load(args.checkpoint, map_location=torch.device(device))
    
    model.load_state_dict(checkpoint['state_dict'])
    

    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [os.path.join(args.input_folder, f) for f in os.listdir(args.input_folder) 
                   if f.lower().endswith(valid_ext)]

    if not image_files:
        raise ValueError("Nenhuma imagem encontrada.")

    predictor = EdgePredictor(model, device)
    
    
    predictor.run_uncertainty(image_files)


if __name__ == '__main__':
    main()