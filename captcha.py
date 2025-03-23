import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import os
import string

from model import CaptchaModel

class Captcha(nn.Module):
    def __init__(self, model, model_path=None):
        super(Captcha, self).__init__()
        self.model = model

        if model_path:
            self.load_model_weights(model_path)

    def load_model_weights(self, model_path):
        """
        Load state dict
        """
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            print(f"Loaded model weights from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model.eval()

    def __call__(self, im_path, save_path=False):
        """
        Algo for inference
        args:
            im_path: .jpg image path to load and infer
            save_path: output file path to save the one-line outcome
        """
        image = Image.open(im_path).convert('RGB')
        
        # Split the image into 5 independent character images
        split_points = [4, 13, 22, 31, 40, 49]
        char_images = [
            image.crop((split_points[i], 0, split_points[i + 1], image.size[1])) 
            for i in range(5)
        ]
        
        predicted_text = ""
        for char_image in char_images:
            char_image = char_image.resize((9, 30))
            char_image = self.transform(char_image)
            
            output = self.model(char_image.unsqueeze(0))
            predicted_char = torch.argmax(output, dim=1)
            predicted_text += self.decode_char(predicted_char.item())
        # print(f"Result: {predicted_text}")
        if save_path:
            with open(save_path, 'w') as f:
                f.write(predicted_text)
        return predicted_text

    def decode_char(self, idx):
        """
        Postprocessing
        """
        alphabet = string.ascii_uppercase  # 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        digits = '0123456789'  # '0123456789'
        
        if idx < 26:
            return alphabet[idx]
        else:
            return digits[idx - 26]
    
    def transform(self, image):
        """
        Image preprocessing
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        return transform(image)
    
if __name__ == "__main__":
    model = CaptchaModel()
    captcha = Captcha(model, 'captcha_model.pth')
    captcha('sampleCaptchas/test/input100.jpg')