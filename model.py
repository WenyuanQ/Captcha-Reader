import os
import torch
import torch.nn as nn
import onnx
from onnxsim import simplify

class CaptchaModel(nn.Module):
    def __init__(self):
        super(CaptchaModel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(64 * 2 * 7, 128)  # Adjusting for 9x30 size, 2 pooling
        # 36 characters A-Z, 0-9
        self.fc2 = nn.Linear(128, 36)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(nn.ReLU()(x))
        
        x = self.conv2(x)
        x = self.pool(nn.ReLU()(x))
        
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        
        return x

def convert_to_onnx(model, input_tensor, ckpt_path='captcha_model.pth', output_path="captcha_model.onnx"):
    model.eval()
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))
        print(f"Loaded model weights from {ckpt_path}")
    torch.onnx.export(model,
                      input_tensor,
                      output_path,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],)
    model_ = onnx.load(output_path)
    model_simplified, check = simplify(model_)
    onnx.save(model_simplified, output_path)
    print(f"Model has been converted to ONNX and saved at {output_path}")
    
if __name__ == "__main__":
    model = CaptchaModel()
    dummy_input = torch.randn(1, 3, 9, 30)
    convert_to_onnx(model, dummy_input)