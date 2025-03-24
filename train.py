from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.optim as optim
import torch.nn as nn
import string

from dataset import CaptchaDataset
from model import CaptchaModel

# preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
])

# dataloader
captcha_dir = 'sampleCaptchas/input'
captcha_dataset = CaptchaDataset(captcha_dir=captcha_dir, transform=transform)
dataloader = DataLoader(captcha_dataset, batch_size=1, shuffle=False)

# init model
model = CaptchaModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 10 epochs
for epoch in range(10):
    model.train()
    running_loss = 0.0
    
    for images, labels in dataloader:
        optimizer.zero_grad()
        
        for i in range(5):
            char_image = images[i]
            output = model(char_image)
            
            # Convert label to numerical representation (A=0, B=1, ..., Z=25, 0=26, ..., 9=35)
            labels_idx = [[string.ascii_uppercase.index(c) if c.isalpha() else 26 + int(c) for c in label] for label in labels]
            labels_idx = torch.tensor(labels_idx)
            loss = criterion(output, labels_idx[:, i])  # loss of each character
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/10], Loss: {running_loss / len(dataloader)}")

# save ckpt
torch.save(model.state_dict(), 'captcha_model.pth')