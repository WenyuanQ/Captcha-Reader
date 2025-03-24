from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class CaptchaDataset(Dataset):
    def __init__(self, captcha_dir, transform=None):
        """
            captcha_dir: Dataset path.
            transform: Preprocessing.
        """
        self.captcha_dir = captcha_dir
        self.transform = transform
        # Get a list of image names in the dataset path
        self.images = [f for f in os.listdir(captcha_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.captcha_dir, self.images[idx])
        image = Image.open(image_path).convert('RGB')

        # Ensure image is of the correct size (60x30)
        image = image.resize((60, 30))

        split_points = [4, 13, 22, 31, 40, 49]
        
        char_images = [
            image.crop((split_points[i], 0, split_points[i + 1], image.size[1])) 
            for i in range(5)
        ]

        if self.transform:
            char_images = [self.transform(char_img) for char_img in char_images]
        
        token0 = os.path.dirname(os.path.dirname(image_path))
        token1 = 'output'
        token2 = self.images[idx].replace('input', 'output').replace('.jpg', '.txt')
        label_path = os.path.join(token0, token1, token2)
        with open(label_path, 'r') as file_:
            label = file_.read()[:5]

        return char_images, label