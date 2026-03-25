import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class PongDataset(Dataset):
    def __init__(self, folder_path, image_size=64):
        self.folder_path = folder_path
        self.images = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
    

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.images[idx])
        # Abrimos com PIL para garantir que venha em RGB
        image = Image.open(img_path).convert('RGB')
        return self.transform(image)