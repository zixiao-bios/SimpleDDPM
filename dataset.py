import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class FFHQDataset(Dataset):
    def __init__(self, transform=None):
        self.dataset = load_dataset("Dmini/FFHQ-64x64", split="train")
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        item = self.dataset[idx]
        image = item['image']
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0  # 返回图像和占位符标签
