import os
from PIL import Image
import random
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class SketchyCOCODataset(Dataset):
    def __init__(self, sketches_dir, positives_dir, negatives_dir, transform=None):
        self.sketches_dir = sketches_dir
        self.positives_dir = positives_dir
        self.negatives_dir = negatives_dir
        self.transform = transform
        self.sketches = os.listdir(sketches_dir)
        
    def __len__(self):
        return len(self.sketches)

    def __getitem__(self, idx):
        sketch_path = os.path.join(self.sketches_dir, self.sketches[idx])
        pos_image_path = os.path.join(self.positives_dir, self.sketches[idx].replace('sketch', 'positive'))
        neg_image_path = os.path.join(self.negatives_dir, random.choice(os.listdir(self.negatives_dir)))
        
        sketch = Image.open(sketch_path).convert('RGB')
        pos_image = Image.open(pos_image_path).convert('RGB')
        neg_image = Image.open(neg_image_path).convert('RGB')
        
        if self.transform:
            sketch = self.transform(sketch)
            pos_image = self.transform(pos_image)
            neg_image = self.transform(neg_image)
        
        return sketch, pos_image, neg_image
