import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class SketchRetrievalModel(nn.Module):
    def __init__(self):
        super(SketchRetrievalModel, self).__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 128)
        
    def forward(self, x):
        return self.backbone(x)

class ThreeTowerModel(nn.Module):
    def __init__(self):
        super(ThreeTowerModel, self).__init__()
        self.sketch_tower = SketchRetrievalModel()
        self.image_tower = SketchRetrievalModel()
        
    def forward(self, sketch, pos_image, neg_image):
        sketch_embed = self.sketch_tower(sketch)
        pos_embed = self.image_tower(pos_image)
        neg_embed = self.image_tower(neg_image)
        return sketch_embed, pos_embed, neg_embed
