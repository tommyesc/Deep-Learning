import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import ThreeTowerModel
from dataset import SketchyCOCODataset
import torchvision.transforms as transforms
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model
model = ThreeTowerModel().to(device)
criterion = nn.TripletMarginLoss(margin=1.0)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dataset and DataLoader
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
dataset = SketchyCOCODataset('data/sketches', 'data/positives', 'data/negatives', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for i, (sketch, pos_image, neg_image) in enumerate(dataloader):
        sketch, pos_image, neg_image = sketch.to(device), pos_image.to(device), neg_image.to(device)
        
        optimizer.zero_grad()
        sketch_embed, pos_embed, neg_embed = model(sketch, pos_image, neg_image)
        
        loss = criterion(sketch_embed, pos_embed, neg_embed)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}")
