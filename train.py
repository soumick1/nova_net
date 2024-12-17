import torch
import torch.nn as nn
from tqdm import tqdm

def train_one_epoch(model, optimizer, data_loader, device, scaler):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    for images, masks in tqdm(data_loader, desc="Training", leave=False):
        images, masks = images.to(device), masks.to(device)
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        total_loss += loss.item()

    return total_loss / len(data_loader)

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    for images, masks in tqdm(data_loader, desc="Validation", leave=False):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        total_loss += loss.item()

    return total_loss / len(data_loader)
