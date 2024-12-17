# NovaNet

NovaNet is a Python package providing a gated multi-scale segmentation architecture tailored for the Pothole600 dataset, along with utilities for training and inference.

## Installation

```bash
pip install nova_net
```

## Usage

### Training
```python
from nova_net.datasets import PotholeSegmentationDataset
from nova_net.transforms import get_train_transforms, get_val_transforms
from nova_net.model import get_model
from nova_net.train import train_one_epoch, evaluate
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = PotholeSegmentationDataset('path/to/train_rgb', 'path/to/train_label', transforms=get_train_transforms())
val_dataset = PotholeSegmentationDataset('path/to/val_rgb', 'path/to/val_label', transforms=get_val_transforms())

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2)

model = get_model(num_classes=2, device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scaler = torch.amp.GradScaler()

for epoch in range(10):
    train_loss = train_one_epoch(model, optimizer, train_loader, device, scaler)
    val_loss = evaluate(model, val_loader, device)
    print(f"Epoch {epoch+1}: Train Loss={train_loss}, Val Loss={val_loss}")
```

### Inference

```python
from nova_net.inference import run_inference

model_ckpt_path = 'best_model.pth'
image_paths = ['path/to/test_img.png']
results = run_inference(model_ckpt_path, image_paths, device=device)

for img_path, pred_mask in results:
    # pred_mask is the predicted segmentation mask
    print(f"Image: {img_path}, Mask shape: {pred_mask.shape}")
```

```yaml

---

### Publishing to PyPI

1. **Build and Upload**:  
   Install `twine`:
   ```bash
   pip install twine
```
