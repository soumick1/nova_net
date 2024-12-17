import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .model import get_model

@torch.no_grad()
def run_inference(model_ckpt_path, image_paths, device=None, score_threshold=0.5):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(num_classes=2, device=device)
    model.load_state_dict(torch.load(model_ckpt_path, map_location=device))
    model.eval()

    transform = A.Compose([
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

    results = []
    for img_path in image_paths:
        img = np.array(Image.open(img_path).convert("RGB"))
        transformed = transform(image=img)
        img_tensor = transformed['image'].unsqueeze(0).to(device)

        outputs = model(img_tensor)
        preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
        results.append((img_path, preds))

    return results
