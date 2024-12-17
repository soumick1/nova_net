import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class PotholeSegmentationDataset(Dataset):
    def __init__(self, rgb_dir, label_dir, transforms=None):
        self.rgb_dir = rgb_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.images = sorted(os.listdir(self.rgb_dir))
        self.labels = sorted(os.listdir(self.label_dir))
        assert len(self.images) == len(self.labels), "Image and label counts do not match!"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.rgb_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(label_path))

        if mask.max() > 1:
            mask = (mask > 0).astype(np.uint8)

        if self.transforms:
            transformed = self.transforms(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
        else:
            img = torch.from_numpy(img.transpose(2,0,1)).float() / 255.0
            mask = torch.from_numpy(mask)

        mask = mask.long()
        return img, mask
