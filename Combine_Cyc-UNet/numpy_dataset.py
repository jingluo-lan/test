import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class NumpyDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train', n_classes=10):
        self.transform = transforms.Compose([transforms.ToTensor()]) if transforms_ is None else transforms_
        self.n_classes = n_classes
        self.files_A = sorted([os.path.join(root, mode + 'A', f) for f in os.listdir(os.path.join(root, mode + 'A')) if f.endswith('.npy')])
        self.files_B = sorted([os.path.join(root, mode + 'B', f) for f in os.listdir(os.path.join(root, mode + 'B')) if f.endswith('.npy')])
        self.label_files = sorted([os.path.join(root, mode + 'labels', f) for f in os.listdir(os.path.join(root, mode + 'labels')) if f.endswith('.npy')])

        print(f"Found {len(self.files_A)} files in {mode}A")
        print(f"Found {len(self.files_B)} files in {mode}B")
        print(f"Found {len(self.label_files)} files in {mode}labels")

    def __getitem__(self, index):
        item_A = np.load(self.files_A[index % len(self.files_A)]).astype(np.float32)
        item_B = np.load(self.files_B[index % len(self.files_B)]).astype(np.float32)
        label = np.load(self.label_files[index % len(self.label_files)]).astype(np.int64)

        # Ensure the data is single-channel and normalized to [0, 1]
        if item_A.ndim == 2:
            item_A = item_A[np.newaxis, :, :]
        if item_B.ndim == 2:
            item_B = item_B[np.newaxis, :, :]

        # Convert to PIL Image
        item_A = Image.fromarray((item_A.squeeze(0) * 255).astype(np.uint8))
        item_B = Image.fromarray((item_B.squeeze(0) * 255).astype(np.uint8))

        # Apply transforms if specified
        item_A = self.transform(item_A)
        item_B = self.transform(item_B)

        # Ensure label values are within the expected range
        label = np.clip(label, 0, self.n_classes - 1)
        label = torch.from_numpy(label)

        return {'A': item_A, 'B': item_B, 'label': label}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
