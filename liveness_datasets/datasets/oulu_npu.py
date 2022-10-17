import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import pandas as pd

class LivenessDataset(Dataset):
    def __init__(self, img_dir, annotations_file, transform=None,
                 depth_transform=None, target_transform=None):
        annotations_path = os.path.join(img_dir, annotations_file)
        self.img_labels = pd.read_csv(
            annotations_path, names=["image_path", "depth_path", "cropped_path", "label"])
        # root dir, contains the data dir and all labels csv files
        self.img_dir = img_dir
        self.transform = transform # image transform
        self.depth_transform = depth_transform # depth map transform
        self.target_transform = target_transform # label transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        depth_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
        image = read_image(img_path)
        depth = read_image(depth_path)
        crop = read_image(cropped_path)
        label = self.img_labels.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
            crop = self.transform(crop)
        if self.depth_transform:
            depth = self.depth_transform(depth)
        if self.target_transform:
            label = self.target_transform(label)
        return image, depth, crop, label
