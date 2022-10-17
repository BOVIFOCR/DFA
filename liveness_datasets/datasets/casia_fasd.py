import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision
import os
import pandas as pd

class CASIAFASDDataset(Dataset):
    def __init__(self, root_dir, subset="train", transform=None,
                 depth_transform=None, target_transform=None,
                 nodepth_path=None):
        self.root_dir = root_dir
        noext = lambda s: s.split('.png')[0] # remove extension from filepath
        # real are 1, 2, HR_1
        get_label = lambda s: not(s[0] in ['1', '2'] or s.startswith('HR_1'))
        self.root_dir = os.path.join(self.root_dir, f"{subset}_release")
        img_paths = []
        suffixes = ["_crop.png", "_depth.png", "_depth_crop.png"]
        for subject in os.listdir(self.root_dir):
            subject_path = os.path.join(self.root_dir, subject)
            for img in os.listdir(subject_path):
                if (any([img.endswith(suffix) for suffix in suffixes])):
                    continue
                img_path = os.path.join(subject_path, img)
                img_paths.append(img_path)

        img_crop_paths = [noext(p) + '_crop.png' for p in img_paths]
        depth_paths = [noext(p) + '_depth.png' for p in img_paths]
        depth_crop_paths = [noext(p) + '_depth_crop.png' for p in img_paths]
        labels = [get_label(p.split('/')[-1]) for p in img_paths]
        if nodepth_path: # depth should be empty for spoofs
            num_samples = len(labels)
            for i in range(num_samples):
                if labels[i] == 0:
                    continue
                depth_paths[i] = nodepth_path
                depth_crop_paths[i] = nodepth_path

        self.img_labels = pd.DataFrame(data={'img_path': img_paths,
                                             'img_crop_path': img_crop_paths,
                                             'depth_path': depth_paths,
                                             'depth_crop_path': depth_crop_paths,
                                             'label': labels})

        self.transform = transform # image transform
        self.has_transform = transform is not None
        self.depth_transform = depth_transform # depth map transform
        self.has_depth_transform = depth_transform is not None
        self.target_transform = target_transform # label transform
        self.has_target_transform = target_transform is not None

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        img_crop_path = self.img_labels.iloc[idx, 1]
        depth_path = self.img_labels.iloc[idx, 2] 
        depth_crop_path = self.img_labels.iloc[idx, 3]
        label = self.img_labels.iloc[idx, 4]

        img = read_image(img_path)
        img_crop = read_image(img_crop_path)
        depth = read_image(depth_path, mode=torchvision.io.ImageReadMode.GRAY)
        depth_crop = read_image(depth_crop_path, mode=torchvision.io.ImageReadMode.GRAY)

        if self.transform:
            img = self.transform(img)
            img_crop = self.transform(img_crop)
        if self.depth_transform:
            trs = self.depth_transform if label else self.transform
            depth = trs(depth) # self.depth_transform(depth)
            depth_crop = trs(depth_crop) # self.depth_transform(depth_crop)
        if self.target_transform:
            label = self.target_transform(label)
        return img, img_crop, depth, depth_crop, label
