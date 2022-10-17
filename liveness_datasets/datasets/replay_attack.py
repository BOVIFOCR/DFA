import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import pandas as pd

class ReplayAttackDataset(Dataset):
    def __init__(self, root_dir, subset="train", transform=None,
                 depth_transform=None, target_transform=None,
                 nodepth_path=None):
        self.root_dir = root_dir
        noext = lambda s: s.split('.png')[0] # remove extension from filepath
        self.root_dir = os.path.join(self.root_dir, subset)
        img_paths = []
        img_crop_paths = []
        depth_paths = []
        depth_crop_paths = []
        labels = []

        attack_fixed_path = os.path.join(self.root_dir, "attack/fixed")
        attack_hand_path = os.path.join(self.root_dir, "attack/hand")
        real_path = os.path.join(self.root_dir, "real")
        paths_labels = [(attack_fixed_path, 1), (attack_hand_path, 1),
                        (real_path, 0)]
        suffixes = ["_crop.png", "_depth.png", "_depth_crop.png"]
        for path, label in paths_labels:
            for img in os.listdir(path):
                if (any([img.endswith(suffix) for suffix in suffixes])):
                    continue
                img_path = os.path.join(path, img)
                img_paths.append(img_path)
                img_crop_paths.append(noext(img_path) + "_crop.png")
                if (not nodepth_path) or label == 0:
                    depth_paths.append(noext(img_path) + "_depth.png")
                    depth_crop_paths.append(noext(img_path) + "_depth_crop.png")
                else:
                    depth_paths.append(nodepth_path)
                    depth_crop_paths.append(nodepth_path)
                labels.append(label)

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
        depth = read_image(depth_path)
        depth_crop = read_image(depth_crop_path)

        if self.transform:
            img = self.transform(img)
            img_crop = self.transform(img_crop)
        if self.depth_transform:
            depth = self.depth_transform(depth)
            depth_crop = self.depth_transform(depth_crop)
        if self.target_transform:
            label = self.target_transform(label)
        return img, img_crop, depth, depth_crop, label
