from torch.utils.data import DataLoader
from torch import nn
from model import FaceModel
from options import opt
import os
import math
from tqdm import tqdm
from tensorboardX import SummaryWriter
from test import eval_model

# our own datasets
from liveness_datasets.datasets.casia_fasd import CASIAFASDDataset
from liveness_datasets import transforms as T
from liveness_datasets import utils as utils
from liveness_datasets.meter import PADMeter

ds_dir = "/home/rgpa18/image_datasets"
nodepth_path = f"{ds_dir}/casia-new/data/attack_depth.png"
writer = SummaryWriter()

best_res = 101
train_batch_size = opt.batch_size
test_batch_size = opt.batch_size

model = FaceModel(opt, isTrain=True, input_nc=3)

# dataset setup
train_ds = CASIAFASDDataset(f"{ds_dir}/casia-new/data/",
                            "train", transform=T.t_src,
                            depth_transform=T.t_depth,
                            nodepth_path=nodepth_path)
val_ds = CASIAFASDDataset(f"{ds_dir}/casia-new/data/", "test",
                          transform=T.t_src, depth_transform=T.t_depth,
                          nodepth_path=nodepth_path)
train_sampler, dev_sampler = utils.split_dataset(train_ds)
train_ldr = DataLoader(train_ds, batch_size=train_batch_size,
                       sampler=train_sampler, num_workers=8)
dev_ldr = DataLoader(train_ds, batch_size=test_batch_size,
                     sampler=dev_sampler, num_workers=8)
val_ldr = DataLoader(val_ds, batch_size=test_batch_size, num_workers=8)

for epoch in tqdm(range(opt.epoch), desc="Epochs", unit="epochs"):
    train_size = math.ceil(.8 * len(train_ds))
    dev_size = math.floor(.2 * len(train_ds))
    val_size = len(val_ds)
    run_one_epoch(epoch, model, train_ldr, train_size, "train", writer)
    run_one_epoch(epoch, model, dev_ldr, dev_size, "dev", writer)
    val_hter = run_one_epoch(epoch, model, val_ldr, val_size, "val", writer)
    if val_hter <= best_res:
        best_name = "best"
        best_res = val_hter
        model.save_networks(best_name)
    filename = "latest"
    model.save_networks(filename)
