from torch.utils.data import DataLoader
from model import FaceModel
from options import opt
import math
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from epoch import run_one_epoch

# our own datasets
from liveness_datasets import protocols
import liveness_datasets.transforms as T

writer = SummaryWriter(log_dir=(None if opt.name is None
                                else f"runs/{opt.name}"))

best_res = 101
train_batch_size = opt.batch_size
test_batch_size = opt.batch_size

model = FaceModel(opt, isTrain=True, input_nc=3)

# dataset setup
trs_img, trs_label, empty_trs = (None,)*3
if opt.noaugment:
    trs_img, trs_label = T.get_transforms(False)
    empty_trs = trs_img
else:
    trs_img, trs_label = T.get_augment(False)
    empty_trs = T.get_empty()

casia_dir = os.path.join(opt.data_dir, "casia-new/data/")
replay_dir = os.path.join(opt.data_dir, "replay-new/data/")

if opt.protocol == "intra-casia-fasd":
    protocol = protocols.IntraCASIAFASDProtocol(
        root_dir=casia_dir, trs=trs_img, trs_label=trs_label,
        trs_test=empty_trs, trs_label_test=trs_label, res=(256, 256),
        res_depth=(32, 32), join_train_and_val=True)
elif opt.protocol == "intra-replay-attack":
    protocol = protocols.IntraReplayAttackProtocol(
        root_dir=replay_dir, trs=trs_img, trs_label=trs_label,
        trs_test=empty_trs, trs_label_test=trs_label, res=(256, 256),
        res_depth=(32, 32), join_train_and_val=True)
elif opt.protocol == "cross-casiafasd-replayattack":
    protocol = protocols.CrossCASIAFASDReplayAttackProtocol(
        from_casia=True, casia_root_dir=casia_dir, replay_root_dir=replay_dir,
        trs=trs_img, trs_label=trs_label, trs_test=empty_trs,
        trs_label_test=trs_label, res=(256, 256), res_depth=(32, 32),
        join_train_and_val=True)
elif opt.protocol == "cross-replayattack-casiafasd":
    protocol = protocols.CrossCASIAFASDReplayAttackProtocol(
        from_casia=False, casia_root_dir=casia_dir, replay_root_dir=replay_dir,
        trs=trs_img, trs_label=trs_label, trs_test=empty_trs,
        trs_label_test=trs_label, res=(256, 256), res_depth=(32, 32),
        join_train_and_val=True)

train_ds, train_sampler = protocol.train
dev_ds, dev_sampler = protocol.val
val_ds, val_sampler = protocol.test

train_ldr = DataLoader(train_ds, batch_size=train_batch_size,
                       sampler=train_sampler, num_workers=8)
dev_ldr = DataLoader(train_ds, batch_size=test_batch_size, sampler=dev_sampler,
                     num_workers=8)
val_ldr = DataLoader(val_ds, sampler=val_sampler, batch_size=test_batch_size,
                     num_workers=8)

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
