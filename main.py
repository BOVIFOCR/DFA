from torch.utils.data import DataLoader
from torch import nn
from model import FaceModel
from options import opt
import os
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
    model.train()
    train_meter = PADMeter()
    train_loss = {"C": 0., "D": 0., "G": 0.}
    for i, data in tqdm(enumerate(train_ldr), desc="Train Batches", unit="batches"):
        model.set_input(data)  # TODO
        model.optimize_parameters()
        class_output = nn.functional.softmax(model.output, dim=1)
        train_meter.update(model.label.cpu().data.numpy(),
                           class_output.cpu().data.numpy())
        losses = model.get_current_losses()
        train_loss['C'] += losses['C']
        train_loss['D'] += losses['D']
        train_loss['G'] += losses['G']
        if i % 10 == 9:
            train_meter.full_update()
            # TODO utils.report_to_tensorboard
            # step epoch * args.batch_size + i
        train_meter.full_update()
        # TODO report_to_tensorboard com step epoch
        train_loss["Sum"] = (train_loss['C'] + train_loss['D']
                             + 0.1 * train_loss['G'])
        if i % 100 == 99:
            # TODO update do meter
            pad_meter_train.get_eer_and_thr()
            pad_meter_train.get_hter_apcer_etal_at_thr(pad_meter_train.threshold)
            pad_meter_train.get_accuracy(pad_meter_train.threshold)
            ret = model.get_current_visuals()
            img_save_dir = os.path.join(opt.checkpoints_dir, opt.name, "res")
            if not os.path.exists(img_save_dir):
                os.makedirs(img_save_dir)
            # TODO log losses e erros
            wandb.log(model.get_current_losses())
            wandb.log({"HTER_train": pad_meter_train.hter, "EER_train": pad_meter_train.eer, "ACC_train": pad_meter_train.accuracy})
            #print('HTER {pad_meter.hter:.4f} EER {pad_meter.eer:.4f} ACC {pad_meter.accuracy:.4f}'.format(
            #     pad_meter=pad_meter_train))
            # vutils.save_image(ret['fake_B'], "%s/epoch_%d_fake.png" % (img_save_dir, e), normalize=True)
            # vutils.save_image(ret['real_B'], "%s/epoch_%d_real.png" % (img_save_dir, e), normalize=True)
    model.eval()
    dev_meter = PADMeter()
    dev_loss = {"C": 0., "D": 0., "G": 0.}
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_ldr),
                             desc="Validation Batches", unit="batches"):
            model.set_input(batch)
            model.forward()
            class_output = nn.functional.softmax(model.output, dim=1)
            dev_meter.update(model.label.cpu().data.numpy(),
                             class_output.cpu().data.numpy())
            losses = model.get_current_losses()
            dev_loss['C'] += losses['C']
            dev_loss['D'] += losses['D']
            dev_loss['G'] += losses['G']
    dev_meter.full_update()
    dev_loss["Sum"] = (dev_loss['C'] + dev_loss['D']
                       + 0.1 * dev_loss['G'])
    for loss_name, loss_value in dev_loss.items():
        dev_loss[loss_name] = loss_value / len(train_ds)

    val_meter = PADMeter
    pad_meter = eval_model(val_ldr, model)

    # TODO log de treino, dev e val
    is_best = pad_meter.hter <= best_res
    best_res = min(pad_meter.hter, best_res)
    if is_best:
        best_name = "best"
        model.save_networks(best_name)
    filename = "latest"
    model.save_networks(filename)
