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

run_dir = os.path.join(opt.checkpoints_dir, opt.name, "runs")
if not os.path.exists(run_dir):
    os.makedirs(run_dir)
writer = SummaryWriter(log_dir=run_dir)

if __name__ == '__main__':
    best_res = 101
    train_batch_size = opt.batch_size
    test_batch_size = opt.batch_size

    model = FaceModel(opt, isTrain=True, input_nc=3)

    # dataset setup
    train_ds = CASIAFASDDataset("/home/rgpa18/image_datasets/casia-new/data/",
                                "train", transform=T.t_src,
                                depth_transform=T.t_depth,
                                nodepth_path=os.path.join(
                                    "/home/rgpa18/image_datasets/casia-new/data/attack_depth.png")
                               )
    val_ds = CASIAFASDDataset(os.path.join("/home/rgpa18/image_datasets/casia-new/data/"),
                                     "test", transform=T.t_src, depth_transform=T.t_depth,
                                     nodepth_path=os.path.join(
                                         "/home/rgpa18/image_datasets/casia-new/data/attack_depth.png")
                                     )
    train_sampler, dev_sampler = utils.split_dataset(train_ds)
    train_ldr = DataLoader(train_ds, batch_size=train_batch_size, sampler=train_sampler, num_workers=8)
    dev_ldr = DataLoader(train_ds, batch_size=test_batch_size,
                                 sampler=dev_sampler, num_workers=8)
    val_ldr = DataLoader(val_ds, batch_size=test_batch_size, num_workers=8)

    for e in tqdm(range(opt.epoch), desc="Epochs", unit="epochs"):
        model.train()
        pad_meter_train = PADMeter()
        for i, data in tqdm(enumerate(train_ldr), desc="Batches", unit="batches"):
            model.set_input(data)
            model.optimize_parameters()
            class_output = nn.functional.softmax(model.output, dim=1)
            pad_meter_train.update(model.label.cpu().data.numpy(),
                             class_output.cpu().data.numpy())

            writer.add_scalars('cls_loss', {'closs': model.get_current_losses()['C']}, i+ len(train_ldr) *e)

            if i %100 ==0:
                pad_meter_train.get_eer_and_thr()
                pad_meter_train.get_hter_apcer_etal_at_thr(pad_meter_train.threshold)
                pad_meter_train.get_accuracy(pad_meter_train.threshold)
                ret = model.get_current_visuals()
                img_save_dir = os.path.join(opt.checkpoints_dir, opt.name, "res")
                if not os.path.exists(img_save_dir):
                    os.makedirs(img_save_dir)
                wandb.log(model.get_current_losses())
                wandb.log({"HTER_train": pad_meter_train.hter, "EER_train": pad_meter_train.eer, "ACC_train": pad_meter_train.accuracy})
                #print('HTER {pad_meter.hter:.4f} EER {pad_meter.eer:.4f} ACC {pad_meter.accuracy:.4f}'.format(
                #     pad_meter=pad_meter_train))
                # vutils.save_image(ret['fake_B'], "%s/epoch_%d_fake.png" % (img_save_dir, e), normalize=True)
                # vutils.save_image(ret['real_B'], "%s/epoch_%d_real.png" % (img_save_dir, e), normalize=True)


        if e%1==0:
            model.eval()
            pad_dev_mater = eval_model(dev_ldr,model)
            pad_meter = eval_model(val_ldr,model)

            pad_meter.get_eer_and_thr()
            pad_dev_mater.get_eer_and_thr()

            pad_meter.get_hter_apcer_etal_at_thr(pad_dev_mater.threshold)
            pad_meter.get_accuracy(pad_dev_mater.threshold)
            wandb.log({"HTER": pad_meter.hter, "EER": pad_meter.eer, "ACC": pad_meter.accuracy})
            # print('HTER {pad_meter.hter:.4f} EER {pad_meter.eer:.4f} ACC {pad_meter.accuracy:.4f}'.format(
            #     pad_meter=pad_meter))
            is_best = pad_meter.hter <= best_res
            best_res = min(pad_meter.hter, best_res)
            if is_best:
                best_name = "best"
                model.save_networks(best_name)

        filename = "lastest"
        model.save_networks(filename)
