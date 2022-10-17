from dataset import AlignedDataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import nn
from model import FaceModel
from options import opt
import torchvision.utils as vutils
import wandb
import os
import numpy as np
import torch
from tqdm import tqdm
from statistics import PADMeter
import logging
from tensorboardX import SummaryWriter
from torch.utils.data.sampler import  WeightedRandomSampler
from test import eval_model

# our own datasets
from liveness_datasets.datasets.casia_fasd import CASIAFASDDataset
import torchvision.transforms as T

file_name = os.path.join(opt.checkpoints_dir, opt.name,"log")
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG,
                    filename= file_name,filemode="w")
run_dir = os.path.join(opt.checkpoints_dir, opt.name,"runs")
if not os.path.exists(run_dir):
    os.makedirs(run_dir)
writer = SummaryWriter(log_dir=run_dir)

wandb.init(project="DFA_fork")

def polarize(x, threshold=0.1):
    return x.masked_fill_(x > threshold, 1)

if __name__ == '__main__':
    best_res = 101
    train_batch_size = opt.batch_size
    test_batch_size = opt.batch_size
    
    model = FaceModel(opt,isTrain = True,input_nc = 3)

    # dataset setup
    transform_src = T.Compose([
        T.ToPILImage(),
        T.Resize((256,256)),
        T.ToTensor(),
    ])

    transform_depth = T.Compose([
        T.ToPILImage(mode="L"),
        T.Resize((256,256)), # another resize will be applied afterwards
        # T.Resize((32,32)),
        T.ToTensor(),
        # T.ConvertImageDtype(torch.float32),
        T.Lambda(lambda x: polarize(x)),
    ])
    train_dataset = CASIAFASDDataset(os.path.join("/home/rgpa18/image_datasets/casia-new/data/"),
                                     "train", transform=transform_src,
                                     depth_transform=transform_depth,
                                     # nodepth_path=os.path.join(
                                        # DATA, "casia-new/data/attack_depth.png")
                                     )
    test_dataset = CASIAFASDDataset(os.path.join("/home/rgpa18/image_datasets/casia-new/data/"),
                                     "test", transform=transform_src,
                                     depth_transform=transform_depth,
                                     # nodepth_path=os.path.join(
                                        # DATA, "casia-new/data/attack_depth.png")
                                     )
    dataset_sz = len(train_dataset)
    indices = list(range(dataset_sz))
    split = int(np.floor(.2 * dataset_sz))
    shuffle = True
    if shuffle:
        np.random.seed(143)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_data_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                                   sampler=train_sampler, num_workers=8)
    dev_data_loader = DataLoader(train_dataset, batch_size=test_batch_size,
                                 sampler=val_sampler, num_workers=8) # using val as dev
    test_data_loader = DataLoader(test_dataset, batch_size=test_batch_size, num_workers=8)

    """
    test_data_loader = DataLoader(AlignedDataset(test_file_list,isTrain = False), batch_size=test_batch_size,
                                   shuffle=True, num_workers=8)
    dev_data_loader = DataLoader(AlignedDataset(dev_file_list,isTrain = False), batch_size=test_batch_size,
                                   shuffle=True, num_workers=8)

    train_dataset = AlignedDataset(train_file_list) 
    train_data_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                                    shuffle = True,num_workers=8)
                                    """

    # wandb.watch(model)
    writer.iter = 0
    for e in tqdm(range(opt.epoch), desc="Epochs", unit="epochs"):
        model.train()
        pad_meter_train = PADMeter()
        for i, data in tqdm(enumerate(train_data_loader), desc="Batches", unit="batches"):
            model.set_input(data)
            model.optimize_parameters()
            class_output = nn.functional.softmax(model.output, dim=1)
            pad_meter_train.update(model.label.cpu().data.numpy(),
                             class_output.cpu().data.numpy())

            writer.add_scalars('cls_loss', {'closs': model.get_current_losses()['C']}, i+ len(train_data_loader) *e)

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
            pad_dev_mater = eval_model(dev_data_loader,model)
            pad_meter = eval_model(test_data_loader,model)

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
