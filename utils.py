import torch
from torchvision.utils import make_grid


def report_loss_to_tensorboard(writer, loss_dict, name, step):
    """report losses from dict to tensorboard writer"""
    for loss_name, value in loss_dict.items():
        writer.add_scalar(f"{name}/{loss_name}_loss", value, step)


def grid_from_batch(imgs, gt_depth_maps, gen_depth_maps):
    cat = torch.cat([imgs, gt_depth_maps, gen_depth_maps], 2)
    return make_grid(cat)
