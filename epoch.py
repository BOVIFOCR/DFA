import torch
from torch import nn
from tqdm import tqdm
from liveness_datasets.meter import PADMeter
from liveness_datasets import utils


def batch_report(meter, model, name, show_grid=False):
    report = {
            "scalars": {
                f"{name}/HTER": meter.hter,
                f"{name}/EER": meter.eer
            }, "grid": {
                "title": '',
                "image": None,
            }
        }
    if show_grid:
        lim = min(10, len(model.real_A))
        grid = utils.grid_from_batch(model.real_A[:lim], model.fake_B[:lim],
                                     model.real_B[:lim])
        title = f"{name}/examples"
        report["grid"]["title"] = title
        report["grid"]["image"] = grid
    return report


def report_to_tensorboard(writer, meter, model, show_grid, name, step):
    report = batch_report(meter, model, name, show_grid)
    for scalar, value in report["scalars"].items():
        writer.add_scalar(scalar, value, step)
    if show_grid:
        writer.add_image(report["grid"]["title"], report["grid"]["image"],
                         step)


def run_one_epoch(epoch, model, ldr, ds_size, mode, writer):
    is_train, is_dev, is_val = (x == mode for x in ["train", "dev", "val"])
    if is_train:
        model.train()
    else:
        model.eval()
    meter = PADMeter()
    loss_dict = {"C": 0., "D": 0., "G": 0.}
    for i, batch in tqdm(enumerate(ldr), desc="Batches", unit="batches"):
        with torch.set_grad_enabled(is_train):
            model.set_input(batch)
            if is_train:
                model.optimize_parameters()
            else:
                model.forward()
            class_output = nn.functional.softmax(model.output, dim=1)
            meter.update(model.label.cpu().data.numpy(),
                         class_output.cpu().data.numpy())
            losses = model.get_current_losses()
            loss_dict['C'] += losses['C']
            loss_dict['D'] += losses['D']
            loss_dict['G'] += losses['G']
            if is_train and i % 10 == 9:
                meter.full_update()
                show_grid = epoch % 10 == 9
                step = epoch * len(batch[0]) + i
                report_to_tensorboard(writer, meter, model, show_grid, mode,
                                      step)
        meter.full_update()
        show_grid = epoch % 10 == 9
        report_name = "train_epoch" if is_train else mode
        report_to_tensorboard(writer, meter, model, show_grid, report_name,
                              epoch)
        loss_dict["Sum"] = (loss_dict['C'] + loss_dict['D']
                            + 0.1 * loss_dict['G'])
        for loss_name, loss_value in loss_dict.items():
            loss_dict[loss_name] = loss_value / ds_size
        utils.report_loss_to_tensorboard(writer, loss_dict, report_name, epoch)
    return meter.hter
