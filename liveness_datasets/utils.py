import torch

def train_loop(X, y, model, loss_fn, optimizer, verbose=True):
    """
    basic training loop for given model and dataset

    taken/modified from pytorch docs
    """
    # compute prediction and loss
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss_value = loss.item()
    loss.backward()
    optimizer.step()

    if verbose:
        print(f"loss: {loss.item()}")
    return loss_value


def test_loop(X, y, model, loss_fn):
    """
    basic testing/validation loop for given model and dataset

    taken/modified from pytorch docs
    """
    tp, fp, tn, fn, test_loss, correct = 0, 0, 0, 0, 0, 0

    with torch.no_grad():
        pred = model(X)
        test_loss += loss_fn(pred, y).item()
        for cls_label, ground_truth in zip(pred.argmax(1), y):
            if cls_label == ground_truth:
                if ground_truth:
                    tp += 1
                else:
                    tn += 1
            else:
                if ground_truth:
                    fn += 1
                else:
                    fp += 1
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    results = {
            "correct": correct,
            "test_loss": test_loss,
            "fp": fp,
            "tp": tp,
            "fn": fn,
            "tn": tn
            }
    
    return results
    
def get_errors(fn=0, tn=0, fp=0, tp=0):
    apcer = fn/(fn+tp)
    bpcer = fp/(fp+tn)
    far = fn/(fn+tp)
    frr = fp/(fp+tn)
    results = {
            "fn": fn,
            "tn": tn,
            "fp": fp,
            "tp": tp,
            "acc": (tp+tn)/(fn+tn+fp+tp),
            "apcer": 100*apcer,
            "bpcer": 100*bpcer,
            "acer": 100*(apcer+bpcer)/2,
            "hter": 100*(far+frr)/2,
            }
    return results

def save_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    """save model and optimizer to checkpoint file"""
    print(f"=> saving checkpoint to {filename}")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(filename, model, optimizer, lr, device):
    """
    load model and optimizer from checkpoint file

    lr: learning rate to be used
    device: "cuda" or "cpu"
    """
    print(f"=> loading checkpoint from {filename}")
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    # lr must be updated for each param group in optimizer, or old lr is kept
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
