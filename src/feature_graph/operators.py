import torch

def enter(mask):
    x = mask.int()
    return torch.cat([torch.tensor([False]), x.diff() == 1])

def exit(mask):
    x = mask.int()
    return torch.cat([torch.tensor([False]), x.diff() == -1])

def between(start, end):
    return (start.cumsum(dim=0) - end.int().cumsum(dim=0)) > 0


operators = {
    'enter': enter,
    'exit': exit
}