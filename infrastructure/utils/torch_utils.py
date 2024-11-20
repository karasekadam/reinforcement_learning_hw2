import torch
import numpy as np


"""
    Functions for going from numpy to torch tensors and backwards.

    detach tensor from computation graph -> copy to cpu and cast to numpy
"""
def to_numpy(torch_tensor):
    return torch_tensor.detach().cpu().numpy()

"""
    Cast numpy arr to tensor on cpu
"""
def to_torch(np_arr):
    return torch.from_numpy(np_arr)
