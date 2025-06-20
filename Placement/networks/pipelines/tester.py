import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from easydict import EasyDict
from utils.utils import LossLogger, compound_annotation
from networks.utils.registry import PIPELINE_DICT
from typing import Tuple, List

@PIPELINE_DICT.register_module
@torch.no_grad()
def test_mono_det(data, module:nn.Module,
                     writer:SummaryWriter, 
                     loss_logger:LossLogger=None, 
                     global_step:int=None, 
                     cfg:EasyDict=None) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    images, P2 = data[0], data[1]

    scores, bbox, obj_index = module([images.cuda().float().contiguous(), torch.tensor(P2).cuda().float()])
    obj_types = [cfg.obj_types[i.item()] for i in obj_index]
    # print("Scores:",scores)

    return (scores, bbox, obj_types)


