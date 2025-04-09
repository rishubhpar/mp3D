import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from easydict import EasyDict
from utils.utils import LossLogger,compound_annotation
from networks.utils.registry import PIPELINE_DICT


@PIPELINE_DICT.register_module
def train_mono_det(data,module,optimizer,writer,loss_logger,global_step,epoch_num,cfg):
    optimizer.zero_grad()
    images,P2,labels,bbox2d,bbox3d,depth=data
    
    # print("Images:",images.shape)
    # print("P2:",P2.shape)
    # print("labels:",labels)
    # print("2dbox:",bbox2d)
    # print("3dbbox:",bbox3d.shape)

    max_length=np.max([len(label) for label in labels])
    # print("Maximum number of objects in all samples:",max_length)
    if max_length==0:
        return None
    annotation=compound_annotation(labels,max_length,bbox2d,bbox3d,cfg.obj_types)
    # print("#######################")
    # # print(annotation.shape)#bs,max_len,12
    # print("Input to the model")
    # print("Images:",images.shape) # b,c,h,w
    # print("Anno:",images.new(annotation).shape)#b,max_obj,12
    # print("p2:",P2.shape)#b,3,4
    # print("depth:",depth.shape)#,h/4,w/4 (because block reduce)
    # module([images.cuda().float().contiguous(),images.new(annotation).cuda().float().contiguous(),P2.cuda(),depth.cuda().contiguous()])
    classification_loss,regression_loss,loss_dict=module([images.cuda().float().contiguous(),images.new(annotation).cuda().float().contiguous(),P2.cuda(),depth.cuda().contiguous()])
    # print("class loss:",classification_loss.shape)
    # print("regression_loss:",regression_loss.shape)
    # # print(classification_loss.shape)
    # # print(regression_loss.shape)
    classification_loss=classification_loss.mean()
    regression_loss=regression_loss.mean()
    if not loss_logger is None:
        loss_logger.update(loss_dict)
    del loss_dict

    if not optimizer is None:
        loss=classification_loss+regression_loss
    
    # print("Total Loss:",loss)
    
    if bool(loss==0):
        del loss,loss_dict
        return
    loss.backward()
    # clip loss norm
    torch.nn.utils.clip_grad_norm_(module.parameters(),cfg.optimizer.clipped_gradient_norm)
    optimizer.step()
    optimizer.zero_grad()
    return (loss)

