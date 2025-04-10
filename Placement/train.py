import os
import sys
import numpy as np
from easydict import EasyDict
from tqdm import tqdm
from fire import Fire
import coloredlogs
import logging
from tqdm import tqdm
import pprint
import torch
import warnings
warnings.filterwarnings("ignore")
from torch.utils.tensorboard import SummaryWriter

#local imports
from preprocess.path_init_ import *
from networks.utils.registry import DETECTOR_DICT, DATASET_DICT, PIPELINE_DICT
from networks.utils.utils import BackProjection, BBox3dProjector, get_num_parameters
from networks.pipelines import train_mono_det
import data.kitti.dataset
from utils.utils import LossLogger,cfg_from_file
from optimizers import build_optimizer
from schedulers import GradualWarmupScheduler,PolyLR,build_scheduler

# torch.cuda.set_device(2)

def main(config="config/config.py", experiment_name="default", world_size=1, local_rank=-1):
    config="config/config.py"
    cfg=cfg_from_file(config)
    # print(cfg)

    #dist training
    world_size=1
    local_rank=-1
    experiment_name="monodtr_moving_det"
    cfg.dist = EasyDict()
    cfg.dist.world_size=world_size  
    cfg.dist.local_rank=local_rank
    is_distributed=local_rank >= 0 # local_rank < 0 -> single training is True
    is_logging=local_rank <= 0 # only log and test with main process
    is_evaluating=local_rank <= 0

    recorder_dir=os.path.join(cfg.path.log_path,experiment_name)
    # print(is_logging)
    if is_logging: # writer exists only if not distributed and local rank is smaller
        ## Clean up the dir if it exists before
        if os.path.isdir(recorder_dir):
            os.system("rm -r {}".format(recorder_dir))
            print("clean up the recorder directory of {}".format(recorder_dir))
        writer=SummaryWriter(recorder_dir)
        ## Record config object using pprint
        formatted_cfg=pprint.pformat(cfg)
        writer.add_text("config.py", formatted_cfg.replace(' ', '&nbsp;').replace('\n', '  \n')) # add space for markdown style in tensorboard text
    else:
        writer=None

    #multigpu and distributed process
    if is_distributed:
        cfg.trainer.gpu = local_rank # local_rank will overwrite the GPU in configure file
        gpu=min(cfg.trainer.gpu,torch.cuda.device_count()-1)
        torch.backends.cudnn.benchmark = getattr(cfg.trainer,'cudnn',False)
        torch.cuda.set_device(gpu)
        if is_distributed:
            torch.distributed.init_process_group(backend='nccl',init_method='env://')
        print(local_rank)


    #dataset and dataloader
    train_dataset=DATASET_DICT[cfg.data.train_dataset](cfg)#3712
    train_dataloader=torch.utils.data.DataLoader(train_dataset,num_workers=cfg.data.num_workers,
    batch_size=cfg.data.batch_size,collate_fn=train_dataset.collate_fn,shuffle=local_rank<0,drop_last=True,
    sampler=torch.utils.data.DistributedSampler(train_dataset,num_replicas=world_size,rank=local_rank,shuffle=False) if local_rank >= 0 else None)
    # print(train_dataset[0].keys())

    # for batch_idx,data in enumerate(train_dataloader):
    #     print(data)
        # print(data)


    val_dataset=DATASET_DICT[cfg.data.val_dataset](cfg,split="validation")#3769
    val_dataloader=torch.utils.data.DataLoader(val_dataset,num_workers=cfg.data.num_workers,
    batch_size=cfg.data.batch_size,collate_fn=val_dataset.collate_fn,shuffle=False,drop_last=True)
    # print(val_dataset[0].keys())

    # detection model (monodtrcore + monodtr det3d head)
    detector=DETECTOR_DICT[cfg.detector.name](cfg.detector)
    # print(detector)

    if is_distributed:
        detector=torch.nn.SyncBatchNorm.convert_sync_batchnorm(detector)
        detector=torch.nn.parallel.DistributedDataParallel(detector.cuda(),device_ids=[gpu],output_device=gpu)
    else:
        detector=detector.cuda()
        detector.train()
    if is_logging:
        string1=detector.__str__().replace(' ', '&nbsp;').replace('\n', '  \n')
        writer.add_text("model structure",string1) # add space for markdown style in tensorboard text
        num_parameters=get_num_parameters(detector)
        print(f'number of parameters of the model:{num_parameters}')

    # # #optimizer and scheduler
    optimizer=build_optimizer(cfg.optimizer,detector)
    scheduler_config=getattr(cfg,'scheduler',None)
    scheduler=build_scheduler(scheduler_config,optimizer)

    # #training
    is_iter_based=getattr(scheduler_config,"is_iter_based",False)
    training_loss_logger=LossLogger(writer, 'train') if is_logging else None
    # print(training_loss_logger)
    # print(is_iter_based)
    if "training_func" in cfg.trainer:
        training_detection=PIPELINE_DICT[cfg.trainer.training_func]
    else:   
        raise KeyError
    # print(training_detection)
        
    print("#########Start of training loop#############")
    def train(detector,optimizer,writer,training_loss_logger,global_step,epoch_num,cfg):
        detector.train()
        for batch_idx,data in tqdm(enumerate(train_dataloader,0),total=len(train_dataloader),smoothing=0.9):
            total_loss=training_detection(data,detector,optimizer,writer,training_loss_logger,global_step,epoch_num,cfg)
            if batch_idx%10==0:
                print("Current loss:",total_loss)
            global_step+=1
            if  is_iter_based:
                scheduler.step()
            if is_logging and global_step%cfg.trainer.disp_iter==0:
                if "total_loss" not in training_loss_logger.loss_stats:
                    print(f"\nIn epoch {epoch_num}, iteration:{batch_idx}, global_step:{global_step}, total_loss not found in logger.")
                else:
                    log_str = 'Epoch: {} | Iteration: {}  | Running loss: {:1.5f}'.format(
                        epoch_num, batch_idx, training_loss_logger.loss_stats['total_loss'].avg)
                    print(log_str, end='\r')
                    writer.add_text("training_log/train", log_str, global_step)
                    training_loss_logger.log(global_step)
        
        if not is_iter_based:
            scheduler.step()
        
        # torch.save(detector.module.state_dict() if is_distributed else detector.state_dict(), os.path.join(cfg.path.checkpoint_path,'{}_{}_latest.pth'.format("detr",cfg.detector.name)))

        if is_logging:
                torch.save(detector.module.state_dict() if is_distributed else detector.state_dict(), os.path.join(cfg.path.checkpoint_path,'{}_{}_latest.pth'.format("smart4_vae",cfg.detector.name)))
        
        if is_logging and (epoch_num + 1) % cfg.trainer.save_iter == 0:
            torch.save(detector.module.state_dict() if is_distributed else detector.state_dict(), os.path.join(cfg.path.checkpoint_path, '{}_{}_{}.pth'.format("smart4_vae",cfg.detector.name,epoch_num)))
        
        if is_distributed:
            torch.distributed.barrier() # wait untill all finish a epoch

        if is_logging:
            writer.flush()
        

    # # function call
    global_step=0
    for epoch_num in range(cfg.trainer.max_epochs):#cfg.trainer.max_epochs):
        print("Epoch number :", epoch_num)
        train(detector,optimizer,writer,training_loss_logger,global_step,epoch_num,cfg)


if __name__ == '__main__':
    Fire(main)