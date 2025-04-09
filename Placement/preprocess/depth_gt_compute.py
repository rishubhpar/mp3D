from tqdm import tqdm
import numpy as np
import os
import pickle
import time
import cv2
from fire import Fire
from typing import List, Dict, Tuple
from copy import deepcopy
import skimage.measure
from numba import jit
import torch

#local imports
from path_init_ import *
from visualization import *
from networks.heads.anchors import Anchors
from networks.utils.utils import calc_iou,BBox3dProjector
from data.pipeline import build_augmentator
from data.kitti.kittidata import kitti_data
from utils.utils import cfg_from_file

@jit(nopython=True,cache=True)
def _lidar2leftcam(pts,Tr_velo_to_cam,R0_rect):
    "R0_rect@Tr_velo_to_cam@pts_lidar"
    hfiller=np.expand_dims(np.ones(pts.shape[0]),axis=1)
    pts_hT=np.hstack((pts,hfiller)).T#(pts,4)
    pts_cam_T=R0_rect@Tr_velo_to_cam@pts_hT
    pts_cam=pts_cam_T.T
    return (pts_cam[:,:3])

@jit(nopython=True,cache=True)
def _leftcam2imgplane(pts,P2):
    "P2@points"
    hfiller=np.expand_dims(np.ones(pts.shape[0]),axis=1)
    pts_hT=np.hstack((pts,hfiller)).T
    pixels_T=P2@pts_hT
    pixels=pixels_T.T
    pixels[:,0]/=pixels[:,2]+1e-6
    pixels[:,1]/=pixels[:,2]+1e-6
    return (pixels[:,:2])


@jit(nopython=True,cache=True)
def generate_depth_from_velo(pc_velo,height,width,Tr_velo_to_cam,R0_rect,P2,base_depth):
    pts_cam=_lidar2leftcam(pc_velo,Tr_velo_to_cam,R0_rect)
    pts_2d=_leftcam2imgplane(pts_cam,P2)
    fov_inds=(pts_2d[:,0]<width-1)&(pts_2d[:,0]>=0)&(pts_2d[:,1]<height-1)&(pts_2d[:,1]>=0)
    fov_inds=fov_inds&(pc_velo[:,0]>2)#select points which are 2 units away from camera
    
    imgfov_pts_2d=pts_2d[fov_inds,:]
    imgfov_pc_rect=pts_cam[fov_inds,:]
    if base_depth is None:
        depth_map=np.zeros((height,width))
    else:
        depth_map=base_depth
    imgfov_pts_2d=imgfov_pts_2d.astype(np.int32)
    for i in range(imgfov_pts_2d.shape[0]):
        depth=imgfov_pc_rect[i,2]
        depth_map[int(imgfov_pts_2d[i,1]),int(imgfov_pts_2d[i,0])]=depth
    return (depth_map)

def denorm(image,rgb_mean,rgb_std):
    image=image*rgb_std+rgb_mean
    image[image>1]=1
    image[image<0]=0
    image*=255
    return (np.array(image,dtype=np.uint8))

def process_train_val_file(cfg):
    train_file=cfg.data.train_split_file
    val_file=cfg.data.val_split_file
    with open(train_file) as f:
        train_lines=f.readlines()
        for i in range(len(train_lines)):
            train_lines[i]=train_lines[i].strip()
    with open(val_file) as f:
        val_lines=f.readlines()
        for i in range(len(val_lines)):
            val_lines[i]=val_lines[i].strip()
    return (train_lines,val_lines)

def compute_depth_for_split(cfg,index_names,data_root_dir,output_dict,data_split,time_display_inter):
    save_dir=os.path.join(cfg.path.preprocessed_path,data_split)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    depth_dir=os.path.join(save_dir,"depth")
    if not os.path.isdir(depth_dir):
        os.mkdir(depth_dir)
    
    N=len(index_names)
    frames=[None]*N
    print("start reading {} data".format(data_split))
    preprocess=build_augmentator(cfg.data.test_augmentation)
    use_right_img=output_dict["image_3"]
    for idx,index_name in tqdm(enumerate(index_names)):
        data_frame=kitti_data(data_root_dir,index_name,output_dict)
    
        if use_right_img:
            calib,image,right_image,label,velo=data_frame.read_data()
        else:
            calib,image,label,velo=data_frame.read_data()
        original_image=image.copy()
        baseline=(calib.P2[0,3]-calib.P3[0,3])/calib.P2[0,0]
        if use_right_img:
            image,image_3,P2,P3=preprocess(original_image,right_image.copy(),p2=deepcopy(calib.P2),p3=deepcopy(calib.P3))
        else:
            image,P2=preprocess(original_image,p2=deepcopy(calib.P2))
        
        print(calib.P2.shape)
        depth_left=generate_depth_from_velo(velo[:,0:3],image.shape[0],image.shape[1],calib.Tr_velo_to_cam,calib.R0_rect,P2,None)
        depth_left=skimage.measure.block_reduce(depth_left,(4,4),np.max)
        file_name=os.path.join(depth_dir,"P2%06d.png"%idx)
        # cv2.imwrite(file_name,depth_left)
        
        if use_right_img:
            depth_right=generate_depth_from_velo(velo[:,0:3],image.shape[0],image.shape[1],calib.Tr_velo_to_cam,calib.R0_rect,P3,None)
            depth_right=skimage.measure.block_reduce(depth_right,(4,4),np.max)
            file_name=os.path.join(depth_dir,"P3%06d.png"%idx)
            # cv2.imwrite(file_name,depth_right)

    print("{} split finished precomputing depth".format(data_split))



cfg=cfg_from_file("")#path for your config file
torch.cuda.set_device(cfg.trainer.gpu)
time_display_inter=100
data_root_dir=cfg.path.data_path
calib_path=os.path.join(data_root_dir,"calib")
list_calib=os.listdir(calib_path)
N=len(list_calib)
use_right_img=False
output_dict={"calib":True,"image":True,"image_3":use_right_img,"label":False,"velodyne":True}
train_names,val_names=process_train_val_file(cfg)
compute_depth_for_split(cfg,train_names,data_root_dir,output_dict,"training",time_display_inter)
print("Depth Preprocessing finished")

