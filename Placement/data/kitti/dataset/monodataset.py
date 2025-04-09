import sys
import os
import torch
import numpy as np
import random

import copy
import csv
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.utils.data
import os
import pickle
import numpy as np
from copy import deepcopy
import sys
from matplotlib import pyplot as plt
import cv2
import math

#local imports
from networks.utils.registry import DATASET_DICT
from networks.utils import BBox3dProjector
from utils.utils import alpha2theta_3d, theta2alpha_3d
from data.kitti.kittidata import kitti_data,kitti_obj,kitti_calib
from data.pipeline import build_augmentator


def add_scaled_jitter(label, jitter):
    temp_label = label
    l = temp_label.l
    w = temp_label.w
    if jitter:
        temp_label.x += w*np.random.uniform(-1.0, 1.0)
        temp_label.z += 2*l*np.random.uniform(-3.0, 3.0) + l//2
    return temp_label

def add_random_jitter(bbox3d, jitter):
    if jitter:
        for i in range(bbox3d.shape[0]):
            locs = bbox3d[i,:]
            locs[0]+=np.random.normal(0.0, 1.0)
            locs[2]+=np.random.normal(0.0, 1.0)
            bbox3d[i,:] = locs

    return bbox3d

def add_jitter(label, jitter):
    temp_label = label
    if jitter:
        temp_label.x += np.random.normal(0.0, 1.0)
        temp_label.z += np.random.normal(0.0, 1.0)
    return temp_label

def threshold_box(bbox, bbox_all, refidx, orientation, thresh):
    
    xref, zref = bbox.x, bbox.z
    wref, lref = bbox.w, bbox.l
    thetaref = bbox.ry
    
    neighbour_boxes = []
    for box_idx in range(len(bbox_all)):
        if box_idx != refidx and orientation[0]<math.degrees(bbox_all[box_idx].ry - thetaref)<orientation[1]:

            x, z = bbox_all[box_idx].x, bbox_all[box_idx].z

            dist = np.sqrt((x-xref)**2 + (z-zref)**2)

            if dist < thresh*lref:
            #if dist < np.inf:
                neighbour_boxes.append(bbox_all[box_idx])

    return neighbour_boxes

def convex_interp(ref_box, neighbour_boxes:list):
    K = len(neighbour_boxes)   # Number of Neighbours
    a = np.random.uniform(low=0.0, high=1.0, size=K)
    a /= np.sum(a)

    if K>0:
        ref_point = np.array([ref_box.x, ref_box.y, ref_box.z])
        convex_interpolated = np.zeros_like(ref_point)
        for k in range(K):
            neighbour = neighbour_boxes[k]
            neighbour_point = np.array([neighbour.x, neighbour.y, neighbour.z])
            convex_interpolated += a[k]*neighbour_point

        b = np.random.uniform(low=0.0, high=1.0, size=2)
        b /= np.sum(b)
        interp_2d = b[0]*ref_point + b[1]*convex_interpolated

        # Interpolated box
        interp_box = copy.deepcopy(ref_box)
        interp_box.x, interp_box.y, interp_box.z = interp_2d[0], interp_2d[1], interp_2d[2]
        return interp_box 
    else:
        return None




# def add_random_jitter(bbox3d,jitter):
#     if jitter:
#         for i in range(bbox3d.shape[0]):
#             locs=bbox3d[i,:]
#             locs[0]+=np.random.normal(0.0, 1.0)
#             locs[2]+=np.random.normal(0.0, 1.0)
#             bbox3d[i,:]=locs
#     return (bbox3d)

"""side note on alpha and roty  
    roty:is the  yaw angle,the angle of the between the object frame axes(drawn by taking the bbox center as origin) and camera axis.
    roty is the angle between the y-axis of the camera system and the object
    alpha:is the angle at which object was observed.
"""


@DATASET_DICT.register_module
class kitti_monodataset(torch.utils.data.Dataset):
    def __init__(self,cfg,split="training"):
        super(kitti_monodataset,self).__init__()
        preprocessed_path=cfg.path.preprocessed_path
        obj_types=cfg.obj_types
        is_train=(split=="training")
        if split=="training":
            imdb_file_path=os.path.join(preprocessed_path,split,"new_imdb_rgb.pkl")
        else:
            imdb_file_path=os.path.join(preprocessed_path,split,"new_imdb_rgb.pkl")
        # print(imdb_file_path)
        p=open(imdb_file_path,"rb")
        self.imdb=pickle.load(open(imdb_file_path,"rb"))
        self.output_dict={"calib":False,"image":True,"label":False,"velodyne":False}
        if is_train:
            self.transform=build_augmentator(cfg.data.train_augmentation)
        else:
            self.transform=build_augmentator(cfg.data.test_augmentation)
        self.projector=BBox3dProjector()
        self.is_train=is_train
        self.obj_types=obj_types
        self.jitter=cfg.jitter
        self.orientation_thresh=cfg.orientation_thresh
        self.box_thresh=cfg.box_thresh
        self.use_right_image=False
        self.is_reproject=getattr(cfg.data,"is_reproject",True) # if reproject 2d
        self.preprocessed_path=preprocessed_path
    
    def _reproject(self,P2,transformed_label):
        original_boxes=np.zeros([len(transformed_label),7])#[x,y,z,w,h,l,alpha]
        bbox3d_state=np.zeros([len(transformed_label),7])#[camera_x, camera_y, z, w, h, l, alpha]
        for obj in transformed_label:
            obj.alpha=theta2alpha_3d(obj.ry,obj.x,obj.z,P2) # check this function
            #this function calculates local yaw from global yaw. (theta=theta_{l}+theta_{ray})
            #theta_{ray} is tan^{-1}(x/z)

            #mean of x,y,z
        # ring the KITTI center up to the middle of the object
        bbox3d_origin=torch.tensor([[obj.x,obj.y-0.5*obj.h,obj.z,obj.w,obj.h,obj.l,obj.alpha] for obj in transformed_label],dtype=torch.float32) # why subtract they y value
        abs_corner,homo_corner,_=self.projector(bbox3d_origin,bbox3d_origin.new(P2))
        for i,obj in enumerate(transformed_label):
            extended_center=np.array([obj.x,obj.y-0.5*obj.h,obj.z,1])[:,np.newaxis]#(4,1)
            extended_bottom=np.array([obj.x,obj.y,obj.z,1])[:,np.newaxis]#(4,1)
            image_center=(P2@extended_center)[:,0] # transform image center to left camera system (3,)
            image_center[0:2]/=image_center[2]#rectify center 
            image_bottom=(P2@extended_bottom)[:,0] # transform image bottom to left camera system (3,)
            image_bottom[0:2]/=image_bottom[2]#rectify bottom
            bbox3d_state[i]=np.concatenate([image_center,[obj.w,obj.h,obj.l,obj.alpha]])#7,1
            # print("herererere",image_center.shape)
            # print("shabasba",extended_bottom.shape)
            original_boxes[i]=np.concatenate([np.array([obj.x,obj.y,obj.z]),[obj.w,obj.h,obj.l,obj.ry]])#7,1
        
        max_xy,_=homo_corner[:,:,0:2].max(dim=1)
        min_xy,_=homo_corner[:,:,0:2].min(dim=1)
        result=torch.cat([min_xy,max_xy],dim=-1)#(:,4)(l,t,b,r)
        bbox_2d=result.cpu().numpy()
        if self.is_reproject:
            for i in range(len(transformed_label)):
                transformed_label[i].bbox_l=bbox_2d[i,0]
                transformed_label[i].bbox_t=bbox_2d[i,1] 
                transformed_label[i].bbox_r=bbox_2d[i,2]
                transformed_label[i].bbox_b=bbox_2d[i,3]

        return (transformed_label,bbox3d_state,original_boxes)
    
    def __len__(self):
        if self.is_train and self.use_right_image:
            return (len(self.imdb)*2)
        else:
            return (len(self.imdb))


    def __getitem__(self,idx):
        kitti_data=self.imdb[idx%len(self.imdb)]
        if idx>=len(self.imdb):
            kitti_data.output_dict={"calib":True,"image":False,"image_3":True,"label":False,"velodyne":False}
            calib,_,image,_,_=kitti_data.read_data()
            calib.P2=calib.P3
        else:
            kitti_data.output_dict=self.output_dict
            _,image,_,_=kitti_data.read_data()
            calib=kitti_data.calib
        calib.image_shape=image.shape
        label=kitti_data.label
        label=[]
        for obj in kitti_data.label:
            if obj.type in self.obj_types:
                label.append(obj)
        # print("ori len:",len(label))
        return_label = label.copy()
        
        ## AUGMENTATION
       
        # SETTING 4
        query_label = copy.deepcopy(label)
        for box_idx in range(len(query_label)):
            #if math.degrees(label[box_idx].ry) > orientation[0] and math.degrees(label[box_idx].ry) < orientation[1]:
            neighbour_boxes = threshold_box(query_label[box_idx], query_label, box_idx, self.orientation_thresh, thresh=self.box_thresh)
            interpolated_box = convex_interp(query_label[box_idx], neighbour_boxes)
            if interpolated_box is not None:
                label.append(interpolated_box)
                # label[box_idx] = interpolated_box
            else:
                temp_box = copy.deepcopy(query_label[box_idx])
                label.append(add_scaled_jitter(temp_box, self.jitter))
                # label[box_idx] = add_scaled_jitter(temp_box, self.jitter)
                del temp_box
        # AUGMENTATION                
        
        # print("augm len:",len(label))
        transformed_image,transformed_P2,transformed_label=self.transform(image,p2=deepcopy(calib.P2),labels=deepcopy(label))
        bbox3d_state=np.zeros([len(transformed_label),7]) #[camera_x, camera_y, z, w, h, l, alpha]
        if len(transformed_label)>0:
            transformed_label,bbox3d_state,ori_boxes=self._reproject(transformed_P2,transformed_label)
        ori_p2=calib.P2
        if self.is_train:
            if(ori_p2[0,3]*transformed_P2[0,3])>=0:#if not mirrored or swaped, depth should base on pointclouds projecting through P2
                depth=cv2.imread(os.path.join(self.preprocessed_path,"training","depth","P2%06d.png"%idx),-1)
            else:# mirrored and swap, depth should base on pointclouds projecting through P3, and also mirrored
                depth=cv2.imread(os.path.join(self.preprocessed_path,"training","depth","P2%06d.png"%idx),-1)
                depth=depth[:,::-1]
                # print("Inside dataset:",depth.shape)
        else:
            depth=None
        bbox2d=np.array([[obj.bbox_l,obj.bbox_t,obj.bbox_r,obj.bbox_b] for obj in transformed_label])
        # bbox3d_state=add_random_jitter(bbox3d_state,jitter=True)
        output_dict={"calib":transformed_P2,"image":transformed_image,"label":[obj.type for obj in transformed_label],"bbox2d":bbox2d,"bbox3d":bbox3d_state,"original_shape":image.shape,"depth":depth,"original_P":calib.P2.copy()}
        return (output_dict)
    
    @staticmethod
    def collate_fn(batch):
        rgb_images=np.array([item["image"] for item in batch])#(bs,h,w,3)
        rgb_images=rgb_images.transpose([0,3,1,2])
        calib=[item["calib"] for item in batch]
        label=[item["label"] for item in batch]
        bbox2ds=[item["bbox2d"] for item in batch]
        bbox3ds=[item["bbox3d"] for item in batch]
        # ori_boxes=[item["original_boxes"] for item in batch]
        depths=[item["depth"] for item in batch]
        if depths[0] is None:
            return (torch.from_numpy(rgb_images).float(),torch.tensor(calib).float(),label,bbox2ds,bbox3ds)#val
        else:
            return (torch.from_numpy(rgb_images).float(),torch.tensor(calib).float(),label,bbox2ds,bbox3ds,torch.tensor(depths).float())



@DATASET_DICT.register_module
class kitti_detrdataset(torch.utils.data.Dataset):
    def __init__(self,cfg,split="training"):
        super(kitti_detrdataset,self).__init__()
        preprocessed_path=cfg.path.preprocessed_path
        obj_types=cfg.obj_types
        is_train=(split=="training")
        if split=="training":
            imdb_file_path=os.path.join(preprocessed_path,split,"new_imdb.pkl")
        else:
            imdb_file_path=os.path.join(preprocessed_path,split,"new_imdb_v2.pkl")
        # print(imdb_file_path)
        p=open(imdb_file_path,"rb")
        self.imdb=pickle.load(open(imdb_file_path,"rb"))
        self.output_dict={"calib":False,"image":True,"label":False,"velodyne":False}
        if is_train:
            self.transform=build_augmentator(cfg.data.train_augmentation)
        else:
            self.transform=build_augmentator(cfg.data.test_augmentation)
        self.projector=BBox3dProjector()
        self.is_train=is_train
        self.obj_types=obj_types
        self.use_right_image=False
        self.is_reproject=getattr(cfg.data,"is_reproject",True) # if reproject 2d
        self.preprocessed_path=preprocessed_path
    
    def _reproject(self,P2,transformed_label):
        bbox3d_state=np.zeros([len(transformed_label),7])#[camera_x, camera_y, z, w, h, l, alpha]
        for obj in transformed_label:
            obj.alpha=theta2alpha_3d(obj.ry,obj.x,obj.z,P2) # check this function
            #this function calculates local yaw from global yaw. (theta=theta_{l}+theta_{ray})
            #theta_{ray} is tan^{-1}(x/z)

            #mean of x,y,z
        bbox3d_origin=torch.tensor([[obj.x,obj.y-0.5*obj.h,obj.z,obj.w,obj.h,obj.l,obj.alpha] for obj in transformed_label],dtype=torch.float32) # why subtract they y value
        abs_corner,homo_corner,_=self.projector(bbox3d_origin,bbox3d_origin.new(P2))
        for i,obj in enumerate(transformed_label):
            extended_center=np.array([obj.x,obj.y-0.5*obj.h,obj.z,1])[:,np.newaxis]#(4,1)
            extended_bottom=np.array([obj.x,obj.y,obj.z,1])[:,np.newaxis]#(4,1)
            image_center=(P2@extended_center)[:,0] # transform image center to left camera system (3,)
            image_center[0:2]/=image_center[2]#rectify center 
            image_bottom=(P2@extended_bottom)[:,0] # transform image bottom to left camera system (3,)
            image_bottom[0:2]/=image_bottom[2]#rectify bottom
            bbox3d_state[i]=np.concatenate([image_center,[obj.w,obj.h,obj.l,obj.alpha]])#7,1
        
        max_xy,_=homo_corner[:,:,0:2].max(dim=1)
        min_xy,_=homo_corner[:,:,0:2].min(dim=1)
        result=torch.cat([min_xy,max_xy],dim=-1)#(:,4)(l,t,b,r)
        bbox_2d=result.cpu().numpy()
        if self.is_reproject:
            for i in range(len(transformed_label)):
                transformed_label[i].bbox_l=bbox_2d[i,0]
                transformed_label[i].bbox_t=bbox_2d[i,1] 
                transformed_label[i].bbox_r=bbox_2d[i,2]
                transformed_label[i].bbox_b=bbox_2d[i,3]

        return (transformed_label,bbox3d_state)
    
    def __len__(self):
        if self.is_train and self.use_right_image:
            return (len(self.imdb)*2)
        else:
            return (len(self.imdb))


    def __getitem__(self,idx):
        kitti_data=self.imdb[idx%len(self.imdb)]
        if idx>=len(self.imdb):
            kitti_data.output_dict={"calib":True,"image":False,"image_3":True,"label":False,"velodyne":False}
            calib,_,image,_,_=kitti_data.read_data()
            calib.P2=calib.P3
        else:
            kitti_data.output_dict=self.output_dict
            _,image,_,_=kitti_data.read_data()
            calib=kitti_data.calib
        calib.image_shape=image.shape
        label=kitti_data.label
        label=[]
        for obj in kitti_data.label:
            if obj.type in self.obj_types:
                label.append(obj)
            
        transformed_image,transformed_P2,transformed_label=self.transform(image,p2=deepcopy(calib.P2),labels=deepcopy(label))
        bbox3d_state=np.zeros([len(transformed_label),7]) #[camera_x, camera_y, z, w, h, l, alpha]
        if len(transformed_label)>0:
            transformed_label,bbox3d_state=self._reproject(transformed_P2,transformed_label)
        ori_p2=calib.P2
        if self.is_train:
            if(ori_p2[0,3]*transformed_P2[0,3])>=0:#if not mirrored or swaped, depth should base on pointclouds projecting through P2
                depth=cv2.imread(os.path.join(self.preprocessed_path,"training","depth","P2%06d.png"%idx),-1)
            else:# mirrored and swap, depth should base on pointclouds projecting through P3, and also mirrored
                depth=cv2.imread(os.path.join(self.preprocessed_path,"training","depth","P2%06d.png"%idx),-1)
                depth=depth[:,::-1]
                # print("Inside dataset:",depth.shape)
        else:
            depth=None
        bbox2d=np.array([[obj.bbox_l,obj.bbox_t,obj.bbox_r,obj.bbox_b] for obj in transformed_label])
        bbox3d_state=add_random_jitter(bbox3d_state,jitter=True)
        output_dict={"calib":transformed_P2,"image":transformed_image,"label":[obj.type for obj in transformed_label],"bbox2d":bbox2d,"bbox3d":bbox3d_state,"original_shape":image.shape,"depth":depth,"original_P":calib.P2.copy()}
        return (output_dict)


    @staticmethod
    def collate_fn(batch):
        rgb_images=np.array([item["image"] for item in batch])#(bs,h,w,3)
        rgb_images=rgb_images.transpose([0,3,1,2])
        calib=[item["calib"] for item in batch]
        label=[item["label"] for item in batch]
        bbox2ds=[item["bbox2d"] for item in batch]
        bbox3ds=[item["bbox3d"] for item in batch]
        depths=[item["depth"] for item in batch]
        if depths[0] is None:
            return (torch.from_numpy(rgb_images).float(),torch.tensor(calib).float(),label,bbox2ds,bbox3ds)#val
        else:
            return (torch.from_numpy(rgb_images).float(),torch.tensor(calib).float(),label,bbox2ds,bbox3ds,torch.tensor(depths).float())





@DATASET_DICT.register_module
class kitti_monotestdataset(kitti_monodataset):
    def __init__(self,cfg,split="test"):
        preprecessed_path=cfg.preprocessed_path
        obj_types=cfg.obj_types
        super(kitti_monotestdataset,self).__init__(cfg,"test")
        is_train=(split=="training")
        imdb_file_path=os.path.join(preprecessed_path,"test","new_imdb.pkl")
        self.imdb=pickle.load(open(imdb_file_path,"rb"))
        self.output_dict={"calib":False,"image":True,"label":False,"velodyne":False}
    
    def __getitem__(self,idx):
        kitti_data=self.imdb[idx%len(self.imdb)]
        # print("this one ")
        kitti_data.output_dict=self.output_dict
        _,image,_,_=kitti_data.read_data()
        calib=kitti_data.calib
        calib.image_shape=image.shape
        transformed_image,transformed_P2=self.transform(image,P2=deepcopy(calib.P2))
        output_dict={"calib":transformed_P2,"image":transformed_image,"original_shape":image.shape,"original_P":calib.P2.copy()}
        return (output_dict)
    
    @staticmethod
    def collate_fn(bacth):
        rgb_images=np.array([item["image"] for item in bacth])#(b,h,w,c)
        rgb_images=rgb_images.transpose(0,3,1,2)#(b,c,h,w)
        calib=[item["calib"] for item in bacth]
        return (torch.from_numpy(rgb_images).float(),calib)



