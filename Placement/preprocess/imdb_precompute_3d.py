import numpy as np
import os
import pickle   
import time
import cv2
from copy import deepcopy
import skimage.measure
import torch
import sys

from path_init_ import *
# from visualization import *
from networks.heads.anchors import Anchors
from networks.utils.utils import calc_iou,BBox3dProjector
from data.pipeline import build_augmentator
from data.kitti.kittidata import kitti_data
from utils.utils import cfg_from_file

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


def read_one_split(cfg,index_names,data_root_dir,output_dict,data_split="training",time_display_inter=100):
    #read one split
    save_dir=os.path.join(cfg.path.preprocessed_path,data_split)
    print("Saving directory",save_dir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if data_split=="training":
        disp_dir=os.path.join(save_dir,"disp")
        if not os.path.isdir(disp_dir):
            os.mkdir(disp_dir)
        # print("Display directory",disp_dir)
    N=len(index_names)
    # print("{} images:".format(data_split),N)
    frames=[None]*N
    # print("start reading {} data".format(data_split))
    anchor_prior=getattr(cfg,"anchor_prior",True)
    # print("Use anchor boxes:",anchor_prior)
    total_objects=[0 for _ in range(len(cfg.obj_types))]
    # print("Total types of objects:",total_objects)
    total_usable_objects=[0 for _ in range(len(cfg.obj_types))]
    # print("Total usable objects:",total_usable_objects)

    # anchors
    if anchor_prior:
        # print("generating anchors")        
        anchor_manager=Anchors(cfg.path.preprocessed_path,readConfigFile=False,**cfg.detector.head.anchors_cfg)
        # print(anchor_manager.pyramid_levels)
        preprocess=build_augmentator(cfg.data.test_augmentation)
        total_objects=[0 for _ in range(len(cfg.obj_types))]
        total_usable_objects=[0 for _ in range(len(cfg.obj_types))]

        len_scale=len(anchor_manager.scales)
        len_ratios=len(anchor_manager.ratios)
        len_level=len(anchor_manager.pyramid_levels)

        examine=np.zeros([len(cfg.obj_types),len_level*len_scale,len_ratios])
        sums=np.zeros([len(cfg.obj_types),len_level*len_scale,len_ratios,3])
        squared=np.zeros([len(cfg.obj_types),len_level*len_scale,len_ratios,3],dtype=np.float64)

        uniform_sum_each_type=np.zeros((len(cfg.obj_types),6),dtype=np.float64)#[z,sin2a,cos2a,w,h,l] bboxes 
        uniform_square_sum_each_type=np.zeros((len(cfg.obj_types),6),dtype=np.float64)

    for i,index_name in enumerate(index_names):
        # print("###########################################################################################################")
        # print("Image number:",index_name)
        #read data
        data_frame=kitti_data(data_root_dir,index_name,output_dict)
        calib,image,label,velo=data_frame.read_data()
        #store list of kitti and calib
        min_z=getattr(cfg.data,"min_z",2)
        # print("min_z",min_z)
        max_trun=getattr(cfg.data,"max_trun",0.8)
        # print("max_trun:",max_trun)
        min_pix_h=getattr(cfg.data,"min_pix_h",25)
        # print("min_pix_h:",min_pix_h)
        #change parameters for pedestrians 
        if data_split=="training":  
            data_frame.label=[obj for obj in label.data if obj.type in cfg.obj_types and obj.height>=min_pix_h and obj.truncated<=max_trun and obj.z>2] # list of cars in each frame
            # if len(data_frame.label)>0:
            #     # print("Number of labels:",len(data_frame.label))
            #     # print("labels:",data_frame.label)
            if anchor_prior:
                for j in range(len(cfg.obj_types)):
                    total_objects[j]+=len([obj for obj in data_frame.label if obj.type==cfg.obj_types[j]])
                    data=np.array([[obj.z,np.sin(2*obj.alpha),np.cos(2*obj.alpha),obj.w,obj.h,obj.l] for obj in data_frame.label if obj.type==cfg.obj_types[j]])#bbox (why 2*alpha)?
                    if data.any():
                        # print("bbox data:",data)
                        # print("bbox sum:",np.sum(data,a   is=0))
                        # print(data.shape) 
                        uniform_sum_each_type[j,:]+=np.sum(data,axis=0)#(add all x,y,z,sin,cos,w,l of the different objects (x1+x2+..),(y1+y2+...))
                        uniform_square_sum_each_type[j,:]+=np.sum(data**2,axis=0)#(add all square x,y,z,sin,cos,w,l of the different objects (x1+x2+..),(y1+y2+...))
        
        else:
            data_frame.label=[obj for obj in label.data if obj.type in cfg.obj_types]
            # print(data_frame.label)
        data_frame.calib=calib
        if data_split=="training" and anchor_prior:
            original_image=image.copy()
            baseline=(calib.P2[0,3]-calib.P3[0,3])/calib.P2[0,0]
            # P2=deepcopy(calib.P2)
            # label=deepcopy(data_frame.label)
            image,P2,label=preprocess(original_image,p2=deepcopy(calib.P2),labels=deepcopy(data_frame.label))
            _,P3=preprocess(original_image,p2=deepcopy(calib.P3))
            # print(image.max())
            # print(image.min())
            # print(image.shape)
            # print(P2.shape)
            
            #stats for positive anchors (using same anchors for all images)
            if len(data_frame.label)>0:
                # print("Anchors start")
                anchors,_=anchor_manager(image[np.newaxis].transpose([0,3,1,2]),torch.tensor(P2).reshape([-1,3,4]))
                # print("Anchors end")
                for j in range(len(cfg.obj_types)):
                    bbox2d=torch.tensor([[obj.bbox_l,obj.bbox_t,obj.bbox_r,obj.bbox_b] for obj in label if obj.type==cfg.obj_types[j]]).cuda()
                    if len(bbox2d) < 1:
                        continue
                    bbox3d=torch.tensor([[obj.x,obj.y,obj.z,np.sin(2*obj.alpha),np.cos(2*obj.alpha)] for obj in label if obj.type==cfg.obj_types[j]]).cuda()
                    #generate one set of 276480 anchors and then use the postive anchors for each object in each frame.
                    # print(anchors.shape)
                    usable_anchors=anchors[0]#(1,n_anchors,4)
                    # print("all anchors:",usable_anchors.shape)
                    # print("all 3d boxes:",bbox3d.shape)
                    # print("all 2d boxes:",bbox2d.shape)
                    IOus=calc_iou(usable_anchors,bbox2d)#(n,k)
                    # print("ious:",IOus.shape)
                    iou_max,iou_argmax=torch.max(IOus,dim=0)#val and idx of bbox which hvae the highest iou with the anchors
                    #if anchor1 has highest iou with box4  then idx=4. (n_objects)
                    # print("bounding box with highest iou with corresponding anchor box:",iou_max.shape,iou_argmax.shape)
                    IoU_max_anchor,IoU_argmax_anchor=torch.max(IOus,dim=1)#vals ,indices , (n_anchor_boxes) 
                    # print("anchor boxes with highest iou with correspodning bounding box:",IoU_max_anchor.shape,IoU_argmax_anchor.shape)
                    num_usable_object=torch.sum(iou_max>cfg.detector.head.loss_cfg.fg_iou_threshold).item()
                    # print("Only using those bounding boxes whose max iou with their corresponding anchor boxes > 0.5:",num_usable_object)
                    total_usable_objects[j]+=num_usable_object# number at each index represents the total number of that objects that have a anchor box with iou>0.5  in all images [ car,pedestrian]

                    #select those anchors only whose max iou with bbox is greater than 0.5
                    positive_anchors_mask=IoU_max_anchor > cfg.detector.head.loss_cfg.fg_iou_threshold
                    # print("1 Anchors where the highest IOU with corresponding bbox is >0.5 ",positive_anchors_mask.shape)
                    # print(IoU_argmax_anchor.shape)
                    # print(positive_anchors_mask.shape)
                    # print(IoU_argmax_anchor[positive_anchors_mask].shape)
                    # print(IoU_argmax_anchor[positive_anchors_mask])
                    #grabbing the boox3d of those anchors that satisfy above condition
                    positive_ground_truth_3d=bbox3d[IoU_argmax_anchor[positive_anchors_mask]].cpu().numpy()
                    # print("positive anchor boxes",positive_ground_truth_3d.shape)
                    used_anchors=usable_anchors[positive_anchors_mask].cpu().numpy()#(x1,y1,x2,y2)
                    # print("used anchors:",used_anchors.shape)
                    sizes_int,ratio_int=anchor_manager.anchors2indexes(used_anchors)#gives the idx of anchor in a multiscale anchor box setup.
                    
                    for k in range(len(sizes_int)):
                        examine[j,sizes_int[k],ratio_int[k]]+=1
                        #add one at those idx where the anchor is used according to the above condition
                        sums[j,sizes_int[k],ratio_int[k]]+=positive_ground_truth_3d[k,2:5]
                        #add at those idx where the anchor is used according to the above condition
                        squared[j,sizes_int[k],ratio_int[k]]+=positive_ground_truth_3d[k,2:5] ** 2
                        #add at those idx where the anchor is used according to the above condition
        frames[i]=data_frame
        # print("Objects:",total_usable_objects) # number at each index represents the total number of that objects in all images [ car,pedestrian]
    
    save_dir=os.path.join(cfg.path.preprocessed_path,data_split)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if data_split=="training" and anchor_prior:
        for j in range(len(cfg.obj_types)):
            global_mean=uniform_sum_each_type[j]/total_objects[j]
            #mean of (z,sin(2alpha),cos(2alpha),w,h,l) of each bounding box for each object.
            print("Mean of the gt bounding box:",global_mean)
            global_var=np.sqrt(uniform_square_sum_each_type[j]/total_objects[j]-global_mean**2)
            #variance of (z,sin(2alpha),cos(2alpha),w,h,l) of each bounding box for each object.
            print("Variance of the gt bounding box:",global_var)


            #getting the corresponding 3dbox of posiitve anchors and then calc the mean and variance 
            avg=sums[j]/(examine[j][:,:,np.newaxis]+1e-8)
            EX_2=squared[j]/(examine[j][:,:,np.newaxis]+1e-8)
            std=np.sqrt(EX_2-avg**2)
            
            #underflow and overflow stuff
            avg[examine[j]<10,:]=-100
            std[examine[j]<10,:]=1e10
            avg[np.isnan(std)]=-100
            std[np.isnan(std)]=1e10
            avg[std<1e-3]=-100
            std[std<1e-3]=1e10

            whl_avg=np.ones([avg.shape[0],avg.shape[1],3])*global_mean[3:6]
            whl_std=np.ones([avg.shape[0],avg.shape[1],3])*global_var[3:6]
            # why is the whl average for the entire set of objects but the avergae for z,sin(2alpha),cos(2alpha) for those boxes which saatisfy above oncdition?
            avg=np.concatenate([avg,whl_avg],axis=2)
            # print(avg.shape)
            std=np.concatenate([std,whl_std],axis=2)

            npy_file=os.path.join(save_dir,"anchor_mean_{}.npy".format(cfg.obj_types[j]))
            # np.save(npy_file,avg)
            std_file=os.path.join(save_dir,"anchor_std_{}.npy".format(cfg.obj_types[j]))
            # np.save(std_file,std)   
    
    pkl_file=os.path.join(save_dir,"new_imdb_rgb.pkl")
    pickle.dump(frames,open(pkl_file,"wb"))
    print("{} split finished precomputing".format(data_split))


#main
def main(config:str="config/config.py"):
    cfg=cfg_from_file(config)#path for your config file
    print(cfg)
    time_display_inter=100# define the inverval displaying time consumed in loop
    data_root_dir=cfg.path.data_path#root path
    print("Root directory:",data_root_dir)
    calib_path=os.path.join(data_root_dir,"calib")
    list_calib=os.listdir(calib_path)
    N=len(list_calib)#7481
    print("Total list of images",N)
    output_dict={"calib":True,"image":True,"label":True,"velodyne":False}
    print("Sample information dictionary",output_dict)
    #names
    train_names,val_names=process_train_val_file(cfg)
    print("Number of train samples:",len(train_names))
    print("Number of validation samples:",len(val_names))
    read_one_split(cfg,train_names,data_root_dir,output_dict,'training',time_display_inter)
    #the shape of the anchor mean and anchor std is (16,3,6)==> [len(cfg.obj_types),len_level*len_scale,len_ratios]

    output_dict={"calib":True,"image":False,"label":True,"velodyne":False}
    read_one_split(cfg,val_names,data_root_dir,output_dict,"validation",time_display_inter)
    print("Preprocessing finished")

if __name__ == '__main__':
    from fire import Fire
    Fire(main)

