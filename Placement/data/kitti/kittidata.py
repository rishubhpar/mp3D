import os
import math
import numpy as np
from numpy.linalg import inv
import os
from typing import Optional
import numpy as np
from PIL import Image
from numba import jit
from numpy.linalg import inv
import cv2

def read_pc_from_bin(bin_path):
    p=np.fromfile(bin_path,dtype=np.float32).reshape(-1,4)
    return (p)

def read_image(img_path):
    return (np.array(Image.open(img_path,"r")))

@jit(nopython=True,cache=True)
def _leftcam2lidar(pts,Tr_velo_to_cam,R0_rect):
    "Tr_velo_to_cam^{-1}@R0_rect^{-1}@pts_cam"
    hfiller=np.expand_dims(np.ones(pts.shape[0]),axis=1)
    pts_hT=np.ascontiguousarray(np.hstack((pts,hfiller)).T)#(pts,4)
    pts_lidar_T=np.ascontiguousarray(np.linalg.inv(Tr_velo_to_cam))@np.ascontiguousarray(np.linalg.inv(R0_rect))@pts_hT#(pts,4)
    pts_lidar=np.ascontiguousarray(pts_lidar_T.T)
    return (pts_lidar[:,:3])

@jit(nopython=True,cache=True)
def _lidar2leftcam(pts,Tr_velo_to_cam,R0_rect):
    "R0_rect@Tr_velo_to_cam@pts_lidar"
    hfiller=np.expand_dims(np.ones(pts.shape[0]),axis=1)
    pts_hT=np.hstack((pts,hfiller)).T#(pts,4)
    pts_cam_T=R0_rect@Tr_velo_to_cam@pts_hT
    pts_cam=pts_cam_T.T
    return (pts_cam[:,:3])

@jit(nopython=True,cache=True)
def _leftcam2imageplane(pts,P2):
    "P2@pts_cam"
    hfiller=np.expand_dims(np.ones(pts.shape[0]),axis=1)
    pts_hT=np.hstack((pts,hfiller)).T
    pixels_T=P2@pts_hT#(pts,3)
    pixels=pixels_T.T 
    pixels[:,0]/=pixels[:,2]+1e-6
    pixels[:,1]/=pixels[:,2]+1e-6
    return (pixels[:,:2])


# from .utils import read_image, read_pc_from_bin, _leftcam2imgplane
# KITTI

class kitti_calib:
    def __init__(self,calib_path):
        self.path=calib_path
        self.data=None
    
    def read_calib_file(self):
        calib=dict()
        with open(self.path,"r") as f:
            str_list=f.readlines()
        str_list=[itm.rstrip() for itm in str_list if itm!='\n']
        for itm in str_list:
            calib[itm.split(":")[0]]=itm.split(":")[1]
        for k,v in calib.items():
            calib[k]=[float(itm) for itm in v.split()]
        self.data=calib
        self.P2=np.array(self.data["P2"]).reshape(3,4)
        self.P3=np.array(self.data["P3"]).reshape(3,4)
        R0_rect=np.zeros([4,4])
        R0_rect[0:3,0:3]=np.array(self.data["R0_rect"]).reshape(3,3)
        R0_rect[3,3]=1
        self.R0_rect=R0_rect

        Tr_velo_to_cam=np.zeros([4,4])
        Tr_velo_to_cam[0:3,:]=np.array(self.data["Tr_velo_to_cam"]).reshape(3,4)
        Tr_velo_to_cam[3,3]=1
        self.Tr_velo_to_cam=Tr_velo_to_cam
        return (self)
    
    def leftcam2lidar(self,pts):
        if self.data is None:
            # print("Read calin file")
            raise RuntimeError
        return (_leftcam2lidar(pts,self.Tr_velo_to_cam,self.R0_rect))
    
    def lidar2leftcam(self,pts):
        if self.data is None:
            # print("Read calib file")
            raise RuntimeError
        return (_lidar2leftcam(pts,self.Tr_velo_to_cam,self.R0_rect))
    
    def leftcam2imageplane(self,pts):
        "P2@pts_cam"
        if self.data is None:
            # print("Read calin file")
            raise RuntimeError
        return (_leftcam2imageplane(pts,self.P2))

class kitti_label:
    def __init__(self,label_path=None):
        self.path=label_path
        self.data=None
    
    def read_label_file(self,no_dontcare=True):
        self.data=[]
        with open(self.path,"r") as f:
            str_list=f.readlines()
        str_list=[itm.strip() for itm in str_list if itm!='\n']
        for s in str_list:
            self.data.append(kitti_obj(s))
        if no_dontcare:
            self.data=list(filter(lambda obj:obj.type!="DontCare",self.data))
        return (self)
    
    def __str__(self):
        s=''
        for obj in self.data:
            s+=obj.__str__() + "\n"
        return (s)
        
    def equal(self,label,acc_cls,rtol):
        if len(self.data)!=len(label.data):
            return False
        if len(self.data)==0:
            return True
        bool_list=[]
        for obj1 in self.data:
            bool_obj1=False
            for obj2 in label.data:
                bool_obj1=bool_obj1 or obj1.equal(obj2,acc_cls,rtol)
            bool_list.append(bool_obj1)
        return (any(bool_list))

    def isempty(self):
        return (self.data is None or len(self.data)==0)


class kitti_obj():
    def __init__(self, s=None):
        self.type=None
        self.truncated=None
        self.occluded=None
        self.alpha=None
        self.bbox_l=None
        self.bbox_t=None
        self.bbox_r=None
        self.bbox_b=None
        self.h=None
        self.w=None
        self.l=None
        self.x=None
        self.y=None
        self.z=None
        self.ry=None
        self.score=None
        if s is None:
            return
        if len(s.split()) == 15: # data
            self.truncated,self.occluded,self.alpha,self.bbox_l,self.bbox_t,self.bbox_r,self.bbox_b,self.h,self.w,self.l,self.x,self.y,self.z,self.ry=[float(itm) for itm in s.split()[1:]]
            self.type=s.split()[0]
        elif len(s.split()) == 16: # result
            self.truncated,self.occluded,self.alpha,self.bbox_l,self.bbox_t,self.bbox_r,self.bbox_b,self.h,self.w,self.l,self.x,self.y,self.z,self.ry,self.score=[float(itm) for itm in s.split()[1:]]
            self.type=s.split()[0]
        else:
            raise NotImplementedError
        self.height=float(self.bbox_b)-float(self.bbox_t)+1

    def __str__(self):
        if self.score is None:
            return "{} {:.2f} {} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(
                self.type,self.truncated,int(self.occluded),self.alpha,\
                self.bbox_l,self.bbox_t,self.bbox_r,self.bbox_b, \
                self.h,self.w,self.l,self.x,self.y,self.z,self.ry)
        else:
            return "{} {:.2f} {} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(
                self.type, self.truncated,int(self.occluded),self.alpha,\
                self.bbox_l,self.bbox_t,self.bbox_r,self.bbox_b, \
                self.h,self.w,self.l,self.x,self.y,self.z,self.ry,self.score)

class kitti_data:
    def __init__(self,root_dir,idx,output_dict=None):
        self.calib_path=os.path.join(root_dir,"calib",idx+".txt")
        #when training with inpainte images
        # self.image2_path=""#path to inpainted images
        self.image2_path=os.path.join(root_dir,"image_2",idx+".png")
        self.image3_path=os.path.join(root_dir,"image_3",idx+".png")
        self.label2_path=os.path.join(root_dir,"label_2",idx+'.txt')
        self.velodyne_path=os.path.join(root_dir,"velodyne",idx+".bin")
        self.output_dict=output_dict
        if self.output_dict is None:
            self.output_dict={"calib":True,"image":True,"image_3":False,"label":True,"velodyne":True}
    
    def read_data(self):
        calib=kitti_calib(self.calib_path).read_calib_file() if self.output_dict["calib"] else None
        image=read_image(self.image2_path) if self.output_dict["image"] else None
        label=kitti_label(self.label2_path).read_label_file() if self.output_dict["label"] else None
        pc=read_pc_from_bin(self.velodyne_path) if self.output_dict["velodyne"] else None
        if "image_3" in self.output_dict and self.output_dict["image_3"]:
            image_3=read_image(self.image3_path) if self.output_dict["image_3"] else None
            return(calib,image,image_3,label,pc)
        return (calib,image,label,pc)
            


