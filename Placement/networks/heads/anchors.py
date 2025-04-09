from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import os
# torch.cuda.set_device(2)



def generate_anchors(base_size=16,ratios=None,scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """
    if ratios is None:
        ratios=np.array([0.5,1,2])
    if scales is None:
        scales=np.array([2**0,2**(1.0/3.0),2**(2.0/3.0)])
    num_anchors=len(ratios)*len(scales)
    anchors=np.zeros((num_anchors,4))#(x_center,y_center,w,h)
    #scale base size
    anchors[:,2:]=base_size*np.tile(scales,(2,len(ratios))).T
    #areas of anchors
    areas=anchors[:,2]*anchors[:,3] #(x_center,y_center,w,h)
    #correct for ratios
    anchors[:,2]=np.sqrt(areas/np.repeat(ratios,len(scales)))
    anchors[:,3]=anchors[:,2]*np.repeat(ratios,len(scales))
    #change format from #(x_center,y_center,w,h) to (l,t,r,b)
    anchors[:,0::2]-=np.tile(anchors[:,2]*0.5,(2, 1)).T
    anchors[:,1::2]-=np.tile(anchors[:,3]*0.5,(2, 1)).T
    
    return(anchors)

def compute_shape(image_shape,pyramid_levels):
    image_shape=np.array(image_shape[:2])
    image_shapes=[(image_shape+2**x-1)//(2**x) for x in pyramid_levels]
    return (image_shapes)

def shift(shape,stride,anchors):
    shift_x=(np.arange(0,shape[1])+0.5)*stride
    shift_y=(np.arange(0,shape[0])+0.5)*stride
    shift_x,shift_y=np.meshgrid(shift_x,shift_y)
    shifts=np.vstack((shift_x.ravel(),shift_y.ravel(),shift_x.ravel(),shift_y.ravel())).transpose()
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A=anchors.shape[0]
    K=shifts.shape[0]
    all_anchors=(anchors.reshape((1,A,4))+shifts.reshape((1,K,4)).transpose((1,0,2)))
    all_anchors=all_anchors.reshape((K*A,4))
    return (all_anchors)

def anchors_for_shape(image_shape,pyramid_levels=None,ratios=None,scales=None,strides=None,sizes=None,shapes_callback=None):
    image_shapes=compute_shape(image_shape,pyramid_levels)
    #compute_all anchors
    all_anchors=np.zeros((0,4))
    for idx,p in enumerate(pyramid_levels):
        anchors=generate_anchors(base_size=sizes[idx],ratios=ratios,scales=scales)
        shifted_anchors=shift(image_shapes[idx],strides[idx],anchors)
        all_anchors=np.append(all_anchors,shifted_anchors,axis=0)
    return (all_anchors)

class Anchors(torch.nn.Module):
    def __init__(self,preprocessed_path,pyramid_levels,strides,sizes,ratios,scales,readConfigFile=1,obj_types=[],filter_anchors=True,filter_y_threshold_min_max=(-0.5,1.8),filter_x_threshold=40.0,anchor_prior_channel=6):
        super(Anchors,self).__init__()
        self.pyramid_levels=pyramid_levels
        self.strides=strides
        self.sizes=sizes
        self.ratios=ratios
        self.scales=scales
        self.shape=None
        self.P2=None
        self.readConfigFile=readConfigFile
        self.scale_step=1/(np.log2(self.scales[1])-np.log2(self.scales[0]))
        if self.readConfigFile:
            self.anchors_mean_original=np.zeros([len(obj_types),len(self.scales)*len(self.pyramid_levels),len(self.ratios),anchor_prior_channel])
            self.anchors_std_original=np.zeros([len(obj_types),len(self.scales)*len(self.pyramid_levels),len(self.ratios),anchor_prior_channel])
            save_dir=os.path.join(preprocessed_path,"training")
            for i in range(len(obj_types)):
                npy_file=os.path.join(save_dir,"anchor_mean_{}.npy".format(obj_types[i]))
                self.anchors_mean_original[i]=np.load(npy_file)#(30,2,6)#[z,  sinalpha, cosalpha, w, h, l,]

                std_file=os.path.join(save_dir,"anchor_std_{}.npy".format(obj_types[i]))
                self.anchors_std_original[i]=np.load(std_file)#(30,2,6)#[z,  sinalpha, cosalpha, w, h, l,]
        self.filter_y_threshold_min_max=filter_y_threshold_min_max
        self.filter_x_threshold=filter_x_threshold

    def anchors2indexes(self,anchors):
        sizes=np.sqrt((anchors[:,2]-anchors[:,0])*(anchors[:,3]-anchors[:,1]))#area
        sizes_diff=sizes-(np.array(self.sizes)*np.array(self.scales))[:,np.newaxis]
        sizes_int=np.argmin(np.abs(sizes_diff),axis=0)

        ratio=(anchors[:,3]-anchors[:,1])/(anchors[:,2]-anchors[:,0])
        ratio_diff=ratio-np.array(self.ratios)[:,np.newaxis]
        ratio_int=np.argmin(np.abs(ratio_diff),axis=0)
        return (sizes_int,ratio_int)
    
    @property
    def num_anchors(self):
        return (len(self.pyramid_levels)*len(self.ratios)*len(self.scales))
    @property
    def num_anchor_per_scale(self):
        return (len(self.ratios)*len(self.scales))
    @staticmethod
    def _deshift_anchors(anchors):
        """shift the anchors to zero base (center of bbox is 0)
        Args:
            anchors: [..., 4] [x1, y1, x2, y2]
        Returns:
            [..., 4] [x1, y1, x2, y2] as with (x1 + x2) == 0 and (y1 + y2) == 0"""
        x1=anchors[...,0]
        y1=anchors[...,1]
        x2=anchors[...,2]
        y2=anchors[...,3]
        center_x=0.5*(x1+x2)
        center_y=0.5*(y1+y2)
        return (torch.stack([x1-center_x,y1-center_y,x2-center_x,y2-center_y],dim=-1))
    
    def forward(self,image,calibs,is_filtering=False):
        shape=image.shape[2:]#(h,w)
        # print("Image shape:",shape)
        if self.shape is None or not(shape==self.shape):
            # print("executing this block")
            self.shape=image.shape[2:] 
            image_shape=image.shape[2:]
            image_shape=np.array(image_shape)
            image_shapes=[(image_shape+2**x-1)//(2**x) for x in self.pyramid_levels]
            # print("Image shapes for pyramid levels:",image_shapes)
            #anchors for all pyramid levels
            all_anchors=np.zeros((0,4)).astype(np.float32)
            for idx,p in enumerate(self.pyramid_levels):
                anchors=generate_anchors(base_size=self.sizes[idx],ratios=self.ratios,scales=self.scales)
                shifted_anchors=shift(image_shapes[idx],self.strides[idx],anchors)
                all_anchors=np.append(all_anchors,shifted_anchors,axis=0)
            if self.readConfigFile:
                sizes_int,ratio_int=self.anchors2indexes(all_anchors)
                self.anchor_means=image.new(self.anchors_mean_original[:,sizes_int,ratio_int])#[types,n,6]
                self.anchor_stds=image.new(self.anchors_std_original[:,sizes_int,ratio_int])#[types,n,6]
                self.anchor_mean_std=torch.stack([self.anchor_means,self.anchor_stds],dim=-1).permute(1,0,2,3)#[n,types,6,2]
            
            all_anchors=np.expand_dims(all_anchors,axis=0)
            if isinstance(image,torch.Tensor):
                self.anchors=image.new(all_anchors.astype(np.float32))#(1,n,4)
            elif isinstance(image,np.ndarray):
                self.anchors=torch.tensor(all_anchors.astype(np.float32)).cuda()
            self.anchors_image_x_center=self.anchors[0,:,0:4:2].mean(dim=1)#n avg of all x 
            self.anchors_image_y_center=self.anchors[0,:,1:4:2].mean(dim=1)#n avg of all y
        
        if calibs is not None and len(calibs)>0:
            P2=calibs#(b,3,4)
            if self.P2 is not None and torch.all(self.P2==P2) and self.P2.shape==P2.shape:
                if self.readConfigFile:
                    return (self.anchors,self.useful_mask,self.anchor_mean_std)
                else:
                    return (self.anchors,self.useful_mask)
            self.P2=P2
            fy=P2[:,1:2,1:2]
            cy=P2[:,1:2,2:3]
            cx=P2[:,0:1,2:3]
            N=self.anchors.shape[1]
            if self.readConfigFile and is_filtering:
                anchors_z=self.anchor_means[:,:,0]
                world_x3d=(self.anchors_image_x_center*anchors_z-anchors_z.new(cx)*anchors_z)/anchors_z.new(fy)#[b,types,n]
                world_y3d=(self.anchors_image_y_center*anchors_z-anchors_z.new(cy)*anchors_z)/anchors_z.new(fy)#[b,types,n]
                self.useful_mask=torch.any((world_y3d > self.filter_y_threshold_min_max[0]) * 
                                            (world_y3d < self.filter_y_threshold_min_max[1]) *
                                            (world_x3d.abs() < self.filter_x_threshold), dim=1)
            else:
                self.useful_mask=torch.ones([len(P2),N],dtype=torch.bool,device="cuda")
            if self.readConfigFile:
                return (self.anchors,self.useful_mask,self.anchor_mean_std)
            else:
                return (self.anchors,self.useful_mask)
        return (self.anchors)

       