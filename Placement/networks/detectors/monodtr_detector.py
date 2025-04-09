import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#local imports
from networks.utils.registry import DETECTOR_DICT
from networks.detectors.monodtr_core import monodtrcore
#detection3dhead
#loss
from networks.heads.detection_3dhead import anchordetection3dhead
from networks.heads.depth_loss import bin_depths, DepthFocalLoss
from networks.heads.detr_loss import build_matcher,detr_loss

@DETECTOR_DICT.register_module
class monodtr(torch.nn.Module):
    def __init__(self,cfg):
        super(monodtr,self).__init__()
        # self.fc_mu=torch.nn.Linear(12,1)
        # self.fc_var=torch.nn.Linear(12,1)
        self.obj_types=cfg.obj_types
        self.build_core(cfg)#everything till the depth transformer
        self.cfg=cfg
        self.build_head(cfg)#detection_3dhead
    
    def build_core(self,cfg):
        self.mono_core=monodtrcore(cfg.mono_backbone)
    
    def build_head(self,cfg):
        self.bbox_head=anchordetection3dhead(**(cfg.head))
        self.depth_loss=DepthFocalLoss(96)
        self.bbox_head=anchordetection3dhead(**(cfg.head))    
        # self.depth_loss=depth_focalloss(96)
    
    def train_forward(self,left_images,annotations,P2,depth_gt=None):
        # print("detector")
        _,features,depth=self.mono_core(dict(image=left_images,P2=P2))
        # print("features:",features.shape)
        # print("depth:",depth.shape)
        depth_output=depth
        # print("inputs:",left_images.shape)
        # print("feats:",features.shape)
        # print("depth:",depth.shape)
        # #head
        cls_preds,reg_preds=self.bbox_head(dict(features=features,P2=P2,image=left_images))
        anchors=self.bbox_head.get_anchor(left_images,P2)
        # print("cls preds:",cls_preds.shape)
        # print("reg presds:",reg_preds.shape)
        # print("anchors:",anchors["anchors"].shape)
        # print("anchor mask",anchors["mask"].shape)
        # print("anchor mean",anchors["anchor_mean_std_3d"].shape)
        cls_loss,reg_loss,loss_dict=self.bbox_head.loss(cls_preds,reg_preds,anchors,annotations,P2)
        # pred_boxes=self.bbox_head.loss(cls_preds,reg_preds,anchors,annotations,P2)
        # print(pred_boxes.shape)
        # self.bbox_head.loss(cls_preds,reg_preds,anchors,annotations,P2)
        # print(cls_loss)
        # print(pred_boxes.shape)
        # print(len(np))
        depth_gt=bin_depths(depth_gt,mode="LID",depth_min=1,depth_max=80,num_bins=96,target=True)

        if reg_loss.mean() > 0 and not depth_gt is None and not depth_output is None:
            depth_gt=depth_gt.unsqueeze(1)
            depth_loss=1.0*self.depth_loss(depth_output, depth_gt)
            loss_dict['depth_loss']=depth_loss
            reg_loss+=depth_loss
            self.depth_output=depth_output.detach()
        else:
            loss_dict['depth_loss']=torch.zeros_like(reg_loss)
        return (cls_loss,reg_loss,loss_dict)

    def test_forward(self,left_images,P2):
        assert left_images.shape[0] == 1 # we recommmend image batch size = 1 for testing
        _,features,_=self.mono_core(dict(image=left_images, P2=P2))
        cls_preds,reg_preds=self.bbox_head(dict(features=features,P2=P2,image=left_images))
        anchors=self.bbox_head.get_anchor(left_images,P2)
        scores,bboxes,cls_indexes=self.bbox_head.get_bboxes(cls_preds,reg_preds,anchors,P2,left_images)
        print("scores",scores.shape)
        return (scores,bboxes,cls_indexes)

    
    def forward(self,inputs):
        if isinstance(inputs,list) and len(inputs)>=3:
            print("###################Training phase#################")
            return (self.train_forward(*inputs))
        else:
            print("###################Testing phase#################")
            return (self.test_forward(*inputs))
       


@DETECTOR_DICT.register_module
class monodtr_detr(torch.nn.Module):
    def __init__(self,cfg):
        super(monodtr_detr,self).__init__()
        self.obj_types=cfg.obj_types
        self.linear_box_layer=torch.nn.Sequential(torch.nn.Linear(256,128),torch.nn.Linear(128,11))
        self.linear_cls_layer=torch.nn.Linear(256,4)
        self.build_core(cfg)#everything till the depth transformer
        self.depth_loss=DepthFocalLoss(96)
        self.loss_dict={}
        self.output={}
        
        #build loss
        self.aux_loss=False
        self.matcher=build_matcher()
        self.weight_dict={"loss_ce":1,"loss_bbox":5}
        # self.weight_dict["loss_giou"]=2

        aux_loss=True
        # if self.aux_loss:
        #     self.aux_weight_dict={}
        #     for i in range(3):
        #         self.aux_weight_dict.update({k + f'_{i}': v for k, v in self.weight_dict.items()})
        #     self.weight_dict.update(self.aux_weight_dict)
        # 'labels
        self.losses=['boxes','labels']
        # num_classes=car,van,truck,bg
        self.detr_criterion=detr_loss(num_classes=3,matcher=self.matcher,weight_dict=self.weight_dict,eos_coef=0.1,losses=self.losses)
        self.cfg=cfg
        
    
    def build_core(self,cfg):
        self.mono_core=monodtrcore(cfg.mono_backbone)
    
    def train_forward(self,left_images,annotations,P2,depth_gt=None):
        # print("detector")
        
        self.sizes=[]
        for j in range(annotations.shape[0]):
            # print(annotations[j].shape)
            real_anns=annotations[j][annotations[j][:,4]!=-1]
            # print(real_anns.shape)
            self.sizes.append(real_anns.shape[0])
        # print("anns:",annotations.shape)
        # print("real boxes:",self.sizes)
        self.loss_dict={}
        annotations=annotations[annotations[:,:,4]!=-1]
        # print(annotations)
        dtr_features,features,depth=self.mono_core(dict(image=left_images,P2=P2))
        # print("features:",dtr_features.shape)
        # print("depth:",depth.shape)
        # sys.exit(1)
        depth_output=depth
        depth_gt=bin_depths(depth_gt,mode="LID",depth_min=1,depth_max=80,num_bins=96,target=True)
        box_op=self.linear_box_layer(dtr_features)
        cls_op=self.linear_cls_layer(dtr_features)
        self.output["box"]=box_op
        self.output["class"]=cls_op

        
        if depth_gt is not None and not depth_output is None:
            depth_gt=depth_gt.unsqueeze(1)
            depth_loss=1.0*self.depth_loss(depth_output, depth_gt)
            self.loss_dict['depth_loss']=depth_loss
            self.depth_output=depth_output.detach()
        
        loss_dict=self.detr_criterion(self.output,annotations,self.sizes)
        # print("Final loss dict:",loss_dict)
        depth_gt=bin_depths(depth_gt,mode="LID",depth_min=1,depth_max=80,num_bins=96,target=True)
        depth_gt=depth_gt.unsqueeze(1   )
        depth_loss=1.0*self.depth_loss(depth_output, depth_gt)
        loss_dict['depth_loss']=depth_loss
        self.depth_output=depth_output.detach()
        
        weight_dict=self.detr_criterion.weight_dict
        # print(weight_dict)
        loss_dict={k:v*weight_dict[k] for k, v in loss_dict.items() if k in weight_dict}
        # print(loss_dict)
        class_loss=loss_dict['loss_ce']
        box_loss=loss_dict['loss_bbox']
        box_loss+=depth_loss
        #the ouput of the transformer number of (bs,box queries,hidden_dimension); in our case that would be(bs,n_pixels,hidden_dim),
        # the class fc_output would  
        return (class_loss,box_loss,loss_dict)


    def test_forward(self,left_images,P2):
        assert left_images.shape[0] == 1 # we recommmend image batch size = 1 for testing
        features,_=self.mono_core(dict(image=left_images, P2=P2))
    
    def forward(self,inputs):
        if isinstance(inputs,list) and len(inputs)>=3:
            # print("###################Training phase#################")
            return (self.train_forward(*inputs))
        else:
            print("###################Testing phase#################")
            return (self.test_forward(*inputs))
       