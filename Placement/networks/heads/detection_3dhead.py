
import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.ops import nms
from easydict import EasyDict
import numpy as np
from typing import List, Tuple, Dict
from shapely.geometry import Polygon
# from pytorch3d.ops import box3d_overlap


#local imports
from networks.heads.anchors import Anchors
from networks.utils.utils import calc_iou,BackProjection,BBox3dProjector
from networks.lib.fast_utils.hill_climbing import post_opt
from networks.utils.utils import ClipBoxes
from networks.lib.blocks import AnchorFlatten
from networks.lib.ops import ModulatedDeformConvPack
from networks.heads.losses import SigmoidFocalLoss,ModifiedSmoothL1Loss,ModifiedHuberLoss,dynamic_huber



class anchordetection3dhead(torch.nn.Module):
    def __init__(self,num_features_in=1024,num_classes=3,num_regression_loss_terms=12,preprocessed_path=" ",anchors_cfg=EasyDict(),layer_cfg=EasyDict(),loss_cfg=EasyDict(),test_cfg=EasyDict(),read_precompute_anchor=True):
        super(anchordetection3dhead,self).__init__()
        
        self.anchors=Anchors(preprocessed_path=preprocessed_path,readConfigFile=read_precompute_anchor,**anchors_cfg)
        
        self.num_classes=num_classes
        self.num_regression_loss_terms=num_regression_loss_terms
        self.decode_before_loss=getattr(loss_cfg,"decode_before_loss",False)
        self.loss_cfg=loss_cfg
        self.test_cfg=test_cfg
        self.build_loss(**loss_cfg)
        self.backprojector=BackProjection()
        self.clipper=ClipBoxes()
        if getattr(layer_cfg,"num_anchors",None) is None:
            layer_cfg["num_anchors"]=self.anchors.num_anchors
        # print(layer_cfg)
        self.init_layers(**layer_cfg)

        #vae layers encoder and decoder
        self.n_boxes=8
        self.latent_dim=256
        self.lamda=0.5
        self.fc_mean=torch.nn.Linear(12,self.latent_dim)
        self.fc_logvar=torch.nn.Linear(12,self.latent_dim)
        self.decoder=torch.nn.Linear(self.latent_dim,12)

    
    def init_layers(self,num_features_in,num_anchors,num_cls_output,num_reg_output,cls_feature_size=1024,reg_feature_size=1024,**kwargs):
        
        self.cls_feature_extraction=torch.nn.Sequential(torch.nn.Conv2d(num_features_in,cls_feature_size,kernel_size=3,padding=1),torch.nn.Dropout2d(0.3),torch.nn.ReLU(inplace=True),torch.nn.Conv2d(cls_feature_size,cls_feature_size,kernel_size=3,padding=1),torch.nn.Dropout2d(0.3),torch.nn.ReLU(inplace=True),torch.nn.Conv2d(cls_feature_size,num_anchors*(num_cls_output),kernel_size=3,padding=1),AnchorFlatten(num_cls_output))#anchorflatten : [B, num_anchors * output_channel, H, W] ==> [B, num_anchors * H * W, output_channel] anchor boxes for every channel
        self.cls_feature_extraction[-2].weight.data.fill_(0)
        self.cls_feature_extraction[-2].bias.data.fill_(0)

        self.reg_feature_extraction=torch.nn.Sequential(ModulatedDeformConvPack(num_features_in,reg_feature_size,3,padding=1),torch.nn.BatchNorm2d(reg_feature_size),torch.nn.ReLU(inplace=True),torch.nn.Conv2d(reg_feature_size,reg_feature_size,kernel_size=3,padding=1),torch.nn.BatchNorm2d(reg_feature_size),torch.nn.ReLU(inplace=True),torch.nn.Conv2d(reg_feature_size,num_anchors*(num_reg_output),kernel_size=3,padding=1),AnchorFlatten(num_reg_output))#anchorflatten : [B, num_anchors * output_channel, H, W] ==> [B, num_anchors * H * W, output_channel] anchor boxes for every channel
        self.reg_feature_extraction[-2].weight.data.fill_(0)
        self.reg_feature_extraction[-2].bias.data.fill_(0)

    def forward(self, inputs):
        #after getting the depth and image features they pass it through a classification and regression head to get a set of features.
        # features have a size of (bs,n_anchors*channel,h,w)  
        cls_preds=self.cls_feature_extraction(inputs['features'])
        # print("cls_preds:",cls_preds.shape)
        reg_preds=self.reg_feature_extraction(inputs['features'])
        # print("box reg preds:",reg_preds.shape)
        return (cls_preds,reg_preds)
    



    def _assign(self,anchor,annotation,bg_iou_threshold=0.0,fg_iou_threshold=0.5,min_iou_threshold=0.0,match_low_quality=True,gt_max_assign_all=True,**kwargs):
        """anchor:[n,4],annotations[n_gt,4]"""
        # print("######################")
        # print("inside assign function")
        N=anchor.shape[0]
        # print("posiitve anchors",N)
        num_gt=annotation.shape[0]
        # print("valid unpadded boxes:",num_gt)
        assigned_gt_inds=anchor.new_full((N,),-1,dtype=torch.long)#indices for each anchor.
        # print("assigned gt indices to each anchor",assigned_gt_inds.shape)
        max_overlaps=anchor.new_zeros((N,))
        # print("max overlaps for each anchor box",max_overlaps.shape)
        assigned_labels=anchor.new_full((N,),-1,dtype=torch.long)#indices of assigned labels for each anchor.
        # print("indices of assigned labels",assigned_labels.shape)
    

        if num_gt==0:
            assigned_gt_inds=anchor.new_full((N,),0,dtype=torch.long)
            return_dict=dict(num_gt=num_gt,assigned_gt_inds=assigned_gt_inds,max_overlaps=max_overlaps,labels=assigned_labels)
            return(return_dict)
        IoU=calc_iou(anchor,annotation[:,:4])
        # print("iou between anchors and gt boxes:",IoU.shape)#(n_anchors,n_gtbox) each anchor will have an iou with each valid gt box.
        #max anchor
        max_overlaps,argmax_ovrelaps=IoU.max(dim=1)#iou value and index of gt box with which every anchor box has highest iou value.  
        #val and idx of anchor which hvae the highest iou with the gtbox
        # print("max overlaps",max_overlaps.shape)    
        # print("idx max overlaps",argmax_ovrelaps.shape)
        #if anchor1 has highest iou with box4  then idx=4 and the iou value with box4#anchor boxes with highest iou with correspodning bounding box
        gt_max_overlaps,gt_argmax_overlaps=IoU.max(dim=0)#iou value and idx of the anchor box with which the gt box has highest iou
        # print("gt max overlap",gt_max_overlaps.shape)
        # print("idx gt max overlaps:",gt_argmax_overlaps.shape)

        #bounding box with highest iou with corresponding anchor box(n_gt)
        #neg
        assigned_gt_inds[(max_overlaps>=0)&(max_overlaps<bg_iou_threshold)]=0
        #pos
        pos_inds=max_overlaps>=fg_iou_threshold #select those boxes which whose max threshold is above
        assigned_gt_inds[pos_inds]=argmax_ovrelaps[pos_inds]+1 #idx of the gt box of pos anchor boxes

        # print("array of 1 and 0 for positive and negative anchors:",assigned_gt_inds.shape)
        # print("checking indices with 1:",torch.where(assigned_gt_inds==1.))
        # print("checking indices with 0:",torch.where(assigned_gt_inds==0.))

        if match_low_quality:
            for i in range(num_gt):
                if gt_max_overlaps[i]>=min_iou_threshold:
                    if gt_max_assign_all:
                        max_iou_inds=IoU[:,i]==gt_argmax_overlaps[i]
                        assigned_gt_inds[max_iou_inds]=i+1
                    else:
                        assigned_gt_inds[gt_argmax_overlaps[i]]=i+1
        assigned_labels=assigned_gt_inds.new_full((N,),-1)
        pos_inds=torch.nonzero(assigned_gt_inds>0, as_tuple=False).squeeze()
        if pos_inds.numel()>0:
            # print(annotation[assigned_gt_inds[pos_inds]-1,4].long())
            assigned_labels[pos_inds]=annotation[assigned_gt_inds[pos_inds]-1,4].long()#4 because class idx is at 4
        return_dict=dict(num_gt=num_gt,assigned_gt_inds=assigned_gt_inds,max_overlaps=max_overlaps,labels=assigned_labels)
        # print("num_gt",num_gt)
        # print("assigned_gt_inds:",torch.unique(assigned_gt_inds)) # idx of the box it belongs to.[-1,1,2,3...... n_box]
        # print("max overlaps",max_overlaps.shape) # overlaps for each anchor.
        # print("assigned labels",torch.unique(assigned_labels)) # class idx for each anchor box. [-1(bg),0(car)]
        # print("### end of assign function")
        return (return_dict)
    
    def build_loss(self,focal_loss_gamma=0.0, balance_weight=[0], L1_regression_alpha=9,**kwargs):
        self.focal_loss_gamma = focal_loss_gamma
        self.register_buffer("balance_weights", torch.tensor(balance_weight, dtype=torch.float32))
        self.loss_cls = SigmoidFocalLoss(gamma=focal_loss_gamma, balance_weights=self.balance_weights)
        self.loss_bbox = ModifiedSmoothL1Loss(L1_regression_alpha)
        # self.loss_huber_bbox=torch.nn.HuberLoss(reduction="none")
        self.loss_mod_huber=ModifiedHuberLoss(0.5)
        self.dynamic_huber=dynamic_huber()
        # print("Using this loss",self.loss_mod_huber)
        regression_weight = kwargs.get("regression_weight", [1 for _ in range(self.num_regression_loss_terms)]) #default 12 only use in 3D
        self.register_buffer("regression_weight", torch.tensor(regression_weight, dtype=torch.float))
        # print("#######################################################3",regression_weight)
        self.alpha_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def _encode(self, sampled_anchors, sampled_gt_bboxes, selected_anchors_3d):
        assert sampled_anchors.shape[0] == sampled_gt_bboxes.shape[0]

        sampled_anchors = sampled_anchors.float()
        sampled_gt_bboxes = sampled_gt_bboxes.float()
        px = (sampled_anchors[..., 0] + sampled_anchors[..., 2]) * 0.5
        py = (sampled_anchors[..., 1] + sampled_anchors[..., 3]) * 0.5
        pw = sampled_anchors[..., 2] - sampled_anchors[..., 0]
        ph = sampled_anchors[..., 3] - sampled_anchors[..., 1]

        gx = (sampled_gt_bboxes[..., 0] + sampled_gt_bboxes[..., 2]) * 0.5
        gy = (sampled_gt_bboxes[..., 1] + sampled_gt_bboxes[..., 3]) * 0.5
        gw = sampled_gt_bboxes[..., 2] - sampled_gt_bboxes[..., 0]
        gh = sampled_gt_bboxes[..., 3] - sampled_gt_bboxes[..., 1]

        targets_dx = (gx - px) / pw
        targets_dy = (gy - py) / ph
        targets_dw = torch.log(gw / pw)
        targets_dh = torch.log(gh / ph)

        targets_cdx = (sampled_gt_bboxes[:, 5] - px) / pw
        targets_cdy = (sampled_gt_bboxes[:, 6] - py) / ph

        targets_cdz = (sampled_gt_bboxes[:, 7] - selected_anchors_3d[:, 0, 0]) / selected_anchors_3d[:, 0, 1]
        targets_cd_sin = (torch.sin(sampled_gt_bboxes[:, 11] * 2) - selected_anchors_3d[:, 1, 0]) / selected_anchors_3d[:, 1, 1]
        targets_cd_cos = (torch.cos(sampled_gt_bboxes[:, 11] * 2) - selected_anchors_3d[:, 2, 0]) / selected_anchors_3d[:, 2, 1]
        targets_w3d = (sampled_gt_bboxes[:, 8]  - selected_anchors_3d[:, 3, 0]) / selected_anchors_3d[:, 3, 1]
        targets_h3d = (sampled_gt_bboxes[:, 9]  - selected_anchors_3d[:, 4, 0]) / selected_anchors_3d[:, 4, 1]
        targets_l3d = (sampled_gt_bboxes[:, 10] - selected_anchors_3d[:, 5, 0]) / selected_anchors_3d[:, 5, 1]

        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh, 
                         targets_cdx, targets_cdy, targets_cdz,
                         targets_cd_sin, targets_cd_cos,
                         targets_w3d, targets_h3d, targets_l3d), dim=1)

        stds = targets.new([0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 1, 1, 1, 1, 1, 1])

        targets = targets.div_(stds)

        targets_alpha_cls = (torch.cos(sampled_gt_bboxes[:, 11:12]) > 0).float()
        return targets, targets_alpha_cls #[N, 4]
    
    def _decode(self, boxes, deltas, anchors_3d_mean_std, label_index, alpha_score):
        std = torch.tensor([0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 1, 1, 1, 1, 1, 1], dtype=torch.float32, device=boxes.device)
        widths  = boxes[..., 2] - boxes[..., 0]
        heights = boxes[..., 3] - boxes[..., 1]
        ctr_x   = boxes[..., 0] + 0.5 * widths
        ctr_y   = boxes[..., 1] + 0.5 * heights

        dx = deltas[..., 0] * std[0]
        dy = deltas[..., 1] * std[1]
        dw = deltas[..., 2] * std[2]
        dh = deltas[..., 3] * std[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w     = torch.exp(dw) * widths
        pred_h     = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        one_hot_mask = torch.nn.functional.one_hot(label_index, anchors_3d_mean_std.shape[1]).bool()
        selected_mean_std = anchors_3d_mean_std[one_hot_mask] #[N]
        mask = selected_mean_std[:, 0, 0] > 0
        
        cdx = deltas[..., 4] * std[4]
        cdy = deltas[..., 5] * std[5]
        pred_cx1 = ctr_x + cdx * widths
        pred_cy1 = ctr_y + cdy * heights
        pred_z   = deltas[...,6] * selected_mean_std[:, 0, 1] + selected_mean_std[:,0, 0]  #[N, 6]
        pred_sin = deltas[...,7] * selected_mean_std[:, 1, 1] + selected_mean_std[:,1, 0] 
        pred_cos = deltas[...,8] * selected_mean_std[:, 2, 1] + selected_mean_std[:,2, 0] 
        pred_alpha = torch.atan2(pred_sin, pred_cos) / 2.0

        pred_w = deltas[...,9]  * selected_mean_std[:, 3, 1] + selected_mean_std[:,3, 0]
        pred_h = deltas[...,10] * selected_mean_std[:,4, 1] + selected_mean_std[:,4, 0]
        pred_l = deltas[...,11] * selected_mean_std[:,5, 1] + selected_mean_std[:,5, 0]

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2,
                                    pred_cx1, pred_cy1, pred_z,
                                    pred_w, pred_h, pred_l, pred_alpha], dim=1)

        pred_boxes[alpha_score[:, 0] < 0.5, -1] += np.pi

        return pred_boxes, mask
    
    def _sample(self, assignment_result, anchors, gt_bboxes):
        """
            Pseudo sampling
        """
        # print("##################")
        # print("#######################inside sampling algorithm")
        # # print(anchors.shape)
        # print(gt_bboxes.shape)
        # print("sample gt box:",gt_bboxes.numel())
        pos_inds = torch.nonzero(
                assignment_result['assigned_gt_inds'] > 0, as_tuple=False
            ).unsqueeze(-1).unique()
        
        # print("indices for those anchor boxes with valid indices >0 ",pos_inds.shape)
        neg_inds = torch.nonzero(
                assignment_result['assigned_gt_inds'] == 0, as_tuple=False
            ).unsqueeze(-1).unique()
        # print("indices for those anchor boxes with valid indices =0 ",neg_inds.shape)
        
        
        
        gt_flags = anchors.new_zeros(anchors.shape[0], dtype=torch.uint8) #
        

        pos_assigned_gt_inds = assignment_result['assigned_gt_inds'] - 1

        if gt_bboxes.numel() == 0:
            pos_gt_bboxes = gt_bboxes.new_zeros([0, 4])
            # print("##########",pos_gt_bboxes.shape)
        else:
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds[pos_inds], :]
            # print(pos_gt_bboxes.shape)
        return_dict = dict(
            pos_inds = pos_inds,
            neg_inds = neg_inds,
            pos_bboxes = anchors[pos_inds],
            neg_bboxes = anchors[neg_inds],
            pos_gt_bboxes = pos_gt_bboxes,
            pos_assigned_gt_inds = pos_assigned_gt_inds[pos_inds],
        )
        return return_dict
    
    def _post_process(self, scores, bboxes, labels, P2s):
        
        N = len(scores)
        bbox2d = bboxes[:, 0:4]
        bbox3d = bboxes[:, 4:] #[cx, cy, z, w, h, l, alpha]

        bbox3d_state_3d = self.backprojector.forward(bbox3d, P2s[0]) #[x, y, z, w, h, l, alpha]
        for i in range(N):
            if bbox3d_state_3d[i, 2] > 3 and labels[i] == 0:
                bbox3d[i] = post_opt(
                    bbox2d[i], bbox3d_state_3d[i], P2s[0].cpu().numpy(),
                    bbox3d[i, 0].item(), bbox3d[i, 1].item()
                )
        bboxes = torch.cat([bbox2d, bbox3d], dim=-1)
        return scores, bboxes, labels

    def get_anchor(self, img_batch, P2):
        is_filtering = getattr(self.loss_cfg, 'filter_anchor', True)
        if not self.training:
            is_filtering = getattr(self.test_cfg, 'filter_anchor', is_filtering)
        # print("#################",is_filtering)
        anchors, useful_mask, anchor_mean_std = self.anchors(img_batch, P2, is_filtering=is_filtering)#is_filtering is True
        return_dict=dict(
            anchors=anchors, #[1, N, 4]
            mask=useful_mask, #[B, N]
            anchor_mean_std_3d = anchor_mean_std  #[N, C, K=6, 2]
        )
        return return_dict

    def _get_anchor_3d(self, anchors, anchor_mean_std_3d, assigned_labels):
        """
            anchors: [N_pos, 4] only positive anchors
            anchor_mean_std_3d: [N_pos, C, K=6, 2]
            assigned_labels: torch.Longtensor [N_pos, ]

            return:
                selected_mask = torch.bool [N_pos, ]
                selected_anchors_3d:  [N_selected, K, 2]
        """
        one_hot_mask = torch.nn.functional.one_hot(assigned_labels, self.num_classes).bool()
        selected_anchor_3d = anchor_mean_std_3d[one_hot_mask]

        selected_mask = selected_anchor_3d[:, 0, 0] > 0 #only z > 0, filter out anchors with good variance and mean
        selected_anchor_3d = selected_anchor_3d[selected_mask]

        return selected_mask, selected_anchor_3d
    
        
    
    def get_bboxes(self, cls_scores, reg_preds, anchors, P2s, img_batch=None):

        #for sampling from vae do a forward pass and then 
        reg_pred=reg_preds[0]
        noise=torch.rand(reg_pred.shape)
        noise[:,7]=0
        noise[:,8]=0
        noise[:,-3:]=0
        noise*=self.lamda
        reg_pred=reg_pred+noise.to(reg_pred.device)
        
        
        assert cls_scores.shape[0] == 1 # batch == 1
        cls_scores = cls_scores.sigmoid()

        cls_score = cls_scores[0][..., 0:self.num_classes]
        alpha_score = cls_scores[0][..., self.num_classes:self.num_classes+1]
        
        
        anchor = anchors['anchors'][0] #[N, 4]
        anchor_mean_std_3d = anchors['anchor_mean_std_3d'] #[N, K, 2]
        useful_mask = anchors['mask'][0] #[N, ]

        anchor = anchor[useful_mask]
        cls_score = cls_score[useful_mask]
        alpha_score = alpha_score[useful_mask]
        reg_pred = reg_pred[useful_mask]
        anchor_mean_std_3d = anchor_mean_std_3d[useful_mask] #[N, K, 2]

        score_thr = getattr(self.test_cfg, 'score_thr', 0.5)
        max_score, label = cls_score.max(dim=-1) 
        # print(max_score)

        high_score_mask = (max_score > score_thr)

        anchor      = anchor[high_score_mask, :]
        anchor_mean_std_3d = anchor_mean_std_3d[high_score_mask, :]
        cls_score   = cls_score[high_score_mask, :]
        alpha_score = alpha_score[high_score_mask, :]
        reg_pred    = reg_pred[high_score_mask, :]
        max_score   = max_score[high_score_mask]
        label       = label[high_score_mask]


        bboxes, mask = self._decode(anchor, reg_pred, anchor_mean_std_3d, label, alpha_score)
        if img_batch is not None:
            bboxes = self.clipper(bboxes, img_batch)
        cls_score = cls_score[mask]
        max_score = max_score[mask]

        cls_agnostic=getattr(self.test_cfg,'cls_agnositc',True) 
        nms_iou_thr=getattr(self.test_cfg,'nms_iou_thr',0.1)
        # nms_iou_thr=0.5
        

        if cls_agnostic:
            print("this block")
            print(bboxes.shape)#n,11 (x, y, w, h, cx, cy, z, sin2a, cos2a, w, h, l)
            # if bboxes.shape[0]>1:
            #     print("w",bboxes[0,[2,3]])
            #     print("h",bboxes[0,3])
            #     print("w",bboxes[0,9])
            #     print("h",bboxes[0,10])
            #     print(max_score.shape)#n
            
            # pred_boxes = torch.stack([pred_boxes_x1 0, pred_boxes_y1 1, pred_boxes_x2, 2 pred_boxes_y2 3,
            #                         pred_cx1 4 , pred_cy1 5 , pred_z 6,
            #                         pred_w 7, pred_h 8 , pred_l, pred_alpha], dim=1)



            # cx=bboxes[:max_score.shape[0],4].unsqueeze(-1)
            # cy=bboxes[:max_score.shape[0],5].unsqueeze(-1)
            # cx1=cx-0.5*bboxes[:max_score.shape[0],7].unsqueeze(-1)
            # cy1=cy-0.5*bboxes[:max_score.shape[0],8].unsqueeze(-1)
            # cx2=cx+0.5*bboxes[:max_score.shape[0],7].unsqueeze(-1)
            # cy2=cy+bboxes[:max_score.shape[0],8].unsqueeze(-1)
            # nms_param=torch.cat([cx1,cy1,cx2,cy2],dim=1)
            # print(nms_param.shape)
            # nms_params=torch.from_numpy(np.array(cx,cy,w,h))

            # keep_inds = nms(nms_param,max_score,nms_iou_thr)
            keep_inds = nms(bboxes[:, :4], max_score, nms_iou_thr)
        else:
            max_coordinate = bboxes.max()
            nms_bbox = bboxes[:, :4] + label.float().unsqueeze() * (max_coordinate)
            keep_inds = nms(nms_bbox, max_score, nms_iou_thr)

        bboxes = bboxes[keep_inds]
        # print("max_scoreL",max_score)
        max_score   = max_score[keep_inds]
        label       = label[keep_inds]


        is_post_opt = getattr(self.test_cfg, 'post_optimization', False)
        # print ("Post optimization:",is_post_opt)
        if is_post_opt:
            max_score, bboxes, label = self._post_process(max_score, bboxes, label, P2s)
        # print("max_scoreL",max_score)
        # print(bboxes.shape)
        return max_score, bboxes, label

    def loss(self, cls_scores, reg_preds, anchors, annotations, P2s , vae_mode=False):
        # print("####################")
        # print("inside loss function")

        batch_size=cls_scores.shape[0]#bs

        anchor=anchors['anchors'][0] #[N,4]
        # print("anchor:",anchor.shape)
        anchor_mean_std_3d=anchors['anchor_mean_std_3d']
        # print("anchor mean",anchor_mean_std_3d.shape)#[N,1,6,2]
        cls_loss=[]
        reg_loss=[]
        number_of_positives=[]
        for j in range(batch_size):
            
            reg_pred=reg_preds[j]#[n_anchors,12]
            # print("single_pred",reg_pred.shape)
            # print("cls scores",cls_scores[j][...,0:self.num_classes].shape)
            cls_score=cls_scores[j][..., 0:self.num_classes]#[n_anchors,1]
            alpha_score=cls_scores[j][..., self.num_classes:self.num_classes+1]#[n_anchors,1]
            # print("alpha_score:",alpha_score.shape)

            # selected by mask
            useful_mask=anchors['mask'][j] #[N] (each index is true or false)
        #     # print("###############,",useful_mask.shape)
            

            anchor_j=anchor[useful_mask]#[n,4](n is the nuimber of 1 in binary maks)
            # print("positive anchors",anchor_j.shape)
            anchor_mean_std_3d_j=anchor_mean_std_3d[useful_mask]
            # print("###############,",reg_pred.shape)
            reg_pred=reg_pred[useful_mask]
            # print("useful reg _preds",reg_pred.shape)
            cls_score=cls_score[useful_mask]
            # print("useful cls scores",cls_score.shape)
            alpha_score=alpha_score[useful_mask]
            # print("useful alpha score",alpha_score.shape)

            # only select useful bbox_annotations
            bbox_annotation=annotations[j, :, :]
            bbox_annotation=bbox_annotation[bbox_annotation[:, 4] != -1]#[k] (here 4 beacause the cls_idx is at 4th index)
            # print("Total unpadded annotation boxes:",bbox_annotation.shape)

            #if there are no valid objects in that sample simply append a 0 in the number of positive samples 
            if len(bbox_annotation) == 0:
                cls_loss.append(torch.tensor(0).cuda().float())
                reg_loss.append(reg_preds.new_zeros(self.num_regression_loss_terms))
                number_of_positives.append(0)
                continue
            
            #passing only postive anchors according to conidition in anchors.py and the valid unpadded bbox_annotation 
            assignement_result_dict=self._assign(anchor_j, bbox_annotation, **self.loss_cfg)
            # print("Result of assignment dictionary",assignement_result_dict.keys())
            # print("num_gt",assignement_result_dict["num_gt"])
            # print("assigned_gt_inds",torch.unique(assignement_result_dict["assigned_gt_inds"]))
            # print("max_overlaps",assignement_result_dict["max_overlaps"].shape)
            # print("labels",torch.unique(assignement_result_dict["labels"]))
            sampling_result_dict=self._sample(assignement_result_dict, anchor_j, bbox_annotation)
            # print("result of sampling dictionary:",sampling_result_dict.keys())
            # print("positive indices :",sampling_result_dict["pos_inds"].shape)
            # print("negative indices:",sampling_result_dict["neg_inds"].shape)
            # print("positive boxes:",sampling_result_dict["pos_bboxes"].shape)
            # print("postive gt boxes:",sampling_result_dict["pos_gt_bboxes"].shape)
            # print("positive gt indices:",sampling_result_dict["pos_assigned_gt_inds"].shape)



        
            num_valid_anchors = anchor_j.shape[0]
            labels = anchor_j.new_full((num_valid_anchors, self.num_classes),
                                    -1, # -1 not computed, binary for each class
                                    dtype=torch.float)

            pos_inds = sampling_result_dict['pos_inds']
            neg_inds = sampling_result_dict['neg_inds']
            
            if len(pos_inds) > 0:
                pos_assigned_gt_label = bbox_annotation[sampling_result_dict['pos_assigned_gt_inds'], 4].long()
                
                selected_mask, selected_anchor_3d = self._get_anchor_3d(
                    sampling_result_dict['pos_bboxes'],
                    anchor_mean_std_3d_j[pos_inds],
                    pos_assigned_gt_label,
                )
                if len(selected_anchor_3d) > 0:
                    pos_inds = pos_inds[selected_mask]
                    pos_bboxes    = sampling_result_dict['pos_bboxes'][selected_mask]
                    pos_gt_bboxes = sampling_result_dict['pos_gt_bboxes'][selected_mask]
                    pos_assigned_gt = sampling_result_dict['pos_assigned_gt_inds'][selected_mask]

                    pos_bbox_targets, targets_alpha_cls = self._encode(
                        pos_bboxes, pos_gt_bboxes, selected_anchor_3d
                    ) #[N, 12], [N, 1]
                    label_index = pos_assigned_gt_label[selected_mask]
                    labels[pos_inds, :] = 0
                    labels[pos_inds, label_index] = 1

                    pos_anchor = anchor[pos_inds]
                    pos_alpha_score = alpha_score[pos_inds]
                    if self.decode_before_loss:
                        pos_prediction_decoded=self._decode(pos_anchor,reg_pred[pos_inds],anchors_3d_mean_std,label_index, pos_alpha_score)
                        pos_target_decoded=self._decode(pos_anchor, pos_bbox_targets,  anchors_3d_mean_std, label_index, pos_alpha_score)
                        reg_loss.append(self.loss_huber_bbox(pos_prediction_decoded, pos_target_decoded)*self.regression_weight).mean(dim=0)
                        # reg_loss.append((self.loss_bbox(pos_prediction_decoded, pos_target_decoded)* self.regression_weight).mean(dim=0))
                    else:
                        #vae stuff
                        # mu=self.fc_mean(reg_pred[pos_inds][:self.n_boxes])
                        # logvar=self.fc_logvar(reg_pred[pos_inds][:self.n_boxes])
                        # epsilon=torch.randn_like(logvar)
                        # # epsilon=torch.randn_like(torch.exp(0.5*logvar))
                        # z=mu+torch.exp(0.5*logvar)*epsilon
                        # # print(z.shape)
                        # output=self.decoder(z)
                        # # print(output.shape)
                        # # print(pos_bbox_targets[:self.n_boxes].shape)
                        # reg_loss_j=self.dynamic_huber(pos_bbox_targets[:self.n_boxes],output)
                        # # reg_loss_j=self.loss_mod_huber(pos_bbox_targets,reg_pred[pos_inds])
                        # #  
                        # alpha_loss_j=self.alpha_loss(pos_alpha_score,targets_alpha_cls)
                        # loss_j=torch.cat([reg_loss_j[:self.n_boxes],alpha_loss_j[:self.n_boxes]],dim=1)*self.regression_weight #[N, 13]
                        # reg_loss.append(loss_j.mean(dim=0))
                        # number_of_positives.append(bbox_annotation.shape[0])
                        
                        
                        
                        # targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh, 
                        # targets_cdx, targets_cdy, targets_cdz,
                        # targets_cd_sin, targets_cd_cos,
                        # targets_w3d, targets_h3d, targets_l3d), dim=1)
                        # reg_pred_new=reg_pred[pos_inds]
                        # noise=torch.rand(reg_pred_new.shape) reg_pred=reg_pred[pos_inds]
                        reg_pred=reg_pred[pos_inds]
                        noise=torch.rand(reg_pred.shape)
                        noise[:,7]=0
                        noise[:,8]=0
                        noise[:,-3:]=0
                        noise*=self.lamda
                        reg_pred=reg_pred+noise.to(reg_pred.device)
                        reg_loss_j=self.loss_bbox(pos_bbox_targets,reg_pred) 
                        alpha_loss_j=self.alpha_loss(pos_alpha_score,targets_alpha_cls)
                        # print("####",alpha_loss_j.shape)
                        loss_j=torch.cat([reg_loss_j, alpha_loss_j],dim=1)*self.regression_weight #[N, 13]
                        reg_loss.append(loss_j.mean(dim=0)) #[13]
                        number_of_positives.append(bbox_annotation.shape[0])
                        


                        #regular stuff
                        # reg_loss_j=self.loss_bbox(pos_bbox_targets,reg_pred[pos_inds]) 
                        # alpha_loss_j=self.alpha_loss(pos_alpha_score,targets_alpha_cls)
                        # # print("####",alpha_loss_j.shape)
                        # loss_j=torch.cat([reg_loss_j, alpha_loss_j],dim=1)*self.regression_weight #[N, 13]
                        # reg_loss.append(loss_j.mean(dim=0)) #[13]
                        # number_of_positives.append(bbox_annotation.shape[0])
                        

                        #dynamic huber
                        # reg_loss_j=self.dynamic_huber(pos_bbox_targets,reg_pred[pos_inds]) 
                        # alpha_loss_j=self.alpha_loss(pos_alpha_score,targets_alpha_cls)
                        # # print("####",alpha_loss_j.shape)
                        # loss_j=torch.cat([reg_loss_j, alpha_loss_j],dim=1)*self.regression_weight #[N, 13]
                        # reg_loss.append(loss_j.mean(dim=0)) #[13]
                        # number_of_positives.append(bbox_annotation.shape[0])

                       
            else:
                reg_loss.append(reg_preds.new_zeros(self.num_regression_loss_terms))
                number_of_positives.append(bbox_annotation.shape[0])

            if len(neg_inds) > 0:
                labels[neg_inds, :] = 0
            cls_loss.append(self.loss_cls(cls_score, labels).sum() / (len(pos_inds) + len(neg_inds)))
        
        weights=reg_pred.new(number_of_positives).unsqueeze(1)#[B, 1]
        cls_loss=torch.stack(cls_loss).mean(dim=0,keepdim=True)
        reg_loss=torch.stack(reg_loss, dim=0)#[B,12]
        weighted_regression_losses=torch.sum(weights*reg_loss/(torch.sum(weights)+1e-6),dim=0)
        reg_loss=weighted_regression_losses.mean(dim=0, keepdim=True)
        # print(cls_loss)
        # print(reg_loss)
        # return None,None,None
        return cls_loss,reg_loss,dict(cls_loss=cls_loss,reg_loss=reg_loss,total_loss=cls_loss+reg_loss)
        



    
