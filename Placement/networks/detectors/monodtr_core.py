import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
import math
import time

from networks.backbones.dla import dla102
from networks.backbones.dlaup import DLAUp
from networks.detectors.dfe import depthawarefe
from networks.detectors.dpe import dpe
from networks.detectors.dtr import depth_transformer


class monodtrcore(torch.nn.Module):
    def __init__(self,backbone_arguments=dict()):
        super(monodtrcore,self).__init__()
        self.backbone=dla102(pretrained=True,return_levels=True)
        channels=self.backbone.channels
        self.first_level=3
        scales=[2**i for i in range(len(channels[self.first_level:]))]
        self.neck=DLAUp(channels[self.first_level:],scales_list=scales)
        self.output_channels_num=256
        self.dpe=dpe(self.output_channels_num)
        self.depth_embed=torch.nn.Embedding(100,self.output_channels_num)        
        self.dtr=depth_transformer(self.output_channels_num)
        self.dfe=depthawarefe(self.output_channels_num)
        self.img_conv=torch.nn.Conv2d(self.output_channels_num,self.output_channels_num,kernel_size=3,padding=1)

    def forward(self,x):
        # print(x.keys())
        # print("#########backbone###############")
        # print("Image:",x["image"].shape)
        x=self.backbone(x["image"])
        # print("backbone:",len(x))
        # print("backbone:")
        # for i in x:
        #     print(i.shape)
        x=self.neck(x[self.first_level:])
        N,C,H,W=x.shape#(bs,256,h//8,w//8)
        # print("Output of backbone (backbone+neck):",x.shape)
        # print("#########depth feature enhancement###############")
        depth,depth_guide,depth_feat=self.dfe(x)
        # print("initial depth for auxillary loss:",depth.shape)
        # print("output of gconv with inital depth after resizing intial depth:",depth_guide.shape)
        # print("final fused feature backbone + feature enhanced depth:",depth_feat.shape)
        # print("#########Creating Depth embedding and transformer input patches###############")
        #depth embeddings are created from indexes of the maximum depth value across channel of the gconv output tensor(initial depth estimate).
        depth_feat=depth_feat.permute(0,2,3,1).view(N,H*W,C)
        # print("Flatten the feature enhanced depth:",depth_feat.shape)
        depth_guide=depth_guide.argmax(1)
        # print("index of maximum element across channel dimension:",depth_guide.shape)
        depth_emb=self.depth_embed(depth_guide).view(N,H*W,C)
        # print("create 256-dim embedding for each max idx across channel of the gconv output",depth_emb.shape)
        depth_emb=self.dpe(depth_emb,(H,W))
        # print("reshaping for transformers (bs,n_patches,emb dim)==>(bs,h*w,d)",depth_emb.shape)
        img_feat=x+self.img_conv(x)
        img_feat=img_feat.view(N,H*W,C)
        # print("backbone output is convolved and added:",img_feat.shape)
        # print("3 inputs to transformer:,")
        # print("Image features (backbone output is convolved and added)",img_feat.shape)
        # print("Depth features from the enhanced depth features ,dfe",depth_feat.shape)
        # print("Depth/pos Embeddings for transformer:",depth_emb.shape)
        # print("#########Transformer###############")
        feat=self.dtr(depth_feat,img_feat,depth_emb)#(bs,n_patches=n_pix,emb_dim=256)
        dtr_feat=feat
        # print("op of dtr:",feat.shape)
        # print("output of transformer patches:",feat.shape)
        feat=feat.permute(0,2,1).view(N,C,H,W)
        # print("#########Output of monodtr core till transformer###############")
        # print("Fused image and depth features from transformer:",feat.shape)
        # print("Inital depth to be used in auxillary loss:",depth.shape)
        return (dtr_feat,feat,depth)




# class DETR(nn.Module):
#     """ This is the DETR module that performs object detection """
#     def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
#         super().__init__()
#         self.num_queries = num_queries
#         self.transformer = transformer
#         hidden_dim = transformer.d_model    
#         self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
#         self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
#         self.query_embed = nn.Embedding(num_queries, hidden_dim)
#         self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
#         self.backbone = backbone
#         self.aux_loss = aux_loss

#     def forward(self, samples: NestedTensor):
#         features, pos = self.backbone(samples)

#         src, mask = features[-1].decompose()
#         assert mask is not None
#         hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

#         outputs_class = self.class_embed(hs)
#         outputs_coord = self.bbox_embed(hs).sigmoid()
#         out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
#         if self.aux_loss:
#             out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
#         return out