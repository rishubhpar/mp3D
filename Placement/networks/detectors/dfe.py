import torch
import torch.nn as nn
import torch.nn.functional as F


class depthawarefe(torch.nn.Module):
    def __init__(self,output_channel_num):
        super(depthawarefe,self).__init__()
        self.output_channel_num=output_channel_num
        self.depth_output=torch.nn.Sequential(torch.nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True),torch.nn.Conv2d(self.output_channel_num,int(self.output_channel_num/2),3,padding=1),torch.nn.BatchNorm2d(int(self.output_channel_num/2)),torch.nn.ReLU(),torch.nn.Conv2d(int(self.output_channel_num/2),96,1))

        self.depth_down=torch.nn.Conv2d(96,12,3,stride=1,padding=1,groups=12)#gconv
        self.acf=dfe_module(256,256)
    
    def forward(self,x):
        # print("depth aware Feature ehnacement module input:",x.shape)
        depth=self.depth_output(x)
        # print("after depth output:",depth.shape)
        N,C,H,W=x.shape
        depth_guide=F.interpolate(depth,size=x.size()[2:],mode="bilinear",align_corners=False)#depth guide is for auxillary depth supervision
        # print("interpolating the depth to origianl shape",depth_guide.shape)
        depth_guide=self.depth_down(depth_guide)#gconv output   
        # print("output of gconv",depth_guide.shape)
        x=x+self.acf(x,depth_guide)
        # print("final depth features after fusing output of backbone and feat enhanced depth:",x.shape)
        return (depth,depth_guide,x)

class dfe_module(torch.nn.Module):
    def __init__(self,in_channels,out_channels):
        super(dfe_module,self).__init__()
        self.softmax=torch.nn.Softmax(dim=-1)
        self.conv1=torch.nn.Sequential(torch.nn.Conv2d(in_channels,out_channels,1,bias=False),torch.nn.BatchNorm2d(out_channels),torch.nn.ReLU(True),torch.nn.Dropout2d(0.2,False))
        self.conv2=torch.nn.Conv2d(out_channels,out_channels,kernel_size=1,stride=1,padding=0)
    
    
    def forward(self,feat_ffm,coarse_x):
        #thw two inputs into the dfe module are the backbone output and the gconv output
        N,D,H,W=coarse_x.size() # initial depth feature
        # print("Input 1  to dfe module output of backbone",feat_ffm.shape)
        # print("Input 2 to dfe module gconv output",coarse_x.shape)#output of gconv output is fed into the module.
        
        feat_ffm=self.conv1(feat_ffm)
        # print("passing the backbone op through a conv X'",feat_ffm.shape)
        #feat_ffm is the output of the two conv's to give X'.
        _,C,_,_=feat_ffm.size()
        #coarse_x is the output of the Gconv
        proj_query=coarse_x.view(N,D,-1)
        # print("Flattening the gconv output:",proj_query.shape)
        proj_key=feat_ffm.view(N,C,-1).permute(0,2,1)
        # print("Flattening the backbone output:",proj_key.shape)
        energy=torch.bmm(proj_query,proj_key)
        # print("first matrix mul(X',gconv op) ; depth prototype",energy.shape)
        energy_new=torch.max(energy,-1,keepdim=True)[0].expand_as(energy) - energy
        attention=self.softmax(energy_new)
        #attention is the depth_protoype in the paper.

        #depth enhancement
        attention=attention.permute(0,2,1)
        proj_value=coarse_x.view(N,D,-1)
        # print("Flattening the Gconv output:",proj_value.shape)
        out=torch.bmm(attention,proj_value)
        # print("Second matrix mul to get F'",out.shape)
        out=out.view(N,C,H,W)
        out=self.conv2(out)
        return (out)


