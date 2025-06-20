import os
import inspect
import math
import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
BatchNorm=torch.nn.BatchNorm2d

def get_model_url(data="imagenet",name="dla34",hash="ba72cf86"):
    return (os.path.join('http://dl.yf.io/dla/models',data,'{}-{}.pth'.format(name,hash)))

def conv3x3(in_planes,out_planes,stride=1):
    return (torch.nn.Conv2d(in_planes,out_planes,kernel_size=3,stride=stride,padding=1,bias=False))

class BasicBlock(torch.nn.Module):
    def __init__(self,inplanes,planes,stride=1,dilation=1):
        super(BasicBlock,self).__init__()
        self.conv1=torch.nn.Conv2d(inplanes,planes,kernel_size=3,stride=stride,padding=dilation,bias=False,dilation=dilation)
        self.bn1=torch.nn.BatchNorm2d(planes)
        self.relu=torch.nn.ReLU(inplace=True)
        
        self.conv2=torch.nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=dilation,bias=False,dilation=dilation)   
        self.bn2=torch.nn.BatchNorm2d(planes)
        self.stride=stride
    
    def forward(self,x,residual=None):
        if residual is None:
            residual=x
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.conv2(out)
        out=self.bn2(out)
        out+=residual
        out=self.relu(out)
        return (out)


class Bottleneck(torch.nn.Module):
    expansion=2
    def __init__(self,inplanes,planes,stride=1,dilation=1):
        super(Bottleneck,self).__init__()
        expansion=Bottleneck.expansion
        bottle_planes=planes//expansion
        self.conv1=torch.nn.Conv2d(inplanes,bottle_planes,kernel_size=1,bias=False)
        self.bn1=BatchNorm(bottle_planes)
        self.conv2=torch.nn.Conv2d(bottle_planes,bottle_planes,kernel_size=3,stride=stride,padding=dilation,bias=False,dilation=dilation)
        self.bn2=BatchNorm(bottle_planes)
        self.conv3=torch.nn.Conv2d(bottle_planes,planes,kernel_size=1,bias=False)
        self.bn3=BatchNorm(planes)
        self.relu=torch.nn.ReLU(inplace=True)
        self.stride=stride
    
    def forward(self,x,residual=None):
        if residual is None:
            residual=x
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        
        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu(out)

        out=self.conv3(out)
        out=self.bn3(out)
        out+=residual
        out=self.relu(out)
        return (out)

class BottleneckX(torch.nn.Module):
    expansion=2
    cardinality=32
    def __init__(self,inplanes,planes,stride=1,dilation=1):
        super(BottleneckX,self).__init__()
        cardinality=BottleneckX.cardinality
        bottle_planes=planes*cardinality//32
        self.conv1=torch.nn.Conv2d(inplanes,bottle_planes,kernel_size=1,bias=False)
        self.bn1=BatchNorm(bottle_planes)
        self.conv2=torch.nn.Conv2d(bottle_planes,bottle_planes,kernel_size=3,stride=stride,padding=dilation,bias=False,dilation=dilation,groups=cardinality)
        self.bn2=BatchNorm(bottle_planes)
        self.conv3=torch.nn.Conv2d(bottle_planes,planes,kernel_size=1,bias=False)
        self.bn3=BatchNorm(planes)
        self.relu=torch.nn.ReLU(inplace=True)
        self.stride=stride
    
    def forward(self,x,residual=None):
        if residual is None:
            residual=x
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        
        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu(out)

        out=self.conv3(out)
        out=self.bn3(out)
        out+=residual
        out=self.relu(out)
        return (out)


class Root(torch.nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,residual):
        super(Root,self).__init__()
        self.conv=torch.nn.Conv2d(in_channels,out_channels,1,stride=1,bias=False,padding=(kernel_size-1)//2)
        self.bn=BatchNorm(out_channels)
        self.relu=torch.nn.ReLU(inplace=True)
        self.residual=residual
    
    def forward(self,*x):
        children=x
        x=self.conv(torch.cat(x,1))
        x=self.bn(x)
        if self.residual:
            x+=children[0]
        x=self.relu(x)
        return (x)

class Tree(torch.nn.Module):
    def __init__(self,levels,block,in_channels,out_channels,stride=1,level_root=False,root_dim=0,root_kernel_size=1,dilation=1,root_residual=False):
        super(Tree,self).__init__()
        if root_dim==0:
            root_dim=2*out_channels
        if level_root:
            root_dim+=in_channels
        if levels==1:
            self.tree1=block(in_channels,out_channels,stride,dilation=dilation)
            self.tree2=block(out_channels,out_channels,1,dilation=dilation)
        else:
            self.tree1=Tree(levels-1,block,in_channels,out_channels,stride,root_dim=0,root_kernel_size=root_kernel_size,dilation=dilation,root_residual=root_residual)
            self.tree2=Tree(levels-1,block,out_channels,out_channels,root_dim=root_dim+out_channels,root_kernel_size=root_kernel_size,dilation=dilation,root_residual=root_residual)
        if levels==1:
            self.root=Root(root_dim,out_channels,root_kernel_size,root_residual)
        self.level_root=level_root
        self.root_dim=root_dim
        self.downsample=None
        self.project=None
        self.levels=levels
        if stride>1:
            self.downsample=torch.nn.MaxPool2d(stride,stride=stride)
        if in_channels!=out_channels:
            self.project=torch.nn.Sequential(torch.nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,bias=False),BatchNorm(out_channels))
    
    def forward(self,x,residual=None,children=None):
        children=[] if children is None else children
        bottom=self.downsample(x) if self.downsample else x
        residual=self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1=self.tree1(x,residual)
        if self.levels==1:
            x2=self.tree2(x1)
            x=self.root(x2,x1,*children)
        else:
            children.append(x1)
            x=self.tree2(x1,children=children)
        return (x)



class DLA(torch.nn.Module):
    def __init__(self,levels,channels,num_classes=1000,block=BasicBlock,residual_root=False,return_levels=False,pool_size=7,linear_root=False):
        super(DLA,self).__init__()
        self.channels=channels
        self.return_levels=return_levels
        self.num_classes=num_classes
        self.base_layer=torch.nn.Sequential(torch.nn.Conv2d(3,channels[0],kernel_size=7,stride=1,padding=3,bias=False),BatchNorm(channels[0]),torch.nn.ReLU(inplace=True))
        self.level0=self._make_conv_level(channels[0],channels[0],levels[0])
        self.level1=self._make_conv_level(channels[0],channels[1],levels[1],stride=2)
        self.level2=Tree(levels[2],block,channels[1],channels[2],2,level_root=False,root_residual=residual_root)
        self.level3=Tree(levels[3],block,channels[2],channels[3],2,level_root=True,root_residual=residual_root)
        self.level4=Tree(levels[4],block,channels[3],channels[4],2,level_root=True,root_residual=residual_root)
        self.level5=Tree(levels[5],block,channels[4],channels[5],2,level_root=True,root_residual=residual_root)
        self.avgpool=torch.nn.AvgPool2d(pool_size)
        self.fc=torch.nn.Conv2d(channels[-1],num_classes,kernel_size=1,stride=1,padding=0,bias=True)

        for m in self.modules():
            if isinstance(m,torch.nn.Conv2d):
                n=m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0,math.sqrt(2./n))
            elif isinstance(m,BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def _make_level(self,block,inplanes,planes,blocks,stride=1):
        downsample=None
        if stride!=1 or inplanes!=planes:
            downsample=torch.nn.Sequential(torch.nn.MaxPool2d(stride,stride=stride),torch.nn.Conv2d(inplanes,planes,kernel_size=1,stride=1,bias=False),BatchNorm(planes))
        layers=[]
        layers.append(block(inplanes,planes,stride,downsample=downsample))
        for i in range(1,blocks):
            layers.append(block(inplanes,planes))
        return (torch.nn.Sequential(*layers))
    
    def _make_conv_level(self,inplanes,planes,convs,stride=1,dilation=1):
        modules=[]
        for i in range(convs):
            modules.extend([torch.nn.Conv2d(inplanes,planes,kernel_size=3,stride=stride if i==0 else 1,padding=dilation,bias=False,dilation=dilation),BatchNorm(planes),torch.nn.ReLU(inplace=True)])
            inplanes=planes
        return (torch.nn.Sequential(*modules))

    def load_pretrained_model(self,data="imagenet",name="dla34",hash='ba72cf86'):
        fc=self.fc
        if name.endswith(".pth"):
            model_weights=torch.load(data+name)
        else:
            model_url=get_model_url(data,name,hash)
            # print(vars(get_model_url))
            # print(inspect.getargvalues(get_model_url))
            # print(model_url)
            model_weights=model_zoo.load_url(model_url)
            # model_weights=torch.load("/home/rishubh/.cache/torch/hub/checkpoints/imagenetdla102.pth")
        num_classes=len(model_weights[list(model_weights.keys())[-1]])
        self.fc=torch.nn.Conv2d(self.channels[-1],num_classes,kernel_size=1,stride=1,padding=0,bias=True)
        self.load_state_dict(model_weights)
        self.fc=fc
    
    def forward(self,x):
        y=[]
        x=self.base_layer(x)
        for i in range(6):
            x=getattr(self,"level{}".format(i))(x)
            y.append(x)
        if self.return_levels:
            return (y)
        else:
            x=self.avgpool(x)
            x=self.fc(x)
            x=x.view(x.size(0),-1)
            return (x)

    
def dla34(pretrained=False,**kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock,**kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
    return model


def dla46_c(pretrained=False, **kwargs):  # DLA-46-C
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 64, 128, 256],
                block=Bottleneck, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla46_c', hash='2bfd52c3')
    return model


def dla46x_c(pretrained=False, **kwargs):  # DLA-X-46-C
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 64, 128, 256],
                block=BottleneckX, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla46x_c', hash='d761bae7')
    return model


def dla60x_c(pretrained=False, **kwargs):  # DLA-X-60-C
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 64, 64, 128, 256],
                block=BottleneckX, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla60x_c', hash='b870c45c')
    return model


def dla60(pretrained=False, **kwargs):  # DLA-60
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla60', hash='24839fc4')
    return model


def dla60x(pretrained=False, **kwargs):  # DLA-X-60
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 128, 256, 512, 1024],
                block=BottleneckX, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla60x', hash='d15cacda')
    return model


def dla102(pretrained=False, **kwargs):  # DLA-102
   Bottleneck.expansion = 2
   model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, residual_root=True, **kwargs)
   if pretrained:
        model.load_pretrained_model(data='imagenet',name='dla102',hash='d94d9790')
   return model


def dla102x(pretrained=False, **kwargs):  # DLA-X-102
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=BottleneckX, residual_root=True, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla102x', hash='ad62be81')
    return model


def dla102x2(pretrained=False, **kwargs):  # DLA-X-102 64
    BottleneckX.cardinality = 64
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=BottleneckX, residual_root=True, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla102x2', hash='262837b6')
    return model


def dla169(pretrained=False, **kwargs):  # DLA-169
    Bottleneck.expansion = 2
    model = DLA([1, 1, 2, 3, 5, 1], [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, residual_root=True, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla169', hash='0914e092')
    return model
