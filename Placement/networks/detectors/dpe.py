import torch
import torch.nn as nn


class dpe(torch.nn.Module):
    def __init__(self,dim,k=3):
        super(dpe,self).__init__()
        self.proj=torch.nn.Conv2d(dim,dim,k,1,k//2,groups=dim)
    def forward(self,x,size):
        B,N,C=x.shape
        H,W=size
        assert N==H*W
        feat=x.transpose(1,2).view(B,C,H,W)
        # print("transposing and reshaping the depth embeddings:",feat.shape)
        x=self.proj(feat)+feat
        # print("applying group conv of dpe to depth embedding and adding to itself,fig 4 paper:",x.shape)
        x=x.view(B,C,-1).transpose(1,2)
        # print("reshaping for transformers (bs,n_patches,emb dim)==>(bs,h*w,d)",x.shape)
        return(x)

    