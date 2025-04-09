import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Module, Dropout


class depth_transformer(torch.nn.Module):
    def __init__(self,output_channels_num):
        super().__init__()
        self.output_channel_num=output_channels_num
        self.encoder=trans_encoder(self.output_channel_num)
        self.decoder=trans_decoder(self.output_channel_num)
    
    def forward(self,depth_feat,context_feat,depth_pos=None):
        context_feat=context_feat+depth_pos
        # print("adding depth embedding to the image features",context_feat.shape)
        context_feat=self.encoder(context_feat)
        #why not add the depth embedding to the depth feats?
        integrated_feat=self.decoder(depth_feat,context_feat)
        return (integrated_feat)



def elu_feature_map(x):
    return F.elu(x) + 1

class trans_encoder(torch.nn.Module):
    def __init__(self,d_model=2048,nhead=8,attention="linear"):
        super(trans_encoder,self).__init__()
        self.dim=d_model//nhead#d_model=dim*nhead
        self.nhead=nhead
        self.q_proj=torch.nn.Linear(d_model,d_model,bias=False)
        self.k_proj=torch.nn.Linear(d_model,d_model,bias=False)
        self.v_proj=torch.nn.Linear(d_model,d_model,bias=False)
        self.attention=linear_attention()
        self.merge=torch.nn.Linear(d_model,d_model,bias=False)
        #ffn
        self.mlp=torch.nn.Sequential(torch.nn.Linear(d_model,d_model*2,bias=False),torch.nn.ReLU(True),torch.nn.Linear(2*d_model,d_model,bias=False))
        #layernorm
        self.norm1=torch.nn.LayerNorm(d_model) 
        self.norm2=torch.nn.LayerNorm(d_model)
        self.drop_path=torch.nn.Identity()
    
    def forward(self,x):
        # print("encoder uses mhsa with the image features and depth embeddings.")
        bs=x.size(0)
        query,key,value=x,x,x
        #msha
        # print("transforming the input patches for multi head key,query and value")
        query=self.q_proj(query).view(bs,-1,self.nhead,self.dim)#(bs,n_pathces,n_heads,dim_per_head)
        key=self.q_proj(key).view(bs,-1,self.nhead,self.dim)#(bs,n_pathces,n_heads,dim_per_head)
        value=self.q_proj(value).view(bs,-1,self.nhead,self.dim)#(bs,n_pathces,n_heads,dim_per_head)
        # print("encoder multi head query,value and keys",query.shape,key.shape,value.shape)
        message=self.attention(query,key,value)#(sftx(kq.T/sqrt(dim))@v)(bs,n_patches,n_head,dim_per_head)
        # print("encoder output of 8 mhsa heads:",message.shape)
        message=self.merge(message.view((bs,-1,self.nhead*self.dim)))#(bs,n_patches,n_heads*dim_per_head)
        # print("encoder reshape into single sequence after mhsa:",message.shape)

        x=x+self.drop_path(self.norm1(message))
        x=x+self.drop_path(self.norm2(self.mlp(message)))
        # print("final encoder op after norm and ffn:",x.shape)
        return (x)  


def elu_feature_map(x):
    return F.elu(x) + 1
# class linear_attention(torch.nn.Module):
class linear_attention(torch.nn.Module):
    def __init__(self,eps=1e-6):
        super().__init__()
        self.feature_map=elu_feature_map
        self.eps=eps
    def forward(self,query,key,value):
        Q=self.feature_map(query)
        K=self.feature_map(key)
        v_length=value.size(1)#n_patches
        value=value/v_length
        KV=torch.einsum("nshd,nshv->nhdv",K,value)  # (L,D)' @ L,V
        Z=1/(torch.einsum("nlhd,nhd->nlh",Q,K.sum(dim=1))+self.eps)
        queried_values=torch.einsum("nlhd,nhdv,nlh->nlhv",Q,KV,Z)*v_length
        return (queried_values.contiguous())

class trans_decoder(torch.nn.Module):
    def __init__(self,d_model,nhead=8,attention="linear"):
        super(trans_decoder,self).__init__()
        self.dim=d_model//nhead
        self.nhead=nhead

        #mhsa
        self.q_proj0=torch.nn.Linear(d_model,d_model,bias=False)
        self.k_proj0=torch.nn.Linear(d_model,d_model,bias=False)
        self.v_proj0=torch.nn.Linear(d_model,d_model,bias=False)
        self.attention0=linear_attention()
        self.merge0=torch.nn.Linear(d_model,d_model,bias=False)

        #mhca
        self.q_proj1=torch.nn.Linear(d_model,d_model,bias=False)
        self.k_proj1=torch.nn.Linear(d_model,d_model,bias=False)
        self.v_proj1=torch.nn.Linear(d_model,d_model,bias=False)
        self.attention1=linear_attention()
        self.merge1=torch.nn.Linear(d_model,d_model,bias=False)

        #ffn
        self.mlp=torch.nn.Sequential(torch.nn.Linear(d_model,d_model*2,bias=False),torch.nn.ReLU(True),torch.nn.Linear(2*d_model,d_model,bias=False))

        #norm
        self.norm0=torch.nn.LayerNorm(d_model)
        self.norm1=torch.nn.LayerNorm(d_model)
        self.norm2=torch.nn.LayerNorm(d_model)
        self.drop_path=torch.nn.Identity()
    
    def forward(self,x,source):
        #source if from the encoder and x is the depth patches
        bs=x.size(0)
        query,key,value=x,x,x

        #mhsa with the encdoer outpuyt 
        query=self.q_proj0(query).view(bs,-1,self.nhead,self.dim)#(bs,n_pathces,n_heads,dim_per_head)
        key=self.k_proj0(key).view(bs,-1,self.nhead,self.dim)#(bs,n_pathces,n_heads,dim_per_head)
        value=self.v_proj0(value).view(bs,-1,self.nhead,self.dim)#(bs,n_pathces,n_heads,dim_per_head)
        # print("decoder multi head query,value and keys",query.shape,key.shape,value.shape)
        message=self.attention0(query, key, value)#(bs,n_pathces,n_heads,dim_per_head)
        # print("deccoder output of 8 mhsa heads:",message.shape)
        message=self.merge0(message.view(bs,-1,self.nhead*self.dim))#(bs,n_patches,n_heads*dim_per_head)
        # print("deccoder reshape into single sequence after mhsa:",message.shape)
        x=x+self.drop_path(self.norm0(message))
        
        #cross attention (in cross attention between the depth and image features the query comes from one sequence and the key, value comes from another)
        query,key,value=x,source,source
        query=self.q_proj1(query).view(bs,-1,self.nhead,self.dim)#(bs,n_pathces,n_heads,dim_per_head)
        key=self.k_proj1(key).view(bs,-1,self.nhead,self.dim)#(bs,n_pathces,n_heads,dim_per_head)
        value=self.v_proj1(value).view(bs,-1,self.nhead,self.dim)#(bs,n_pathces,n_heads,dim_per_head)
        # print("decoder multi head query,value and keys",query.shape,key.shape,value.shape)
        message=self.attention1(query,key,value)#(bs,n_pathces,n_heads,dim_per_head)
        # print("deccoder output of 8 mhca heads:",message.shape)
        message=self.merge1(message.view(bs,-1,self.nhead*self.dim))#(bs,n_patches,n_heads*dim_per_head)
        # print("deccoder reshape into single sequence after mhca:",message.shape)
        x=x+self.drop_path(self.norm1(message))
        x=x+self.drop_path(self.norm2(self.mlp(x)))#(bs,n_patches,n_heads*dim_per_head)
        # print(x.shape)
        return (x)#(bs,n_patches,n_heads*dim_per_head)

