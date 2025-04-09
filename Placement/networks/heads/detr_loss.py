import numpy as np
import sys
import typing
import itertools
import torch
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
from metrics.giou import calc_iou



@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



def hungarian_bf(cost_matrix):
    height,width=cost_matrix.shape
    print(cost_matrix)
    print(height,width)
    minimum_cost=float("inf")
    if height>=width:
        for idx in itertools.permutations(list(range(height)),min(height,width)):
            print("row idx:",idx)
            row_idx=idx
            col_idx=list(range(width))
            print("col idx:",col_idx)
            print("elements:",cost_matrix[row_idx,col_idx])#selecting possible sqaure sub matrices and caculating the sum of the left diagonal
            cost=cost_matrix[row_idx,col_idx]
            print("cost:",cost_matrix[row_idx,col_idx].sum())
            if cost<=minimum_cost:
                minimum_cost=cost
                optimal_row_idx=row_idx
                optimal_col_idx=col_idx
    
    if height<width:
        for idx in itertools.permutations(list(range(width)),min(height,width)):
            print("row idx:",idx)
            col_idx=idx
            row_idx=list(range(height))
            print("col idx:",col_idx)
            print("elements:",cost_matrix[row_idx,col_idx])#selecting possible sqaure sub matrices and caculating the sum of the left diagonal
            cost=cost_matrix[row_idx,col_idx]
            print("cost:",cost_matrix[row_idx,col_idx].sum())
            if cost<=minimum_cost:
                minimum_cost=cost
                optimal_row_idx=row_idx
                optimal_col_idx=col_idx
    return (optimal_row_idx,optimal_col_idx)


# cost_matrix=np.array([[108,125,150],[150,135,175],[200,500,230],[765,734,589]])
# hungarian_bf(cost_matrix)


class hungarian_matcher(torch.nn.Module):
    def __init__(self,cost_class,cost_bbox,cost_giou):
        super().__init__()
        self.cost_class=cost_class
        self.cost_bbox=cost_bbox
        self.cost_giou=cost_giou
        assert cost_class!=0 or cost_bbox!=0 or cost_giou!=0,"all costs cant be 0"

    @torch.no_grad()
    def forward(self,outputs,targets,sizes):
        filler=np.NaN
        """outputs[pred_logits]=bs,n_queries,n_classes(3+1)
        outputs[pred_boxes]=bs,n_queries,(11)[l,t,b,r,x,y,z,w,h,l,alpha]
        targets[labels]tensor of dim n_target boxes, targets[boxes] are (n_target,11)"""
        bs,n_queries=outputs["class"].shape[0:2]
        # print("assignment loss")
        out_prob=outputs["class"].flatten(0,1).softmax(-1)#(bs*n_queries,n_classes)
        # out_prob=out_prob.unsqueeze(0)
        # print("pred class:",out_prob.shape)
        out_bbox=outputs["box"].flatten(0,1)#(bs*n_queries,11)
        # out_bbox=out_bbox.unsqueeze(0)
        # print("pred box:",out_bbox.shape)
        # targets=targets.unsqueeze(0)
        # print(targets.shape)
        target_ids=[i[4] for i in targets]
        target_ids=torch.tensor(target_ids)
        # print("gt target_class:",target_ids.shape)
        # mask=torch.ones(targets.shape).bool()
        # mask[:,4]=0
        # target_boxes=targets[mask]  
        # target_boxes=target_boxes.view(mask.shape[0],mask.shape[1]-1)
        target_boxes=torch.cat([targets[:,0:4],targets[:,4+1:]],dim=1)
        # print(target_boxes)
        # print("gt target_boxes:",target_boxes.shape)

        # # target_ids=torch.cat([v["labels"] for v in targets])
        # # target_boxes=torch.cat([v["boxes"] for v in targets])
        
        cost_class=-out_prob[:,target_ids.long()]
        # print("Class cost:",cost_class.shape)
        cost_bbox=torch.cdist(out_bbox,target_boxes,p=1)#l1 loss use huber loss also.
        # print("bbox cost:",cost_bbox.shape)
        # # cost_giou=calc_iou(out_bbox[0:6],target_boxes[0:6])
        
        # # #build cost matrix
        cost_matrix=self.cost_bbox*cost_bbox + self.cost_class*cost_class #+ self.cost_giou*cost_giou
        cost_matrix=cost_matrix.view(bs,n_queries,-1).cpu()
        # print(cost_matrix.shape)
        indices=[c[i] for i,c in enumerate(cost_matrix.split(sizes,-1))]
        matched_indices=[]
        for i in indices:
            if i.shape[1]!=0:
                matched_indices.append(linear_sum_assignment(i))
            else:
                pass
                # matched_indices.append((np.array([filler]),np.array([filler])))
        # for i,j in matched_indices:
        #     print(i,j)
        return [(torch.as_tensor(i,dtype=torch.int64),torch.as_tensor(j,dtype=torch.int64)) for i,j in matched_indices]


def build_matcher():
    return (hungarian_matcher(cost_class=1,cost_bbox=5,cost_giou=2))


class detr_loss(torch.nn.Module):
    def __init__(self,num_classes,matcher,weight_dict,eos_coef,losses):
        super().__init__()
        self.num_classes=num_classes
        self.matcher=matcher
        self.weight_dict=weight_dict
        self.eos_coef=eos_coef
        self.losses=losses
        empty_weight=torch.ones(self.num_classes+1)
        empty_weight[-1]=self.eos_coef
        self.register_buffer('empty_weight',empty_weight)
    
    def get_loss(self,loss,outputs,targets,indices,num_boxes):
        loss_map={'boxes':self.loss_boxes,'labels':self.loss_labels}
        return loss_map[loss](outputs,targets,indices,num_boxes)

    def _get_src_permutation_idx(self,indices):
        #for explanation of this function check commend
        batch_idx=torch.cat([torch.full_like(src,i) for i,(src,_) in enumerate(indices)])
        src_idx=torch.cat([src for (src,_) in indices])
        return (batch_idx,src_idx)


    def loss_boxes(self,outputs,targets,indices,num_boxes):
        assert 'box' in outputs
        idx=self._get_src_permutation_idx(indices)
        # print(idx)
        # print(outputs['box'].shape)
        src_boxes=outputs['box'][idx]
        # print(src_boxes.shape)    
        # print(src_boxes)
        # print("ok till here")
        # for i in idx:
        #     print(i)
        # print(outputs['box'][idx[0][0],idx[1][0],:])
        # print(outputs['box'][idx[0][-1],idx[1][-1],:])
        # print("####",targets.shape)
        target_boxes=torch.cat([targets[i,:] for t,(_,i) in zip(targets,indices)])
        target_boxes=torch.cat([target_boxes[:,0:4],target_boxes[:,4+1:]],dim=1)
        # print(target_boxes.shape)
        # print(src_boxes.shape)
        loss_bbox=F.l1_loss(src_boxes,target_boxes,reduction='none')
        # print(num_boxes)
        losses={}
        losses['loss_bbox']=loss_bbox.sum()/num_boxes
        return (losses)

    def _get_tgt_permutation_idx(self,indices):
        batch_idx=torch.cat([torch.full_like(tgt,i) for i,(_,tgt) in enumerate(indices)])
        tgt_idx=torch.cat([tgt for (_,tgt) in indices])
        return (batch_idx,tgt_idx)


    def loss_labels(self,outputs,targets,indices,num_boxes,log=True):
        assert "class" in outputs
        src_logits=outputs['class']
        # print("class logits:",src_logits.shape)
        idx=self._get_src_permutation_idx(indices)
        # print(idx)
        target_classes_o=torch.cat([targets[i,4].long() for t,(_,i) in zip(targets,indices)])
        # print(target_classes_o)
        target_classes=torch.full(src_logits.shape[:2],self.num_classes,dtype=torch.int64,device=src_logits.device)
        target_classes[idx]=target_classes_o
        # print(torch.where(target_classes==1))
        loss_ce=F.cross_entropy(src_logits.transpose(1,2),target_classes,self.empty_weight)  
        losses={'loss_ce':loss_ce}
        losses["class_error"]=100-accuracy(src_logits[idx],target_classes_o)[0]
        return (losses)


    def forward(self,outputs,targets,sizes):
        # out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        outputs_without_aux={k:v for k,v in outputs.items() if k!="aux_outputs"}
        # print(outputs_without_aux.keys())
        # print("boxes:",outputs_without_aux["box"].shape)
        # print("classes",outputs_without_aux["class"].shape)
        # print("targets:",targets.shape)
        #retrieve indices with best match
        indices=self.matcher(outputs_without_aux,targets,sizes)
        # print(indices)
        #compute average number of boxes 
        num_boxes=sum(sizes)
        # num_boxes=sum(len(t["labels"]) for t in targets)
        # print("here")
        num_boxes=torch.as_tensor([num_boxes],dtype=torch.float,device=next(iter(outputs.values())).device)
        num_boxes=torch.clamp(num_boxes,min=1).item()
        # print("here")

        losses={}
        for loss in self.losses:
            losses.update(self.get_loss(loss,outputs,targets,indices,num_boxes))
        return (losses)




#     def forward(self, outputs, targets):
#         """ This performs the loss computation.
#         Parameters:
#              outputs: dict of tensors, see the output specification of the model for the format
#              targets: list of dicts, such that len(targets) == batch_size.
#                       The expected keys in each dict depends on the losses applied, see each loss' doc
#         """
#         outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

#         # Retrieve the matching between the outputs of the last layer and the targets
#         indices = self.matcher(outputs_without_aux, targets)

#         # Compute the average number of target boxes accross all nodes, for normalization purposes
#         num_boxes = sum(len(t["labels"]) for t in targets)
#         num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
#         if is_dist_avail_and_initialized():
#             torch.distributed.all_reduce(num_boxes)
#         num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

#         # Compute all the requested losses
#         losses = {}
#         for loss in self.losses:
#             losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

#         # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
#         if 'aux_outputs' in outputs:
#             for i, aux_outputs in enumerate(outputs['aux_outputs']):
#                 indices = self.matcher(aux_outputs, targets)
#                 for loss in self.losses:
#                     if loss == 'masks':
#                         # Intermediate masks losses are too costly to compute, we ignore them.
#                         continue
#                     kwargs = {}
#                     if loss == 'labels':
#                         # Logging is enabled only for the last layer
#                         kwargs = {'log': False}
#                     l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
#                     l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
#                     losses.update(l_dict)

#         return losses


# #build loss
# matcher=build_matcher()
# weight_dict={"loss_ce":1,"loss_bbox":5}
# weight_dict["loss_giou"]=2

# aux_loss=True #noqa
# if aux_loss:
#     aux_weight_dict={}
#     for i in range(5):
#         aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
#     weight_dict.update(aux_weight_dict)

# losses=['labels','boxes','cardinality']
# # num_classes=car,van,truck,bg
# criterion=detr_loss(num_classes=3+1,matcher=matcher,weight_dict=weight_dict,eos_coef=0.1,losses=losses)
