import torch
import torch.nn as nn
import numpy as np
def cross_entropy(predict,target):
    predict=torch.clamp(predict,min=1e-6,max=1-1e-6)
    loss=-1*(target*torch.log(predict)).sum(-1)
    return loss
class FocalLossV1(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean'):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCELoss(reduction='none')

    def forward(self, logits, label):  
        # label_sum=(1-label).sum(dim=-1) 
        logits=torch.clamp(logits,min=1e-6,max=1-1e-6)
        pt=(logits*label).sum(dim=-1)
        # pt=(1-logits*label)
        # pt=(1-logits)*label+logits*(1-label)
        loss=-1*torch.pow((1-pt),self.gamma)*torch.log(pt)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

class CAD_loss(nn.Module):
    def __init__(self, alpha):
        super(CAD_loss, self).__init__()
        self.alpha = alpha
        self.ce_creation=nn.BCELoss()
        self.focalloss=FocalLossV1()
        
  
    def forward(self, score_base,  fore_weights, label):#score_base:(batch,classes),fore_weights(batch,num_segments,1),label(batch,classes)
        loss = {}
        max_label=label.sum(1)
        fore_weights=fore_weights.squeeze()
        k_att=torch.topk(fore_weights,1)[0].mean(dim=-1)
        att_loss=self.ce_creation(k_att,max_label)
        #hinge_loss
        # normal_indices=(label.sum(1)==0).view(-1).nonzero().flatten()
        # abnormal_indices=(label.sum(1)==1).view(-1).nonzero().flatten()
        # normal_scores=fore_weights[normal_indices]
        # abnormal_scores=fore_weights[abnormal_indices]
        # normal_scores_maxes=normal_scores.max(dim=1)[0]
        # abnormal_scores_maxes=abnormal_scores.max(dim=1)[0]
        # hinge_loss=1-abnormal_scores_maxes.mean()+normal_scores_maxes.mean()
        
        #JS 


        loss_base=self.ce_creation(score_base,label)
        # print(loss_base)
        sparse_loss=torch.mean(torch.norm(fore_weights,p=1,dim=1))
        loss_total=loss_base +sparse_loss*self.alpha+att_loss
        loss["loss_base"] = loss_base
        loss["sparse_loss"] = sparse_loss
        loss["att_loss"] = att_loss
       
        loss["loss_total"] = loss_total
        
        return loss_total, loss

def train(net, train_loader, loader_iter, optimizer, criterion, logger, step):
    net.train()
    try:
        _data, _label,_,_,_ = next(loader_iter)
    except:
        loader_iter = iter(train_loader)
        _data, _label,_,_,_= next(loader_iter)

    _data = _data.cuda()
    _label = _label.cuda()

    optimizer.zero_grad()

    score_base, fore_weights = net(_data)

    cost, loss = criterion(score_base, fore_weights,_label)

    cost.backward()
    optimizer.step()

    for key in loss.keys():
        
        logger.add_scalar(key, loss[key].cpu().item(), step)
        print(step,',',key,":",loss[key].cpu().item())
if __name__ == "__main__":
    A=torch.randn(100,14)
    B=torch.randn(100,14)
    c=cross_entropy(A,B)
    print(c.size())