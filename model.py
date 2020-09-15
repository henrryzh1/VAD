import torch
import  math
import torch.nn as nn
from torch.nn.parameter import Parameter
class Correct(nn.Module):
    def __init__(self,cas_len,clas_len):
        super(Correct,self).__init__()
        self.cas_len=cas_len
        self.clas_len=clas_len
        self.weight=Parameter(torch.FloatTensor(cas_len,clas_len))
        self.reset_parameters()
        self.sigmoid=nn.Sigmoid()
    def reset_parameters(self):
        stdv=1./math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv,stdv)
    def forward(self,cas,main_feature):
        output=torch.matmul(torch.matmul(cas,self.weight),main_feature)
        output=self.sigmoid(output)
        return output
    
class CAD(nn.Module):
    def __init__(self,len_feature,num_classes):
        super(CAD,self).__init__()
        self.len_feature=len_feature
        self.num_classes=num_classes
        self.softmax=nn.Softmax(dim=-1)
        self.sigmoid=nn.Sigmoid()
        # self.attention=nn.Sequential(
        #     nn.Linear(self.len_feature,256),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.6),
        #     nn.Linear(256,1),
        #     nn.Sigmoid()
        # )
        self.attention=nn.Sequential(
            nn.Linear(self.len_feature,256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128,1),
            nn.Sigmoid()
        )
        self.classifier=nn.Sequential(
            nn.Linear(self.len_feature,256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128,1 )
        )
    def forward(self,x):
        att=self.attention(x)    
        cas=self.classifier(x)
        cas=torch.softmax(cas,dim=1)
        predict=(att*cas).sum(1)
        # refine=torch.cosine_similarity(x,predict,dim=-1)
        
        return predict,att


if __name__ == "__main__":
    model=CAD(1024,14)
    A=torch.ones(3,400,1024)
    s=model.forward(A)
    print(s[0].size())