# import os
# import torch

# import numpy as np
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import classification_report,classification,accuracy_score
# data_path='dataset/'
# split_path = os.path.join(data_path, '{}_Annotation.txt'.format('Test'))
# split_file = open(split_path, 'r')
# vid_list = []
# for line in split_file:
#     vid_list.append(line.strip().replace('.mp4','').split())
# split_file.close()
# start_end_couples=[]
# for index in range(len(vid_list)):
#     anomalies_frames = [int(x) for x in vid_list[index][3:]]
#     start_end_couples.append(anomalies_frames)
# import torch
# label=torch.randn(10,14)
# fore_weights=torch.randn(10,3,1)
# max_att=fore_weights.squeeze().max(dim=-1)[0]
# min_att=fore_weights.squeeze().min(dim=-1)[0]
# hinge_loss=label[:,-1]*(1-max_att+min_att)+(1-label[:,-1])*(max_att-min_att)
# print(hinge_loss.size())
# A=torch.Tensor([[1,2,0],[2,1,1]])
# print(A[:,-1])
# def focal_loss(score_base,label,gamma,alpha):
#         y1=-label*(1-score_base)**gamma*torch.log(score_base)
#         y0=-(1-label)*(score_base**gamma)*torch.log(1-score_base)
#         loss=alpha*y1+(1-alpha)*y0
#         return loss.sum(dim=1).mean()
# if __name__ == "__main__":
#     score=torch.rand(5,4)
#     label=torch.Tensor([[1,0,0,0],[0,1,0,1],[1,0,0,0],[0,0,0,1],[0,0,0,0]])
#     loss=focal_loss(score,label,gamma=2,alpha=0.3)
#     print(loss)
import torch.nn as nn
import torch
loss = nn.CrossEntropyLoss()
input = torch.randn(100, 14, requires_grad=True).cuda()
target = torch.randint(100).random_(5).cuda()
output = loss(input, target)
print(output)