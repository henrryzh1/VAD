import numpy as np
from sklearn.metrics import confusion_matrix
def loadtxt(dir):
    A=[]
   
    i=0
    s=0
    with open(dir,'r') as f:
        lines=f.read().splitlines()
        for line in lines:
            data=line.split()
            if "Normal" in data[0]:
                A.append([i,i+int(data[1])])
                s=s+int(data[1])
            i=i+int(data[1])
    return A

if __name__ == "__main__":
    A=loadtxt("Test_Annotation.txt")
    pre=np.load("frame_pre.npy")
    pred=[]
    len1=0
    for index in A:
        temp=pre[index[0]:index[1]]
        len1=len(temp)+len1
        for i,data in enumerate(temp):
            pred.append(data)
    pred=np.array(pred)
    new_gt=np.zeros(len(pred))
    pred[pred>=0.5]=1
    pred[pred<0.5]=0
    conf_mat=confusion_matrix(new_gt,pred,labels=[1,0])
    print(conf_mat)
    print(conf_mat[1][0]/len(pred))
