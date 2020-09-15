import torch
from matplotlib import pyplot as plt
from options import *
from config import *
from model import *
import torch.nn as nn
import numpy as np
import utils
import os
from tensorboardX import SummaryWriter
from ucf_crime_features import *
from sklearn.metrics import roc_curve,auc,classification_report,accuracy_score,confusion_matrix
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
def accuracy(pre,gt):
    correct=pre.equal
def test(net, config, logger, test_loader, test_info, step, model_file=None):
    with torch.no_grad():
        net.eval()

        if model_file is not None:
            net.load_state_dict(torch.load(model_file))

        final_res = {}
        final_res['version'] = 'VERSION 1.3'
        final_res['results'] = {}
        final_res['external_data'] = {'used': True, 'details': 'Features from I3D Network'}

        num_correct = 0.
        num_total = 0.

        load_iter = iter(test_loader)
        labels=None
        predicts=None
        frame_gt=None
        frame_predict=None
        for i in range(len(test_loader.dataset)):

            _data, _label, temp_anno, frames, vid_num_seg = next(load_iter)
            
            _data = _data.cuda()
            _label = _label.cuda()
            pre,att = net(_data)

            att=att.view(-1).cpu().numpy()   
            pre=torch.where(pre<config.class_thresh,torch.zeros_like(pre),torch.ones_like(pre))
            
            label=_label.view(-1).cpu().numpy()   
            pre=pre.view(-1).cpu().numpy()   
            if labels is None:
                labels = label
                predicts = pre
            else:
                labels = np.concatenate([labels, label])
                predicts = np.concatenate([predicts, pre])

            frames=frames.item()
            temp_anno=np.reshape(temp_anno.cpu().numpy(),(-1))
           
            fgt_=np.zeros(frames)
            fpre_=np.zeros(frames)
           
            for i in range(vid_num_seg):
                if temp_anno[0]!=-1:
                    fgt_[temp_anno[0]-1:temp_anno[1]-1]=1
                    if temp_anno[2]!=-1:
                        fgt_[temp_anno[2]-1:temp_anno[3]-1]=1
                frame_start=i*16
                frame_end=(i+1)*16
                fpre_[frame_start:frame_end]=att[i]
            if frame_gt is None:
                    frame_gt = fgt_
                    frame_predict = fpre_
            else:
                frame_gt = np.concatenate([frame_gt, fgt_])
                frame_predict = np.concatenate([frame_predict, fpre_])
        # np.save('frame_level/frame_pre.npy',frame_predict)
        # np.save('frame_level/frame_gt.npy',frame_gt)
        fpr,tpr,thres=roc_curve(frame_gt,frame_predict)
        auc_score=auc(fpr,tpr)

        classes=[ 'Abnormal','Normal']
        accuracy=accuracy_score(labels,predicts)

        #测试阶段
        con_m=confusion_matrix(y_true=labels,y_pred=predicts)
        # print(con_m)
        # utils.plot_conf(con_m,classes,"Anomaly Confusion Matrix")
        
        # plt.show()
        # plt.savefig("AD_cm.png",format='png')
        print(accuracy) 

        report=classification_report(labels,predicts,target_names=classes)
        
        logger.add_scalar('auc_socre',auc_score,step)
        # logger.log_value('accuracy',accuracy,step)
        logger.add_scalar('accuracy',accuracy,step)
        test_info["step"].append(step)
        test_info["auc"].append(auc_score)
        test_info["accuracy"].append(accuracy)
        test_info["report"].append(report)
if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        pdb.set_trace()
    config = Config(args)
    logger=SummaryWriter("logs/")
    test_info = {"step": [], "auc":[],"accuracy": [],"report":[]}
    net = CAD(config.len_feature, config.num_classes)
    net = net.cuda()
    test_loader = data.DataLoader(
        UCF_crime(data_path=config.data_path, mode='Test_Annotation',
                        modal=config.modal, feature_fps=config.feature_fps,
                        num_segments=config.num_segments, len_feature=config.len_feature,
                        seed=config.seed, sampling='uniform'),
            batch_size=1,
            shuffle=False, num_workers=config.num_workers,
            worker_init_fn=4)      
    test(net, config, logger, test_loader, test_info, 1,model_file='models/CADnet/CAD_4.pkl')
