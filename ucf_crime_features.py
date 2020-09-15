import torch.utils.data as data
import os
import csv
import json
import numpy as np
import torch
import pdb
import time
import random
import utils
import config
class UCF_crime(data.Dataset):
    def __init__(self, data_path, mode, modal, feature_fps, num_segments, len_feature, sampling, seed=-1, supervision='weak'):
        if seed >= 0:
            utils.set_seed(seed)

        self.mode = mode
        self.modal = modal
        self.feature_fps = feature_fps
        self.num_segments = num_segments
        self.len_feature = len_feature

        if self.modal == 'all':
            self.feature_path = []
            for _modal in ['rgb', 'flow']:
                self.feature_path.append(os.path.join(data_path, 'features'))
        else:
            self.feature_path = os.path.join(data_path, 'features')

        split_path = os.path.join(data_path, '{}.txt'.format(self.mode))
        split_file = open(split_path, 'r')
        self.vid_list = []
        for line in split_file:
            self.vid_list.append(line.strip().replace('.mp4','').split())
        split_file.close()

        self.class_name_to_idx = dict((v, k) for k, v in config.class_dict.items())        
        self.num_classes = len(self.class_name_to_idx.keys())

        self.supervision = supervision
        self.sampling = sampling


    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        data, vid_num_seg = self.get_data(index)
        label,frames,temp_anno = self.get_label(index,self.mode)

        return data, label, temp_anno, frames, vid_num_seg

    def get_data(self, index):
        vid_name = self.vid_list[index][0]

        vid_num_seg = 0

        if self.modal == 'all':
            rgb_feature = np.load(os.path.join(self.feature_path[0],
                                    vid_name + '.npz.npy')).astype(np.float32)
            flow_feature = np.load(os.path.join(self.feature_path[1],
                                    vid_name + '.npz.npy')).astype(np.float32)

            vid_num_seg = rgb_feature.shape[0]

            if self.sampling == 'random':
                sample_idx = self.random_perturb(rgb_feature.shape[0])
            elif self.sampling == 'uniform':
                sample_idx = self.uniform_sampling(rgb_feature.shape[0])
            else:
                raise AssertionError('Not supported sampling !')

            rgb_feature = rgb_feature[sample_idx]
            flow_feature = flow_feature[sample_idx]

            feature = np.concatenate((rgb_feature, flow_feature), axis=1)
        else:
            feature = np.load(os.path.join(self.feature_path,
                                    vid_name + '.npz.npy')).astype(np.float32)

            vid_num_seg = feature.shape[0]

            if self.sampling == 'random':
                sample_idx = self.random_perturb(vid_num_seg)
                feature = feature[sample_idx]
            elif self.sampling == 'uniform':
                pass
                # sample_idx = self.uniform_sampling(vid_num_seg)
            else:
                raise AssertionError('Not supported sampling !')

            
        return torch.from_numpy(feature), vid_num_seg

    def get_label(self, index,mode):
        vid_name = self.vid_list[index][0]
        frames=int(self.vid_list[index][1])
        label = np.zeros(1, dtype=np.float32)
        if 'Normal' not in vid_name:
            vid=vid_name.split('/')[0]
            label[0] = 1
        # else:
        #     label=(1./13)*np.ones((13),dtype=np.float32)  
        # else:
        #     label[-1]=1

        start_end_couples=[]
        if mode=='Train':
            return label,torch.Tensor(0),torch.Tensor(0)
        else:
            anomalies_frames = [int(x) for x in self.vid_list[index][3:]]
            start_end_couples.append(anomalies_frames) 
            start_end_couples=torch.from_numpy(np.array(start_end_couples))        
            return label,frames,start_end_couples


    def random_perturb(self, length):
        if self.num_segments == length:
            return np.arange(self.num_segments).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        for i in range(self.num_segments):
            if i < self.num_segments - 1:
                if int(samples[i]) != int(samples[i + 1]):
                    samples[i] = np.random.choice(range(int(samples[i]), int(samples[i + 1]) + 1))
                else:
                    samples[i] = int(samples[i])
            else:
                if int(samples[i]) < length - 1:
                    samples[i] = np.random.choice(range(int(samples[i]), length))
                else:
                    samples[i] = int(samples[i])
        return samples.astype(int)


    def uniform_sampling(self, length):
        if self.num_segments == length:
            return np.arange(self.num_segments).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        samples = np.floor(samples)
        return samples.astype(int)