import librosa
import os, pickle, argparse
import numpy as np
import torch
import torch.utils.data as data
import python_speech_features

from tqdm import tqdm
from glob import glob
import python_speech_features
from torchvision import transforms

import sys
sys.path.append("..")
from utils import get_landmark_seq, get_mfcc_seq
import pdb


class ALnet_dataset(data.Dataset):
    def __init__(self, dataset_dir, split, window_size, step):
        self.dataset_dir = dataset_dir
        self.split = split
        self.window_size = window_size
        self.step = step
        self.landmark_seq = get_landmark_seq(self.dataset_dir, self.split, self.window_size)
        self.mfcc_seq = get_mfcc_seq(self.dataset_dir, self.split, self.window_size)
        
        self.input_mfcc_all = np.concatenate(self.mfcc_seq, axis=0)      # (43225, 28, 12)
        self.landmark_all = np.concatenate(self.landmark_seq, axis=0)    # (43225, 136)

        self.landmark_norm = transforms.ToTensor()(self.landmark_all).squeeze()
        # self.landmark_norm = self.landmark_all / 255
        
    def __getitem__(self, index):
        
        input_mfcc = self.input_mfcc_all[index*self.step: index*self.step + self.window_size, :, :]
        landmark_seq = self.landmark_norm[index*self.step: index*self.step + self.window_size, :]
        
        # landmark_seq_136 = landmark_seq_68_2.reshape(landmark_seq.shape[0], -1)
    
        num = np.random.randint(0, self.window_size)
        single_landmark = landmark_seq[num: num+1, :]
        
        # if self.transform is not None:
        #     single_landmark = self.transform(single_landmark)
        #     landmark_seq_in = []
        #     for num in range(self.window_size):
        #         landmark_num = landmark_seq[num: num+1, :]
        #         landmark_transform = self.transform(landmark_num)
        #         landmark_seq_in.append(landmark_transform)
        #     landmark_seq_in = np.concatenate(landmark_seq_in, axis=0)

        return single_landmark, input_mfcc, landmark_seq
    
    def __len__(self):
        return int((self.input_mfcc_all.shape[0] - self.window_size) / self.step)