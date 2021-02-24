from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import PatchGenerator, padding, read_csv, read_csv_complete, read_csv_complete_apoe, get_AD_risk
import random
import pandas as pd
import csv


class MLP_Data(Dataset):
    def __init__(self, Data_dir, exp_idx, stage, roi_threshold, roi_count, choice, seed=1000):
        random.seed(seed)
        self.exp_idx = exp_idx
        self.Data_dir = Data_dir
        self.roi_threshold = roi_threshold
        self.roi_count = roi_count
        if choice == 'count':
            self.select_roi_count()
        else:
            self.select_roi_thres()
        if stage in ['train', 'valid', 'test']:
            self.path = './lookupcsv/exp{}/{}.csv'.format(exp_idx, stage)
        else:
            self.path = './lookupcsv/{}.csv'.format(stage)
        self.Data_list, self.Label_list, self.demor_list = read_csv_complete(self.path)
        self.risk_list = [get_AD_risk(np.load(Data_dir+filename+'.npy'))[self.roi] for filename in self.Data_list]
        self.in_size = self.risk_list[0].shape[0]

    def select_roi_thres(self):
        self.roi = np.load('./DPMs/fcn_exp{}/train_MCC.npy'.format(self.exp_idx))
        self.roi = self.roi > self.roi_threshold
        for i in range(self.roi.shape[0]):
            for j in range(self.roi.shape[1]):
                for k in range(self.roi.shape[2]):
                    if i%3!=0 or j%2!=0 or k%3!=0:
                        self.roi[i,j,k] = False

    def select_roi_count(self):
        self.roi = np.load('./DPMs/fcn_exp{}/train_MCC.npy'.format(self.exp_idx))
        tmp = []
        for i in range(self.roi.shape[0]):
            for j in range(self.roi.shape[1]):
                for k in range(self.roi.shape[2]):
                    if i%3!=0 or j%2!=0 or k%3!=0: continue
                    tmp.append((self.roi[i,j,k], i, j, k))
        tmp.sort()
        tmp = tmp[-self.roi_count:]
        self.roi = self.roi != self.roi
        for _, i, j, k in tmp:
            self.roi[i,j,k] = True

    def __len__(self):
        return len(self.Data_list)

    def __getitem__(self, idx):
        label = self.Label_list[idx]
        risk = self.risk_list[idx]
        demor = self.demor_list[idx]
        return risk, label, np.asarray(demor).astype(np.float32)

    def get_sample_weights(self):
        count, count0, count1 = float(len(self.Label_list)), float(self.Label_list.count(0)), float(self.Label_list.count(1))
        weights = [count / count0 if i == 0 else count / count1 for i in self.Label_list]
        return weights, count0 / count1


class MLP_Data_apoe(MLP_Data):
    def __init__(self, Data_dir, exp_idx, stage, roi_threshold, roi_count, choice, seed=1000):
        super().__init__(Data_dir, exp_idx, stage, roi_threshold, roi_count, choice, seed)
        self.Data_list, self.Label_list, self.demor_list = read_csv_complete_apoe(self.path)

if __name__ == "__main__":
    data = MLP_Data(Data_dir='./DPMs/cnn_exp1/', exp_idx=1, stage='train')
    dataloader = DataLoader(data, batch_size=10, shuffle=False)
    for risk, label, demor in dataloader:
        print(risk.shape, label, demor)