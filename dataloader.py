from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import PatchGenerator, padding, read_csv, read_csv_complete, read_csv_complete_apoe, get_AD_risk
import random
import pandas as pd
import csv

"""
dataloaders are defined in this scripts:
    1. FCN dataloader (data split into 60% train, 20% validation and 20% testing)
        (a). Training stage:    use random patches to train classification FCN model
        (b). Validation stage:  forward whole volume MRI to FCN to get Disease Probability Map (DPM). use MCC of DPM as criterion to save model parameters
        (c). Testing stage:     get all available DPMs for the development of MLP

    2. MLP dataloader (use the exactly same split as FCN dataloader)
        (a). Training stage:    train MLP on DPMs from the training portion
        (b). Validation stage:  use MCC as criterion to save model parameters
        (c). Testing stage:     test the model on ADNI_test, NACC, FHS and AIBL datasets

    3. CNN dataloader (baseline classification model to be compared with FCN+MLP framework)
        (a). Training stage:    use whole volume to train classification FCN model
        (b). Validation stage:  use MCC as criterion to save model parameters
        (c). Testing stage:     test the model on ADNI_test, NACC, FHS and AIBL datasets
"""

#Check if CNN_Data is going to be ever needed in this and how
#Recheck where it is getting in the data from

class FCN_Data(Dataset):
    def __init__(self,
                 Data_dir,
                 exp_idx,
                 stage,
                 whole_volume=False,
                 seed=1000,
                 patch_size=47,
                 transform=Augment()):

        """
        :param Data_dir:      data path
        :param exp_idx:       experiment index maps to different data splits
        :param stage:         stage could be 'train', 'valid', 'test' and etc ...
        :param whole_volume:  if whole_volume == True, get whole MRI;
                              if whole_volume == False and stage == 'train', sample patches for training
        :param seed:          random seed
        :param patch_size:    patch size has to be 47, otherwise model needs to be changed accordingly
        :param transform:     transform is about data augmentation, if transform == None: no augmentation
                              for more details, see Augment class
        """

        FCN_Data.__init__(self, Data_dir, exp_idx, stage, seed)
        self.stage = stage
        self.transform = transform
        self.whole = whole_volume
        self.patch_size = patch_size
        self.patch_sampler = PatchGenerator(patch_size=self.patch_size)

    def __getitem__(self, idx):
        label = self.Label_list[idx]
        if self.stage == 'train' and not self.whole:
            data = np.load(self.Data_dir + self.Data_list[idx] + '.npy', mmap_mode='r').astype(np.float32)
            patch = self.patch_sampler.random_sample(data)
            if self.transform:
                patch = self.transform.apply(patch).astype(np.float32)
            patch = np.expand_dims(patch, axis=0)
            return patch, label
        else:
            data = np.load(self.Data_dir + self.Data_list[idx] + '.npy').astype(np.float32)
            data = np.expand_dims(padding(data, win_size=self.patch_size // 2), axis=0)
            return data, label


if __name__ == "__main__":
    data = FCN_Data(Data_dir='./DPMs/cnn_exp1/', exp_idx=1, stage='train')
    dataloader = DataLoader(data, batch_size=10, shuffle=False)
    for risk, label, demor in dataloader:
        print(risk.shape, label, demor)

