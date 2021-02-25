import os
import numpy as np
from model import _CNN, _FCN, _MLP_A, _MLP_B, _MLP_C, _MLP_D
from utils import matrix_sum, get_accu, get_MCC, get_confusion_matrix, write_raw_score, DPM_statistics, timeit, read_csv
from dataloader import CNN_Data, FCN_Data, MLP_Data, MLP_Data_apoe, CNN_MLP_Data
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np

"""
model wrapper class are defined in this scripts which includes the following methods:
    1. init: initialize dataloader, model
    2. train:
    3. valid:
    4. test:
    5. ...
    1. FCN wrapper
    2. MLP wrapper
    3. CNN wrapper
"""



class FCN_Wrapper:

    def __init__(self, fil_num,
                       drop_rate,
                       seed,
                       batch_size,
                       balanced,
                       Data_dir,
                       exp_idx,
                       model_name,
                       metric,
                       patch_size):

        """
            :param fil_num:    output channel number of the first convolution layer
            :param drop_rate:  dropout rate of the last 2 layers, see model.py for details
            :param seed:       random seed
            :param batch_size: batch size for training FCN
            :param balanced:   balanced could take value 0 or 1, corresponding to different approaches to handle data imbalance,
                               see self.prepare_dataloader for more details
            :param Data_dir:   data path for training data
            :param exp_idx:    experiment index maps to different data splits
            :param model_name: give a name to the model
            :param metric:     metric used for saving model during training, can take 'accuracy' or 'MCC'
                               for example, if metric == 'accuracy', then the time point where validation set has best accuracy will be saved
            :param patch_size: size of patches for FCN training, must be 47. otherwise model has to be changed accordingly
        """

        self.seed = seed
        self.exp_idx = exp_idx
        self.Data_dir = Data_dir
        self.patch_size = patch_size
        self.model_name = model_name
        self.eval_metric = get_accu if metric == 'accuracy' else get_MCC
        self.model = _FCN(num=fil_num, p=drop_rate).cuda()
        self.prepare_dataloader(batch_size, balanced, Data_dir)
        if not os.path.exists('./checkpoint_dir/'): os.mkdir('./checkpoint_dir/')
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        if not os.path.exists('./DPMs/'): os.mkdir('./DPMs/')
        self.DPMs_dir = './DPMs/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.DPMs_dir):
            os.mkdir(self.DPMs_dir)

   
    def test_and_generate_DPMs(self):
        print('testing and generating DPMs ... ')
        self.model.load_state_dict(torch.load('{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch)))
        self.fcn = self.model.dense_to_conv()
        self.fcn.train(False)
        with torch.no_grad():
            for stage in ['train', 'valid', 'test', 'AIBL', 'NACC', 'FHS']:
                Data_dir = self.Data_dir
                if stage in ['AIBL', 'NACC', 'FHS']:
                    Data_dir = Data_dir.replace('ADNI', stage)
                data = FCN_Data(Data_dir, self.exp_idx, stage=stage, whole_volume=True, seed=self.seed, patch_size=self.patch_size)
                filenames = data.Data_list
                dataloader = DataLoader(data, batch_size=1, shuffle=False)
                DPMs, Labels = [], []
                for idx, (inputs, labels) in enumerate(dataloader):
                    inputs, labels = inputs.cuda(), labels.cuda()
                    DPM = self.fcn(inputs, stage='inference').cpu().numpy().squeeze()
                    np.save(self.DPMs_dir + filenames[idx] + '.npy', DPM)
                    DPMs.append(DPM)
                    Labels.append(labels)
                matrix, ACCU, F1, MCC = DPM_statistics(DPMs, Labels)
                np.save(self.DPMs_dir + '{}_MCC.npy'.format(stage), MCC)
                np.save(self.DPMs_dir + '{}_F1.npy'.format(stage),  F1)
                np.save(self.DPMs_dir + '{}_ACCU.npy'.format(stage), ACCU)
                print(stage + ' confusion matrix ', matrix, ' accuracy ', self.eval_metric(matrix))
        print('DPM generation is done')


if __name__ == "__main__":
    pass
