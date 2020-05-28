import os
import glob
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torch.autograd import Variable


class SelfDataset(Dataset):
    def __init__(self, data_path, label_path, logger=None):
        self.data = np.load(data_path, allow_pickle=True)
        self.label = np.load(label_path, allow_pickle=True)
        if logger is not None:
            logger.info(f'Finished creating an instance of {self.__class__.__name__} with {len(self.label)} examples')

    def __len__(self):
        return len(self.label)

    def __getitem__(self,i):
        data = self.data[i]
        label = self.label[i]
        
        data = torch.Tensor(data)
        
        data_fixed = torch.zeros(768, 300)
        cut = min(768, data.size(0))
        data_fixed[:cut, :] = data[:cut, :]
        data = data_fixed

        label = torch.from_numpy(np.array(label)).long()

        return data, label


if __name__ == "__main__":

    dataset = SelfDataset('./dataset/data_train.npy', './dataset/label_train.npy')
    data, label = dataset.__getitem__(0)
    print(data.shape)
    print(label.shape)
