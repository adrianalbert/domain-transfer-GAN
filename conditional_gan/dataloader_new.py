import torch
from torch.utils import data
import os.path
import random
import torchvision.transforms as transforms
import numpy as np
from numpy import inf

from skimage.transform import resize

DEV_SIZE = 200



class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, inputs, outputs):
        'Initialization'
        self.input_labels = inputs
        self.output_labels = outputs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample


        # Load data and get label
        load_string = "../datasets/livneh_loader/"
        temp = np.concatenate((np.load(load_string + var + "/" + str(index) + ".npy") for var in self.input_labels), axis = 0)
        print(temp.shape)
        X = torch.from_numpy(temp)
        temp = np.concatenate((np.load(load_string + var + "/" + str(index) + ".npy") for var in self.output_labels), axis = 0)
        y = torch.from_numpy(temp)

        return X, y





class AlignedIterator(data.Dataset)):
    """Iterate multiple ndarrays (e.g. images and labels) IN THE SAME ORDER
    and return tuples of minibatches"""

    def __init__(self, nc4A, nc4B, **kwargs):
        super(AlignedIterator, self).__init__()

        assert data_A.shape[0] == data_B.shape[0], 'passed data differ in number!'
        self.data_A = data_A
        self.data_B = data_B

        self.num_samples = data_A.shape[0]

        batch_size = kwargs.get('batch_size', 100)
        shuffle = kwargs.get('shuffle', False)

        self.n_batches = self.num_samples // batch_size
        if self.num_samples % batch_size != 0:
            self.n_batches += 1

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.reset()

    def __iter__(self):
        return self

    def reset(self):
        if self.shuffle:
            self.data_indices = np.random.permutation(self.num_samples)
        else:
            self.data_indices = np.arange(self.num_samples)
        self.batch_idx = 0

    def __next__(self):
        if self.batch_idx == self.n_batches:
            self.reset()
            raise StopIteration


        idx = self.batch_idx * self.batch_size
        chosen_indices = self.data_indices[idx:idx+self.batch_size]
        self.batch_idx += 1

        return {'A': torch.from_numpy(self.data_A[chosen_indices]),
                'B': torch.from_numpy(self.data_B[chosen_indices])}

    def __len__(self):
        return self.num_samples
