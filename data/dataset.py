'''
    Module contains Dataset class, collate function for DataLoader and loader getter function.

    * MiniFlickrDataset loads data from pickle file and returns image embedding and caption.
    * cl_fn is used to process batch of data and return tensors.
    * get_loader returns DataLoader object.
'''

import os
import pickle

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer

class MiniFlickrDataset(Dataset):
    def __init__(self, path): 
        # check if file is file
        if not os.path.isfile(path):
            raise OSError('Dataset file not found. Downloading...')

        with open(path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]