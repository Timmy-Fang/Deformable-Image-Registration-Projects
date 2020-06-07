#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 10:59:56 2018

@author: user

make a data set according to the mutual info threshold
"""

import os
import itertools as it
import numpy as np
#import pandas as pd

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
#import torch.nn.functional as F

import transform
from utils import visual

class Volumes(data.Dataset):
    """
    create a data set of 3D CT image saved in .npy format.
    :param:
        str root: the folder of image files;
        int case_id: the index of case for the validation; 1-10
        train: if True(default), load  images of 9 cases 
            by taking random timepoints per patient as fi
xed and moving images;
            else, load images of the remaining one case; 
        transform: transform the images 
    """
    def __init__(self, root, case_id=1, train=True, transform=None):
        self.root = root
        self.case_id = case_id
        self.train = train
        self.transform = transform 
        self.dset_list_name = 'train_val_list_case%g.pth' % case_id
        if os.path.exists(root + self.dset_list_name):
            image_list = torch.load(root + self.dset_list_name)
            self.image_list = image_list['train' if self.train else 'val']
        else:
            self.image_list = self._make_dataset()
        
    def __getitem__(self, index):
        pairs = self.image_list[index]
        mov = np.load(pairs[0])
        ref = np.load(pairs[1])
        mov = np.expand_dims(mov, 0)#shape(1, D, H, W)
        ref = np.expand_dims(ref, 0)
        if self.transform is not None:
            mov = self.transform(mov)
            ref = self.transform(ref)
        return mov, ref
        
    def __len__(self):
        return len(self.image_list)
    
    def _make_dataset(self):
        """
        split the data set into train and val set according to case_id.
        """
        samples_train, samples_val = [], []

        for index in range(1, 11):#11
            if index == self.case_id:
                case = 'case%g' % self.case_id
                path = self.root + case + '/'
                mov_fname = path + case + '_T00.npy'
                ref_fname = path + case + '_T50.npy'
                sample = [mov_fname, ref_fname]
                samples_val.append(sample)
            else:
                path = self.root + 'case%g/' % index
                dcm_list = os.listdir(path)
                for pairs in it.permutations(dcm_list, 2):
                    mov_fname = path + pairs[0]
                    ref_fname = path + pairs[1]
                    sample = [mov_fname, ref_fname]
                    samples_train.append(sample)
            
        #save .npy file
        samples = {'train': samples_train,
                   'val': samples_val}
        torch.save(samples, self.root + self.dset_list_name)
        
        return samples_train if self.train else samples_val

#root = '../dir/data1/'
#Transform = transforms.Compose([transform.OneNorm(),
#                                transform.ToTensor()])
#dset = Volumes(root, 1, train=True, transform=Transform)
#loader = data.DataLoader(dset, batch_size=1, shuffle=True)
#print("Dataset: %d" % len(dset))  # num of samples
#mov, ref = next(iter(loader))
#
#visual.view_slice(mov.numpy()[0, 0], 35)
#visual.view_slice(ref.numpy()[0, 0], 35)

#import networks
#net = networks.snet()
#w, f= net(mov, ref)
#
#visual.view_slice(w.detach().numpy()[0, 0], 35)
#