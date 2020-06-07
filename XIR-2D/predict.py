#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 16:53:34 2018

@author: user

test model prediction
"""
import os
import sys
sys.path.append('E:/IR/xir')
sys.path.append('E:/IR/xir/xir_wi')
sys.path.append('E:/IR/xir/ext')
sys.path.append('E:/IR/xir/utils')
#import time
import numpy as np
from pathlib import Path
import SimpleITK as sitk
import matplotlib.pyplot as plt

import torch
#import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
#import torchvision
import torchvision.transforms as transforms

import networks
import dataset
import transform
#import train_utils_wi
from utils import train_utils as tn_util
from utils import dataset_utils as ds_util
from utils import test_utils as tt_util
from ext import loss
#%%
'''dataset loading'''
torch.cuda.manual_seed(0)
img_size = 576
batch_size = 1
max_disp = 6
win_size = [17, 17]
val_ratio = 0.1
modified = True #model
train = True
'''data'''
refFolder = 'E:/IR/xir/data/raw_data/RegResult_JiaDing613/SourceRefImg586/'
movFolder = 'E:/IR/xir/data/raw_data/RegResult_JiaDing613/Reg_Out_FFD/'
badsegFolder = 'E:/IR/xir/data/raw_data/RegResult_JiaDing613/RefLungSeg/BadSegCases/'
segmaskFolder = 'E:/IR/xir/data/raw_data/RegResult_JiaDing613/RefLungSeg/'
diffimgFolder = 'E:/IR/xir/data/raw_data/RegResult_JiaDing613/DifferAfter_ROI/'

Folder = {'ref': refFolder,
          'mov': movFolder,
          'badseg': badsegFolder,
          'segmask': segmaskFolder,
          'diffimg': diffimgFolder}
Transform=transforms.Compose([transform.OneNorm(),
#                              transform.addGaussianNoise(),
                              transform.CenterCrop(),
                              transform.ToTensor()])
smTransform=transforms.Compose([transform.MaskDilate()])

dset = '_train_' if train else '_val_'# for files saving
pdt_dset = dataset.Images(Folder, train, val_ratio, True, Transform, smTransform)
loader = data.DataLoader(pdt_dset, batch_size, shuffle=True) 
print("pdt_dset: %d" %len(pdt_dset))
'''model'''
model = networks.xirnet_wi(img_size, max_disp, modified)#.cuda()
#model.apply(train_utils.weights_init)
criterion = loss.LCC(win_size)#.cuda()
criterion1 =  loss.Grad('l2')#.cuda()

#%%weights
'''dset split'''
#####no tanh, win=9
#WEIGHTS_PATH = 'weights/lcc-5/weights_adam-wi-seg-bs16-lr-3-lamb5-win9-unet-epoch300/'
#weights_fname = 'weights-286--0.180--0.191.pth'
#####no tanh, win=13
#WEIGHTS_PATH = 'weights/lcc-5/weights_adam-wi-seg-bs16-lr-3-lamb5-win13-unet-epoch300/'
#weights_fname = 'weights-298--0.214--0.224.pth'
#####no tanh, win=17
#WEIGHTS_PATH = 'weights/lcc-5/weights_adam-wi-seg-bs16-lr-3-lamb5-win17-unet-epoch300/'
#weights_fname = 'weights-284--0.244--0.255.pth'
#####tanh, win=17, disp=6
WEIGHTS_PATH = 'weights/lcc-5/weights_adam-wi-seg-bs16-lr-3-lamb5-win17-munet-epoch300-disp6/'
weights_fname = 'weights-284--0.247--0.256.pth'
startEpoch = tn_util.load_weights(model, WEIGHTS_PATH + weights_fname)            
#%% save predictions folder
saveFolder = 'results/predict/weights_adam-wi-seg-bs16-lr-3-lamb5-win17-munet-epoch300-disp6/'
warpFolder = 'warped/'
diffbFolder = 'diff_before/'
diffaFolder = 'diff_after/'
flowFolder = 'flow/'
Path(saveFolder).mkdir(exist_ok=True)
Path(saveFolder + warpFolder).mkdir(exist_ok=True)
Path(saveFolder + diffbFolder).mkdir(exist_ok=True)
Path(saveFolder + diffaFolder).mkdir(exist_ok=True)
Path(saveFolder + flowFolder).mkdir(exist_ok=True)
#%% '''predict'''             
model.eval()
#mov, ref, mask, name, diff = next(iter(loader))
for idx in range(len(loader.dataset)):

    mov, ref, mask, name, diff = loader.dataset[idx]
    print(name)
    mov = Variable(mov.unsqueeze(0)) #.cuda()
    ref = Variable(ref.unsqueeze(0)) #.cuda()
    mask = Variable(torch.Tensor(np.expand_dims(mask, 0))) #.cuda()
    
#    mov = Variable(mov.cuda()) 
#    ref = Variable(ref.cuda())
#    mask = Variable(mask.cuda())
    warped, ref0, flow = model(mov, ref, mask)
    
    loss = criterion(warped, ref0)
    loss1 = criterion1(flow)
    mov, ref = mov*mask, ref*mask
    loss_orig = criterion(mov, ref)
    
    #data
    m = mov.data#.cpu()
    r = ref.data#.cpu()
    w = warped.data#.cpu()
    flow = flow.data#.cpu()
    grid = tt_util.flow2grid(flow)
    diffb = tt_util.DiffAdjust((m-r).numpy()[0,0])
    diffa = tt_util.DiffAdjust((w-r).numpy()[0,0])
    lb = loss_orig.item()
    la = loss.item()
    grad = loss1.item()
    #plot
#    tt_util.view_pred(m, r, diff, grid, diffb, diffa, lb, la, grad, name)
    ############save files########################
    # name
    name = os.path.splitext(name)[0] #name remove .dcm
    flow_name = name + dset + 'grad_{:.4f}'.format(grad) + '.npy'
    diffb_name = name + dset + 'lossb_{:.4f}'.format(lb) + '.dcm'
    diffa_name = name + dset + 'lossa_{:.4f}'.format(la) + '.dcm'
    warp_name = name + dset + '.dcm'
    #flow
    np.save(saveFolder + flowFolder + flow_name, flow.numpy()[0])
    #diffa, diffb
    mask = mask.cpu().numpy()[0, 0]
    maski = 1 - mask
    max, min = tt_util.get_max_min(diff, mask)
    diffa = diffa * (max-min) + min
    diffb = diffb * (max-min) + min
    diffa = diff * maski + diffa * mask
    diffb = diff * maski + diffb * mask
    diffa_save = sitk.GetImageFromArray(diffa.astype('int16')[0])
    diffb_save = sitk.GetImageFromArray(diffb.astype('int16')[0])
    sitk.WriteImage(diffa_save, saveFolder + diffaFolder + diffa_name)
    sitk.WriteImage(diffb_save, saveFolder + diffbFolder + diffb_name)
    #warped
    mov_list = os.listdir(movFolder)
    mov_name_list = [i.split('_')[0] for i in mov_list]
    mov_idx = mov_name_list.index(name.split('_')[0])
    mov_name = mov_list[mov_idx]
    movimg, _, _ = ds_util.loadDCM(movFolder + mov_name)
    cencrop = transform.CenterCrop()
    movimg_cc = cencrop(np.expand_dims(movimg,0))
    warped = w.numpy()[0]
    max, min = tt_util.get_max_min(movimg_cc, mask)
    warped = warped * (max-min) + min
    warped = movimg_cc * maski + warped * mask
    warped_save = sitk.GetImageFromArray(warped.astype('int16')[0])
    sitk.WriteImage(warped_save, saveFolder + warpFolder + warp_name)