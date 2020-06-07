#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 15:35:36 2018

@author: user
"""
'''
python=3.5.4, torch=0.4.1, cv2=3.4.2
'''
import sys
sys.path.append('/home/user/fangqiming/xir')
sys.path.append('/home/user/fangqiming/xir/xir_wi')
sys.path.append('/home/user/fangqiming/xir/ext')
sys.path.append('/home/user/fangqiming/xir/utils')
import time
from pathlib import Path
import argparse

#import torch
import torch.optim as optim
import torch.utils.data as data
#import torchvision
import torchvision.transforms as transforms

import networks
import dataset
import transform
import train_utils_wi
from utils import train_utils
#from utils import dataset_utils as ds_util
from ext import loss

parser = argparse.ArgumentParser(description='Training codes')
parser.add_argument('-b', '--batch', default=16, help='batch size')
parser.add_argument('-l', '--lr', default=0.001, help='learning rate')
parser.add_argument('-e', '--epoch', default=300, help='training epochs')
parser.add_argument('-d', '--lamb', default=5, help='lambda, balance the losses.')
parser.add_argument('-w', '--window', default=9, type=int, 
                    help='window size, in the LCC loss')
parser.add_argument('-m', '--modified', default=False, type=bool,
                    help='if modified network')
args = parser.parse_args()
'''dataset loading'''
#torch.cuda.manual_seed(0)

img_size = 576
val_ratio = 0.1
max_disp = 6
#w_decay = 1e-4
#Epoch_step_size = 10
#Gamma = 0.6

#distinguish the saved losses
optim_label = 'adam-real' 
loss_label = optim_label + '-bs%d-lr%.4f-lamb%g'%(args.batch, args.lr, args.lamb)

WEIGHTS_PATH = 'weights_adam/'
Path(WEIGHTS_PATH).mkdir(exist_ok=True)
LOSSES_PATH = 'losses/'
Path(LOSSES_PATH).mkdir(exist_ok=True)
RESULTS_PATH = 'results/'
Path(RESULTS_PATH).mkdir(exist_ok=True)

'''log file'''
f = open(WEIGHTS_PATH + 'README.txt', 'w')

refFolder = '../data/raw_data/RegResult_JiaDing613/SourceRefImg586/'
movFolder = '../data/raw_data/RegResult_JiaDing613/Reg_Out_FFD/'
badsegFolder = '../data/raw_data/RegResult_JiaDing613/RefLungSeg/BadSegCases/'
segmaskFolder = '../data/raw_data/RegResult_JiaDing613/RefLungSeg/'
diffimgFolder = '../data/raw_data/RegResult_JiaDing613/DifferAfter_ROI/'

Folder = {'ref': refFolder,
          'mov': movFolder,
          'badseg': badsegFolder,
          'segmask': segmaskFolder,
          'diffimg': diffimgFolder}
Transform=transforms.Compose([transform.OneNorm(),
#                              transform.addGaussianNoise(),
                              transform.CenterCrop(),
                              transform.ToTensor()])

smTransform=transforms.Compose([transform.MaskDilate()])

#dset = dataset.Images(Folder, False, Transform, smTransform)
##dataset random split
#val_ = int(val_ratio*len(dset))
#train_dset, val_dset = data.random_split(dset, [len(dset)-val_, val_])

train_dset = dataset.Images(Folder, True, val_ratio, False, False, Transform, smTransform)
val_dset = dataset.Images(Folder, False, val_ratio, False, False, Transform, smTransform)

train_loader = data.DataLoader(train_dset, args.batch, shuffle=True)
val_loader = data.DataLoader(val_dset, args.batch, shuffle=True) 

print("Train dset: %d" %len(train_dset))
print("Val dset: %d" %len(val_dset))
print("Train dset: %d" %len(train_dset), file=f)
print("Val dset: %d" %len(val_dset), file=f)
#mov, ref = next(iter(train_loader))  

'''Train'''
model = networks.xirnet_wi(img_size, max_disp, args.modified).cuda()
#model.apply(train_utils.weights_init)

optimizer = optim.Adam(model.parameters(), lr=args.lr)#, weight_decay=0.9
#optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.95) 
#optimizer = optim.RMSprop(model.parameters(), lr=LR, weight_decay=w_decay)#, momentum=0.95
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=Epoch_step_size, gamma=Gamma)

'''lcc + grad'''
criterion = {'lcc': loss.LCC([args.window]*2).cuda(),
             'grad': loss.Grad('l2').cuda(),
             'lambda': args.lamb}

losses = []
for epoch in range(1, args.epoch+1):
    since = time.time()
    
#    scheduler.step()# adjust lr
    ### Train ###
    trn_loss, trn_lcc, trn_mae = train_utils_wi.train(
        model, train_loader, optimizer, criterion, epoch)
    print('Epoch {:d}\nTrain - Loss: {:.4f} | Lcc: {:.4f} | MAE: {:.4f}'.format(
        epoch, trn_loss, trn_lcc, trn_mae))    
    time_elapsed = time.time() - since  
    print('Train Time {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Epoch {:d}\nTrain - Loss: {:.4f} | Lcc: {:.4f} | MAE: {:.4f}'.format(
        epoch, trn_loss, trn_lcc, trn_mae), file=f)      
    print('Train Time {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60), file=f)

    ### Val ###
    val_loss, val_lcc, val_mae = train_utils_wi.test(
            model, val_loader, criterion, epoch)    
    print('Val - Loss: {:.4f} | Lcc: {:.4f} | MAE: {:.4f}'.format(
            val_loss, val_lcc, val_mae))
    time_elapsed = time.time() - since  
    print('Total Time {:.0f}m {:.0f}s\n'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Val - Loss: {:.4f} | Lcc: {:.4f} | MAE: {:.4f}'.format(
            val_loss, val_lcc, val_mae), file=f)
    print('Total Time {:.0f}m {:.0f}s\n'.format(
        time_elapsed // 60, time_elapsed % 60), file=f)
    
    ### Checkpoint ###    
    train_utils.save_weights(model, epoch, val_loss, val_lcc, WEIGHTS_PATH)
        
    ### Save/Plot loss ###
    loss_info = [epoch, trn_loss, trn_lcc, val_loss, val_lcc]
    losses.append(loss_info)
    
train_utils.save_loss(losses, loss_label, LOSSES_PATH)
f.close()
 
###plot loss curve
#train_utils.load_loss('losses-adam-real-bs16-lr0.0010-lamb5.pth', '/home/user/fangqiming/xir/xir_wi/'+LOSSES_PATH, True)

