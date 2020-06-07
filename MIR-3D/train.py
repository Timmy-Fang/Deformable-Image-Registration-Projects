#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 17:51:17 2019

@author: user
"""
'''
python=3.5.4, torch=0.4.1, cv2=3.4.2
'''
import time
import argparse
from pathlib import Path
#import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

import networks
import dataset
import transform
from utils import train_utils
#from utils import dataset_utils
from ext import loss

'''dataset loading'''
torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser(description='Training codes')
parser.add_argument('-v', '--val', default=8, type=int,
                    help='the case index of validation')
parser.add_argument('-b', '--batch', default=4, type=int,
                    help='batch size')
parser.add_argument('-l', '--lr', default=0.001, type=float, 
                    help='learning rate')
parser.add_argument('-e', '--epoch', default=50, type=int, 
                    help='training epochs')
parser.add_argument('-d', '--lamb', default=0, type=float,
                    help='lambda, balance the losses.')
#parser.add_argument('-a', '--alpha', default=2, type=float,
#                    help='alpha, balance the flow loss.')
parser.add_argument('-w', '--win', default=[5,5,5], type=int, 
                    help='window size, in the LCC loss')
parser.add_argument('-i', '--image', default=[96, 208, 272], type=int, 
                    help='image size')
args = parser.parse_args()

#distinguish the saved losses
optim_label = 'adam' 
loss_label = optim_label + '-val%g-bs%g-lr%.4f-lamb%g-win%g-epoch%g'%(
        args.val, args.batch, args.lr, args.lamb, args.win[0], args.epoch)

WEIGHTS_PATH = 'weights-adam/'
Path(WEIGHTS_PATH).mkdir(exist_ok=True)
LOSSES_PATH = 'losses/'
Path(LOSSES_PATH).mkdir(exist_ok=True)
RESULTS_PATH = 'results/'
Path(RESULTS_PATH).mkdir(exist_ok=True)
'''log file'''
f = open(WEIGHTS_PATH + 'README.txt', 'w')

root = '../dir/data1/'
Transform = transforms.Compose([transform.OneNorm(),
                                transform.ToTensor()])
train_dset = dataset.Volumes(root, args.val, train=True, transform=Transform)
val_dset = dataset.Volumes(root, args.val, train=False, transform=Transform)
train_loader = data.DataLoader(train_dset, args.batch, shuffle=True)
val_loader = data.DataLoader(val_dset, args.batch, shuffle=True)

print("Train dset: %d" %len(train_dset))
print("Val dset: %d" %len(val_dset))

'''Train'''
model = networks.snet(ndown=3, img_size=args.image).cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr)#, weight_decay=0.9
#optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.95) 
#optimizer = optim.RMSprop(model.parameters(), lr=LR, weight_decay=w_decay)#, momentum=0.95
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=Epoch_step_size, gamma=Gamma)

'''lcc + grad'''
criterion = {'lcc': loss.LCC(args.win).cuda(),
             'mse': torch.nn.MSELoss().cuda(),
             'lambda': args.lamb}

losses = []
for epoch in range(1, args.epoch+1):
    since = time.time()
    
#    scheduler.step()# adjust lr
    ### Train ###
    trn_loss, trn_lcc, trn_mae = train_utils.train(
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
    val_loss, val_lcc, val_mae = train_utils.test(
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
    
train_utils.save_loss(losses, loss_label, LOSSES_PATH, RESULTS_PATH)
f.close()
   
###plot loss curve
#train_utils.load_loss('losses-adam-val1-bs2-lr0.0001-lamb0-win9.pth', LOSSES_PATH, True)
 