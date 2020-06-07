
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 15:43:55 2018

@author: user
"""
#import numpy as np
#import visdom
#viz = visdom.Visdom()

import torch
from torch.autograd import Variable
#import torch.nn.functional as F

#from utils import train_utils as tn_util
#from utils import dataset_utils as ds_util        
    
def train(model, trn_loader, optimizer, criterion, epoch):
    model.train()
    trn_loss, trn_lcc, trn_mae = 0, 0, 0 
    i = 0
    for idx, (mov, ref) in enumerate(trn_loader): #, mask
        mov = Variable(mov.cuda())
        ref = Variable(ref.cuda())
#        mask = Variable(mask.cuda())

        optimizer.zero_grad()
        warped, ref, flow = model(mov, ref) #, mask
#        loss = criterion(mov, ref)
#        output.register_hook(lambda g: print(g))#print intermediate output grad.

        loss1 = criterion['lcc'](warped, ref)
        loss2 = criterion['grad'](flow)
        loss = loss1 + criterion['lambda'] * loss2
        if loss1.item() <= -1:
            i += 1
            print('negative loss, index: {}, loss: {:4f}'.format(idx, loss1.item()))
            continue
        else:
            loss.backward()
            optimizer.step()
            trn_loss += loss.item()
            trn_lcc += loss1.item()
            trn_mae += torch.mean(torch.abs(flow)).item()

    trn_loss /= len(trn_loader) - i
    trn_lcc /= len(trn_loader) - i
    trn_mae /= len(trn_loader) - i
    return trn_loss, trn_lcc, trn_mae

def test(model, test_loader, criterion, epoch):
    model.eval()
    test_loss, test_lcc, test_mae = 0, 0, 0
    i = 0
    for mov, ref in test_loader: #, mask
        mov = Variable(mov.cuda()) 
        ref = Variable(ref.cuda())
#        mask = Variable(mask.cuda())
        warped, ref, flow = model(mov, ref)#, mask
        
        loss1 = criterion['lcc'](warped, ref)
        loss2 = criterion['grad'](flow)
        loss = loss1 + criterion['lambda'] * loss2
        if loss1.item() <= -1:
            i += 1
            print('negative loss, loss: {:4f}'.format(loss1.item()))
            continue
        else:
            test_loss += loss.item()
            test_lcc += loss1.item()
            test_mae += torch.mean(torch.abs(flow)).item()
        
    test_loss /= len(test_loader) - i
    test_lcc /= len(test_loader) - i
    test_mae /= len(test_loader) - i
    return test_loss, test_lcc, test_mae