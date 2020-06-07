# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 16:29:46 2018

@author: qiming.fang
"""
import os
import shutil
#import random
#import numpy as np
import matplotlib.pyplot as plt

import torch
#import torch.nn as nn
#import torch.nn.functional as F
#from torch.autograd import Variable

def save_weights(model, epoch, loss, err, WEIGHTS_PATH):
    weights_fname = 'weights-%d-%.3f-%.3f.pth' % (epoch, loss, err)
    weights_fpath = os.path.join(WEIGHTS_PATH, weights_fname)
    torch.save({
            'startEpoch': epoch,
            'loss':loss,
            'error': err,
            'state_dict': model.state_dict()
        }, weights_fpath)
    shutil.copyfile(weights_fpath, WEIGHTS_PATH+'latest.pth')

def load_weights(model, fpath):
    print("loading weights '{}'".format(fpath))
    weights = torch.load(fpath, map_location='cpu')
    startEpoch = weights['startEpoch']
    model.load_state_dict(weights['state_dict'])
    print("loaded weights (lastEpoch {}, loss {}, error {})"
          .format(startEpoch-1, weights['loss'], weights['error']))
    return startEpoch

def save_loss(losses, loss_label, LOSSES_PATH, RESULTS_PATH, plot=True):
    loss_fname = 'losses-' + loss_label + '.pth'
    loss_fpath = os.path.join(LOSSES_PATH, loss_fname)
    torch.save(losses, loss_fpath)
    if plot:
        plot_loss_lcc(losses, RESULTS_PATH)
        
def load_loss(loss_fname, LOSSES_PATH, RESULTS_PATH, plot=True):
    loss_fpath = os.path.join(LOSSES_PATH, loss_fname)
    losses = torch.load(loss_fpath)
    if plot:
        plot_loss_lcc(losses, RESULTS_PATH)
        
def adjust_lr(lr, decay, optimizer, cur_epoch, n_epochs):
    """Sets the learning rate to the initially
        configured `lr` decayed by `decay` every `n_epochs`"""
    new_lr = lr * (decay ** (cur_epoch // n_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def plot_loss(losses):
    leng = len(losses)
    trn_loss, val_loss = [], []
    epochs = []
    for i in range(leng):
        epochs.append(losses[i][0])
        trn_loss.append(losses[i][1])
        val_loss.append(losses[i][3])
    #plot loss
    plt.figure('Loss/Error curves')
    plt.title('Loss curve')
    plt.plot(epochs, trn_loss, label='train')
    plt.plot(epochs, val_loss, label='val')
#    plt.ylim((0, 0.15))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()  
    plt.grid(True)
    plt.tight_layout()
    plt.show() 
    
def plot_loss_mae(losses):
    leng = len(losses)
    trn_loss, val_loss = [], []
    epochs = []
    trn_ed, val_ed = [], []
    for i in range(leng):
        epochs.append(losses[i][0])
        trn_loss.append(losses[i][1])
        trn_ed.append(losses[i][2])
        val_loss.append(losses[i][3])
        val_ed.append(losses[i][4])
    #plot loss
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Loss & MAE curve')
    line1 = ax1.plot(epochs, trn_loss, label='train loss')
    line2 = ax1.plot(epochs, val_loss, label='val loss')
    ax2 = ax1.twinx()
    line3 = ax2.plot(epochs, trn_ed, '-r', label='train mae')
    line4 = ax2.plot(epochs, val_ed, '-b', label='val mae')
    #set legend in the fig
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    fig.legend(lines, labels, loc=1, 
               bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    ax2.set_ylabel('mae')
    ax1.grid(True)
    plt.tight_layout()
    plt.show()
    
def plot_loss_lcc(losses, RESULTS_PATH):
    leng = len(losses)
    trn_loss, val_loss = [], []
    epochs = []
    trn_ed, val_ed = [], []
    for i in range(leng):
        epochs.append(losses[i][0])
        trn_loss.append(losses[i][1])
        trn_ed.append(losses[i][2])
        val_loss.append(losses[i][3])
        val_ed.append(losses[i][4])
    #plot loss
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Loss curves')
    ax1.plot(epochs, trn_loss, label='train loss')
    ax1.plot(epochs, val_loss, label='val loss')
    ax1.plot(epochs, trn_ed, '-r', label='train lcc')
    ax1.plot(epochs, val_ed, '-b', label='val lcc')
    #set legend in the fig
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    ax1.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_PATH + 'newlossfig.png')
    plt.show()

def train(model, trn_loader, optimizer, criterion, epoch):
    model.train()
    trn_loss, trn_lcc, trn_mae = 0, 0, 0 
    for _, (mov, ref) in enumerate(trn_loader):
        mov = mov.cuda()
        ref = ref.cuda()
        
        optimizer.zero_grad()
        warped, flow = model(mov, ref)

        loss1 = criterion['lcc'](warped, ref)
        loss2 = criterion['mse'](warped, ref)
        loss = loss1 + criterion['lambda'] * loss2

        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        
        trn_loss += loss.item()
        trn_lcc += loss1.item()
        trn_mae += torch.mean(torch.abs(flow)).item()

    trn_loss /= len(trn_loader)
    trn_lcc /= len(trn_loader)
    trn_mae /= len(trn_loader)
    return trn_loss, trn_lcc, trn_mae

def test(model, test_loader, criterion, epoch):
    model.eval()
    test_loss, test_lcc, test_mae = 0, 0, 0
    for mov, ref in test_loader:
        mov = mov.cuda() 
        ref = ref.cuda()

        with torch.no_grad():
            warped, flow = model(mov, ref)
            
            loss1 = criterion['lcc'](warped, ref)
            loss2 = criterion['mse'](warped, ref)
            loss = loss1 + criterion['lambda'] * loss2
            
        torch.cuda.empty_cache()
        
        test_loss += loss.item()
        test_lcc += loss1.item()
        test_mae += torch.mean(torch.abs(flow)).item()
        
    test_loss /= len(test_loader)
    test_lcc /= len(test_loader)
    test_mae /= len(test_loader)
    return test_loss, test_lcc, test_mae

def train0(model, trn_loader, optimizer, criterion, epoch):
    model.train()
    trn_loss, trn_lcc, trn_mae = 0, 0, 0 
    for _, (mov, ref) in enumerate(trn_loader):
        mov = mov.cuda()
        ref = ref.cuda()
        
        optimizer.zero_grad()
        warped, flow, warped3, flow3 = model(mov, ref)

        loss1 = criterion['lcc'](warped, ref)
        loss2 = criterion['grad'](flow)
        loss3 = criterion['lcc'](warped3, ref)
        loss = criterion['alpha']*loss3 + (loss1 + criterion['lambda']*loss2)
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        
        trn_loss += loss1.item()
        trn_lcc += loss3.item()
        trn_mae += torch.mean(torch.abs(flow+flow3)).item()

    trn_loss /= len(trn_loader)
    trn_lcc /= len(trn_loader)
    trn_mae /= len(trn_loader)
    return trn_loss, trn_lcc, trn_mae

def test0(model, test_loader, criterion, epoch):
    model.eval()
    test_loss, test_lcc, test_mae = 0, 0, 0
    for mov, ref in test_loader:
        mov = mov.cuda()#Variable() 
        ref = ref.cuda()#Variable()

        with torch.no_grad():
            warped, flow, warped3, flow3 = model(mov, ref)
            
            loss1 = criterion['lcc'](warped, ref)
#            loss2 = criterion['grad'](flow3)
            loss3 = criterion['lcc'](warped3, ref)
#            loss = criterion['alpha']*loss3 + loss1 + criterion['lambda']*loss2
            
        torch.cuda.empty_cache()
        
        test_loss += loss1.item()
        test_lcc += loss3.item()
        test_mae += torch.mean(torch.abs(flow+flow3)).item()
        
    test_loss /= len(test_loader)
    test_lcc /= len(test_loader)
    test_mae /= len(test_loader)
    return test_loss, test_lcc, test_mae
 
