#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 15:21:40 2018
@author: user
from: https://github.com/voxelmorph/voxelmorph/blob/master/src/losses.py
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
 
class LCC(nn.Module):
    """
    local (over window) normalized cross correlation (square)
    """
    def __init__(self, win=[9, 9], eps=1e-5):
        super(LCC, self).__init__()
        self.win = win
        self.eps = eps
        
    def forward(self, I, J):
        I2 = I.pow(2)
        J2 = J.pow(2)
        IJ = I * J
        
        filters = Variable(torch.ones(1, 1, self.win[0], self.win[1]))
        if I.is_cuda:#gpu
            filters = filters.cuda()
        padding = (self.win[0]//2, self.win[1]//2)
        
        I_sum = F.conv2d(I, filters, stride=1, padding=padding)
        J_sum = F.conv2d(J, filters, stride=1, padding=padding)
        I2_sum = F.conv2d(I2, filters, stride=1, padding=padding)
        J2_sum = F.conv2d(J2, filters, stride=1, padding=padding)
        IJ_sum = F.conv2d(IJ, filters, stride=1, padding=padding)
        
        win_size = self.win[0]*self.win[1]
 
        u_I = I_sum / win_size
        u_J = J_sum / win_size
        
        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size
 
        cc = cross*cross / (I_var*J_var + self.eps)#np.finfo(float).eps
        lcc = -1.0 * torch.mean(cc) + 1
        return lcc
    
class GCC(nn.Module):
    """
    global normalized cross correlation (sqrt)
    """
    def __init__(self):
        super(GCC, self).__init__()
 
    def forward(self, I, J):
        I2 = I.pow(2)
        J2 = J.pow(2)
        IJ = I * J
        #average value
        I_ave, J_ave= I.mean(), J.mean()
        I2_ave, J2_ave = I2.mean(), J2.mean()
        IJ_ave = IJ.mean()
        
        cross = IJ_ave - I_ave * J_ave
        I_var = I2_ave - I_ave.pow(2)
        J_var = J2_ave - J_ave.pow(2)
 
#        cc = cross*cross / (I_var*J_var + np.finfo(float).eps)#1e-5
        cc = cross / (I_var.sqrt() * J_var.sqrt() + np.finfo(float).eps)#1e-5
 
        return -1.0 * cc + 1
    
class Grad(nn.Module):
    """
    N-D gradient loss
    """
    def __init__(self, penalty='l2'):
        super(Grad, self).__init__()
        self.penalty = penalty
    
    def _diffs(self, y):#y shape(bs, nfeat, vol_shape)
        ndims = y.ndimension() - 2
        df = [None] * ndims
        for i in range(ndims):
            d = i + 2#y shape(bs, c, d, h, w)
            # permute dimensions to put the ith dimension first
#            r = [d, *range(d), *range(d + 1, ndims + 2)]
            y = y.permute(d, *range(d), *range(d + 1, ndims + 2))
            dfi = y[1:, ...] - y[:-1, ...]
            
            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
#            r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(*range(1, d + 1), 0, *range(d + 1, ndims + 2))
        
        return df
    
    def forward(self, pred):
        ndims = pred.ndimension() - 2
        if pred.is_cuda:
            df = Variable(torch.zeros(1).cuda())
        else:
            df = Variable(torch.zeros(1))
        for f in self._diffs(pred):
            if self.penalty == 'l1':
                df += f.abs().mean() / ndims
            else:
                assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
                df += f.pow(2).mean() / ndims
        return df
 
class Bend_Penalty(nn.Module):
    """
    Bending Penalty of the spatial transformation (2D)
    """
    def __init__(self):
        super(Bend_Penalty, self).__init__()
    
    def _diffs(self, y, dim):#y shape(bs, nfeat, vol_shape)
        ndims = y.ndimension() - 2
        d = dim + 2
        # permute dimensions to put the ith dimension first
#       r = [d, *range(d), *range(d + 1, ndims + 2)]
        y = y.permute(d, *range(d), *range(d + 1, ndims + 2))
        dfi = y[1:, ...] - y[:-1, ...]
        
        # permute back
        # note: this might not be necessary for this loss specifically,
        # since the results are just summed over anyway.
#       r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
        df = dfi.permute(*range(1, d + 1), 0, *range(d + 1, ndims + 2))
        
        return df
    
    def forward(self, pred):#shape(B,C,H,W)
        Ty = self._diffs(pred, dim=0)
        Tx = self._diffs(pred, dim=1)
        Tyy = self._diffs(Ty, dim=0)
        Txx = self._diffs(Tx, dim=1)
        Txy = self._diffs(Tx, dim=0)
        p = Tyy.pow(2).mean() + Txx.pow(2).mean() + 2 * Txy.pow(2).mean()
        
        return p
 
class IDloss(nn.Module):
    """
    loss between affine transformation and identity transf.
    """
    def __init__(self, penalty='l1'):
        super(IDloss, self).__init__()
        self.penalty = penalty
        self.id = torch.FloatTensor([1, 0, 0, 0, 1, 0])
 
    def forward(self, theta):
        if theta.is_cuda:
            ID = Variable(self.id.cuda())
        else:
            ID = Variable(self.id)
        ID = ID.repeat(theta.size(0), 1).view(theta.shape)
        if self.penalty == 'l1':
            loss = torch.mean(torch.abs(theta - ID))
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            loss = torch.mean(torch.pow(theta - ID, 2))
        return loss
 
#test 
#a=torch.zeros(1, 2, 30, 40)
#grad=Grad()
#loss = grad(a)
        
#l = Bend_Penalty()
#a=torch.zeros(1, 2, 30, 40)
#c=l(a)
#print(c)