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
#from torch.autograd import Variable
#import numpy as np

class LCC(nn.Module):
    """
    local (over window) normalized cross correlation (square)
    """
    def __init__(self, win=None, eps=1e-5):
        super(LCC, self).__init__()
        self.win = win
        self.eps = eps
        
    def forward(self, I, J):
        ndims = I.ndimension() - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
        
        # set window size
        if self.win is None:
            self.win = [9] * ndims

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I.pow(2)
        J2 = J.pow(2)
        IJ = I*J

        # compute filters
        filters = torch.ones([1, 1, *self.win])#Variable()
        win_size = torch.prod(torch.Tensor(self.win), dtype=torch.float)
        if I.is_cuda:#gpu
            filters = filters.cuda()
            win_size = win_size.cuda()
        padding = self.win[0]//2
        
        I_sum = conv_fn(I, filters, stride=1, padding=padding)
        J_sum = conv_fn(J, filters, stride=1, padding=padding)
        I2_sum = conv_fn(I2, filters, stride=1, padding=padding)
        J2_sum = conv_fn(J2, filters, stride=1, padding=padding)
        IJ_sum = conv_fn(IJ, filters, stride=1, padding=padding)

        u_I = I_sum/win_size
        u_J = J_sum/win_size
        
        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross*cross / (I_var*J_var + self.eps)#1e-5
        return - torch.mean(cc) + 1
    
class GCC(nn.Module):
    """
    global normalized cross correlation (sqrt)
    """
    def __init__(self, eps=1e-5):
        super(GCC, self).__init__()
        self.eps = eps
        
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
        cc = cross / (I_var.sqrt() * J_var.sqrt() + self.eps)#1e-5

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
            d = i + 1
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
            df = torch.zeros(1).cuda()#Variable()
        else:
            df = torch.zeros(1)#Variable()
        for f in self._diffs(pred):
            if self.penalty == 'l1':
                df += f.abs().mean() / ndims
            else:
                assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
                df += f.pow(2).mean() / ndims
        return df

#test        
#a = Variable(torch.rand((2,3,32,32,32)), requires_grad=True)
#grad=Grad()
#loss = grad(a)
#loss.backward()

#in1 = Variable(torch.rand((2,1,32,32,32)))
#in2 = Variable(torch.rand((2,1,32,32,32)))
#lcc = LCC()
#loss = lcc(in1, in2)