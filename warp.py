#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 11:13:22 2018
@author: user
https://github.com/NVlabs/PWC-Net/blob/master/PyTorch/models/PWCNet.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
 
class Warper2d(nn.Module):
    def __init__(self, img_size):
        super(Warper2d, self).__init__()
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
#        img_src: [B, 1, H1, W1] (source image used for prediction, size 32)
        img_smp: [B, 1, H2, W2] (image for sampling, size 44)
        flow: [B, 2, H1, W1] flow predicted from source image pair
        """
        self.img_size = img_size
        H, W = img_size, img_size
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,H,W)
        yy = yy.view(1,H,W)
        self.grid = torch.cat((xx,yy),0).float() # [2, H, W]
            
    def forward(self, flow, img):
        grid = self.grid.repeat(flow.shape[0],1,1,1)#[bs, 2, H, W]
        if img.is_cuda:
            grid = grid.cuda()
#        if flow.shape[2:]!=img.shape[2:]:
#            pad = int((img.shape[2] - flow.shape[2]) / 2)
#            flow = F.pad(flow, [pad]*4, 'replicate')#max_disp=6, 32->44
        vgrid = Variable(grid, requires_grad = False) + flow
 
        # scale grid to [-1,1] 
#        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/(W-1)-1.0 #max(W-1,1)
#        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/(H-1)-1.0 #max(H-1,1)
        vgrid = 2.0*vgrid/(self.img_size-1)-1.0 #max(W-1,1)
 
        vgrid = vgrid.permute(0,2,3,1)        
        output = F.grid_sample(img, vgrid)
#        mask = Variable(torch.ones(img.size())).cuda()
#        mask = F.grid_sample(mask, vgrid)
#        
#        mask[mask<0.9999] = 0
#        mask[mask>0] = 1
        
        return output#*mask
 
class Warper3d(nn.Module):
    def __init__(self, img_size):
        super(Warper3d, self).__init__()
        """
        warp an image, according to the optical flow
        image: [B, 1, D, H, W] image for sampling
        flow: [B, 3, D, H, W] flow predicted from source image pair
        """
        self.img_size = img_size
        D, H, W = img_size
        # mesh grid 
        xx = torch.arange(0, W).view(1,1,-1).repeat(D,H,1).view(1,D,H,W)
        yy = torch.arange(0, H).view(1,-1,1).repeat(D,1,W).view(1,D,H,W)
        zz = torch.arange(0, D).view(-1,1,1).repeat(1,H,W).view(1,D,H,W)
        self.grid = torch.cat((xx,yy,zz),0).float() # [3, D, H, W]
            
    def forward(self, img, flow):
        grid = self.grid.repeat(flow.shape[0],1,1,1,1)#[bs, 3, D, H, W]
#        mask = torch.ones(img.size())
        if img.is_cuda:
            grid = grid.cuda()
#            mask = mask.cuda()
        vgrid = grid + flow
 
        # scale grid to [-1,1]
        D, H, W = self.img_size
        vgrid[:,0] = 2.0*vgrid[:,0]/(W-1)-1.0 #max(W-1,1)
        vgrid[:,1] = 2.0*vgrid[:,1]/(H-1)-1.0 #max(H-1,1)
        vgrid[:,2] = 2.0*vgrid[:,2]/(D-1)-1.0 #max(H-1,1)
 
        vgrid = vgrid.permute(0,2,3,4,1)#[bs, D, H, W, 3]        
        output = F.grid_sample(img, vgrid, padding_mode='border')#, mode='nearest'
#        mask = F.grid_sample(mask, vgrid)#, mode='nearest'        
#        mask[mask<0.9999] = 0
#        mask[mask>0] = 1
        
        return output#*mask