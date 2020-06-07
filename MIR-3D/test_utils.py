# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 11:13:14 2019

@author: qiming.fang
"""
import numpy as np
import matplotlib.pyplot as plt

import torch

# plot def
def flow2grid(flow):
    '''input:tensor flow, shape->(bs, 2, h, w), unit-pixel.
    output:tensor grid, shape->(bs, h, w, 2), range(-1, 1)'''
    H, W = flow.shape[2], flow.shape[3]
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,H,W)
    yy = yy.view(1,H,W)
    grid = torch.cat((xx,yy),0).float().repeat(flow.shape[0],1,1,1)
    
    vgrid = grid + flow
#    scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/(W-1)-1.0 #max(W-1,1)
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/(H-1)-1.0 #max(H-1,1)
    vgrid = vgrid.permute(0,2,3,1)
    return vgrid    

def view_pred(mov, ref, warped, grid, diffb, diffa, lossb, lossa, grad, label=None):
    '''input, bs=1
    mov, ref, warped: tensor, shape(bs, 1, h, w)
    grid: tensor, shape(bs, h, w, 2)
    lossb, lossa: float numpy
    label: string
    '''
    if label is not None:
        fig_label = label
    else:
        fig_label = 'sample'
    fig_label = fig_label+ '-lcc_bf:{:.4f},lcc_af:{:.4f},grad:{:.4f}'.format(
            lossb, lossa, grad)
    fig = plt.figure(fig_label)
    ax1 = fig.add_subplot(231)
    ax1.set_title('mov')
    ax1.imshow(mov.numpy()[0,0], 'gray')
    ax2 = fig.add_subplot(232)
#    ax2.set_title('warped')
#    ax2.imshow(warped.numpy()[0,0], 'gray')
    ax2.set_title('diff')
    ax2.imshow(warped[0], 'gray')
    ax3 = fig.add_subplot(233)
    ax3.set_title('ref')
    ax3.imshow(ref.numpy()[0,0], 'gray')
    #diff
    ax3 = fig.add_subplot(234)
    ax3.set_title('before Reg')
    ax3.imshow(diffb, 'gray')
    ax3 = fig.add_subplot(235)
    ax3.set_title('after Reg')
    ax3.imshow(diffa, 'gray')
    #grid2contour  shape(h, w, 2)
    grid0 = grid[0]
    assert grid0.ndimension() == 3
    ax4 = fig.add_subplot(236)
    ax4.set_title('deform field')
    x = np.arange(-1, 1, 2/grid0.size(0))
    y = np.arange(-1, 1, 2/grid0.size(1))
    X, Y = np.meshgrid(x, y)
    Z1 = grid0.numpy()[:,:,0] + 2#remove the dashed line
    Z1 = Z1[::-1]#vertial flip
    Z2 = grid0.numpy()[:,:,1] + 2
    ax4.contour(X, Y, Z1, 15, colors='k')
#    plt.clabel(CS, fontsize=9, inline=1)
    ax4.contour(X, Y, Z2, 15, colors='k')
#    plt.clabel(CS, fontsize=9, inline=1)
    ax4.set_xticks(()), ax4.set_yticks(())
            
    plt.tight_layout()
    plt.show()

def compare_diff(diff, diffb, diffa, lossb, lossa, grad, label=None):
    '''
    compare the diff images that before Reg and after Reg.\
    :params:
        diff: tensor, shape(1, 1, h, w)
        diffa, diffb: array, shape(h, w)
        lossb, lossa: float numpy
        label: string
    '''
    if label is not None:
        fig_label = label
    else:
        fig_label = 'sample'
    fig_label = fig_label+ '-lcc_bf:{:.4f},lcc_af:{:.4f},grad:{:.4f}'.format(
            lossb, lossa, grad)
    fig = plt.figure(fig_label)
    #diff
    ax1 = fig.add_subplot(131)
    ax1.set_title('before Reg')
    ax1.imshow(diffb, 'gray')
    ax2 = fig.add_subplot(132)
    ax2.set_title('diff image')
    ax2.imshow(diff.numpy()[0, 0], 'gray')
    ax3 = fig.add_subplot(133)
    ax3.set_title('after Reg')
    ax3.imshow(diffa, 'gray')
            
#    plt.tight_layout()
    plt.show()

def DiffAdjust(diff, cut=[0.001, 0.999]):
    #param: numpy.ndarray diff
    eps = 1e-5
    copy = diff.copy()
    # cut
    sort0 = sorted(copy.flatten().tolist())
    min_idx = int(len(sort0) * cut[0])
    max_idx = int(len(sort0) * cut[1])
    max = sort0[max_idx]
    min = sort0[min_idx]
    
    copy[copy>=max] = max
    copy[copy<=min] = min
            
    copy -= min
    ad_diff = copy /(max - min + eps)
    return ad_diff

def compute_tre(mov_lmk, ref_lmk, spacing):
    #TRE, unit: mm

    diff = (ref_lmk - mov_lmk) * spacing
    diff = torch.Tensor(diff)
    tre = diff.pow(2).sum(1).sqrt()
    mean, std = tre.mean(), tre.std()
    return mean, std, diff
 