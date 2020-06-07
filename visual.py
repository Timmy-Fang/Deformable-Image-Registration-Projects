#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 19:44:03 2019
@author: user
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
 
import torch
 
#% plot def
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
  
def view_comp(mov, fix, warped, diffgt, diffb, diffa, label=None, Path=None, save=False):
    '''input, bs=1
    mov, diffgt, ... : array, shape(h, w)
    label: string
    '''
    if label is None:
        label = 'sample' 
    fig = plt.figure(label)
    ax1 = fig.add_subplot(231)
    ax1.set_title('moving')
    ax1.imshow(mov, 'gray')
    plt.axis('off')
    ax2 = fig.add_subplot(232)
    ax2.set_title('warped')
    ax2.imshow(warped, 'gray')
    plt.axis('off')
    ax3 = fig.add_subplot(233)
    ax3.set_title('fixed')
    ax3.imshow(fix, 'gray')
    plt.axis('off')
    #diff
    ax3 = fig.add_subplot(234)
    ax3.set_title('before Reg')
    ax3.imshow(diffb, 'gray')
    plt.axis('off')
    ax3 = fig.add_subplot(235)
    ax3.set_title('after Reg-mi')
    ax3.imshow(diffgt, 'gray')
    plt.axis('off')
    ax4 = fig.add_subplot(236)
    ax4.set_title('after Reg')
    ax4.imshow(diffa, 'gray')
    plt.axis('off')
    if save:
        plt.savefig(Path + label, dpi=300)
    else:
        plt.show()
    
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
 
def get_max_min(diff, mask):
    '''
    get the max & min value of the image under the given mask.
    :param:
        diff: numpy, shape(1, 576, 576)
        mask: numpy, size(576, 576)
    '''
    masked = diff[0] * mask
    flat = masked.flatten().astype('int')
    rmzero = np.delete(flat, np.where(flat==0)) # remove zero value
    sort0 = np.array(sorted(rmzero.tolist()))
    max, min = sort0.max(), sort0.min()
    return max, min
 
'''modified from: https://github.com/dykuang/Medical-image-registration/'''
#==============================================================================
# Calculate the Determinent of Jacobian of the transformation
#==============================================================================
def Get_Jac(displacement):
    '''
    the expected input: displacement of shape(batch, H, W, D, channel)
    but the input dim: (batch, channel, D, H, W)
    '''
    displacement = np.transpose(displacement, (0,3,4,2,1))
    D_y = (displacement[:,1:,:-1,:-1,:] - displacement[:,:-1,:-1,:-1,:])
    D_x = (displacement[:,:-1,1:,:-1,:] - displacement[:,:-1,:-1,:-1,:])
    D_z = (displacement[:,:-1,:-1,1:,:] - displacement[:,:-1,:-1,:-1,:])
 
    D1 = (D_x[...,0]+1)*((D_y[...,1]+1)*(D_z[...,2]+1) - D_y[...,2]*D_z[...,1])
    D2 = (D_x[...,1])*(D_y[...,0]*(D_z[...,2]+1) - D_y[...,2]*D_z[...,0])
    D3 = (D_x[...,2])*(D_y[...,0]*D_z[...,1] - (D_y[...,1]+1)*D_z[...,0])
    
    D = D1-D2+D3#np.abs()
    
    return D
 
#==============================================================================
# plot an array of images for comparison
#==============================================================================    
 
#def show_sample_slices(sample_list, name_list, Jac = False, cmap = 'gray', attentionlist=None):
#    num = len(sample_list)
#    fig, ax = plt.subplots(1, num)
#    for i in range(num):
#        if Jac:
#            im = ax[i].imshow(sample_list[i], cmap, norm=MidpointNormalize(midpoint=1))
#        else:
#            im = ax[i].imshow(sample_list[i], cmap)
#        ax[i].set_title(name_list[i])
#        ax[i].axis('off')
#        if attentionlist:
#            ax[i].add_artist(attentionlist[i])
#    plt.subplots_adjust(wspace=0.1)
#    cbar = plt.colorbar(im, ax=[ax[i] for i in range(num)], shrink=0.35)
#    cbar.set_label('Jacobian determinant')
#    cbar.set_ticks(np.linspace(0, 5, 6, endpoint=True))
 
#==============================================================================
# Define a custom colormap for visualiza Jacobian
#==============================================================================
# Example of making your own norm.  Also see matplotlib.colors.
# From Joe Kington: This one gives two different linear ramps:
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)
 
    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        masked = np.interp(value, x, y)
        less = np.ma.masked_less(value, self.vmin)
        masked += less.mask * value
        return np.ma.masked_array(masked)
 
def show_sample_jac(sample, name, under='lime'):
    #figure
    cmap = plt.cm.bwr
    cmap.set_under(under)
    vmin, vmax = sample.min(), sample.max()
    vmin = vmin if vmin>0 else 0
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=1, clip=False)
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(sample, cmap=cmap, norm=norm)
    ax.set_title(name)
    ax.axis('off')
    #colorbar
    extend='min' if sample.min()<0 else 'neither'
    cbar = plt.colorbar(im, ax=ax, shrink=0.5,extend=extend)
    tick = np.array([vmin] + np.arange(1, vmax).tolist() + [vmax])
    vmin0 = np.floor(np.array(vmin*10))/10
    vmax0 = np.floor(np.array(vmax*10))/10
    ticklabel = [str(vmin0)]
    for i in np.arange(1, vmax):
        ticklabel += [str(i)]
    cbar.set_label('Jacobian determinant')
    cbar.set_ticks(tick)
    cbar.set_ticklabels(ticklabel+[vmax0])
 
#count negative values
def count_jac(jac):
    #count the num of negative jac values, and output its percent
    jac[jac>=0] = 0
    jac[jac<0] = 1
    num = np.sum(jac)
    percent = num / np.size(jac)
    return num, percent
 
def Norm(array, maxv, minv, scale):
    # normalize the array by max-min.
    array[array>=maxv] = maxv
    array[array<=minv] = minv
    array_n = (array-minv)/(maxv-minv)
    array_n0 = array_n * scale[1] + scale[0]
    #let max and min as 1 and 0, respectively.
    array_n0[0,0] = 0
    array_n0[-1,-1] = 1
    return array_n
 
def ContrastAdjust(diffmi, diffb, diffa, mask, scale=[0.02,0.98]):
    #param: numpy.ndarray of the same shape
    masked = diffb * mask
    maxv = masked.max()
    masked[masked==0] ==maxv
    minv = masked.min()
    diffmi_n = Norm(diffmi, maxv, minv, scale)
    diffb_n = Norm(diffb, maxv, minv, scale)
    diffa_n = Norm(diffa, maxv, minv, scale)       
    return diffmi_n, diffb_n, diffa_n