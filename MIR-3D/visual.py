#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 20:54:41 2019

@author: user
modified from: https://github.com/dykuang/Medical-image-registration/
"""
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

from utils import test_utils

def view_slice(image, index, label=None):
    if label is not None:
        title = label + '_slice_%g' % index
    else:
        title = 'slice_%g' % index
    plt.figure()
    plt.title(title)
    plt.imshow(image[index], 'gray')
    plt.show()

def view_diff(m, r, w, index):
    
    diff0 = m - r
#    ad_diff0 = test_utils.DiffAdjust(diff0[index])
    ad_diff0 = diff0[index]
    diff1 = w - r
#    ad_diff1 = test_utils.DiffAdjust(diff1[index])
    ad_diff1 = diff1[index]
    fig = plt.figure()
    ax1 = fig.add_subplot(2,3,1)
    ax1.imshow(r[index], 'gray')
    ax1.set_title('fixed')
    ax1.axis('off')
    ax2 = fig.add_subplot(2,3,2)
    ax2.imshow(m[index], 'gray')
    ax2.set_title('moving')
    ax2.axis('off')
    ax3 = fig.add_subplot(2,3,3)
    ax3.imshow(w[index], 'gray')
    ax3.set_title('moved')
    ax3.axis('off')
    ax5 = fig.add_subplot(2,3,5)
    ax5.imshow(ad_diff0, 'gray')
    ax5.set_title('before reg')
    ax5.axis('off')
    ax6 = fig.add_subplot(2,3,6)
    ax6.imshow(ad_diff1, 'gray')
    ax6.set_title('after reg')
    ax6.axis('off')
         
    plt.show()
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

    D1 = (D_x[...,0]+1)*((D_y[...,1]+1)*(D_z[...,2]+1) - D_z[...,1]*D_y[...,2])
    D2 = (D_x[...,1])*(D_y[...,0]*(D_z[...,2]+1) - D_y[...,2]*D_x[...,0])
    D3 = (D_x[...,2])*(D_y[...,0]*D_z[...,1] - (D_y[...,1]+1)*D_z[...,0])

    D = np.abs(D1-D2+D3)
    
    return D

#==============================================================================
# plot an array of images for comparison
#==============================================================================    

def show_sample_slices(sample_list, name_list, Jac = False, cmap = 'gray', attentionlist=None):
    num = len(sample_list)
    fig, ax = plt.subplots(1, num)
    for i in range(num):
        if Jac:
            im = ax[i].imshow(sample_list[i], cmap, norm=MidpointNormalize(midpoint=1))
        else:
            im = ax[i].imshow(sample_list[i], cmap)
        ax[i].set_title(name_list[i])
        ax[i].axis('off')
        if attentionlist:
            ax[i].add_artist(attentionlist[i])
    plt.subplots_adjust(wspace=0.1)
    cbar = plt.colorbar(im, ax=[ax[i] for i in range(num)], shrink=0.35)
    cbar.set_label('Jacobian determinant')
    cbar.set_ticks(np.linspace(0, 5, 6, endpoint=True))

#==============================================================================
# Define a custom colormap for visualiza Jacobian
#==============================================================================
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

#==============================================================================
#  Overlay two images contained in numpy arrays  
#==============================================================================
def overlay(img1, img2, cmap1=None, cmap2=None, alpha=0.4, Jac = False):
#    plt.figure()
    plt.imshow(img1, cmap=cmap1)
    if Jac:
        plt.imshow(img2, cmap=cmap2, norm=MidpointNormalize(midpoint=1),alpha=alpha)
    plt.imshow(img2, cmap=cmap2, alpha=alpha)
    plt.axis('off')

def corr_plot(diff_br, diff_ar, mode='tre', title=None, xlabel=None, ylabel=None):
    '''plot x or y correlation scatter
    params:
        tre_br: TRE before registration, ndarray of shape(300, 3)
        tre_ar: TRE after registration, ndarray of shape(300, 3)
        title: the title of plot, default:'Target-Prediction Correlation'
        mode: 'tre' or 'xyz', default:'tre'
    '''
    assert mode in {'tre', 'xyz'}
    if title == None:
        title='TRE scatterplot'
    if xlabel == None:
        xlabel = 'TRE before registration (mm)'
    if ylabel == None:
        ylabel = 'TRE after registration (mm)'
    if mode=='tre':
        tre_br = diff_br.pow(2).sum(1).sqrt()
        tre_ar = diff_ar.pow(2).sum(1).sqrt()
        plt.figure()
        plt.scatter(tre_br, tre_ar, s=5, alpha=0.2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim((-2, 33))
        plt.ylim((-2, 33))
        plt.title(title)
#        plt.grid(True)
    else:
        fig, ax = plt.subplots(1,3,figsize=[15,5])
#        diag_x, diag_y = [-6,6], [-6,6]#diagonal line
#        ax1.plot(diag_x, diag_y, 'b')
#        ax2.plot(diag_x, diag_y, 'b')
        ax[0].scatter(diff_br[:,0], diff_ar[:,0], s=5, alpha=0.2)
        ax[0].set_title('X direction')
#        ax[0].set_xlim((-5,5))
#        ax[0].set_ylim((-5,5))
        ax[0].set_xlabel(xlabel)
        ax[0].set_ylabel(ylabel)
        ax[1].scatter(diff_br[:,1], diff_ar[:,1], s=5, alpha=0.5)
        ax[1].set_title('Y direction')
#        ax[1].set_xlim((-5,5))
#        ax[1].set_ylim((-5,5))
        ax[1].set_xlabel(xlabel)
#        ax[1].set_ylabel('TRE after registration (mm)')
        ax[2].scatter(diff_br[:,2], diff_ar[:,2], s=5, alpha=0.5)
        ax[2].set_title('Z direction')
#        ax[2].set_xlim((-12,5))
#        ax[2].set_ylim((-12,5))
#        fig.suptitle(title)
        ax[2].set_xlabel(xlabel)
#        ax[1].set_ylabel('TRE after registration (mm)')
#        ax[0].grid(True)
#        ax[1].grid(True)
#        ax[2].grid(True)
    
#    plt.tight_layout()
    plt.show()