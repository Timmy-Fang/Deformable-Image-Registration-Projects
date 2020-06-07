
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 22:26:10 2019

@author: user
"""
#import os 
import numpy as np
#from pathlib import Path
#import matplotlib.pyplot as plt

#import torch

import transform
from utils import test_utils, dataset_utils, visual
#% load image
case = 8 #1-10, for validation
assert case in range(1, 11), 'Error: case_index is out of range.'

root = '/home/user/fangqiming/data/DIRLab/'
path = root + 'Case%gPack/Images_dcm/' % case
#save path
#save_root = 'E:/IR/dir/data/'
#case_dir = save_root + 'case%g/' % case
#Path(case_dir).mkdir(exist_ok=True)

mov_fname = 'case%g_T00.dcm' % case
ref_fname = 'case%g_T50.dcm' % case                
mov_arr, spacing0 = dataset_utils.load_dcm(path + mov_fname)
ref_arr, _ = dataset_utils.load_dcm(path + ref_fname)
mov_arr = np.flip(mov_arr, 1)# z,y,x (D,H,W)
ref_arr = np.flip(ref_arr, 1)
spacingf = np.flip(spacing0, axis=0)

#show
#visual.view_slice(mov_arr, 70, 'mov')
#visual.view_slice(ref_arr, 70, 'ref')
#%% preprocessing
#1.clamp, one norm
Threshold = [-1000, 500]#[-1000, -200]
mov_arr0 = mov_arr - 1024 #units: HU
ref_arr0 = ref_arr - 1024 #units: HU
norm = transform.OneNorm(Threshold)
mov1 = norm(mov_arr0)
ref1 = norm(ref_arr0) 

##2.resample
new_spacing= [2.5,1,1]
mov1rs, spacing1 = transform.ReSample(mov1, spacingf, new_spacing)
ref1rs, _ = transform.ReSample(ref1, spacingf, new_spacing)

###3.crop
img_dims = [96, 208, 272]
Residual = [[0,0,0],[0,-15,0],[0,0,0],[0,0,0],[0,10,0],
            [0,-20,20], [0,-20,10], [0,-65,0], [-16,-30,0], [-10,-20,0]]
residual = Residual[case-1]
cc = transform.CenterCrop(residual)

mov1rs = mov1rs.astype('float32')
ref1rs = ref1rs.astype('float32')
mov1cc, delta = cc(mov1rs, img_dims)
ref1cc, _ = cc(ref1rs, img_dims)
# change the coordinate system
delta0 = np.flip(delta, 0)

#show
#visual.view_slice(mov1cc, 60, 'mov')
#visual.view_slice(ref1cc, 60, 'ref')
#%% load flow
lamb = 10
#alpha = 0.1

RESULTS_PATH = 'results/'
flow_folder = RESULTS_PATH + 'flow/'
#save flow
flow_fname = 'flow_case%g_lamb%g_reso2_noseg.npy' % (case, lamb)
#flow_fname = 'flow_case%g_lamb%g_alp%g.npy' % (case, lamb, alpha)
flow = np.load(flow_folder+flow_fname)

#%% load landmarks
if case in range(1, 6):
    lmk_path = root + 'Case%gPack/ExtremePhases/' % case
    mov_lmk_fname = 'Case%g_300_T00_xyz.txt' % case
    ref_lmk_fname = 'Case%g_300_T50_xyz.txt' % case
else:
    lmk_path = root + 'Case%gPack/extremePhases/' % case
    mov_lmk_fname = 'case%g_dirLab300_T00_xyz.txt' % case
    ref_lmk_fname = 'case%g_dirLab300_T50_xyz.txt' % case

mov_lmk = np.loadtxt(lmk_path+mov_lmk_fname, dtype=int)
ref_lmk = np.loadtxt(lmk_path+ref_lmk_fname, dtype=int)

# preprocessing landmarks
resize_factor = spacingf / spacing1 #new_spacing
resize_factor0 = np.flip(resize_factor, 0)
spacing1f = np.flip(spacing1, 0)

mov_lmk0 = (mov_lmk-1) * resize_factor0 + delta0
ref_lmk0 = (ref_lmk-1) * resize_factor0 + delta0

#flow sampling
ref_lmk_index = np.round(ref_lmk0).astype('int32')

ref_lmk1 = ref_lmk0.copy()
for i in range(300):
    wi, hi, di = ref_lmk_index[i]
    w0, h0, d0 = flow[0, :, di, hi, wi]
    ref_lmk1[i] += [w0, h0, d0]
#    break
    
# compute TRE
#no reg
tre_mean_bf, tre_std_bf, diff_br = test_utils.compute_tre(mov_lmk0, ref_lmk0, spacing1f)
print('TRE-before reg, mean: {:.2f},std: {:.2f}'.format(
        tre_mean_bf, tre_std_bf))
#with reg
tre_mean_af, tre_std_af, diff_ar = test_utils.compute_tre(mov_lmk0, ref_lmk1, spacing1f)
print('TRE-after reg, mean: {:.2f},std: {:.2f}'.format(
        tre_mean_af, tre_std_af))

#mean, std of 10 cases

visual.corr_plot(diff_br, diff_ar, mode='tre')

#%% test landmarks
#lmk_id = 30
##before resampling
#lm1_mov = mov_lmk[lmk_id]-1
#lm1_ref = ref_lmk[lmk_id]-1
#
#fig, ax = plt.subplots(1, 2)
#ax[0].imshow(mov1[lm1_mov[2]], cmap='gray')
#ax[0].scatter([lm1_mov[0]], [lm1_mov[1]], 50, color='red')
#ax[0].set_title('mov')
#ax[1].imshow(ref1[lm1_ref[2]], cmap='gray')
#ax[1].scatter([lm1_ref[0]], [lm1_ref[1]], 50, color='red')
#ax[1].set_title('ref')
#plt.show()
#    
##after resampling
#mov_lmk_int = np.round(mov_lmk0).astype('int32')
#ref_lmk_int = np.round(ref_lmk0).astype('int32')
#
#lm1_mov0 = mov_lmk_int[lmk_id]
#lm1_ref0 = ref_lmk_int[lmk_id]
#fig, ax = plt.subplots(1, 2)
#ax[0].imshow(mov1cc[lm1_mov0[2]], cmap='gray')
#ax[0].scatter([lm1_mov0[0]], [lm1_mov0[1]], 50, color='red')
#ax[0].set_title('mov')
#ax[1].imshow(ref1cc[lm1_ref0[2]], cmap='gray')
#ax[1].scatter([lm1_ref0[0]], [lm1_ref0[1]], 50, color='red')
#ax[1].set_title('ref')
#plt.show()