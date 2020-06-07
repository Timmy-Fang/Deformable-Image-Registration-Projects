#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 09:55:47 2019

@author: user
"""
import numpy as np
import time
from pathlib import Path
#import matplotlib.pyplot as plt

import torch
#import torch.optim as optim
#import torch.utils.data as data
import torchvision.transforms as transforms

import networks
#import dataset
import transform
from utils import train_utils, visual, test_utils
from ext import loss, warp
#%%
'''dataset loading'''
torch.backends.cudnn.benchmark = True

img_size = [96, 208, 272]
#batch_size = 1 
#win_size = [5,5,5]
val_index = 8

#root = '../dir/data3/'
Transform = transforms.Compose([transform.OneNorm(),
                                transform.ToTensor()])
#val_dset = dataset.Volumes(root, val_index, train=False, transform=Transform)
#val_loader = data.DataLoader(val_dset, batch_size, shuffle=True)
#print("Val dset: %d" %len(val_dset))

path = '../dir/data1/case%g/' % val_index
mov_fname = 'case%g_T00.npy' % val_index
ref_fname = 'case%g_T50.npy' % val_index
mov = np.load(path + mov_fname)
ref = np.load(path + ref_fname)
mov = np.expand_dims(mov, 0)#shape(1, D, H, W)
ref = np.expand_dims(ref, 0)

mov0 = Transform(mov)
ref0 = Transform(ref)

mov = mov0.unsqueeze(0).cuda()
ref = ref0.unsqueeze(0).cuda()

#model = networks.dirnet(img_size).cuda()
model = networks.snet(ndown=3, img_size=img_size).cuda()

#criterion0 = loss.LCC(win_size).cuda()
#criterion1 = torch.nn.MSELoss().cuda()
#%% weights
'''trilinear, no seg, reso 1, lcc+mse'''
#lambda 0, bs 4, win 5
#WEIGHTS_PATH = 'weights/tri_reso1/weights-adam-val1-bs4-lr0.0010-lamb0-win5-epoch50-reso1-hu500-noseg-lcc-mse/'
#weights_fname = 'weights-45-0.388-0.388.pth'
#lambda 0, bs 4, win 5, val 2
#WEIGHTS_PATH = 'weights/tri_reso1/weights-adam-val2-bs4-lr0.0010-lamb0-win5-epoch50-reso1-hu500-noseg-lcc-mse/'
#weights_fname = 'weights-33-0.412-0.412.pth'
#lambda 0, bs 4, win 5, val 3
#WEIGHTS_PATH = 'weights/tri_reso1/weights-adam-val3-bs4-lr0.0010-lamb0-win5-epoch50-reso1-hu500-noseg-lcc-mse/'
#weights_fname = 'weights-39-0.412-0.412.pth'
#lambda 0, bs 4, win 5, val 4
#WEIGHTS_PATH = 'weights/tri_reso1/weights-adam-val4-bs4-lr0.0010-lamb0-win5-epoch50-reso1-hu500-noseg-lcc-mse/'
#weights_fname = 'weights-44-0.483-0.483.pth'
#lambda 0, bs 4, win 5, val 5
#WEIGHTS_PATH = 'weights/tri_reso1/weights-adam-val5-bs4-lr0.0010-lamb0-win5-epoch50-reso1-hu500-noseg-lcc-mse/'
#weights_fname = 'weights-45-0.456-0.456.pth'
#lambda 0, bs 4, win 5, val 6
#WEIGHTS_PATH = 'weights/tri_reso1/weights-adam-val6-bs4-lr0.0010-lamb0-win5-epoch50-reso1-hu500-noseg-lcc-mse/'
#weights_fname = 'weights-40-0.597-0.597.pth'
#lambda 0, bs 4, win 5, val 7
#WEIGHTS_PATH = 'weights/tri_reso1/weights-adam-val7-bs4-lr0.0010-lamb0-win5-epoch50-reso1-hu500-noseg-lcc-mse/'
#weights_fname = 'weights-49-0.568-0.568.pth'
#lambda 0, bs 4, win 5, val 8
WEIGHTS_PATH = 'weights/tri_reso1/weights-adam-val8-bs4-lr0.0010-lamb0-win5-epoch50-reso1-hu500-noseg-lcc-mse/'
weights_fname = 'weights-49-0.656-0.656.pth'
##lambda 0, bs 4, win 5, val 9
#WEIGHTS_PATH = 'weights/tri_reso1/weights-adam-val9-bs4-lr0.0010-lamb0-win5-epoch50-reso1-hu500-noseg-lcc-mse/'
#weights_fname = 'weights-47-0.530-0.530.pth'
#lambda 0, bs 4, win 5, val 10
#WEIGHTS_PATH = 'weights/tri_reso1/weights-adam-val10-bs4-lr0.0010-lamb0-win5-epoch50-reso1-hu500-noseg-lcc-mse/'
#weights_fname = 'weights-50-0.551-0.551.pth'

#lambda 0, bs 4, win 9, val 8
#WEIGHTS_PATH = 'weights/tri_reso1/weights-adam-val8-bs4-lr0.0010-lamb0-win9-epoch50-reso1-hu500-noseg-lcc-mse/'
#weights_fname = 'weights-47-0.506-0.506.pth'

#lambda 10, bs 4, win 5
#WEIGHTS_PATH = 'weights/tri_reso1/weights-adam-val1-bs4-lr0.0010-lamb10-win5-epoch50-reso1-hu500-noseg-lcc-mse/'
#weights_fname = 'weights-46-0.402-0.388.pth'
#lambda 20, bs 4, win 5
#WEIGHTS_PATH = 'weights/tri_reso1/weights-adam-val1-bs4-lr0.0010-lamb20-win5-epoch50-reso1-hu500-noseg-lcc-mse/'
#weights_fname = 'weights-35-0.415-0.389.pth'
'''trilinear, no seg, reso 1, lcc+mse, maxpool'''
#lambda 0, bs 4, win 5, val 8
#WEIGHTS_PATH = 'weights/tri1_max/weights-adam-val8-bs4-lr0.0010-lamb0-win5-epoch50-reso1-hu500-noseg-lcc-mse/'
#weights_fname = 'weights-47-0.650-0.650.pth'
'''trilinear, no seg, reso 1, mse+lcc'''
#lambda 0, bs 4, win 5
#WEIGHTS_PATH = 'weights/tri_reso1/weights-adam-val1-bs4-lr0.0010-lamb0-win5-epoch50-reso1-hu500-noseg-mse-lcc/'
#weights_fname = 'weights-49-0.001-0.001.pth'
'''trilinear, no seg, reso 1, lcc+mse, down 34'''
#lambda 0, bs 4, win 5
#WEIGHTS_PATH = 'weights/tri_reso1/weights-adam-val1-bs4-lr0.0010-lamb0-win5-epoch50-reso1-hu500-noseg-lcc-mse-down34/'
#weights_fname = 'weights-44-0.407-0.407.pth'
#lambda 0, bs 4, win 5, val8
#WEIGHTS_PATH = 'weights/tri_reso1/weights-adam-val8-bs4-lr0.0010-lamb0-win5-epoch50-reso1-hu500-noseg-lcc-mse-down34/'
#weights_fname = 'weights-47-0.689-0.689.pth'

'''trilinear, no seg, reso 1, lcc+mse, down 4'''
#lambda 0, bs 4, win 5
#WEIGHTS_PATH = 'weights-adam/'
#weights_fname = 'weights-28-0.433-0.433.pth'
#lambda 0, bs 4, win 5, val 8
#WEIGHTS_PATH = 'weights/tri_reso1/weights-adam-val8-bs4-lr0.0010-lamb0-win5-epoch50-reso1-hu500-noseg-lcc-mse-down4/'
#weights_fname = 'weights-37-0.710-0.710.pth'

startEpoch = train_utils.load_weights(model, WEIGHTS_PATH + weights_fname)
#%% '''predict''' 
it = 1        
warper = warp.Warper3d(img_size)
flow = torch.zeros([1,3]+img_size).cuda()
#evaluate model for it times
model.eval()
since = time.time()
with torch.no_grad():
    warped = mov
    for _ in range(it):
        _, flow0 = model(warped, ref)
        flow += flow0
        warped = warper(mov, flow)
        
time_elapsed = time.time() - since  
print('Prediction Time {:.0f}m {:.04f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
                
m = mov.data.cpu().numpy()[0, 0]
r = ref.data.cpu().numpy()[0, 0]
w = warped.data.cpu().numpy()[0, 0]
flow = flow.data.cpu()

#    loss0 = criterion0(warped, ref)
#    loss1 = criterion1(warped, ref)
#    loss_orig = criterion0(mov, ref)
#data
#lb = loss_orig.item()
#la = loss0.item()
#la1 = loss1.item()
#plot
#slice_id = 70
#visual.view_slice(m, slice_id, 'moving')
#visual.view_slice(r, slice_id, 'fixed')
#visual.view_slice(w, slice_id, 'moved')

#%% show jacobian
#jac = visual.Get_Jac(flow.numpy())[0]
#jaclist = [jac[:,:,60], jac[:,:,80]]
#namelist = ['lambda0_60', 'lambda0_80']
#jaclist = [jac[:, :, 50], jac[:, :, 60]]

#visual.show_sample_slices(jaclist, namelist, Jac=True, cmap='bwr')
#% show diff image
visual.view_diff(m, r, w, 65)
#%% save warped, flow, jac
lamb = 10

RESULTS_PATH = 'results/'
flow_folder = RESULTS_PATH + 'flow/'
#jac_folder = RESULTS_PATH + 'jac/'
#warpped_folder = RESULTS_PATH + 'warpped/'
#mov_folder = RESULTS_PATH + 'mov/'
#ref_folder = RESULTS_PATH + 'ref/'
Path(flow_folder).mkdir(exist_ok=True)
#Path(jac_folder).mkdir(exist_ok=True)
#Path(warpped_folder).mkdir(exist_ok=True)
#Path(mov_folder).mkdir(exist_ok=True)
#Path(ref_folder).mkdir(exist_ok=True)
#save flow
flow_fname = 'flow_case%g_lamb%g_reso2_noseg.npy' % (val_index, lamb)
np.save(flow_folder+flow_fname, flow.numpy())
##save jac
#jac_fname = 'jac_case%g_lamb%g_reso2_noseg.npy' % (val_index, lamb)
#np.save(jac_folder+jac_fname, jac)
##save warpped, mov, ref
#warpped_fname = 'warpped_case%g_lamb%g_reso2_noseg.npy' % (val_index, lamb)
#np.save(warpped_folder+warpped_fname, w)
#
#mov_fname = 'mov_case%g.npy' % val_index
#np.save(mov_folder+mov_fname, m)
#ref_fname = 'ref_case%g.npy' % val_index
#np.save(ref_folder+ref_fname, r)