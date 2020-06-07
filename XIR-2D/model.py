#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 11:32:27 2018

@author: user
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ext import warp
#from utils import train_utils as tn_utils

class conv_block(nn.Module):
    def __init__(self, inChan, outChan, stride=1):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(inChan, outChan, kernel_size=3, stride=stride, padding=1, bias=True),
#                nn.BatchNorm2d(outChan),
                nn.LeakyReLU(0.2, inplace=True)
                )
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                #default: mode='fan_in', nonlinearity='leaky_relu'
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        x = self.conv(x)

        return x

class Unet(nn.Module):
    def __init__(self, enc_nf=[2,16,32,32,64,64], dec_nf=[64,32,32,32,16,2]):
        super(Unet, self).__init__()
        """
        unet architecture(voxelmorph). 
        :param enc_nf: list of encoder filters. right now it needs to be 1x5. 
            e.g. [2,16,32,32,64,64]
        :param dec_nf: list of decoder filters. right now it must be 1x5
            e.g. [64,32,32,32,16,2]
        """
        self.inconv = conv_block(enc_nf[0], enc_nf[1])
        self.down1 = conv_block(enc_nf[1], enc_nf[2], 2)
        self.down2 = conv_block(enc_nf[2], enc_nf[3], 2)
        self.down3 = conv_block(enc_nf[3], enc_nf[4], 2)
        self.down4 = conv_block(enc_nf[4], enc_nf[5], 2)
        self.up1 = conv_block(enc_nf[-1], dec_nf[0])
        self.up2 = conv_block(dec_nf[0]+enc_nf[4], dec_nf[1])
        self.up3 = conv_block(dec_nf[1]+enc_nf[3], dec_nf[2])
        self.up4 = conv_block(dec_nf[2]+enc_nf[2], dec_nf[3])
        self.same_conv = conv_block(dec_nf[3]+enc_nf[1], dec_nf[4])
        self.outconv = nn.Conv2d(
                dec_nf[4], dec_nf[5], kernel_size=3, stride=1, padding=1, bias=True)
        #init last_conv
        self.outconv.weight.data.normal_(mean=0, std=1e-5)
        if self.outconv.bias is not None:
            self.outconv.bias.data.zero_()

    def forward(self, x):
        # down-sample path (encoder)
        skip1 = self.inconv(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        skip4 = self.down3(skip3)
        x = self.down4(skip4)
        # up-sample path (decoder)
        x = self.up1(x)
#        x = self.upsample(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat((x, skip4), 1)
        x = self.up2(x)
#        x = self.upsample(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat((x, skip3), 1)
        x = self.up3(x)
#        x = self.upsample(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat((x, skip2), 1)
        x = self.up4(x)
#        x = self.upsample(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat((x, skip1), 1)
        x = self.same_conv(x)
        x = self.outconv(x)
        
        return x

class convblock(nn.Module):
    def __init__(self, inChan, outChan, stride=1):
        super(convblock, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(inChan, outChan, kernel_size=3, stride=stride, padding=1, bias=True),
                nn.BatchNorm2d(outChan),
                nn.LeakyReLU(0.2, inplace=True)
#                nn.ReLU(inplace=True)
                )
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2)#
                #default: mode='fan_in', nonlinearity='leaky_relu'
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        x = self.conv(x)

        return x

class mUnet(nn.Module):
    def __init__(self, max_disp=12, enc_nf=[2,16,32,32,64,64], dec_nf=[64,32,32,32,16,2]):
        super(mUnet, self).__init__()
        """
        modified unet architecture(voxelmorph). 
        :param enc_nf: list of encoder filters. right now it needs to be 1x5. 
            e.g. [2,16,32,32,64]
        :param dec_nf: list of decoder filters. right now it must be 1x5
            e.g. [32,32,32,16,2]
        """
        self.max_disp = max_disp
        self.inconv = convblock(enc_nf[0], enc_nf[1])
        self.down1 = convblock(enc_nf[1], enc_nf[2], 2)
        self.down2 = convblock(enc_nf[2], enc_nf[3], 2)
        self.down3 = convblock(enc_nf[3], enc_nf[4], 2)
        self.down4 = convblock(enc_nf[4], enc_nf[5], 2)
        self.up1 = convblock(enc_nf[-1], dec_nf[0])
        self.up2 = convblock(dec_nf[0]+enc_nf[4], dec_nf[1])
        self.up3 = convblock(dec_nf[1]+enc_nf[3], dec_nf[2])
        self.up4 = convblock(dec_nf[2]+enc_nf[2], dec_nf[3])
        self.same_conv = convblock(dec_nf[3]+enc_nf[1], dec_nf[4])
        self.outconv = nn.Conv2d(
                dec_nf[4], dec_nf[5], kernel_size=3, stride=1, padding=1, bias=True)
        self.tanh = nn.Tanh()
        #init outconv
        self.outconv.weight.data.normal_(mean=0, std=1e-5)
        if self.outconv.bias is not None:
            self.outconv.bias.data.zero_()

    def forward(self, x):
        # down-sample path (encoder)
        skip1 = self.inconv(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        skip4 = self.down3(skip3)
        x = self.down4(skip4)
        # up-sample path (decoder)
        x = self.up1(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat((x, skip4), 1)
        x = self.up2(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat((x, skip3), 1)
        x = self.up3(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat((x, skip2), 1)
        x = self.up4(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat((x, skip1), 1)
        x = self.same_conv(x)
        x = self.outconv(x)
        x = self.tanh(x) * self.max_disp
        
        return x

class xirnet_wi(nn.Module):
    def __init__(self, img_size, max_disp, modified=True, 
                 enc_nf=[2,16,32,32,64,64], dec_nf=[64,32,32,32,16,2]):
        super(xirnet_wi, self).__init__()
        if modified:
            self.unet = mUnet(max_disp, enc_nf, dec_nf)
        else:
            self.unet = Unet(enc_nf, dec_nf)
        self.warper = warp.Warper2d(img_size)
        
    def forward(self, mov, ref):#, mask
#        mov = mov * mask
#        ref = ref * mask
        image = torch.cat((mov, ref), 1)
        flow = self.unet(image)#(bs, 2, 32, 32)
        warped = self.warper(flow, mov)
#        warped = warped * mask
        return warped, ref, flow

#a=xirnet_wi(img_size=512)
#in1 = Variable(torch.rand((2,1,512,512)))
#in2 = Variable(torch.rand((2,1,512,512)))
#b,c,d=a(in1, in2)