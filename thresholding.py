import skimage.io        as skio
import numpy             as np

from skimage.measure import regionprops
from scipy           import ndimage

import logging
import cv2
import os


# This section is the segmentation routine ... I need to add the FogBank code here probably
def _fill(img_lbl, img_dil, **kwargs):
    img_sum = np.sum((img_lbl > 0) * img_dil)
    for ii in range(10):
        B = ndimage.maximum_filter(img_lbl, 3)
        B[img_lbl != 0] = img_lbl[img_lbl != 0]
        img_lbl = B * (img_dil > 0)

        img_sum_tmp = np.sum((img_lbl > 0) * img_dil)
        if  img_sum_tmp == img_sum:
            break
        img_sum = img_sum_tmp 

    return img_lbl

def boxcar(img, boxcar_size = 30, **kwargs):
    
    if img.dtype != "float32":
        img = img.astype(np.float32)
    
    # Low-pass
    kern    = np.ones((2*boxcar_size, 2*boxcar_size))
    kern    = kern / np.sum(kern)
    img_bkg = cv2.filter2D(img, -1, kern)
    img_sub = img - img_bkg
    img_sub[img_sub < 0] = 0
    return img_sub

def gauss(img, sigma = 8, **kwargs):
    
    if img.dtype != "float32":
        img = img.astype(np.float32)

    # gaussian filter
    mesh     = np.arange(- 4 * sigma, 4 * sigma + 1, 1)
    x, y     = np.meshgrid(mesh, mesh)
    gauss    = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    gauss    = gauss / np.sum(gauss)
    img_filt = cv2.filter2D(img, -1, gauss)
    img_sub  = img - img_filt
    img_sub[img_sub < 0] = 0
    return img_sub

def Threshold(img, **kwargs):
    # Default arguments
    if kwargs.get('min_size') is None:
        kwargs["min_size"]    = 80
    
    if kwargs.get('sigma') is None:
        kwargs["sigma"]       = 3
    
    if kwargs.get('boxcar_size') is None:
        kwargs["boxcar_size"] = 30
    
    
    if img.dtype != "float32":
        img = img.astype(np.float32)
    
    # gaussian filter
    img_sub  = gauss(img, **kwargs)
    
    # boxcar filter
    img_sub = boxcar(img_sub, **kwargs)
    
    # filter out threshold image
    kern         = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_bin      = cv2.morphologyEx((img_sub > 0).astype(np.uint8), cv2.MORPH_OPEN, kern, iterations = 2)
    img_lbl      = cv2.connectedComponents(img_bin)[1]
    
    # TODO: Add in fogbank section here to further separate objects ...
    
    # Get the original threshold to fill out foreground
    kwargs['boxcar_size'] = 2 * kwargs['boxcar_size']
    img_dil      = boxcar(img, **kwargs) > 0
    img_filled   = _fill(img_lbl, img_dil)
    
    # remove small objects
    reg = regionprops(img_filled)
    for r in reg:
        if not (r.area < kwargs.get('min_size')):
            continue
        
        coords                                 = r.coords
        img_filled[coords[:, 0], coords[:, 1]] = 0
        
    img_filled   = _fill(img_filled, img_dil)
    
    return img_filled.astype(np.uint16)
