#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 6/24/2021 11:29 AM
# @Author: yzf
"""Visualize segmentation results for performance comparison"""
import nibabel as nib
import cv2
from visualizers.image_tools import *

def get_score_map(score, TYPE=cv2.COLORMAP_JET):
    score = norm_score(score)
    score_cmap = cv2.applyColorMap((score * 255.99).astype(np.uint8), TYPE)
    return score_cmap

def get_nii_data(file):
    img = nib.load(file)
    arr = img.get_fdata()  # h, w, d
    return arr

def norm_score(score):
    """Min-max scaler"""
    return (score - score.min()) / max(score.max() - score.min(), 1e-5)

def clip_intensity(arr):
    arr = np.clip(arr, -250., 200.)
    arr = norm_score(arr)
    return (arr * 255.).astype(np.uint8)

def get_score_map(score, TYPE=cv2.COLORMAP_JET):
    score = norm_score(score)
    score_cmap = cv2.applyColorMap((score * 255.99).astype(np.uint8), TYPE)
    return score_cmap

# img = './demo/img0001.nii.gz'
# seg = './demo/btcv_high_resolution_pseg0001.nii.gz'
# edge = './demo/btcv_high_resolution_edge0001.nii.gz'

img = './demo/img0005.nii.gz'
seg = './demo/btcv_high_resolution_pseg0005.nii.gz'
edge = './demo/btcv_high_resolution_edge0005.nii.gz'

img = get_nii_data(img)
seg = get_nii_data(seg)
edge = get_nii_data(edge)
assert img.shape == seg.shape == edge.shape

h, w, d = img.shape
v_images = []

for ind in range(d):
    im = np.rot90(img[..., ind])
    se = np.rot90(seg[..., ind])
    ed = np.rot90(edge[..., ind])

    im = clip_intensity(im)
    imRGB = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB);
    getScoreMap = lambda x: cv2.addWeighted(get_score_map(x),
                                            0.8, imRGB, 0.2, 0)

    se = getScoreMap(se)
    ed = getScoreMap(ed)

    h_images = [im, se, ed]
    v_images.append(imhstack(h_images, height=120))
v_images = imvstack(v_images)
imwrite(os.path.join('./demo', 'abdominal_ct.jpg'), v_images)


