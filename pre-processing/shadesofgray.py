# https://www.kaggle.com/code/apacheco/shades-of-gray-color-constancy/notebook

import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from skimage.io import imread, imsave


def shade_of_gray_cc(img, mask, power=6, gamma=None):
    """
    img (numpy array): the original image with format of (h, w, c)
    power (int): the degree of norm, 6 is used in reference paper
    gamma (float): the value of gamma correction, 2.2 is used in reference paper
    """
    img[~np.repeat(mask[:,:,np.newaxis], 3, axis=-1).astype(bool)] = 0
    img_dtype = img.dtype

    if gamma is not None:
        img = img.astype('uint8')
        look_up_table = np.ones((256, 1), dtype='uint8') * 0
        for i in range(256):
            look_up_table[i][0] = 255 * pow(i / 255, 1 / gamma)
        img = cv2.LUT(img, look_up_table)

    img = img.astype('float32')
    img_power = np.power(img, power)
    rgb_vec = np.power(np.mean(img_power, (0, 1)), 1 / power)
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec / rgb_norm
    rgb_vec = 1 / (rgb_vec * np.sqrt(3))
    img = np.multiply(img, rgb_vec)

    # Andrew Anikin suggestion
    img = np.clip(img, a_min=0, a_max=255)

    return img.astype(img_dtype)

if __name__=='__main__':
    img_dir = "/home/zengguangjie/ISIC2016/huang_hairremoval"
    corner_mask_dir = "/home/zengguangjie/ISIC2016/dark_corner_artifact/masks"
    gt_dir = "/home/zengguangjie/ISIC2016/Training_GroundTruth"
    circle_mask_dir = "/home/zengguangjie/ISIC2016/circle_color_chart/masks"
    shadesofgray_dir = "/home/zengguangjie/ISIC2016/shadesofgray_colorconstancy"
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        mask_path = os.path.join(circle_mask_dir, img_name)
        shadesofgray_path = os.path.join(shadesofgray_dir, img_name)
        img = imread(img_path)
        mask = imread(mask_path).astype(bool).astype(int)
        img_cc = shade_of_gray_cc(img, mask)
        imsave(shadesofgray_path, img_cc)