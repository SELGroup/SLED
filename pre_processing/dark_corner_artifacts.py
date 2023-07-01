import numpy as np
import os, os.path
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2 as cv
from skimage import io


def remove_dark_corner_PH2():
    dir_path = "E:\medi_vision\skin lesion\PH2\PH2Dataset"
    for img_name in os.listdir(os.path.join(dir_path, "slrmsr_colorconstancy")):
        img_path = os.path.join(dir_path, "slrmsr_colorconstancy", img_name)
        print(img_path)

        img = io.imread(img_path)
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        img_thresh = 100
        ret, thresh_img = cv.threshold(gray, img_thresh, 255, cv.THRESH_BINARY)
        contours, hierarchy = cv.findContours(thresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        img_contours = np.zeros(img.shape)

        contoursB = []
        big_contour = []
        max = 0
        for i in contours:
            area = cv.contourArea(i) #--- find the contour having biggest area ---
            if(area > max):
                max = area
                big_contour = i
                contoursB.append(i)

        mask = np.ones(img.shape)
        mask = cv.drawContours(mask, contoursB, -1, 0, cv.FILLED)

        (x,y), radius = cv.minEnclosingCircle(big_contour)
        center = (int(x), int(y))
        radius = int(radius) - 2
        this_contour = cv.circle(np.ones(img.shape, dtype=np.uint8)*255,center,radius,(0,0,0),-1)
        mask_path = os.path.join(dir_path, "dark_corner_artifact", "masks", img_name)
        io.imsave(mask_path, (255 - this_contour)[:,:,0])

        output = img.copy()
        output[this_contour.astype(np.bool_)] = 0
        output_path = os.path.join(dir_path, "dark_corner_artifact", "images", img_name)
        io.imsave(output_path, output)

def remove_dark_corner_ISIC2016_training():
    dir_path = "E:\medi_vision\skin lesion\ISIC2016"
    for img_name in os.listdir(os.path.join(dir_path, "Training_Data")):
        img_path = os.path.join(dir_path, "Training_Data", img_name)
        img = io.imread(img_path)
        height, width, depth = img.shape
        img_contracted = img[1:height-1, 1:width-1, :]
        gray = cv.cvtColor(img_contracted, cv.COLOR_RGB2GRAY)
        img_thresh = 50
        ret, thresh_img = cv.threshold(gray, img_thresh, 255, cv.THRESH_BINARY)
        contours, hierarchy = cv.findContours(thresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        contoursB = []
        big_contour = []
        max = 0
        for i in contours:
            area = cv.contourArea(i) #--- find the contour having biggest area ---
            if(area > max):
                max = area
                big_contour = i
                contoursB.append(i)

        (x,y), radius = cv.minEnclosingCircle(big_contour)
        center = (int(x+1), int(y+1))
        radius = int(radius)
        print(radius, np.sqrt((height-2)**2+(width-2)**2)/2-2)
        if radius >= (np.sqrt((height-2)**2+(width-2)**2)/2-2):
            radius = radius + 10
        this_contour = cv.circle(np.ones(img.shape, dtype=np.uint8)*255,center,radius,(0,0,0),-1)

        mask_path = os.path.join(dir_path, "dark_corner_artifact", "masks", img_name)
        io.imsave(mask_path, (255 - this_contour)[:,:,0])
        output = img.copy()
        output[this_contour.astype(np.bool_)] = 0
        output_path = os.path.join(dir_path, "dark_corner_artifact", "images", img_name)
        io.imsave(output_path, output)

def remove_dark_corner_ISIC2016_test():
    dir_path = "E:\medi_vision\skin lesion\ISIC2016_test"
    for img_name in os.listdir(os.path.join(dir_path, "resized")):
        img_path = os.path.join(dir_path, "resized", img_name)
        img = io.imread(img_path)
        height, width, depth = img.shape
        img_contracted = img[1:height-1, 1:width-1, :]
        gray = cv.cvtColor(img_contracted, cv.COLOR_RGB2GRAY)

        img_thresh = 50
        ret, thresh_img = cv.threshold(gray, img_thresh, 255, cv.THRESH_BINARY)
        contours, hierarchy = cv.findContours(thresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contoursB = []
        big_contour = []
        max = 0
        for i in contours:
            area = cv.contourArea(i) #--- find the contour having biggest area ---
            if(area > max):
                max = area
                big_contour = i
                contoursB.append(i)

        (x,y), radius = cv.minEnclosingCircle(big_contour)
        center = (int(x+1), int(y+1))
        radius = int(radius)
        print(radius, np.sqrt((height-2)**2+(width-2)**2)/2-2)
        if radius >= (np.sqrt((height-2)**2+(width-2)**2)/2-2):
            radius = radius + 10
        this_contour = cv.circle(np.ones(img.shape, dtype=np.uint8)*255,center,radius,(0,0,0),-1)
        mask_path = os.path.join(dir_path, "dark_corner_artifact", "masks", img_name)
        io.imsave(mask_path, (255 - this_contour)[:,:,0])

        output = img.copy()
        output[this_contour.astype(np.bool_)] = 0
        output_path = os.path.join(dir_path, "dark_corner_artifact", "images", img_name)
        io.imsave(output_path, output)


if __name__=='__main__':
    remove_dark_corner_ISIC2016_test()
