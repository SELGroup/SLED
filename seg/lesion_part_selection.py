from skimage.color import rgb2gray
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import label, opening
import cv2 as cv

# Closeness of connected regions against the center of the image.
# Method from paper "Automatic segmentation of dermoscopy images using saliency combined with adaptive
# thresholding based wavelet transform".

# Generate a gaussian kernel, h, w are the height and width of the kernel,
# h_sig, w_sig are \sigma of height and width.
def gkern(h, w, h_sig, w_sig):
    hx = np.linspace(-(h-1)/2, (h-1)/2, 1)
    wx = np.linspace(-(w-1)/2, (w-1)/2, 1)
    h_gauss = np.exp(-0.5*np.square(hx) / np.square(h_sig))
    w_gauss = np.exp(-0.5*np.square(wx) / np.square(w_sig))
    kernel = np.outer(h_gauss, w_gauss)
    kernel = kernel / np.sum(kernel)
    return kernel

# Select the connected component with largest region score.
def lesion_selection_with_loc(seg, mask):
    height, width = seg.shape
    L = np.maximum(height, width)

    center = (int(seg.shape[1]/2), int(seg.shape[0]/2))
    radius = int(0.80 * np.maximum(seg.shape[0], seg.shape[1]) / 2)
    circle = cv.circle(np.zeros(seg.shape, dtype=np.uint8), center, radius, (1), -1)
    mask_expanded = circle
    rectangle = cv.rectangle(np.zeros(seg.shape, dtype=np.uint8),
                             (int(0.1 * width), int(0.1 * height)), (int(0.9 * width), int(0.9 * height)), color=(1),
                             thickness=-1)
    mask_expanded = 1 - ((1 - mask_expanded) + (1 - rectangle)).astype(bool).astype(int)

    sMask = np.logical_and((seg > 0), mask)
    slabel = label(sMask)
    g_kernel = gkern(height, width, 0.15 * L, 0.15 * L)

    max = 0
    iMax = -1
    for i in range(1, slabel.max()+1):
        parti = (slabel == i) * mask_expanded.astype(bool).astype(int)
        scorei = np.sum(g_kernel * parti)
        if scorei > max:
            max = scorei
            iMax = i

    seg_final = slabel==iMax
    seg_final = binary_fill_holes(seg_final)
    # seg_final = opening(seg_final, selem=disk(8))
    return seg_final

# The darker class C0 is the lesion region. For structural entropy guided segmentation.
def background_selection_darkest(img, seg, mask):
    img_gray = rgb2gray(img)
    intensity = np.ones(seg.max()+1)*255
    for i in range(0, seg.max()+1):
        parti = np.logical_and((seg==i), mask)
        intensity[i] = np.sum(img_gray[parti]) / np.sum(parti)

    background = np.logical_and(~(seg==np.argmin(intensity)), mask).astype(int)
    return background