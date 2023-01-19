# Color chart removal based on paper "Histogram based circle detection"
import matplotlib.pyplot as plt
import numpy as np
import numba as nb
from skimage.transform import rescale, resize
import cv2 as cv
import jpype
import os
from skimage.io import imread, imsave
from skimage import segmentation, color
from skimage.measure import regionprops
from algo.graph_construction import img_knn_affinity
from algo.iterative_refinement_SE import refinement_SE
from algo.iterative_refinement_SE import merging
from concurrent import futures

# Contour should be the contour of a region which possibly contains a circle or a partial circle
# in format of a list of [x_pos, y_pos]
@nb.jit(nopython=True)
def histogram_based_circle_detection(img, contour, min_dist, max_dist, step, thresh):
    # print(contour)
    height, width = img.shape[0], img.shape[1]
    # accumulator_map = np.zeros((height, width), dtype=float)
    # radius_map = np.zeros((height, width), dtype=float)
    circles = []
    for i in range(height):
        for j in range(width):
            ds = np.sqrt(np.sum(np.square(np.array([j,i]) - contour), axis=-1))
            hist, bin_edges = np.histogram(ds, bins=int((max_dist-min_dist)/step), range=(min_dist, max_dist))
            max_perimeter = np.max(hist)
            # accumulator_map[i,j] = max_perimeter
            max_index = np.argmax(hist)
            radius = bin_edges[max_index]+step/2
            # radius_map[i,j] = radius

            # angle = max_perimeter / radius
            # # if max_perimeter > 0:
            # #     print(i,j,max_perimeter, radius, angle)
            # if angle > thresh:
            #     circles.append((i,j,radius))
            # print(hist)
            if max_perimeter / len(contour) > thresh:
                circles.append(np.array([i, j, radius]))
                print(max_perimeter / len(contour))
                # circles[(i,j,radius)] = max_perimeter / len(contour)
    # circles = sorted(circles.items(), key=lambda x:x[1], reverse=True)
    return circles

def detect_circle(seg, mask):
    if seg.shape[0]*seg.shape[1] < 2000*3000:
        return mask
    ds_scale = 0.1
    max_radius = int(np.minimum(1500, np.minimum(seg.shape[0]/2, seg.shape[1]/2)))
    min_radius = 500
    bin_step = 20
    pad_width = max_radius
    thresh = 0.289

    circle_masks = []
    for i in range(0, seg.max()+1):
        lbl = np.logical_and((seg == i), mask).astype(int)
        lbl = np.pad(lbl, pad_width=((pad_width, pad_width), (pad_width, pad_width)), mode='constant', constant_values=0)
        lbl = rescale(lbl.astype(bool), scale=ds_scale, anti_aliasing=False, channel_axis=None)
        lbl = (lbl * 255).astype(np.uint8)
        # plt.imshow(lbl, cmap='gray')
        # plt.colorbar()
        # plt.title("detect circle parti")
        # plt.show()
        contours, hierarchy = cv.findContours(lbl, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        big_contour = []
        max = 0
        for contour_j in contours:
            area = cv.contourArea(contour_j)  # --- find the contour having biggest area ---
            if (area > max):
                max = area
                big_contour = np.squeeze(contour_j)
        if max < 80000 * ds_scale*ds_scale:
            continue
        circles_i = histogram_based_circle_detection(lbl, big_contour, min_radius*ds_scale, max_radius*ds_scale, bin_step*ds_scale, thresh)
        circle_mask = np.zeros_like(mask)
        for circle_i in circles_i:
            circle_i[0] /= ds_scale
            circle_i[1] /= ds_scale
            circle_i[2] /= ds_scale
            circle_i[0] -= pad_width
            circle_i[1] -= pad_width
            circle_mask_i = cv.circle(np.zeros_like(mask, dtype=np.uint8), (int(circle_i[1]), int(circle_i[0])),
                                                  int(circle_i[2]), (1), -1)
            if np.sum(circle_mask_i.astype(bool).astype(int)) > 0.5*np.sum(mask.astype(bool).astype(int)):
                continue
            circle_mask = np.logical_or(circle_mask, circle_mask_i)
            circle_masks.append(circle_mask)
        # plt.imshow(circle_mask, cmap='gray')
        # plt.colorbar()
        # plt.show()
        # circles.extend(circles_i)
    circle_mask = np.zeros_like(mask)
    for circle_mask_i in circle_masks:
        circle_mask = np.logical_or(circle_mask, circle_mask_i)
    circle_mask = ~circle_mask
    mask = np.logical_and(circle_mask, mask).astype(int)
    return mask

def get_circle_mask(img_name):
    if not jpype.isJVMStarted():
        jpype.startJVM(jpype.getDefaultJVMPath(), classpath=['./algo/Merging.jar'])
    img_dir = "./data/ISIC2016_test/Test_Data"
    resized_dir = "./data/ISIC2016_test/resized"
    corner_mask_dir = "./data/ISIC2016_test/dark_corner_artifact/masks"
    circle_dir = "./data/ISIC2016_test/circle_color_chart"
    # for img_name in os.listdir(img_dir):
    #     if img_name != 'ISIC_0009160.jpg':
    #         continue
    resized_path = os.path.join(resized_dir, img_name)
    print(img_name)
    img_path = os.path.join(img_dir, img_name)
    img = imread(img_path)
    corner_mask_path = os.path.join(corner_mask_dir, img_name)
    corner_mask = imread(corner_mask_path).astype(bool).astype(int)
    corner_mask = resize(corner_mask.astype(bool), (img.shape[0], img.shape[1])).astype(int)
    seg_sp = segmentation.slic(img, mask=corner_mask, compactness=10, n_segments=400,
                               start_label=0)
    num_seg = np.amax(seg_sp) + 1
    centroids = np.zeros([num_seg, 2])
    for region in regionprops(seg_sp):
        if region.label >= 0:
            centroids[int(region.label)] = region.centroid
    avgcolor_segsp = color.label2rgb(seg_sp, img, kind='avg', bg_label=-1)
    height, width = seg_sp.shape
    avgcolors = np.zeros([num_seg, 3])
    for h in range(height):
        for w in range(width):
            if int(seg_sp[h, w]) >= 0:
                avgcolors[int(seg_sp[h, w])] = avgcolor_segsp[h, w, :]
    adj = img_knn_affinity(avgcolors, 30, 50, centroids, 0.3*(img.shape[0]+img.shape[1]))
    y = merging(adj, img_name)
    z = refinement_SE(adj, y)
    seg_iter = seg_sp.copy()
    for h in range(height):
        for w in range(width):
            node_label = int(seg_sp[h, w])
            if node_label >= 0:
                seg_iter[h, w] = z[node_label]
            else:
                seg_iter[h, w] = node_label

    mask = detect_circle(seg_iter, corner_mask)
    mask = np.logical_and(mask, corner_mask).astype(int)
    mask = resize(mask, imread(corner_mask_path).shape)
    circle_mask_path = os.path.join(circle_dir, "masks", img_name)
    imsave(circle_mask_path, mask.astype(bool))
    circle_img_path = os.path.join(circle_dir, "images", img_name)
    circle_img = imread(resized_path)
    circle_img[~np.repeat(mask[:,:,np.newaxis].astype(bool), 3, axis=-1).astype(bool)] = 0
    imsave(circle_img_path, circle_img)
    plt.imshow(~(mask.astype(bool)), cmap='gray')
    plt.show()
    # jpype.shutdownJVM()

if __name__=='__main__':

    img_name = "ISIC_0001242.jpg"
    get_circle_mask(img_name)
