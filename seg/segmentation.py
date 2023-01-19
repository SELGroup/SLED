from concurrent import futures
from seg.ght import ght_thresh_img
import numpy as np
import os
from skimage import io, color, segmentation
from skimage.measure import regionprops
from algo.graph_construction import img_knn_affinity, knn_affinity
from algo.iterative_refinement_SE import merging, refinement_SE
from skimage.filters import threshold_otsu
from seg.lesion_part_selection import lesion_selection_with_loc, background_selection_darkest
from sklearn.metrics import confusion_matrix
import argparse
import jpype
import numpy.ma as ma
from seg.classifier import lesion_prob, max_between_variance_channel
from skimage.segmentation import find_boundaries, mark_boundaries
from skimage.morphology import label

# SLED_SS segmentation.
def SLED_single_scale(img, mask, img_name, slic_nsegments, args):
    args.slic_nsegments = slic_nsegments
    if not jpype.isJVMStarted():
        jpype.startJVM(jpype.getDefaultJVMPath(), classpath=['./algo/Merging.jar'])
    mask = mask.astype(bool).astype(int)
    seg_sp = segmentation.slic(img, mask=mask, compactness=args.slic_compatness, n_segments=args.slic_nsegments,
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

    if args.adj_type == 'img_knn':
        adj = img_knn_affinity(avgcolors, args.self_tuning_k, args.knn_k, centroids,
                               args.centroid_thresh * (img.shape[0] + img.shape[1]))
    elif args.adj_type == 'knn':
        adj = knn_affinity(avgcolors, args.sigma, args.knn_k, centroids,
                           args.centroid_thresh * (img.shape[0] + img.shape[1]))
    else:
        raise NotImplementedError

    y = merging(adj, img_name, int(args.slic_nsegments))
    z = refinement_SE(adj, y)
    seg_iter = seg_sp.copy()
    for h in range(height):
        for w in range(width):
            node_label = int(seg_sp[h, w])
            if node_label >= 0:
                seg_iter[h, w] = z[node_label]
            else:
                seg_iter[h, w] = node_label

    seg_iter = label(seg_iter, background=-1)

    # ------------------------------bisection--------------------------------------
    avgcolor_segiter = color.label2rgb(seg_iter, img, kind='avg', bg_label=-1)
    target_channel, variance = max_between_variance_channel(avgcolor_segiter, mask)
    gray_segiter = avgcolor_segiter[:, :, target_channel]
    masked_gray_segiter = ma.masked_array(data=gray_segiter, mask=(1 - mask)).compressed()
    thresh = threshold_otsu(masked_gray_segiter)
    seg_bisection = (gray_segiter <= thresh).astype(int)
    background = background_selection_darkest(img, seg_bisection, mask)

    score_map = None
    if args.multi_scale:
        X = avgcolors
        score_map = lesion_prob(seg_sp, background, X, mask, args)
        score_map = (score_map - np.amin(score_map)) / (np.amax(score_map) - np.amin(score_map))

    return score_map, background, variance, img_name

# SLED_MS segmentation.
def SLED_multi_scale(img, mask, img_name, args):
    scoremap_global = np.zeros_like(mask, dtype=float)
    scoremap_list = []
    w_list = []
    variance_list = []

    to_do = []
    with futures.ProcessPoolExecutor(max_workers=len(range(args.nsegments_start, args.nsegments_end, 50))) as executor:
        for n_segments in range(args.nsegments_start, args.nsegments_end, 50):
            job = executor.submit(SLED_single_scale, img, mask, img_name, n_segments, args)
            to_do.append(job)
        for future in futures.as_completed(to_do):
            scoremap, _, variance, _ = future.result()
            scoremap_list.append(scoremap)
            variance_list.append(variance)
    for variance in variance_list:
        w = np.exp((variance - np.min(variance_list))/np.min(variance_list))
        w_list.append(w)
    for w, scoremap in zip(w_list, scoremap_list):
        scoremap_global += w*scoremap
    scoremap_global /= np.sum(w_list)
    thresh = ght_thresh_img(scoremap_global[mask > 0].ravel(), args)
    score_otsu = scoremap_global >= thresh
    score_otsu = np.logical_and(score_otsu, mask)
    seg_final = lesion_selection_with_loc(score_otsu, mask)
    return seg_final, scoremap_global, img_name


