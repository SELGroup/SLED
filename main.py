import argparse
import os
import matplotlib.pyplot as plt
from skimage import io
from skimage.segmentation import mark_boundaries, find_boundaries
from skimage.transform import resize
from sklearn.metrics import confusion_matrix
import numpy as np
from seg.segmentation import SLED_multi_scale, SLED_single_scale
from pre_processing.dullrazor import dullrazor
from pre_processing.shadesofgray import shade_of_gray_cc

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['PH2', 'ISIC2016', 'ISIC2017', 'ISIC2018'], default='ISIC2016')
# superpixel arguments
parser.add_argument('--superpixel', choices=['slic'], default='slic')
parser.add_argument('--slic_compactness', default=10)
parser.add_argument('--slic_nsegments', default=400)
# graph construction arguments
parser.add_argument('--adj_type', choices=['img_knn', 'knn'], default='img_knn')
parser.add_argument('--self_tuning_k', default=30)
parser.add_argument('--knn_k', default=50)
parser.add_argument('--centroid_thresh', default=0.3)
parser.add_argument('--knn_sigma', default=30)
# multi scale mechanism arguments
parser.add_argument('--contamination', default=0.1)
parser.add_argument('--nsegments_start', default=200)
parser.add_argument('--nsegments_end', default=700)
parser.add_argument('--outlier_detection', choices=['OCSVM', 'IFOREST', 'PCA', 'KNN', 'ECOD', 'COPOD', 'CBLOF'], default='IFOREST')
parser.add_argument('--multi_scale', default=True)
# thresholding arguments
parser.add_argument('--ght_nu', default=1e30)
parser.add_argument('--ght-tau', default=0.1)
parser.add_argument('--ght_kappa', default=1e-30)
parser.add_argument('--ght_omega', default=0.9)

parser.add_argument('--width', default=768)
parser.add_argument('--height', default=560)

args = parser.parse_args()

# SLED_MS on PH2 dataset.
def SLED_PH2():
    img_dir = "./exmaple_data/PH2/images"
    mask_dir = "./exmaple_data/PH2/dark_corner_artifact"
    gt_dir = "./exmaple_data/PH2/ground_truth"
    result_dir = "./exmaple_data/PH2/result"
    for img_name in os.listdir(img_dir):
        print(img_name)
        img_path = os.path.join(img_dir, img_name)
        img = io.imread(img_path)
        mask_path = os.path.join(mask_dir, img_name)
        mask = io.imread(mask_path).astype(bool).astype(int)
        gt_path = os.path.join(gt_dir, img_name)
        gt = io.imread(gt_path).astype(bool)
        img = resize(img, (args.height, args.width))
        mask = resize(mask.astype(bool), (args.height, args.width)).astype(int)
        gt = resize(gt.astype(bool), (args.height, args.width)).astype(int)
        img = dullrazor((img * 255).astype(np.uint8))
        img = shade_of_gray_cc(img, mask)
        result_path = os.path.join(result_dir, img_name)
        visualization_path = os.path.join(result_dir, img_name.split('.')[0]+"_visualization.png")
        seg_final, scoremap_global, img_name = SLED_multi_scale(img, mask, img_name, args)
        tn, fp, fn, tp = confusion_matrix(gt.flatten(), seg_final.astype(bool).flatten()).ravel()
        DI = 2 * tp / (2 * tp + fn + fp)

        marked_boundaries = mark_boundaries(img, seg_final, color=(1, 0, 0))
        gt_boundaries = find_boundaries(gt)
        marked_boundaries[gt_boundaries] = (0, 1, 0)
        fig, axes = plt.subplots(2,2, sharex="all", sharey="all")
        axes[0,0].imshow(img)
        axes[0,1].imshow(gt, cmap='gray')
        axes[1,0].imshow(marked_boundaries)
        axes[1,1].imshow(seg_final, cmap='gray')
        axes[0,0].axis('off')
        axes[0, 1].axis('off')
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')
        axes[0,0].set_title("Dermoscopic image")
        axes[0, 1].set_title("Ground truth")
        axes[1, 0].set_title("Segmentation result")
        axes[1, 1].set_title("Segmentation mask")
        plt.suptitle("{}: DI={:.4f}".format(img_name, DI))
        plt.savefig(visualization_path)
        plt.show()
        io.imsave(result_path, seg_final)

# SLED_MS on the ISIC 2016 dataset.
def SLED_ISIC2016():
    img_dir = "./exmaple_data/ISIC2016/images"
    mask_dir = "./exmaple_data/ISIC2016/circle_color_chart"
    gt_dir = "./exmaple_data/ISIC2016/ground_truth"
    result_dir = "./exmaple_data/ISIC2016/result"
    for img_name in os.listdir(img_dir):
        print(img_name)
        img_path = os.path.join(img_dir, img_name)
        img = io.imread(img_path)
        mask_path = os.path.join(mask_dir, img_name)
        mask = io.imread(mask_path).astype(bool).astype(int)
        gt_path = os.path.join(gt_dir, img_name.split('.')[0]+"_Segmentation.png")
        gt = io.imread(gt_path).astype(bool)
        img = resize(img, (args.height, args.width))
        mask = resize(mask.astype(bool), (args.height, args.width)).astype(int)
        gt = resize(gt.astype(bool), (args.height, args.width)).astype(int)
        img = dullrazor((img * 255).astype(np.uint8))
        img = shade_of_gray_cc(img, mask)
        result_path = os.path.join(result_dir, img_name)
        visualization_path = os.path.join(result_dir, img_name.split('.')[0]+"_visualization.png")
        seg_final, scoremap_global, img_name = SLED_multi_scale(img, mask, img_name, args)
        tn, fp, fn, tp = confusion_matrix(gt.flatten(), resize(seg_final.astype(bool), gt.shape).flatten()).ravel()
        DI = 2 * tp / (2 * tp + fn + fp)

        marked_boundaries = mark_boundaries(img, seg_final, color=(1, 0, 0))
        gt_boundaries = find_boundaries(resize(gt.astype(bool), seg_final.shape))
        marked_boundaries[gt_boundaries] = (0, 1, 0)
        fig, axes = plt.subplots(2,2, sharex="all", sharey="all")
        axes[0,0].imshow(img)
        axes[0,1].imshow(resize(gt.astype(bool), seg_final.shape), cmap='gray')
        axes[1,0].imshow(marked_boundaries)
        axes[1,1].imshow(seg_final, cmap='gray')
        axes[0,0].axis('off')
        axes[0, 1].axis('off')
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')
        axes[0,0].set_title("Dermoscopic image")
        axes[0, 1].set_title("Ground truth")
        axes[1, 0].set_title("Segmentation result")
        axes[1, 1].set_title("Segmentation mask")
        plt.suptitle("{}: DI={:.4f}".format(img_name, DI))
        plt.savefig(visualization_path)
        plt.show()
        io.imsave(result_path, seg_final)

# SLED_MS on the test set of the ISIC 2017 dataset.
def SLED_ISIC2017():
    img_dir = "./exmaple_data/ISIC2017/images"
    mask_dir = "./exmaple_data/ISIC2017/dark_corner_artifact"
    gt_dir = "./exmaple_data/ISIC2017/ground_truth"
    result_dir = "./exmaple_data/ISIC2017/result"
    for img_name in os.listdir(img_dir):
        print(img_name)
        img_path = os.path.join(img_dir, img_name)
        img = io.imread(img_path)
        mask_path = os.path.join(mask_dir, img_name)
        mask = io.imread(mask_path).astype(bool).astype(int)
        gt_path = os.path.join(gt_dir, img_name.split('.')[0]+"_segmentation.png")
        gt = io.imread(gt_path).astype(bool)
        img = resize(img, (args.height, args.width))
        mask = resize(mask.astype(bool), (args.height, args.width)).astype(int)
        gt = resize(gt.astype(bool), (args.height, args.width)).astype(int)
        img = dullrazor((img * 255).astype(np.uint8))
        img = shade_of_gray_cc(img, mask)
        result_path = os.path.join(result_dir, img_name)
        visualization_path = os.path.join(result_dir, img_name.split('.')[0]+"_visualization.png")
        seg_final, scoremap_global, img_name = SLED_multi_scale(img, mask, img_name, args)
        tn, fp, fn, tp = confusion_matrix(gt.flatten(), resize(seg_final.astype(bool), gt.shape).flatten()).ravel()
        DI = 2 * tp / (2 * tp + fn + fp)

        marked_boundaries = mark_boundaries(img, seg_final, color=(1, 0, 0))
        gt_boundaries = find_boundaries(resize(gt.astype(bool), seg_final.shape))
        marked_boundaries[gt_boundaries] = (0, 1, 0)
        fig, axes = plt.subplots(2,2, sharex="all", sharey="all")
        axes[0,0].imshow(img)
        axes[0,1].imshow(resize(gt.astype(bool), seg_final.shape), cmap='gray')
        axes[1,0].imshow(marked_boundaries)
        axes[1,1].imshow(seg_final, cmap='gray')
        axes[0,0].axis('off')
        axes[0, 1].axis('off')
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')
        axes[0,0].set_title("Dermoscopic image")
        axes[0, 1].set_title("Ground truth")
        axes[1, 0].set_title("Segmentation result")
        axes[1, 1].set_title("Segmentation mask")
        plt.suptitle("{}: DI={:.4f}".format(img_name, DI))
        plt.savefig(visualization_path)
        plt.show()
        io.imsave(result_path, seg_final)

# SLED_MS on the test set of the ISIC 2018 dataset.
def SLED_ISIC2018():
    img_dir = "./exmaple_data/ISIC2018/images"
    mask_dir = "./exmaple_data/ISIC2018/dark_corner_artifact"
    gt_dir = "./exmaple_data/ISIC2018/ground_truth"
    result_dir = "./exmaple_data/ISIC2018/result"
    for img_name in os.listdir(img_dir):
        print(img_name)
        img_path = os.path.join(img_dir, img_name)
        img = io.imread(img_path)
        mask_path = os.path.join(mask_dir, img_name)
        mask = io.imread(mask_path).astype(bool).astype(int)
        gt_path = os.path.join(gt_dir, img_name.split('.')[0]+"_segmentation.png")
        gt = io.imread(gt_path).astype(bool)
        img = resize(img, (args.height, args.width))
        mask = resize(mask.astype(bool), (args.height, args.width)).astype(int)
        gt = resize(gt.astype(bool), (args.height, args.width)).astype(int)
        img = dullrazor((img * 255).astype(np.uint8))
        img = shade_of_gray_cc(img, mask)
        result_path = os.path.join(result_dir, img_name)
        visualization_path = os.path.join(result_dir, img_name.split('.')[0]+"_visualization.png")
        seg_final, scoremap_global, img_name = SLED_multi_scale(img, mask, img_name, args)
        tn, fp, fn, tp = confusion_matrix(gt.flatten(), resize(seg_final.astype(bool), gt.shape).flatten()).ravel()
        DI = 2 * tp / (2 * tp + fn + fp)

        marked_boundaries = mark_boundaries(img, seg_final, color=(1, 0, 0))
        gt_boundaries = find_boundaries(resize(gt.astype(bool), seg_final.shape))
        marked_boundaries[gt_boundaries] = (0, 1, 0)
        fig, axes = plt.subplots(2,2, sharex="all", sharey="all")
        axes[0,0].imshow(img)
        axes[0,1].imshow(resize(gt.astype(bool), seg_final.shape), cmap='gray')
        axes[1,0].imshow(marked_boundaries)
        axes[1,1].imshow(seg_final, cmap='gray')
        axes[0,0].axis('off')
        axes[0, 1].axis('off')
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')
        axes[0,0].set_title("Dermoscopic image")
        axes[0, 1].set_title("Ground truth")
        axes[1, 0].set_title("Segmentation result")
        axes[1, 1].set_title("Segmentation mask")
        plt.suptitle("{}: DI={:.4f}".format(img_name, DI))
        plt.savefig(visualization_path)
        plt.show()
        io.imsave(result_path, seg_final)

def SLED():
    if args.dataset == 'PH2':
        SLED_PH2()
    elif args.dataset == 'ISIC2016':
        SLED_ISIC2016()
    elif args.dataset == 'ISIC2017':
        SLED_ISIC2017()
    elif args.dataset == 'ISIC2018':
        SLED_ISIC2018()

if __name__=='__main__':
    SLED()
    # SLED_PH2()
    # SLED_ISIC2016()
    # SLED_ISIC2017()
    # SLED_ISIC2018()