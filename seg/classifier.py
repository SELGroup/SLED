import numpy as np
from numpy import ma
from skimage.filters.thresholding import threshold_otsu
from pyod.models.iforest import IForest
from pyod.models.pca import PCA
from pyod.models.ocsvm import OCSVM
from pyod.models.cblof import CBLOF
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from pyod.models.knn import KNN

# select a channel that best separate the structural entropy based segmentation result
def max_between_variance_channel(img, mask):
    assert img.ndim == 3
    max_variance = 0
    max_channel = 0
    for channeli in range(img.shape[2]):
        img_gray = img[:,:,channeli]
        masked = ma.masked_array(data=img_gray, mask=(1-(mask.astype(bool).astype(int)))).compressed()
        thresh = threshold_otsu(masked)
        C0 = masked[masked <= thresh]
        C1 = masked[masked > thresh]
        w0 = C0.shape[0]
        w1 = C1.shape[0]
        u0 = np.mean(C0)
        u1 = np.mean(C1)
        between_variance = w0*w1*np.square(u0-u1)
        if between_variance > max_variance:
            max_variance = between_variance
            max_channel = channeli
    return max_channel, max_variance

def lesion_prob(seg_sp, background, X, mask, args):
    score_map = np.zeros_like(seg_sp, dtype=float)
    train_indices = np.unique(seg_sp[np.logical_and(background.astype(bool), ~(seg_sp==-1))])
    assert not (-1 in train_indices)
    test_indices = np.unique(seg_sp[np.logical_and(~(background.astype(bool)), ~(seg_sp==-1))])
    assert not (-1 in test_indices)

    X_train = X[train_indices, :]
    X_test = X[test_indices, :]

    # outlier detection methods
    if args.outlier_detection == "OCSVM":
        clf = OCSVM(contamination=args.contamination)
    elif args.outlier_detection == "IFOREST":
        clf = IForest(contamination=args.contamination)
    elif args.outlier_detection == "PCA":
        clf = PCA(contamination=args.contamination)
    elif args.outlier_detection == "KNN":
        clf = KNN(contamination=args.contamination)
    elif args.outlier_detection == "ECOD":
        clf = ECOD(contamination=args.contamination)
    elif args.outlier_detection == "COPOD":
        clf = COPOD(contamination=args.contamination)
    elif args.outlier_detection == "CBLOF":
        clf = CBLOF(contamination=args.contamination)
    else:
        raise NotImplementedError
    clf.fit(X_train)
    y_train_pred = clf.labels_
    y_train_scores = clf.decision_scores_
    y_test_pred = clf.predict(X_test)
    y_test_scores = clf.decision_function(X_test)

    for i, index in enumerate(train_indices):
        score_map[seg_sp==index] = y_train_scores[i]

    for i, index in enumerate(test_indices):
        score_map[seg_sp==index] = y_test_scores[i]

    return score_map