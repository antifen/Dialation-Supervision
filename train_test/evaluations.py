
import os
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from skimage import filters
from matplotlib import pyplot as plt
from hausdorff import hausdorff_distance

""" 
confusion matrix
P\L     P    N 
P      TP    FP 
N      FN    TN 
"""

def pixel_values_in_mask(true_vessels, pred_vessels, mask, train_or, dataset):
    if train_or == 'val':
        true_vessels = np.squeeze(true_vessels)
        pred_vessels = np.squeeze(pred_vessels)

        if dataset == 'HRF-AV':
            true_vessels = (true_vessels[mask[0, ...] != 0])
            pred_vessels = (pred_vessels[mask[0, ...] != 0])
        else:
            true_vessels = (true_vessels[mask != 0])
            pred_vessels = (pred_vessels[mask != 0])

        assert np.max(pred_vessels) <= 1.0 and np.min(pred_vessels) >= 0.0
        assert np.max(true_vessels) == 1.0 and np.min(true_vessels) == 0.0

    return true_vessels.flatten(), pred_vessels.flatten()

def AUC_ROC(true_vessel_arr, pred_vessel_arr):
    AUC_ROC = roc_auc_score(true_vessel_arr, pred_vessel_arr)
    return AUC_ROC

def threshold_by_otsu(pred_vessels):
    threshold = filters.threshold_otsu(pred_vessels)
    pred_vessels_bin = np.zeros(pred_vessels.shape)
    pred_vessels_bin[pred_vessels >= threshold] = 1
    return pred_vessels_bin
    return threshold

def AUC_PR(true_vessel_img, pred_vessel_img):
    precision, recall, _ = precision_recall_curve(true_vessel_img.flatten(), pred_vessel_img.flatten(), pos_label=1)
    AUC_prec_rec = auc(recall, precision)
    return AUC_prec_rec

def roc_pr_curve(y_true, y_scores, path_experiment):
    '''
    :param y_true:          is 0 or 1 only.
    :param y_scores:        probability values likes : [0.8223, 0.08, 0.07,0.90.....]
    :param path_experiment: save roc and pr curve path
    :return:                saved two curves
    '''
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    AUC_ROC = auc(fpr, tpr)

    plt.figure()                           # create figures
    plt.plot(fpr, tpr, 'b', label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], 'r--')
    plt.savefig(path_experiment + 'ROC.png')
    plt.close()
    precision, recall, threshold = precision_recall_curve(y_true, y_scores)
    precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
    recall = np.fliplr([recall])[0]        # so the array is increasing (you won't get negative AUC)
    AUC_prec_rec = np.trapz(precision, recall)
    plt.figure()
    plt.plot(recall, precision, 'b', label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
    plt.title('Precision - Recall curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], 'r--')
    plt.savefig(path_experiment + "Precision_Recall.png")
    return AUC_prec_rec, AUC_ROC

def hausdorff_dis(true_vessel_arr, pred_vessel_arr):
    # when hausdorff is smaller, it means that the segmented results is more similar to label !
    mask_vessel = np.zeros(shape=(true_vessel_arr.shape[0], 2))
    pred_vessel = np.zeros(shape=(pred_vessel_arr.shape[0], 2))
    for j in range(0, true_vessel_arr.shape[0]):
        if true_vessel_arr[j] == 1:
            mask_vessel[j, 1] = 1
        if true_vessel_arr[j] == 0:
            mask_vessel[j, 0] = 1
        pred_vessel[j, 1] = pred_vessel_arr[j]
        pred_vessel[j, 0] = 1 - pred_vessel_arr[j]
    return hausdorff_distance(mask_vessel.transpose(), pred_vessel.transpose(), distance="euclidean")

def misc_measures(true_vessel_arr, pred_vessel_arr, need_threshold = False):
    if need_threshold is True:
        pred_vessel_arr = threshold_by_otsu(pred_vessel_arr)
    cm = confusion_matrix(true_vessel_arr, pred_vessel_arr)
    mse = mean_squared_error(true_vessel_arr, pred_vessel_arr)
    try:
        acc = 1. * (cm[0, 0] + cm[1, 1]) / np.sum(cm)
        sensitivity = 1. * cm[1, 1] / (cm[1, 0] + cm[1, 1])
        specificity = 1. * cm[0, 0] / (cm[0, 1] + cm[0, 0])
        precision = 1. * cm[1, 1] / (cm[1, 1] + cm[0, 1])
        G = np.sqrt(sensitivity * specificity)
        F1_score_2 = 2 * precision * sensitivity / (precision + sensitivity)
        iou = 1. * cm[1, 1] / (cm[1, 0] + cm[0, 1] + cm[1, 1])
        dis = hausdorff_dis(true_vessel_arr, pred_vessel_arr)
        return np.array([acc, sensitivity, specificity, precision, G, F1_score_2, mse, iou, dis])
    except:
        print('Wrong have happened in evaluation process, please check each evaluation metrics carefully!')
        return np.array([ 0, 0, 0, 0, 0, 0, 0, 0, 0])

'''
the follow two functions are for multi-segmentation issue like MO dataset or BrainTS2017 and so on.
'''

def change_one_hot(truth, preds, value):
    truth = truth.copy()
    preds = preds.copy()
    size = np.sum(truth == value)
    truth [np.where(truth != value)] = 0
    truth [np.where(truth == value) ]= 1
    preds [np.where(preds != value)] = 0
    preds [np.where(preds == value) ]= 1
    return truth, preds, size

def get_my_evaluation(true_vessel_arr, pred_vessel_arr):
    truth_a,preds_a,size_a = change_one_hot(true_vessel_arr, pred_vessel_arr, 1)
    truth_v,preds_v,size_v = change_one_hot(true_vessel_arr, pred_vessel_arr, 3)
    truth_u,preds_u,size_u = change_one_hot(true_vessel_arr, pred_vessel_arr, 2)
    return ((misc_measures(truth_a, preds_a)*size_a + misc_measures(truth_v, preds_v)*size_v
           +misc_measures(truth_u, preds_u)*size_u)/(size_a+size_v+size_u))

if __name__ == '__main__':

    '''
    0: background  1:artery   2:uncertain  3:vein
    '''
    true_vessel_arr = np.array([1,0,1,1,0,1,2,0,0,1,3,2,2,1,3,0,0,2,0,0])
    pred_vessel_arr = np.array([0,0,3,1,2,1,1,0,2,1,3,2,3,1,1,0,0,2,3,1])
    p_all = []
    print(misc_measures(true_vessel_arr,pred_vessel_arr))
    array_metrics = get_my_evaluation(true_vessel_arr, pred_vessel_arr)
    print(array_metrics)
    print(AUC_PR(true_vessel_arr, pred_vessel_arr))

    '''
    0: background   1:vessel   0-1:  predictions
    '''
    true_vessel_arr = np.array([0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1])
    pred_vessel_arr = np.array([0.1,0,0.7,1,0,0.6,0.8,0,0,0.4,0.5,0.6,0.3,0.4,0.3,0,0.4,0.2,0.1,0.95])
    pred_vessel_arr = threshold_by_otsu(pred_vessel_arr)
    # pred_vessel_arr[np.where(pred_vessel_arr >= 0.5)] = 1
    # pred_vessel_arr[np.where(pred_vessel_arr <  0.5)] = 0
    array_metrics = misc_measures(true_vessel_arr, pred_vessel_arr)
    print(array_metrics)


    pass