import sys
sys.path.append('../')                  # import Constant file

import  numpy as np
import os
import matplotlib.pyplot as plt
from  PIL import  Image
import cv2
from dilation_supervised import Constants
from dilation_supervised.data_process.data_ultils import deformation_set, read_all_images, data_shuffle

path_images_drive = '../dataset1/DRIVE/training/images/'
path_gt_drive = '../dataset1/DRIVE/training/1st_manual/'
path_images_test_drive = '../dataset1/DRIVE/test/images/'
path_gt_test_drive = '../dataset1/DRIVE/test/1st_manual/'
path_images_val_drive = '../dataset1/DRIVE/val/images/'
path_gt_val_drive = '../dataset1/DRIVE/val/1st_manual/'


def read_numpy_into_npy(arrays, path):
    np.save(path, arrays)
    print('have saved all arrays in to path ', path)

def load_from_npy(npy_path):
    arrays = np.load(npy_path)
    print('have loaded all arrays from ', npy_path)
    return arrays

def read_drive_images(size_h,size_w, path_images, path_gt,total_imgs, mask_ch =1):

    all_images = np.empty(shape=(total_imgs, size_h, size_w, 3))
    all_masks  = np.empty(shape=(total_imgs, size_h, size_w, mask_ch))
    all_images = read_all_images(path_images, all_images,size_h,size_w)
    all_masks  = read_all_images(path_gt, all_masks,size_h,size_w)
    print('============= have read all images ==============')
    return all_images, all_masks


def data_auguments(aug_num,size_h, size_w,path_images, path_gt,total_imgs, mask_ch, augu=True):

    all_images, all_masks = read_drive_images(size_h, size_w,path_images, path_gt,total_imgs, mask_ch)         # original data
    if augu is False:
        return all_images, all_masks
    # print('image and gt shape is:', all_images.shape, all_masks.shape)
    img_list = []
    gt_list = []
    for nums in range(0, aug_num):
        for i_d in range(0, all_images.shape[0]):
            aug_img, aug_gt = deformation_set(all_images[i_d, :, :, :, ], all_masks[i_d, :, :, :, ])
            print(aug_gt.shape,'---------')
            img_list.append(np.expand_dims(aug_img, axis=0))
            gt_list.append(np.expand_dims(aug_gt, axis=0))
    img_au = np.concatenate(img_list, axis=0)
    gt_au = np.concatenate(gt_list, axis=0)
    # print(img_au.shape, gt_au.shape)
    # visualize(group_images(all_masks, 5), './image_test')
    return img_au,gt_au

def data_for_train(aug_num,size_h, size_w,path_images, path_gt,total_imgs, mask_ch,augu):
    all_images, all_masks = data_auguments(aug_num, size_h, size_w, path_images, path_gt,total_imgs, mask_ch,augu)
    print('image and gt shape is:', all_images.shape, all_masks.shape)
    img = np.array(all_images, np.float32).transpose(0,3,1,2) / 255.0
    mask = np.array(all_masks, np.float32).transpose(0,3,1,2) / 255.0

    if mask_ch ==1:
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0
    #  data shuffle
    index = np.arange(img.shape[0])
    np.random.shuffle(index)
    img  = img[index, :, :, :]
    mask = mask[index, :, :]
    return img, mask

def save_drive_data(mum_arg = 3):
    images, mask = data_for_train(mum_arg, Constants.resize_drive,Constants.resize_drive,
                                  path_images_drive, path_gt_drive, 20, mask_ch=1,augu=True)
    images_test, mask_test = data_for_train(mum_arg, Constants.resize_drive,Constants.resize_drive,
                                  path_images_test_drive, path_gt_test_drive, 20, mask_ch=1,augu=False)
    # images_val, mask_val = data_for_train(mum_arg, Constants.resize_drive, Constants.resize_drive,
    #                               path_images_test_drive, path_gt_val_drive, 2, mask_ch=1, augu=False)

    try:
        read_numpy_into_npy(images,Constants.path_image_drive)
        read_numpy_into_npy(mask, Constants.path_label_drive)
        read_numpy_into_npy(images_test,Constants.path_test_image_drive)
        read_numpy_into_npy(mask_test, Constants.path_test_label_drive)
        # read_numpy_into_npy(images_val, Constants.path_val_image_drive)
        # read_numpy_into_npy(mask_val, Constants.path_val_label_drive)
        print('========  all drive train and test data has been saved ! ==========')
    except:
        print(' file save exception has happened! ')


def one_hot_drive_color(array):
    array_c = array.copy()
    for m in range(0, array.shape[0]):
        for i in range(0, array.shape[2]):
            for j in range(0, array.shape[3]):
                if (array[m,0,i,j]==1 and array[m,1,i, j] == 0 and array[m,2,i, j] == 0 ):
                    array_c[m,0,i, j] = 1   # red
                elif (array[m,1,i, j] == 1 and array[m,0,i, j] == 0 and array[m,2,i, j] == 0):
                    array_c[m,0,i,j] = 2    # green
                elif (array[m,2,i, j] == 1 and array[m,1,i, j] == 0 and array[m,0,i, j] == 0):
                    array_c[m,0,i,j] = 3    # blue
                elif (array[m,0,i,j] == 1  and array[m,1,i, j] == 1 and array[m,2,i, j] == 1):
                    array_c[m,0,i,j] = 4    # white
                else:
                    array_c[m,0,i, j] = 0   # black
    return array_c[:,0,:,:,]

def check_bst_data():
    a=load_from_npy(Constants.path_image_drive)
    b=load_from_npy(Constants.path_label_drive)
    c=load_from_npy(Constants.path_test_image_drive)
    d=load_from_npy(Constants.path_test_label_drive)
    # e = load_from_npy(Constants.path_test_image_drive)
    # f = load_from_npy(Constants.path_test_label_drive)
    print(a.shape, b.shape, c.shape, d.shape)
    print(np.max(a),np.max(b),np.max(c), np.max(d))


if __name__ == '__main__':
    save_drive_data()
    check_bst_data()
    pass