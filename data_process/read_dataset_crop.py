import sys
import torch
sys.path.append('../')
sys.path.append('/root/dilation_supervised/')

import  numpy as np
import os
import matplotlib.pyplot as plt
from  PIL import  Image
import cv2
from dilation_supervised import Constants
from dilation_supervised.data_process.data_ultils import  read_all_images, data_shuffle
from dilation_supervised.data_process.retinal_process import rgb2gray, clahe_equalized, dataset_normalized, adjust_gamma
from skimage import feature
from skimage.segmentation import  find_boundaries


# four datasets' read_path
path_images_drive = './dataset1/DRIVE/training/images/'
path_gt_drive = './dataset1/DRIVE/training/1st_manual/'
path_images_test_drive = './dataset1/DRIVE/test/images/'
path_gt_test_drive = './dataset1/DRIVE/test/1st_manual/'
path_images_val_drive = '../dataset1/DRIVE/val/images/'
path_gt_val_drive = '../dataset1/DRIVE/val/1st_manual/'

# path_images_drive = './dataset1/STARE/training/images/'
# path_gt_drive = './dataset1/STARE/training/1st_manual/'
# path_images_test_drive = './dataset1/STARE/test/images/'
# path_gt_test_drive = './dataset1/STARE/test/1st_manual/'
# path_images_val_drive = '../dataset1/STARE/val/images/'
# path_gt_val_drive = '../dataset1/STARE/val/1st_manual/'

# path_images_drive = './dataset1/CHASEDB/training/images/'
# path_gt_drive = './dataset1/CHASEDB/training/1st_manual/'
# path_images_test_drive = './dataset1/CHASEDB/test/images/'
# path_gt_test_drive = './dataset1/CHASEDB/test/1st_manual/'
# path_images_val_drive = '../dataset1/CHASEDB/val/images/'
# path_gt_val_drive = '../dataset1/CHASEDB/val/1st_manual/'

# path_images_drive = './dataset1/HRF/training/images/'
# path_gt_drive = './dataset1/HRF/training/1st_manual/'
# path_images_test_drive = './dataset1/HRF/test/images/'
# path_gt_test_drive = './dataset1/HRF/test/1st_manual/'
# path_images_val_drive = '../dataset1/HRF/val/images/'
# path_gt_val_drive = '../dataset1/HRF/val/1st_manual/'


def read_numpy_into_npy(arrays, path):
    np.save(path, arrays)
    print('have saved all arrays in to path ', path)

def load_from_npy(npy_path):
    arrays = np.load(npy_path)
    print('have loaded all arrays from ', npy_path)
    return arrays

def read_drive_images(size_h,size_w, path_images, path_gt, total_imgs, mask_ch =1):
    all_images = np.empty(shape=(total_imgs, size_h, size_w, 3))
    all_masks = np.empty(shape=(total_imgs, size_h, size_w, 1))  # 3 for DCA1 dataset
    all_images = read_all_images(path_images, all_images,size_h, size_w,type ='non_resize')
    all_masks = read_all_images(path_gt, all_masks, size_h, size_w,type ='non_resize')

    # all_masks = all_masks[:, :, :, 1, ]   # DCA1 dataset
    # all_masks = np.expand_dims(all_masks, axis=3) # DCA1 dataset

    print('============= have read all images ==============')
    return all_images, all_masks

def gaussian_noise(img, mean, sigma):
    noise = np.random.normal(mean, sigma, img.shape)
    gaussian_out = img + noise
    gaussian_out = np.clip(gaussian_out, 0, 1)
    gaussian_out = np.uint8(gaussian_out)
    return gaussian_out 

def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)
    return image, mask

def randomHorizontalFlip(image, mask):
    if np.random.random() < 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)
    return  image, mask

def randomVerticleFlip(image, mask):
    if np.random.random() < 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    return image, mask

def crop_images(image, mask, crop_size = Constants.resize_drive):
    select_id = np.random.randint(0, 4)
    d_h, d_w, h, w = image.shape[0] - crop_size, image.shape[1] - crop_size,image.shape[0],image.shape[1]
    crop_lu_im,  crop_lu_ma = image[d_h:h, d_w:w, :,], mask[d_h:h, d_w:w, :,]
    crop_ld_im,  crop_ld_ma = image[d_h:h, 0:w-d_w, :, ], mask[d_h:h, 0:w-d_w, :, ]
    crop_ru_im,  crop_ru_ma = image[0:h - d_h, d_w:w, :, ], mask[0:h - d_h, d_w:w, :, ]
    crop_rd_im,  crop_rd_ma = image[0:h - d_h, 0:w-d_w, :, ], mask[0:h - d_h, 0:w-d_w, :, ]

    crop_img, crop_mask =None, None
    if select_id ==0:
        crop_img, crop_mask = np.expand_dims(crop_lu_im, axis=0), np.expand_dims(crop_lu_ma, axis=0)
    if select_id ==1:
        crop_img, crop_mask = np.expand_dims(crop_ld_im, axis=0), np.expand_dims(crop_ld_ma, axis=0)
    if select_id ==2:
        crop_img, crop_mask = np.expand_dims(crop_ru_im, axis=0), np.expand_dims(crop_ru_ma, axis=0)
    if select_id ==3:
        crop_img, crop_mask = np.expand_dims(crop_rd_im, axis=0), np.expand_dims(crop_rd_ma, axis=0)
    return crop_img, crop_mask

#HRF数据集裁剪
def crop_order_images(image, mask, crop_size = 960, row = 3, col = 4,  mode = 'randoms', rands = 12):
    image, mask = torch.tensor(image.copy()),torch.tensor(mask.copy())
    image = torch.unsqueeze(image.permute((2,0,1)), dim=0)
    mask = torch.unsqueeze(mask.permute((2, 0, 1)), dim=0)
    # print(image.size(), mask.size())
    if mode =='orders':
        image, s_h, s_w= padding_img(image, crop_size, row, col)
        mask, s_h, s_w = padding_img(mask, crop_size, row, col)
        assert (crop_size > s_h and crop_size > s_w)
        img_set, mask_set = [], []
        for m in range(0, row):
            for n in range(0, col):
                dh = m * s_h
                dw = n * s_w
                img_set.append(image[:,:,dh:dh+crop_size, dw:dw+crop_size])
                mask_set.append(mask[:, :, dh:dh + crop_size, dw:dw + crop_size])
        return torch.cat([img_set[i] for i in range(0, len(img_set))], dim=0), \
               torch.cat([mask_set[i] for i in range(0, len(img_set))], dim=0),
    elif mode == 'randoms':
        # random select center point to expand patches ! (compliment)
        import  random
        img_set, mask_set = [], []
        for i in range(0, rands):
            center_y = random.randint(crop_size//2, image.size()[2] - crop_size//2)
            center_x = random.randint(crop_size//2, image.size()[3] - crop_size//2)
            crops_img = image[:,:,center_y - crop_size//2:center_y + crop_size//2, center_x - crop_size//2:center_x + crop_size//2]
            img_set.append(crops_img)
            crops_mask = mask[:,:,center_y - crop_size//2:center_y + crop_size//2,center_x - crop_size//2:center_x + crop_size//2]
            mask_set.append(crops_mask)
        return torch.cat([img_set[i].permute((0,2,3,1)) for i in range(0, len(img_set))], dim=0)\
            ,torch.cat([mask_set[i].permute((0,2,3,1)) for i in range(0, len(img_set))], dim=0)

def padding_img(image, crop_size, rows, clos):
    pad_h, pad_w = (image.size()[2] - crop_size)%(rows-1), (image.size()[3] - crop_size)%(clos-1)
    image = padding_hw(image, dims='h', ns= 0 if pad_h==0 else rows -1 -pad_h)
    image = padding_hw(image, dims='w', ns= 0 if pad_w==0 else clos -1 -pad_w)
    return image.to(device), (image.size()[2] - crop_size)//(rows-1), (image.size()[3] - crop_size)//(clos-1)

def padding_hw(img, dims = 'h', ns = 0):
    if ns ==0:
        return img
    else:
        after_expanding = None
        if dims == 'h':
            pad_img = torch.zeros_like(img[:,:,0 : ns,:,])
            after_expanding = torch.cat([img, pad_img], dim=2)
        elif dims == 'w':
            pad_img = torch.zeros_like(img[:,:,:,0:ns])
            after_expanding = torch.cat([img, pad_img], dim=3)
        return after_expanding


def deformation_set(image, mask,
                           shift_limit=(-0.2, 0.2),
                           scale_limit=(-0.2, 0.2),
                           rotate_limit=(-180.0, 180.0),
                           aspect_limit=(-0.1, 0.1),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    print('deformation_set size check: ', image.shape, mask.shape)

    start_angele, per_rotate = -180, 10
    rotate_num = - start_angele // per_rotate * 2
    image_set, mask_set, image_full_set, mask_full_set = [], [], [], []
    for rotate_id in range(0, rotate_num):
        masks = mask
        img = image
        height, width, channel = img.shape
        sx, sy = 1., 1.
        angle = start_angele + rotate_id * per_rotate
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)
        cc = np.cos(angle / 180 * np.pi) * sx
        ss = np.sin(angle / 180 * np.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])
        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height],])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])
        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        img = cv2.warpPerspective(img, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(0, 0, 0,))
        masks = cv2.warpPerspective(masks, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(0, 0, 0,))
        #以上是随机移位缩放旋转l
        img, masks = randomHorizontalFlip(img, masks)
        #以上是随即水平旋转
        img, masks = randomVerticleFlip(img, masks)
        #以上是随机垂直旋转
        masks = np.expand_dims(masks, axis=2)               #

        # image_set.append(np.expand_dims(img, axis=0))
        # mask_set.append(np.expand_dims(masks, axis=0))

        crop_im, crop_ma = crop_images(img, masks)#DRIVE CHASEDB STARE

        image_set.append(crop_im)
        mask_set.append(crop_ma)
        # print(img.shape, masks.shape,'====================')
    aug_img  = np.concatenate([image_set[i] for i in range(0, len(image_set))],axis=0)
    aug_mask = np.concatenate([mask_set[i] for i in range(0, len(mask_set))], axis=0)

    return aug_img, aug_mask

def data_auguments(aug_num,size_h, size_w,path_images, path_gt, total_imgs, mask_ch, augu=True):

    all_images, all_masks = read_drive_images(size_h, size_w,path_images, path_gt, total_imgs, mask_ch)   # original data
    if augu is False:
        return all_images, all_masks
    # print('image and gt shape is:', all_images.shape, all_masks.shape)
    img_list = []
    gt_list = []
    for nums in range(0, aug_num):
        for i_d in range(0, all_images.shape[0]):
            aug_img, aug_gt = deformation_set(all_images[i_d, :, :, :, ], all_masks[i_d, :, :, :, ])
            print(aug_img.shape,'---------', aug_gt.shape)
            img_list.append(aug_img)
            gt_list.append(aug_gt)

    img_au = np.concatenate(img_list, axis=0)
    gt_au = np.concatenate(gt_list, axis=0)



    return img_au, gt_au

def data_for_train(aug_num,size_h, size_w,path_images, path_gt, total_imgs, mask_ch,augu):
    all_images, all_masks = data_auguments(aug_num, size_h, size_w, path_images, path_gt, total_imgs, mask_ch,augu)
    print('image and gt shape is:', all_images.shape, all_masks.shape)
    # img = np.array(all_images, np.float32).transpose(0,3,1,2) / 255.0
    # mask = np.array(all_masks, np.float32).transpose(0,3,1,2) / 255.0
    img = np.array(all_images, np.float32).transpose(0, 3, 1, 2)
    mask = np.array(all_masks, np.float32).transpose(0, 3, 1, 2)

    if mask_ch ==1:
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0

    index = np.arange(img.shape[0])
    np.random.shuffle(index)
    img = img[index, :, :, :]
    mask = mask[index, :, :]

    return img, mask


def save_drive_data(mum_arg = 1):
    images, mask= data_for_train(mum_arg, Constants.size_h_drive,Constants.size_w_drive,
                                  path_images_drive, path_gt_drive, 20, mask_ch=1, augu=True)
    images_test, mask_test = data_for_train(mum_arg, Constants.size_h_drive,Constants.size_w_drive,
                                  path_images_test_drive, path_gt_test_drive, 20, mask_ch=1, augu=False)
    images_val, mask_val = data_for_train(mum_arg, Constants.size_h_drive, Constants.size_w_drive,
                                            path_images_val_drive, path_gt_val_drive, 2, mask_ch=1, augu=False)

    images, mask = data_shuffle(images, mask)
    images_test,mask_test = data_shuffle(images_test,mask_test)


    try:
        read_numpy_into_npy(images, Constants.path_image_drive)
        read_numpy_into_npy(mask, Constants.path_label_drive)
        read_numpy_into_npy(images_test, Constants.path_test_image_drive)
        read_numpy_into_npy(mask_test, Constants.path_test_label_drive)
        read_numpy_into_npy(images_val, Constants.path_val_image_drive)
        read_numpy_into_npy(mask_val, Constants.path_val_label_drive)

        print('========  all drive train and test data has been saved ! ==========')
    except:
        print(' file save exception has happened! ')

    pass


def check_bst_data():
    a = load_from_npy(Constants.path_image_drive)
    b = load_from_npy(Constants.path_label_drive)
    c = load_from_npy(Constants.path_test_image_drive)
    d = load_from_npy(Constants.path_test_label_drive)

    print(a.shape, b.shape, c.shape, d.shape)
    print(np.max(a), np.max(b), np.max(c), np.max(d))


if __name__ == '__main__':
    save_drive_data()
    check_bst_data()
    pass
