
from PIL import Image
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage import distance_transform_edt
import  cv2
from skimage.transform import rotate
from skimage.transform import rescale, resize
import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import ImageEnhance


def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)
    return image, mask

def randomHorizontalFlip(image, mask):
    if np.random.random() < 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)
    return image, mask

def randomVerticleFlip(image, mask):
    if np.random.random() < 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    return image, mask

def elastic_transform(image, mask, alpha, sigma, alpha_affine, random_state=None):
    # Function to distort image  alpha = im_merge.shape[1]*2、sigma=im_merge.shape[1]*0.08、alpha_affine=sigma
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if np.random.random() < 0.5:
        # mask = np.expand_dims(mask, axis=2)
        # print(mask.shape,'??????')
        cc = mask.shape[2]
        if cc==1:
            mask = np.concatenate([mask, mask, mask], axis =2)

        if random_state is None:
            random_state = np.random.RandomState(None)

        shape = image.shape
        shape_size = shape[:2]   #(512,512)表示图像的尺寸
        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        # pts1为变换前的坐标，pts2为变换后的坐标，范围为什么是center_square+-square_size？
        # 其中center_square是图像的中心，square_size=512//3=170
        pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                           center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        # Mat getAffineTransform(InputArray src, InputArray dst)  src表示输入的三个点，dst表示输出的三个点，获取变换矩阵M
        M = cv2.getAffineTransform(pts1, pts2)  #获取变换矩阵
        #默认使用 双线性插值，
        image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(mask, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
        # # random_state.rand(*shape) 会产生一个和 shape 一样打的服从[0,1]均匀分布的矩阵
        # * 2 - 1 是为了将分布平移到 [-1, 1] 的区间
        # 对random_state.rand(*shape)做高斯卷积，没有对图像做高斯卷积，为什么？因为论文上这样操作的
        # 高斯卷积原理可参考：https://blog.csdn.net/sunmc1204953974/article/details/50634652
        # 实际上 dx 和 dy 就是在计算论文中弹性变换的那三步：产生一个随机的位移，将卷积核作用在上面，用 alpha 决定尺度的大小
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dz = np.zeros_like(dx)  #构造一个尺寸与dx相同的O矩阵
        # np.meshgrid 生成网格点坐标矩阵，并在生成的网格点坐标矩阵上加上刚刚的到的dx dy
        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))  #网格采样点函数
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
        # indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
        pp = None
        if cc==1:
            pp =  np.expand_dims(map_coordinates(mask, indices, order=1, mode='reflect').reshape(shape)[:,:,1], axis=2)
        else:
            pp = map_coordinates(mask, indices, order=1, mode='reflect').reshape(shape)
        return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), pp

    else:
        return image, mask

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    images = image
    if np.random.random() < u:
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        images = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        # image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return images

def deformation_set(image, mask,
                           shift_limit=(-0.1, 0.1),
                           scale_limit=(-0.2, 0.2),
                           rotate_limit=(-180.0, 180.0),
                           aspect_limit=(-0.1, 0.1),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    # print('--------------', image.shape, mask.shape)
    masks = mask
    img = image
    if np.random.random() < u:
        height, width, channel = img.shape
        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)


        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        img = cv2.warpPerspective(img, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        masks = cv2.warpPerspective(masks, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))
        # print('--------**-------',image.shape, mask.shape)
        if len(masks.shape)==2:
            masks = np.expand_dims(masks, axis=2)
        # print('-------<>------',image.shape, masks.shape)

        img, masks = randomHorizontalFlip(img, masks)
        img, masks = randomVerticleFlip(img, masks)
        img, masks = randomRotate90(img, masks)
        # print(img.shape, masks.shape,'====================')
        img, masks = elastic_transform(img, masks, img.shape[1] * 2, img.shape[1] * 0.08,
                                           img.shape[1] * 0.08)
        # print(img.shape, masks.shape, '=========...===========')
        # img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
        # masks = np.array(masks, np.float32).transpose(2, 0, 1) / 255.0
        # img = np.array(img, np.float32).transpose(1, 2, 0)
        # masks = np.array(masks, np.float32).transpose(1, 2, 0)
        # print(img.shape, masks.shape,'================')

    return img, masks

def read_all_images(path_images, all_images,size_h, size_w, type ='resize'):

    '''
    :param path_images:  the path of data array
    :param all_images:   all_images is numpy array to contain all images channel 1 and 3 is different
    :return:             N H W C
    '''
    index = 0
    for _, _, content in os.walk(path_images):
        if type is 'polyp':
            content.sort(key=lambda x: int(x[0:x.find('.')]))
        else:
            content.sort(key=lambda x: int(x[0:x.find('_')]))

        print('read content sequence is : \n', content)
        for p_img in list(content):
                img = Image.open(path_images + p_img)   # DRIVE CHASEDB STARE HRF dataset
                # img = cv2.imread(path_images + p_img)     # DCA1 dataset
                # print(p_img)
                # enh_con = ImageEnhance.Sharpness(img)
                # img     = enh_con.enhance(factor=2.1)   # ADD augment
                if type == 'resize':
                    img = img.resize((size_h, size_w))
                img_arr = np.asarray(img)
                # img_arr = randomHueSaturationValue(img_arr)  # add
                if(len(img_arr.shape)==2):   # (1024, 1024)
                    new_img_arr = np.reshape(img_arr,(img_arr.shape[0],img_arr.shape[1], 1)) # (1024, 1024, 1)
                    all_images[index, :, :, :, ] = new_img_arr
                else:
                    all_images[index, :, :, :,] = img_arr           # (1024, 1024, 3)
                index =index+1
    print(' this directory of ({}) has total {} images and images tensor is {}'.format(path_images, index, all_images.shape))
    # visualize(group_images(all_images, 5),'./testimages2'+str(np.random.randint(1,100)))
    return all_images

def group_images(data, per_row):
    '''
    :param data:     both N H W C and N C H W is OK ！
    :param per_row:  every images number in a row
    :return:         images channel last
    '''
    # print(data.shape[0],'----')
    assert data.shape[0]%per_row==0
    if (data.shape[1]==1 or data.shape[1]==3):
        data = np.transpose(data,(0,2,3,1))      # change data format into channel last !
    all_stripe = []
    for i in range(int(data.shape[0]/per_row)):
        stripe = data[i*per_row]
        for k in range(i*per_row+1, i*per_row+per_row):
            stripe = np.concatenate((stripe,data[k]),axis=1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    for i in range(1,len(all_stripe)):
        totimg = np.concatenate((totimg,all_stripe[i]),axis=0)
    return totimg

# visualize image (as PIL image, NOT as matplotlib!)
def visualize(data, filename):
    '''
    :param data:     input is 3d tensor of a image,whose size is (H*W*C)
    :param filename:
    :return:         saved into filename positions
    '''
    assert (len(data.shape) == 3)  # height*width*channels
    # print data
    if data.shape[2] == 1:  # in case it is black and white
        data = np.reshape(data, (data.shape[0], data.shape[1]))
    if np.max(data) > 1:
        img = Image.fromarray(data.astype(np.uint8))  # the image is already 0-255
    else:
        img = Image.fromarray((data * 255).astype(np.uint8))  # the image is between 0-1
    img.save(filename + '.png')
    return img


# color : red, yellow, green, blue
def get_color(category):
    # category can be 2, 3, 4
    # Two Category is black-white
    if category == 2:
        return [(0, 0, 0), (255, 255, 255)]
    # Three Category is red-green-blue
    elif category == 3:
        return [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    # Four category is red-green-blue-yello
    else:
        return [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 0)]

def label2rgb(imgs, category):
    if category > 4:
        print("ERROR: at most 4 categories")
        exit()
    assert (len(imgs.shape) == 4)
    result = np.zeros((imgs.shape[0], 3, imgs.shape[2], imgs.shape[3]))
    color = get_color(category)
    for k in range(imgs.shape[0]):
        for i in range(imgs.shape[2]):
            for j in range(imgs.shape[3]):
                c = int(imgs[k, 0, i, j])
                # 3 channels
                for m in range(3):
                    result[k, m, i, j] = color[c][m]
    return result

def data_shuffle(img, mask):
    index = np.arange(img.shape[0])
    np.random.shuffle(index)
    img  = img[index, :, :, :]
    mask = mask[index, :, :]
    return img, mask

def normal01():
    import torchvision.transforms as transforms
    transfrom = transforms.Compose([
            transforms.ToTensor(),         # height * width * channel -> channel * height * width
            transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
        ])
    return transfrom


if __name__ == '__main__':
    s = normal01()(Image.open('./tempt/cat.png'))
    visualize(s.permute((1,2,0)).numpy(),'./tempt/dog')

    print('---end---', s.size())

    pass