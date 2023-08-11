import numpy as np
import random
from PIL import Image
import cv2
import os
import random
from PIL import Image
import cv2
import os


def visualize(data,filename):
    '''
    :param data:      the visual data must be channel last ! H W C
    :param filename:
    :return:          save into filename path .png format !
    '''
    assert (len(data.shape)==3) #height*width*channels
    img = None
    if data.shape[2]==1:  #in case it is black and white
        data = np.reshape(data,(data.shape[0],data.shape[1]))
    if np.max(data)>1:
        img = Image.fromarray(data.astype(np.uint8))   #the image is already 0-255
    else:
        img = Image.fromarray((data*255).astype(np.uint8))  #the image is between 0-1
    img.save(filename + '.png')
    # print('===========================>visualize function have saved into ',filename + '.png')
    return img

def group_images(data, per_row):
    '''
    :param data:     both N H W C and N C H W is OK ！
    :param per_row:  every images number in a row
    :return:         images channel last
    '''
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

''' 
this method use to color visual predict difference with gt
blue represent FP and red represent TN 
'''

def retina_color(pathpre, pathgt, pathsave):
    img_test_path = pathpre
    img_test_gt = pathgt
    predicts = Image.open(img_test_path)
    gts = Image.open(img_test_gt)
    predicts_img = np.asarray(predicts)
    gt_img  = np.asarray(gts)
    predicts_img = np.reshape(predicts_img,(predicts_img.shape[0], predicts_img.shape[1],1))
    gt_img = np.reshape(gt_img, (gt_img.shape[0], gt_img.shape[1], 1))
    # print('predict size is :',predicts_img.shape, 'predict max value is :', np.max(predicts_img))
    # print('gt size is: ', gt_img.shape)
    new_color_retina = np.zeros(shape=(gt_img.shape[0], gt_img.shape[1], 3))
    thereshold = 0.3
    # print(np.max(predicts_img), np.max(gt_img))
    for pixel_x in range(0, gt_img.shape[0]):
        for pixel_y in range(0,gt_img.shape[1]):
            if predicts_img[pixel_x,pixel_y,0]>=thereshold:
                if gt_img[pixel_x,pixel_y,0]>=0.5:
                    new_color_retina[pixel_x,pixel_y,1] = 255
                else:
                    new_color_retina[pixel_x, pixel_y,2] = 255
            if gt_img[pixel_x,pixel_y,0]>=0.5:
                if  predicts_img[pixel_x,pixel_y,0] >= thereshold:
                    new_color_retina[pixel_x,pixel_y,1]=255
                else:
                    new_color_retina[pixel_x, pixel_y, 0] = 255
    # print('new color image shape is: ',new_color_retina.shape)
    visualize(new_color_retina, pathsave)
    # print(np.max(predicts_img), np.max(gt_img))
    return

def retina_color_different(predicts, gts, pathsave):
    predicts_img = np.asarray(predicts)
    gt_img  = np.asarray(gts)
    predicts_img = np.reshape(predicts_img,(predicts_img.shape[0], predicts_img.shape[1],1))
    gt_img = np.reshape(gt_img, (gt_img.shape[0], gt_img.shape[1], 1))
    # print('predict size is :',predicts_img.shape, 'predict max value is :', np.max(predicts_img))
    # print('gt size is: ', gt_img.shape)
    new_color_retina = np.zeros(shape=(gt_img.shape[0], gt_img.shape[1], 3))
    thereshold = 0.5
    # print(np.max(predicts_img), np.max(gt_img))
    for pixel_x in range(0, gt_img.shape[0]):
        for pixel_y in range(0,gt_img.shape[1]):
            if predicts_img[pixel_x,pixel_y,0] >= thereshold:
                if gt_img[pixel_x,pixel_y,0] >= thereshold:
                    new_color_retina[pixel_x,pixel_y, 1] = 255
                else:
                    new_color_retina[pixel_x, pixel_y, 2] = 255
            if gt_img[pixel_x,pixel_y,0] >= thereshold:
                if  predicts_img[pixel_x,pixel_y,0] >= thereshold:
                    new_color_retina[pixel_x,pixel_y,1] = 255
                else:
                    new_color_retina[pixel_x, pixel_y, 0] = 255
    # print('new color image shape is: ',new_color_retina.shape)
    visualize(new_color_retina, pathsave)
    # print(np.max(predicts_img), np.max(gt_img))
    return


if __name__ == '__main__':
    pathpre = './retina/10.png'              # 注意： 分割之后的概率图，值是 0-1之间
    pathgt = './retina/10_manual1.png'       # 对应的ground-truth
    pathsave = '../img4/predict_comp304'     # 差异图保存路径
    retina_color(pathpre,pathgt,pathsave)

    # test_nuclei()

    pass