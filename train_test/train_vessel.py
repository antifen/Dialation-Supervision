import sys
sys.path.append('../')
sys.path.append('/root/dilation_supervised/')
sys.path.append('../data_process/')
sys.path.append('../networks/mine/')


from dilation_supervised import Constants
import numpy as np
import torch
import cv2
import math
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import torch.nn as nn
import torch.optim as optims
from torch import optim
import torch.utils.data as data
import torch.nn.functional as F
from skimage.morphology import binary_dilation, binary_erosion
from skimage.morphology import disk
from skimage.segmentation import find_boundaries
from time import time
from dilation_supervised.data_process.data_load import ImageFolder, get_drive_data
from dilation_supervised.networks.mine.net import UNet
# from dilation_supervised.networks.common.LL import LL2
# from dilation_supervised.networks.othernets.DenseUnet import Dense_Unet
from dilation_supervised.train_test.losses import loss_ce, DiceLoss
from dilation_supervised.train_test.eval_test import val_vessel
from torch.utils.tensorboard import SummaryWriter
from dilation_supervised.train_test.help_functions import platform_info, check_size
from dilation_supervised.train_test.evaluations import threshold_by_otsu
from torchinfo import summary
from PIL import Image


learning_rates = Constants.learning_rates
gcn_model = False


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
        img = Image.fromarray(data.astype(np.uint8))          # the image is already 0-255
    else:
        img = Image.fromarray((data * 255).astype(np.uint8))  # the image is between 0-1
    img.save(filename + '.png')
    return img

def load_model(path):
    net = torch.load(path)
    return net

def adjust_lr(optimizer, base_lr, iter, max_iter, power=0.9):
    lr = base_lr * (1 - float(iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def update_lr1(optimizer, old_lr, ratio):
    for param_group in optimizer.param_groups:
        param_group['lr'] = old_lr / ratio
    print('update learning rate: %f -> %f' % (old_lr, old_lr / ratio))
    return old_lr / ratio

def update_lr2(epoch, optimizer, total_epoch=Constants.TOTAL_EPOCH):
    new_lr = learning_rates * (1 - epoch / total_epoch)
    for p in optimizer.param_groups:
        p['lr'] = new_lr

def optimizer_net(net1, optimizers1,  criterion, images, masks, ch):
    optimizers1.zero_grad()

    pred0, pred1, pred2 = net1(images)
    #添加边界权重
    # M = masks.cpu().numpy().astype(np.uint8)
    # weight = np.zeros_like(M)
    # z1 = np.zeros_like(M)
    # z2 = np.zeros_like(M)
    # for i_d in range(0, M.shape[0]):
    #     z1[i_d] = find_boundaries(M[i_d, :, :, :, ].reshape(512, 512), connectivity=1, mode='inner')
    #     z2[i_d] = find_boundaries(M[i_d, :, :, :, ].reshape(512, 512), connectivity=1, mode='outer')
    #     weight[i_d] = z1[i_d] + z2[i_d]
    # weight = torch.from_numpy(weight).cuda().float()

    #第二个输出的通道
    # MM = pred1.clone().detach()
    # MM1 = abs(MM-masks)
    # zero = torch.zeros_like(MM)
    # one = zero + 1
    # new = torch.where(MM1 <= 0.7, MM, abs(MM-one))
    # new[new >= 0.5] = 1
    # new[new < 0.5] = 0
    # MM2 = abs(new - masks)
    # weight_unreal = torch.where(MM2 >= 0.4, one, zero)

    #假阳性 假阴性权重
    # M2 = pred2.clone().detach()
    # M2[M2 >= 0.5] = 1
    # M2[M2 < 0.5] = 0
    # weight_err = (~(M2.int() == masks.int())).int()

    # weight1 = 1 + 10 * weight
    # weight_unreal = 1 + 5 * weight_unreal
    # weight1 = weight1.reshape((-1,))
    # weight_unreal = weight_unreal.reshape((-1,))

    # loss1 = loss_ce(pred1, masks, nn.BCELoss(weight=weight1), ch)
    # loss2 = loss_ce(pred2, masks, nn.BCELoss(), ch)
    # loss3 = loss_ce(pred2, MM, criterion, ch)
    #
    # loss = loss1 + loss2 + loss3
#################################################################################################################################################
    masks255 = (masks*255).cpu().numpy()
    masks_dilation = binary_dilation(masks255)
    masks_dilation = (torch.from_numpy(masks_dilation).cuda()).int()

    dilation_edge = masks_dilation - masks
    edge_weight = (1 + 5*masks + 5*dilation_edge).reshape((-1,))

    loss0 = loss_ce(pred0, masks, criterion, ch)
    loss00 = DiceLoss(pred0, masks)
    loss1 = loss_ce(pred1, masks_dilation, criterion, ch)
    loss2 = loss_ce(pred2, masks, nn.BCELoss(weight=edge_weight), ch)
    loss01 = DiceLoss(pred2, masks)

    loss = loss0 + loss00 + loss1 + loss2 + loss01
    loss.backward()
    optimizers1.step()
    return pred2, loss

def optimizer_net_adaptive_threshold(net1, optimizers1,  criterion, images, masks, ch):# This for train an adaptive threshold
    optimizers1.zero_grad()
    net = load_model('/root/dilation_supervised/log/weights_save/DRIVE/*.iter5')
    net.eval()
    with torch.no_grad():
        images00, images0, images = net(images)

    H = images.clone().detach()

    threshold_map = net1(H)
    pred = 1 / (torch.exp(-(H - threshold_map) * 50) + 1)

    loss = loss_ce(pred, masks, criterion, ch)

    loss.backward()
    optimizers1.step()
    return pred, loss

def visual_preds(preds, is_preds=True):  # This for multi-classification
    rand_arr = torch.rand(size=(preds.size()[1], preds.size()[2], 3))
    color_preds = torch.zeros_like(rand_arr)
    outs = preds.permute((1, 2, 0))  # N H W C
    if is_preds is True:
        outs_one_hot = torch.argmax(outs, dim=2)
    else:
        outs_one_hot = outs.reshape((preds.size()[1], preds.size()[2]))
    for H in range(0, preds.size()[1]):
        for W in range(0, preds.size()[2]):
            if outs_one_hot[H, W] == 1:
                color_preds[H, W, 0] = 255
            if outs_one_hot[H, W] == 2:
                color_preds[H, W, 1] = 255
            if outs_one_hot[H, W] == 3:
                color_preds[H, W, 2] = 255
            if outs_one_hot[H, W] == 4:
                color_preds[H, W, 0] = 255
                color_preds[H, W, 1] = 255
                color_preds[H, W, 2] = 255
    return color_preds.permute((2, 0, 1))

def train_model(learning_rates):
    writer = SummaryWriter(comment=f"MyDRIVETrain01", flush_secs=1)
    tic = time()
    total_epoch = Constants.TOTAL_EPOCH
    train_epoch_best_loss = Constants.INITAL_EPOCH_LOSS
    ch = Constants.BINARY_CLASS
    criterion = nn.BCELoss()
    net1 = UNet().to(device)
    summary(net1, input_data=torch.rand(Constants.BATCH_SIZE, 1, 512, 512))
#     net1 = load_model('/root/dilation_supervised/log/weights_save/DRIVE/*.iter5').to(device)
#     net2 = LL2(1).to(device)
    optimizers1 = optims.Adam(net1.parameters(), lr=learning_rates, betas=(0.9, 0.999),weight_decay=1e-5)
    trains, val = get_drive_data()
    dataset = ImageFolder(trains[0], trains[1])
    data_loader = data.DataLoader(dataset, batch_size = Constants.BATCH_SIZE, shuffle=True, num_workers=8)

    rand_img, rand_label, rand_pred = None, None, None
    for epoch in range(1, total_epoch + 1):
        net1.train(mode=True)
        # net2.train(mode=True)
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        index = 0
        for img, mask in data_loader_iter:

            img = img.to(device)
            mask = mask.to(device)
            # pred, train_loss = optimizer_net1(net2, optimizers2, criterion, img, mask, ch)
            pred, train_loss = optimizer_net(net1, optimizers1, criterion, img, mask, ch)
            train_epoch_loss += train_loss.item()
            index = index + 1
            if np.random.rand(1) > 0.4 and np.random.rand(1) < 0.8:
                rand_img, rand_label, rand_pred = img, mask, pred

        train_epoch_loss = train_epoch_loss / len(data_loader_iter)
        writer.add_scalar('Train/loss', train_epoch_loss, epoch)
        if ch ==1:      # for [N,1,H,W]
            rand_pred_cpu = rand_pred[0, :, :, :].detach().cpu().reshape((-1,)).numpy()
            rand_pred_cpu = threshold_by_otsu(rand_pred_cpu)
            new_mask = rand_label[0, :, :, :].cpu().reshape((-1,)).numpy()
            writer.add_scalar('Train/acc', rand_pred_cpu[np.where(new_mask == rand_pred_cpu)].shape[0] / new_mask.shape[0], epoch)  # for [N,H,W,1]
        if ch ==2:      # for [N,2,H,W]
            new_mask = rand_label[0, :, :, :].cpu().reshape((-1,))
            new_pred = torch.argmax(rand_pred[0, :, :, :].permute((1, 2, 0)), dim=2).detach().cpu().reshape((-1,))
            t = new_pred[torch.where(new_mask == new_pred)].size()[0]
            writer.add_scalar('Train/acc', t / new_pred.size()[0], epoch)

        platform_info(epoch, tic, train_epoch_loss, Constants.IMG_SIZE, optimizers1)
        # platform_info(epoch, tic, train_epoch_loss, Constants.IMG_SIZE, optimizers2)
        if epoch % 10 == 1:
            writer.add_image('Train/image_origins', rand_img[0, :, :, :], epoch)
            writer.add_image('Train/image_labels', rand_label[0, :, :, :], epoch)
            if ch == 1:  # for [N,1,H,W]
                writer.add_image('Train/image_predictions', rand_pred[0, :, :, :], epoch)
            if ch == 2:  # for [N,2,H,W]
                  writer.add_image('Train/image_predictions', torch.unsqueeze(torch.argmax(rand_pred[0, :, :, :], dim=0), 0),
                             epoch)
        update_lr2(epoch, optimizers1)  # modify  lr
        # update_lr2(epoch, optimizers2)


        print('************ start to validate current model {}.iter performance ! ************'.format(epoch))
        acc, sen, f1score, val_loss = val_vessel(net1, val[0], val[1], val[0].shape[0], epoch)
        writer.add_scalar('Val/accuracy', acc, epoch)
        writer.add_scalar('Val/sensitivity', sen, epoch)
        writer.add_scalar('Val/f1score', f1score, epoch)
        writer.add_scalar('Val/val_loss', val_loss, epoch)



        model_name = Constants.saved_path + "{}.iter5".format(epoch)
        torch.save(net1, model_name)


    print('***************** Finish training process ***************** ')

if __name__ == '__main__':
    RANDOM_SEED = 42  # any random number
    # RANDOM_SEED = 3407
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # CPU
        torch.cuda.manual_seed(seed)  # GPU
        torch.cuda.manual_seed_all(seed)  # All GPU
        os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
        torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
        torch.backends.cudnn.benchmark = False  # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现


    set_seed(RANDOM_SEED)
    train_model(learning_rates)
    pass





