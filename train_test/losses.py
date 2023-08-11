import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')

'''
files to devise different loss strategy !
'''

def myLoss(predict, target, target_yuan):
    epsilon = 1e-5
    assert predict.size() == target.size(), "the size of predict and target must be equal."
    num = predict.size(0)

    pre = predict.view(num, -1)
    tar = target.view(num, -1)
    tar_yuan = target_yuan.view(num, -1)

    intersection = (pre * tar).sum(-1).sum()  # multiply flags and labels
    union = (pre + tar).sum(-1).sum()

    loss1 = 2 * (intersection + epsilon) / (union + epsilon)
    loss2 = torch.sum(pre*tar)/torch.sum(tar)
    loss = 2 * (torch.sum(tar)/torch.sum(tar_yuan)) - loss1 - loss2
    return loss

def similar(target):
    b, h, w = target.shape[0], target.shape[2], target.shape[3]
    conv_op1 = nn.Conv2d(1, 25, kernel_size=5, padding=2, dilation=1, bias=False)

    kernel1 = np.array([[[-1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                        [[0, -1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                        [[0, 0, -1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                        [[0, 0, 0, -1, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, -1], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0], [-1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0], [0, -1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0], [0, 0, -1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0], [0, 0, 0, -1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0], [0, 0, 0, 0, -1], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-1, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, -1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, -1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, -1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [-1, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, -1, 0, 0, 0], [0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, -1, 0, 0], [0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, -1, 0], [0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, -1], [0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [-1, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, -1, 0, 0, 0]],
                        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, -1, 0, 0]],
                        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, -1, 0]],
                        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, -1]],
                        ], dtype='float32')

    # kernel1 = np.array([[[-1, 0, 0], [0, 0, 0], [0, 0, 0]],
    #                     [[0, -1, 0], [0, 0, 0], [0, 0, 0]],
    #                     [[0, 0, -1], [0, 0, 0], [0, 0, 0]],
    #                     [[0, 0, 0], [-1, 0, 0], [0, 0, 0]],
    #                     [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    #                     [[0, 0, 0], [0, 0, -1], [0, 0, 0]],
    #                     [[0, 0, 0], [0, 0, 0], [-1, 0, 0]],
    #                     [[0, 0, 0], [0, 0, 0], [0, -1, 0]],
    #                     [[0, 0, 0], [0, 0, 0], [0, 0, -1]],
    #                     ], dtype='float32')

    kernel1 = kernel1.reshape((25, 1, 5, 5))
    kernel1 = torch.from_numpy(kernel1)
    kernel1 = kernel1.to(device)
    conv_op1.weight.data = kernel1
    kernel1 = abs(conv_op1(target).view(b, 25, h, w))
    kernel1 = torch.where(kernel1 > 0.5, 0, 1)

    return kernel1

def confidence(predict):
    b, h, w = predict.shape[0], predict.shape[2], predict.shape[3]
    confidence = abs(predict.clone().detach() - 0.5)
    binary = torch.where(confidence <= 0.2, 0, 1).float()
    unfold = nn.Unfold(5, dilation=1, padding=2, stride=1)
    confidence_kernel = unfold(binary).reshape(b, 1, 25, h*w).permute(0, 1, 3, 2).reshape(b, 1, h*w, 5, 5).reshape(b, 25, h, w)
    binary_before_kernel = unfold(confidence).reshape(b, 1, 25, h*w).permute(0, 1, 3, 2).reshape(b, 1, h*w, 5, 5).reshape(b, 25, h, w)
    return confidence_kernel, binary_before_kernel

def accuracyy(predict, target):
    b, h, w = predict.shape[0], predict.shape[2], predict.shape[3]
    predict0 = predict.clone().detach()
    predict0[predict0 >= 0.5] = 1
    predict0[predict0 < 0.5] = 0
    acc = (predict0.int() == target.int()).float()
    unfold = nn.Unfold(5, dilation=1, padding=2, stride=1)
    acc_kernel = unfold(acc).reshape(b, 1, 25, h*w).permute(0, 1, 3, 2).reshape(b, 1, h*w, 5, 5).reshape(b, 25, h, w)
    return acc_kernel

def myLoss1(predict, target):
    b, h, w = predict.shape[0], predict.shape[2], predict.shape[3]
    confidence_kernel, binary_before_kernel = confidence(predict)
    kernel = (similar(target)*confidence_kernel*accuracyy(predict, target)).unsqueeze(1)
    # kernel1 = torch.where((kernel * predict) >= 0.5, 1, 0)
    kernel = (binary_before_kernel.unsqueeze(1)) * kernel
    max_kernel, _ = torch.max(kernel, dim=2, keepdim=True)
    i_s = torch.where(max_kernel <= 0, 0, 1)
    kernel_roi = torch.where((kernel/(max_kernel+1e-8).expand_as(kernel)) >= 1, 1.0, 0.0)
    kernel_roi = kernel_roi * i_s.expand_as(kernel_roi)
    number = torch.sum(kernel_roi, dim=2, keepdim=True)
    kernel_roi = kernel_roi/(number + 1e-8)
    kernel_middle = torch.where((kernel/(max_kernel+1e-8)) >= 1, 0, 0)###############感兴趣位置
    kernel_middle.cuda()
    kernel_middle[:, :, 12, :, :] = 1#############中间位置
    kernel_middle = kernel_middle * i_s.expand_as(kernel_middle)

    unfold = nn.Unfold(5, dilation=1, padding=2, stride=1)
    predict_unfold = unfold(predict).reshape(b, 1, 25, h*w).permute(0, 1, 3, 2).reshape(b, 1, h*w, 5, 5).reshape(b, 1, 25, h, w)
    target_unfold = unfold(target).reshape(b, 1, 25, h * w).permute(0, 1, 3, 2).reshape(b, 1, h * w, 5, 5).reshape(b, 1, 25, h, w)

    mse_middle = torch.mean((abs(torch.mul(kernel_middle, predict_unfold)-torch.mul(kernel_middle, target_unfold))).sum(dim=2))
    mse_roi = torch.mean((abs(torch.mul(kernel_roi, predict_unfold) - torch.mul(kernel_roi, target_unfold))).sum(dim=2))
    loss = (1/((1-mse_roi+1e-4)*(1-mse_middle+1e-4)))-(1/(1+1e-4)**2)


    # y = torch.mul(kernel_roi, predict_all2).sum(dim=2)
    # lg_x_ya = torch.mul(kernel_middle, predict2).sum(dim=2)
    # lg_x_ya = lg_x_ya.clamp(min=1e-8, max=1.0)
    # lg_x = torch.log(lg_x_ya)
    # one_y = 1 - y
    # lg_one_x_ya = torch.tensor(1 - torch.mul(kernel_middle, predict2).sum(dim=2))
    # lg_one_x_ya = lg_one_x_ya.clamp(min=1e-8, max=1.0)
    # lg_one_x = torch.log(lg_one_x_ya)
    # loss_d = torch.mean(-(y * lg_x + one_y * lg_one_x))
    #
    # kernel1 = (kernel1 * kernel_roi).sum(dim=2)
    # y_ya = y.clamp(min=1e-8, max=1.0)
    # one_y_ya = one_y.clamp(min=1e-8, max=1.0)
    # loss_wall = torch.mean(-(kernel1 * torch.log(y_ya) + (1 - kernel1) * torch.log(one_y_ya)))
    #
    # loss = loss_d + 5 * loss_wall
    return loss

def myloss2(predict, target):
    epsilon = 1e-5
    avg_op = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)
    predict_avg = avg_op(predict)
    target_avg = avg_op(target)
    predict_add = predict_avg * 49
    target_add = target_avg * 49
    t = (target + epsilon) / (target_add + epsilon)
    p = (predict + epsilon) / (predict_add + epsilon)
    score1 = ((p - t) ** 2).mean()
    score2 = ((predict_avg - target_avg) ** 2).mean()
    score = score1 + score2
    return score

def DiceLoss(predict, target):
    epsilon = 1e-5
    assert predict.size() == target.size(), "the size of predict and target must be equal."
    num = predict.size(0)

    pre = predict.view(num, -1)
    tar = target.view(num, -1)

    intersection = (pre * tar).sum(-1).sum()      # multiply flags and labels
    union = (pre + tar).sum(-1).sum()

    score = 1 - 2 * (intersection + epsilon) / (union + epsilon)

    return score


class FocalLoss(nn.Module):
    def __init__(self, alpha = 0.5, gamma = 2, logits = False, reduce = True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class weighted_entropy(nn.Module):
    '''
    pred  : N, C,
    label : N, -1
    '''
    def __init__(self, need_soft_max = True):
        super(weighted_entropy, self).__init__()
        self.need_soft_max = need_soft_max
        pass

    def forward(self, pred, label):
        if self.need_soft_max is True:
            preds = F.softmax(pred, dim=1)
        else:
            preds = pred
        epusi  = 1e-10
        counts = torch.rand(size=(2,))
        counts[0] = label[torch.where(label == 0)].size(0)
        counts[1] = label[torch.where(label == 1)].size(0)
        N = label.size()[0]
        weights = counts[1]
        weights_avg = 1 - weights / N
        loss = weights_avg * torch.log(preds[:,1] + epusi) + (1 - weights_avg) * torch.log(1 - preds[:,1] + epusi)
        loss = - torch.mean(loss)
        return loss

def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2).mean()

def loss_ce(preds, masks, criterion, selected_mode = 1):

    '''
    :param preds:       N  H  W  C
    :param masks:       N  1  H  W
    :param criterion:   This is used to calculate nn.cross-entropy() or nn.BCE-loss(), both is OK !
    :return:            criterion (N*H*W, C  and  N,-1)
    '''
    if selected_mode == 1:       # when choose nn.BCELoss() masks should be float !
        masks = torch.tensor(masks, dtype=torch.float)
    outs = preds.permute((0, 2, 3, 1))         # N H W C
    outs = outs.reshape((-1, outs.size()[3]))  # N*H*W, C
    if selected_mode == 1:
        outs = outs.reshape((-1,))
    masks = masks.reshape((-1,))               # N,1,H,W ===> N,-1
    if selected_mode == 2:
        masks = torch.tensor(masks, dtype=torch.long)
    return criterion(outs, masks)

def loss_ce1(preds, masks, criterion1, selected_mode = 1):

    '''
    :param preds:       N  H  W  C
    :param masks:       N  1  H  W
    :param criterion:   This is used to calculate nn.cross-entropy() or nn.BCE-loss(), both is OK !
    :return:            criterion (N*H*W, C  and  N,-1)
    '''
    if selected_mode == 1:       # when choose nn.BCELoss() masks should be float !
        masks = torch.tensor(masks, dtype=torch.float)
    outs = preds.permute((0, 2, 3, 1))         # N H W C
    outs = outs.reshape((-1, outs.size()[3]))  # N*H*W, C
    if selected_mode == 1:
        outs = outs.reshape((-1,))
    masks = masks.reshape((-1,))               # N,1,H,W ===> N,-1
    if selected_mode == 2:
        masks = torch.tensor(masks, dtype=torch.long)
    return criterion1(outs, masks)

def loss_ce_ds(preds, masks, criterion, selected_mode = 2):
    # this is used to calculate cross-entropy with many categories !
    if selected_mode ==1:       # when choose nn.BCELoss() masks should be float !
        masks = torch.tensor(masks, dtype=torch.float)
    outs0 = preds[0].permute((0, 2, 3, 1))  # N H W C
    outs0 = outs0.reshape((-1, outs0.size()[3]))  # N*H*W, C
    outs1 = preds[1].permute((0, 2, 3, 1))  # N H W C
    outs1 = outs1.reshape((-1, outs1.size()[3]))  # N*H*W, C
    outs2 = preds[2].permute((0, 2, 3, 1))  # N H W C
    outs2 = outs2.reshape((-1, outs2.size()[3]))  # N*H*W, C
    outs3 = preds[3].permute((0, 2, 3, 1))  # N H W C
    outs3 = outs3.reshape((-1, outs3.size()[3]))  # N*H*W, C
    masks = masks.reshape((-1,))  # N,1,H,W ===> N,-1
    masks = torch.tensor(masks, dtype=torch.long)
    loss = 0.25 * criterion(outs0, masks) + 0.5 * criterion(outs1, masks) + \
           0.75 * criterion(outs2, masks) + 1.0 * criterion(outs3, masks)
    return loss

def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2).mean()

if __name__ == '__main__':
    labels = torch.tensor([0, 1, 1, 0, 1, 1])
    pred = torch.tensor([[0.3, 0.7], [0.3, 0.7], [0.3, 0.7], [0.4, 0.6], [0.3, 0.7], [0.3, 0.7]])
    pred2 = torch.tensor([0.3, 0.7, 0.6, 0.2, 0.5, 0.9])

    print(weighted_entropy(need_soft_max = False)(pred,labels))
    print(DiceLoss()(pred2, labels))
    print(FocalLoss()(pred2, torch.tensor(labels, dtype=torch.float)))
    print(nn.CrossEntropyLoss()(pred, labels))

    pass