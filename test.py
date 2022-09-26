import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
# from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
from torchvision import transforms
from PIL import Image
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
import socket
from datetime import datetime

import torch
from tensorboardX import SummaryWriter
import tqdm
import torch.nn as nn
from torch.nn import functional as F
import math
import torch.backends.cudnn as cudnn
from torch.nn import init
from model import R2AttU_Net, U_Net
from vision_transformer import SwinUnet

ROOT_path = 'H:/研究生课程/大数据分析/xm/train_dataset/root'


class DefaultConfig(object):
    num_epochs = 150
    epoch_start_i = 0
    checkpoint_step = 5
    validation_step = 1
    crop_height = 416
    crop_width = 416
    batch_size = 2
    # 训练集所在位置，根据自身训练集位置进行修改
    data = r'H:/研究生课程/大数据分析/xm/train_dataset/'

    log_dirs = os.path.join(ROOT_path, 'Log/OCT')

    lr = 1e-4
    lr_mode = 'poly'
    net_work = 'BaseNet'
    # net_work= 'MSSeg'  #net_work= 'UNet'

    momentum = 0.9  #
    weight_decay = 1e-4  #

    mode = 'train'
    num_classes = 2

    k_fold = 4
    test_fold = 4
    num_workers = 0

    cuda = '0'
    use_gpu = True
    pretrained_model_path = os.path.join(ROOT_path, 'pretrained', 'resnet34-333f7ec4.pth')
    save_model_path = r'H:\研究生课程\大数据分析\xm\train_dataset\save'


from torch.autograd import Variable


class DiceLoss(nn.Module):
    def __init__(self, smooth=0.01):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        input = torch.sigmoid(input)
        Dice = Variable(torch.Tensor([0]).float()).cuda()
        intersect = (input * target).sum()
        union = torch.sum(input) + torch.sum(target)
        Dice = (2 * intersect + self.smooth) / (union + self.smooth)
        dice_loss = 1 - Dice
        return dice_loss


class Multi_DiceLoss(nn.Module):
    def __init__(self, class_num=5, smooth=0.001):
        super(Multi_DiceLoss, self).__init__()
        self.smooth = smooth
        self.class_num = class_num

    def forward(self, input, target):
        input = torch.exp(input)
        Dice = Variable(torch.Tensor([0]).float()).cuda()
        for i in range(0, self.class_num):
            input_i = input[:, i, :, :]
            target_i = (target == i).float()
            intersect = (input_i * target_i).sum()
            union = torch.sum(input_i) + torch.sum(target_i)
            dice = (2 * intersect + self.smooth) / (union + self.smooth)
            Dice += dice
        dice_loss = 1 - Dice / (self.class_num)
        return dice_loss


class EL_DiceLoss(nn.Module):
    def __init__(self, class_num=4, smooth=1, gamma=0.5):
        super(EL_DiceLoss, self).__init__()
        self.smooth = smooth
        self.class_num = class_num
        self.gamma = gamma

    def forward(self, input, target):
        input = torch.exp(input)
        self.smooth = 0.
        Dice = Variable(torch.Tensor([0]).float()).cuda()
        for i in range(1, self.class_num):
            input_i = input[:, i, :, :]
            target_i = (target == i).float()
            intersect = (input_i * target_i).sum()
            union = torch.sum(input_i) + torch.sum(target_i)
            if target_i.sum() == 0:
                dice = Variable(torch.Tensor([1]).float()).cuda()
            else:
                dice = (2 * intersect + self.smooth) / (union + self.smooth)
            Dice += (-torch.log(dice)) ** self.gamma
        dice_loss = Dice / (self.class_num - 1)
        return dice_loss


def augmentation():
    # augment images with spatial transformation: Flip, Affine, Rotation, etc...
    # see https://github.com/aleju/imgaug for more details
    pass


def augmentation_pixel():
    # augment images with pixel intensity transformation: GaussianBlur, Multiply, etc...
    pass


class Data(torch.utils.data.Dataset):
    Unlabelled = [0, 0, 0]
    sick = [255, 255, 255]
    COLOR_DICT = np.array([Unlabelled, sick])

    def __init__(self, dataset_path, scale=(320, 320), mode='train'):
        super().__init__()
        self.mode = mode
        self.img_path = dataset_path + '/img'
        self.mask_path = dataset_path + '/mask'
        self.image_lists, self.label_lists = self.read_list(self.img_path)
        self.resize = scale
        self.flip = iaa.SomeOf((2, 5), [
            iaa.PiecewiseAffine(scale=(0, 0.1), nb_rows=4, nb_cols=4, cval=0),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.1),
            iaa.Affine(rotate=(-20, 20),
                       scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
            iaa.OneOf([
                iaa.GaussianBlur((0, 1.0)),  # blur images with a sigma between 0 and 3.0
                iaa.AverageBlur(k=(3, 5)),  # blur image using local means with kernel sizes between 2 and 7
                iaa.MedianBlur(k=(3, 5)),  # blur image using local medians with kernel sizes between 2 and 7
            ]),
            iaa.contrast.LinearContrast((0.5, 1.5))],
                               random_order=True)

        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        # load image and crop
        img = Image.open(self.image_lists[index]).convert('RGB')
        img = img.resize(self.resize)
        img = np.array(img)
        labels = self.label_lists[index]
        # load label
        if self.mode != 'test':
            label_ori = Image.open(self.label_lists[index]).convert('RGB')
            label_ori = label_ori.resize(self.resize)
            label_ori = np.array(label_ori)
            label = np.ones(shape=(label_ori.shape[0], label_ori.shape[1]), dtype=np.uint8)

            # convert RGB  to one hot

            for i in range(len(self.COLOR_DICT)):
                equality = np.equal(label_ori, self.COLOR_DICT[i])
                class_map = np.all(equality, axis=-1)
                label[class_map] = i

            # augment image and label
            if self.mode == 'train':
                seq_det = self.flip.to_deterministic()  # 固定变换
                segmap = ia.SegmentationMapsOnImage(label, shape=label.shape)
                img = seq_det.augment_image(img)
                label = seq_det.augment_segmentation_maps([segmap])[0].get_arr().astype(np.uint8)

            label_img = torch.from_numpy(label.copy()).float()
            if self.mode == 'val':
                img_num = len(os.listdir(os.path.dirname(labels)))
                labels = label_img, img_num
            else:
                labels = label_img
        imgs = img.transpose(2, 0, 1) / 255.0
        img = torch.from_numpy(imgs.copy()).float()  # self.to_tensor(img.copy()).float()
        return img, labels

    def __len__(self):
        return len(self.image_lists)

    def read_list(self, image_path):
        fold = os.listdir(image_path)
        # fold = sorted(os.listdir(image_path), key=lambda x: int(x[-2:]))
        # print(fold)

        img_list = []
        label_list = []
        if self.mode == 'train':
            for item in fold:
                name = item.split('.')[0]
                img_list.append(os.path.join(image_path, item))
                label_list.append(os.path.join(image_path.replace('img', 'mask'), '{}.png'.format(name)))


        elif self.mode == 'val':
            for item in fold:
                name = item.split('.')[0]
                img_list.append(os.path.join(image_path, item))
                label_list.append(os.path.join(image_path.replace('img', 'mask'), '{}.png'.format(name)))


        elif self.mode == 'test':
            for item in fold:
                name = item.split('.')[0]
                img_list.append(os.path.join(image_path, item))
                label_list.append(os.path.join(image_path.replace('img', 'mask'), '{}.png'.format(name)))

        return img_list, label_list


import shutil
import os.path as osp


def save_checkpoint(state, best_pred, epoch, is_best, checkpoint_path, filename='./checkpoint/checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, osp.join(checkpoint_path, 'best.pth'))
    # shutil.copyfile(filename, osp.join(checkpoint_path, 'model_ep{}.pth'.format(epoch+1)))


def adjust_learning_rate(opt, optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs(step = 30)
    """
    if opt.lr_mode == 'step':
        lr = opt.lr * (0.1 ** (epoch // opt.step))
    elif opt.lr_mode == 'poly':
        lr = opt.lr * (1 - epoch / opt.num_epochs) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(opt.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def one_hot_it(label, label_info):
    # return semantic_map -> [H, W, num_classes]
    semantic_map = []
    for info in label_info:
        color = label_info[info]
        # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
        equality = np.equal(label, color)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map


def compute_score(predict, target, forground=1, smooth=1):
    score = 0
    count = 0
    target[target != forground] = 0
    predict[predict != forground] = 0
    assert (predict.shape == target.shape)
    overlap = ((predict == forground) * (target == forground)).sum()  # TP
    union = (predict == forground).sum() + (target == forground).sum() - overlap  # FP+FN+TP
    FP = (predict == forground).sum() - overlap  # FP
    FN = (target == forground).sum() - overlap  # FN
    TN = target.shape[0] * target.shape[1] - union  # TN

    # print('overlap:',overlap)
    dice = (2 * overlap + smooth) / (union + overlap + smooth)

    precsion = ((predict == target).sum() + smooth) / (target.shape[0] * target.shape[1] + smooth)

    jaccard = (overlap + smooth) / (union + smooth)

    Sensitivity = (overlap + smooth) / ((target == forground).sum() + smooth)

    Specificity = (TN + smooth) / (FP + TN + smooth)

    return dice, precsion, jaccard, Sensitivity, Specificity


def eval_multi_seg(predict, target, num_classes):
    # pred_seg=torch.argmax(torch.exp(predict),dim=1).int()
    pred_seg = predict.data.cpu().numpy()
    label_seg = target.data.cpu().numpy().astype(dtype=np.int)
    assert (pred_seg.shape == label_seg.shape)
    acc = (pred_seg == label_seg).sum() / (pred_seg.shape[0] * pred_seg.shape[1] * pred_seg.shape[2])

    # Dice = []
    # Precsion = []
    # Jaccard = []
    # Sensitivity=[]
    # Specificity=[]

    # n = pred_seg.shape[0]
    Dice = []
    True_label = []
    TP = FPN = 0
    for classes in range(1, num_classes):
        overlap = ((pred_seg == classes) * (label_seg == classes)).sum()
        union = (pred_seg == classes).sum() + (label_seg == classes).sum()
        Dice.append((2 * overlap + 0.1) / (union + 0.1))
        True_label.append((label_seg == classes).sum())
        TP += overlap
        FPN += union

    return Dice, True_label, acc, 2 * TP / (FPN + 1)

    # for i in range(n):
    #     dice,precsion,jaccard,sensitivity,specificity= compute_score(pred_seg[i],label_seg[i])
    #     Dice.append(dice)
    #     Precsion .append(precsion)
    #     Jaccard.append(jaccard)
    #     Sensitivity.append(sensitivity)
    #     Specificity.append(specificity)

    # return Dice,Precsion,Jaccard,Sensitivity,Specificity


def eval_seg(predict, target, forground=1):
    pred_seg = torch.round(torch.sigmoid(predict)).int()
    pred_seg = pred_seg.data.cpu().numpy()
    label_seg = target.data.cpu().numpy().astype(dtype=np.int)
    assert (pred_seg.shape == label_seg.shape)

    Dice = []
    Precsion = []
    Jaccard = []
    n = pred_seg.shape[0]

    for i in range(n):
        dice, precsion, jaccard = compute_score(pred_seg[i], label_seg[i])
        Dice.append(dice)
        Precsion.append(precsion)
        Jaccard.append(jaccard)

    return Dice, Precsion, Jaccard


def batch_pix_accuracy(pred, label, nclass=1):
    if nclass == 1:
        pred = torch.round(torch.sigmoid(pred)).int()
        pred = pred.cpu().numpy()
    else:
        pred = torch.max(pred, dim=1)
        pred = pred.cpu().numpy()
    label = label.cpu().numpy()
    pixel_labeled = np.sum(label >= 0)
    pixel_correct = np.sum(pred == label)

    assert pixel_correct <= pixel_labeled, \
        "Correct area should be smaller than Labeled"

    return pixel_correct, pixel_labeled


def batch_intersection_union(predict, target, nclass):
    """Batch Intersection of Union
    Args:
        predict: input 4D tensor
        target: label 3D tensor
        nclass: number of categories (int),note: not include background
    """
    if nclass == 1:
        pred = torch.round(torch.sigmoid(predict)).int()
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        area_inter = np.sum(pred * target)
        area_union = np.sum(pred) + np.sum(target) - area_inter

        return area_inter, area_union


def test(model, dataloader, args, save_path):
    print('start test!')
    with torch.no_grad():
        model.eval()
        # precision_record = []
        tq = tqdm.tqdm(dataloader, desc='\r')
        tq.set_description('test')
        comments = os.getcwd().split('\\')[-1]
        for i, (data, label_path) in enumerate(tq):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                # label = label.cuda()
            aux_pred, predict = model(data)
            predict = torch.argmax(torch.exp(predict), dim=1)
            pred = predict.data.cpu().numpy()
            pred_RGB = Data.COLOR_DICT[pred.astype(np.uint8)]

            for index, item in enumerate(label_path):
                img = Image.fromarray(pred_RGB[index].squeeze().astype(np.uint8))
                _, name = os.path.split(item)

                img.save(os.path.join(save_path, name))
                # tq.set_postfix(str=str(save_img_path))
        tq.close()


if __name__ == '__main__':
    save_path = r'H:\研究生课程\大数据分析\xm\save'
    model_path = r'H:\研究生课程\大数据分析\xm\train_dataset\save\checkpoint_latest.pth'
    dataset_test = Data(os.path.join(DefaultConfig.data, 'train'), scale=(DefaultConfig.crop_width,
                                                                          DefaultConfig.crop_height), mode='test')
    args = DefaultConfig()
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model_all = {'R2AttU_Net': R2AttU_Net(), 'SwinUnet': SwinUnet()}
    model = model_all['SwinUnet']
    cudnn.benchmark = True
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    test(model, dataloader_test, args, save_path)
