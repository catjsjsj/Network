#!/usr/bin/python
# -*- encoding: utf-8 -*-
from PIL import Image
import numpy as np
from torch.autograd import Variable
from FusionNet1 import FusionNet3
from TaskFusion_dataset import Fusion_dataset
import argparse
import datetime
import time
import logging
import os.path as osp
import os
from logger import setup_logger
from cityscapes import CityScapes
from loss import OhemCELoss, Fusionloss
from optimizer import Optimizer
import torch
from torch.utils.data import DataLoader
import warnings
import pytorch_msssim

warnings.filterwarnings('ignore')

def parse_args():
    parse = argparse.ArgumentParser()
    return parse.parse_args()


def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def train_fusion(num=0, logger=None):  # 这个是生成器
    lr_start = 0.001
    modelpth = './model'
    Method = 'Fusion'
    modelpth = os.path.join(modelpth, Method)
    fusionmodel = eval('FusionNet3')(output=1)
    fusionmodel.cuda()
    fusionmodel.train()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, fusionmodel.parameters()), lr=lr_start)
    if num > 0:
        fusion_model_path = './model/Fusion/fusion_model.pth'
        fusionmodel.load_state_dict(torch.load(fusion_model_path))

    train_dataset = Fusion_dataset('train')
    print("the training dataset is length:{}".format(train_dataset.length))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    train_loader.n_iter = len(train_loader)
    criteria_fusion = Fusionloss()
    loss_msssim = pytorch_msssim.msssim

    epoch = 10
    st = glob_st = time.time()
    logger.info('Training Fusion Model start~')
    for epo in range(0, epoch):
        if num == 0:
            lr_start = 0.001
            lr_decay = 0.75
            lr_this_epo = lr_start * lr_decay ** (epo - 1)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_epo
        elif num > 0:
            lr_start = 0.0001
            lr_decay = 1
            lr_this_epo = lr_start * lr_decay ** (num - 1)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_epo
        for it, (image_vis, image_ir, label, name) in enumerate(train_loader):
            image_vis = Variable(image_vis).cuda()
            image_vis_ycrcb = RGB2YCrCb(image_vis)
            image_ir = Variable(image_ir).cuda()
            logits_g = fusionmodel(image_vis_ycrcb, image_ir)
            optimizer.zero_grad()

            loss_fusion, loss_in, loss_grad = criteria_fusion(
                image_vis_ycrcb, image_ir, label, logits_g
            )

            max_image = torch.max(image_ir, image_vis_ycrcb[:, :1, :, :])
            msssim_loss_temp2 = 1 - loss_msssim(logits_g, image_ir, normalize=True)
            msssim_loss_temp1 = 1 - loss_msssim(logits_g, image_vis_ycrcb[:, :1], normalize=True)
            mx_ssim = 1 - loss_msssim(logits_g, max_image, normalize=True)

            loss_total = loss_fusion
            loss_total.backward()
            optimizer.step()

            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            now_it = train_loader.n_iter * epo + it + 1
            eta = int((train_loader.n_iter * epoch - now_it)
                      * (glob_t_intv / (now_it)))
            eta = str(datetime.timedelta(seconds=eta))
            if now_it % 10 == 0:
                msg = ', '.join(
                    [
                        'step: {it}/{max_it}',
                        'l_total: {loss_total:.4f}',
                        'l_in: {loss_in:.4f}',
                        'l_grad: {loss_grad:.4f}',
                        'msssim_ir: {msssim_loss_ir:.4f}',
                        'msssim_vis: {msssim_loss_vis:.4f}',
                        'mx_loss: {max_image:.4f}',
                        'eta: {eta}',
                        'time: {time:.4f}',
                    ]
                ).format(
                    it=now_it,
                    max_it=train_loader.n_iter * epoch,
                    loss_total=loss_total.item(),
                    loss_in=loss_in.item(),
                    loss_grad=loss_grad.item(),
                    msssim_loss_ir=msssim_loss_temp2,
                    msssim_loss_vis=msssim_loss_temp1,
                    max_image=mx_ssim,
                    time=t_intv,
                    eta=eta,
                )
                logger.info(msg)
                st = ed
    fusion_model_file = os.path.join(modelpth, 'fusion_model.pth')
    torch.save(fusionmodel.state_dict(), fusion_model_file)
    logger.info("Fusion Model Save to: {}".format(fusion_model_file))
    logger.info('\n')


def run_fusion(type='train'):
    fusion_model_path = './model/Fusion/fusion_model.pth'
    fused_dir = os.path.join('./MSRS/Fusion', type, 'MSRS')
    os.makedirs(fused_dir, mode=0o777, exist_ok=True)
    fusionmodel = eval('FusionNet3')(output=1)
    fusionmodel.eval()
    if args.gpu >= 0:
        fusionmodel.cuda(args.gpu)
    fusionmodel.load_state_dict(torch.load(fusion_model_path))
    print('done!')
    test_dataset = Fusion_dataset(type)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    with torch.no_grad():
        for it, (images_vis, images_ir, labels, name) in enumerate(test_loader):  # name:['00001D.png', '00002D.png']
            images_vis = Variable(images_vis)
            images_ir = Variable(images_ir)
            if args.gpu >= 0:
                images_vis = images_vis.cuda(args.gpu)
                images_ir = images_ir.cuda(args.gpu)
            images_vis_ycrcb = RGB2YCrCb(images_vis)
            logits = fusionmodel(images_vis_ycrcb, images_ir)
            fusion_ycrcb = torch.cat(
                (logits, images_vis_ycrcb[:, 1:2, :,:], images_vis_ycrcb[:, 2:, :, :]),
                dim=1
            )
            fusion_image = YCrCb2RGB(fusion_ycrcb)

            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(
                fusion_image < zeros, zeros, fusion_image)
            fused_image = fusion_image.cpu().numpy()
            fused_image = fused_image.transpose((0, 2, 3, 1))
            fused_image = (fused_image - np.min(fused_image)) / (
                    np.max(fused_image) - np.min(fused_image)
            )
            fused_image = np.uint8(255.0 * fused_image)
            for k in range(len(name)):
                image = fused_image[k, :, :, :]
                image = image.squeeze()
                image = Image.fromarray(image)
                save_path = os.path.join(fused_dir, name[k])
                image.save(save_path)
                print('Fusion {0} Sucessfully!'.format(save_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='SeAFusion')
    parser.add_argument('--batch_size', '-B', type=int, default=16)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=4)
    args = parser.parse_args()
    logpath = './logs'
    logger = logging.getLogger()
    setup_logger(logpath)
    for i in range(0, 3):
        train_fusion(i, logger)
        print("|{0} Train Fusion Model Sucessfully~!".format(i + 1))
    print("training Done!")