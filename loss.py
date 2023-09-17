#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_msssim





class OhemCELoss(nn.Module):#其中 thresh 表示的是，损失函数大于多少的时候，会被用来做反向传播。n_min 表示的是，在一个 batch 中，最少需要考虑多少个样本。
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')#设置 reduction 为 none，保留每个元素的损失，返回的维度为 N\times H\times W。

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)#将预测的损失拉平为一个长向量，每个元素为一个 pixel 的损失。
        loss, _ = torch.sort(loss, descending=True)#将预测的损失拉平为一个长向量，每个元素为一个 pixel 的损失。
        if loss[self.n_min] > self.thresh:#最少考虑 n_min 个损失最大的 pixel，如果前 n_min 个损失中最小的那个的损失仍然大于设定的阈值，那么取实际所有大于该阈值的元素计算损失：loss=loss[loss>thresh]。
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)#这些 hard example 的损失的均值作为最终损失


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(nn.FocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss

class NormalLoss(nn.Module):
    def __init__(self,ignore_lb=255, *args, **kwargs):
        super( NormalLoss, self).__init__()
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels)
        return torch.mean(loss)


class Fusionloss(nn.Module):

    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()
        self.L2 = nn.MSELoss(reduction='none',reduce=True, size_average=True)

    def forward(self,image_vis,image_ir,labels,generate_img):
        # image_y = image_vis[:, :1, :, :]
        # loss_in = F.l1_loss(image_y, generate_img)
        # ir_grad = self.sobelconv(image_y)
        # generate_img_grad = self.sobelconv(generate_img)
        # loss_grad = F.l1_loss(ir_grad, generate_img_grad)
        # loss_tal = loss_grad+loss_in
        #
        # return  loss_tal, loss_in,loss_grad
        loss_msssim = pytorch_msssim.msssim
        # return loss_tal, loss_in, loss_grad
        image_y=image_vis[:,:1,:,:]
        x_in_max=torch.max(image_y,image_ir)
        loss_in=F.l1_loss(x_in_max,generate_img)  # 结构损失
        # loss_in=self.L2(generate_img,x_in_max)
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)

        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint= torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)#这里还可以改成ir的特征的梯度
        # s_max = torch.max(y_grad,ir_grad)
        # loss_s = pytorch_msssim.msssim(s_max, generate_img_grad)

        loss_total=loss_in+loss_grad#原本系数是5


        # ssim_vis = SSIM(generate_img,image_y)
        # ssim_ir = SSIM(generate_img,image_ir)
        # ssim = torch.max(ssim_ir, ssim_vis)
        return loss_total,loss_in,loss_grad
        # loss_in = F.l1_loss(image_ir, generate_img)
        # ir_grad = self.sobelconv(image_ir)
        # generate_img_grad=self.sobelconv(generate_img)
        # loss_grad = F.l1_loss(ir_grad, generate_img_grad)
        # loss_total = loss_in + loss_grad
        # return loss_total,loss_in,loss_grad

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)#增加维度
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()#将参数转化为可学习的，false不可训练

        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

if __name__ == '__main__':
    pass

