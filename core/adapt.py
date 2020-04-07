"""Adversarial adaptation to train target encoder."""

import os

import torch
import torch.optim as optim
from torch import nn

import params
from utils import make_variable


def train_tgt(src_encoder, tgt_encoder, critic,
              src_data_loader, tgt_data_loader):
    """Train encoder for target domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    tgt_encoder.train() #相当于生成器
    critic.train()      #相当于鉴别器

    # setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()    #交叉熵损失函数
    # optimizer_tgt = optim.Adam(tgt_encoder.parameters(),
    #                            lr=params.c_learning_rate,
    #                            betas=(params.beta1, params.beta2))
    optimizer_tgt = optim.RMSprop(tgt_encoder.parameters(),
                                  lr = params.c_learning_rate,
                                  alpha=0.9)

    # optimizer_critic = optim.Adam(critic.parameters(),
    #                               lr=params.d_learning_rate,
    #                               betas=(params.beta1, params.beta2))
    optimizer_critic = optim.RMSprop(critic.parameters(),
                                     lr = params.d_learning_rate,
                                     alpha=0.9)
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))

    ####################
    # 2. train network #
    ####################


    for epoch in range(params.num_epochs):
        # zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        for step, ((images_src, _), (images_tgt, _)) in data_zip:  #对于每个样本，先训练辨别器D，再训练生成器G
            # print(step)
            ###########################
            # 2.1 train discriminator # 训练辨别器D（critic）
            ###########################

            # make images variable
            images_src = make_variable(images_src)
            images_tgt = make_variable(images_tgt)

            # zero gradients for optimizer   清除上一轮的梯度
            optimizer_critic.zero_grad()

            # extract and concat features
            feat_src = src_encoder(images_src)
            feat_tgt = tgt_encoder(images_tgt)
            feat_concat = torch.cat((feat_src, feat_tgt), 0)   #纵向叠加在一起
            # print('编码过后的大小为',feat_src.shape,feat_tgt.shape)

            # predict on discriminator辨别器的预测值
            pred_concat = critic(feat_concat.detach())   #域分类器的预测值，不求叠加之前的两个网络梯度，只求叠加之后的critic的梯度

            # prepare real and fake label辨别器的期望值
            label_src = make_variable(torch.ones(feat_src.size(0)).long())   #对于源域，希望为1,即最小化和1的距离
            label_tgt = make_variable(torch.zeros(feat_tgt.size(0)).long())  #对于目标域，希望为0，即最小化和0的距离
            label_concat = torch.cat((label_src, label_tgt), 0)  #同样纵向叠加

            # compute loss for critic   辨别器损失函数D_loss
            loss_critic = criterion(pred_concat, label_concat)
            loss_critic.backward()

            # optimize critic
            optimizer_critic.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])   #返回每一行中最大值的那个元素的索引（返回最大元素在这一行的列索引）
            acc = (pred_cls == label_concat).float().mean()

            ############################
            # 2.2 train target encoder # 训练生成器G，即目标域编码器target encoder
            ############################

            # zero gradients for optimizer  再次清除上一轮的残余值
            optimizer_critic.zero_grad()  #应该要加这一步
            optimizer_tgt.zero_grad()

            # extract and target features
            feat_tgt = tgt_encoder(images_tgt)

            # predict on discriminator
            pred_tgt = critic(feat_tgt)

            # prepare fake labels
            label_tgt = make_variable(torch.ones(feat_tgt.size(0)).long())  #生成器想愚弄判别器，被判别为1，及最小化和1的距离

            # compute loss for target encoder   生成器损失函数G_loss
            loss_tgt = criterion(pred_tgt, label_tgt)
            loss_tgt.backward()

            # optimize target encoder
            optimizer_tgt.step()

            #######################
            # 2.3 print step info #
            #######################
            # if ((step + 1) % params.log_step == 0):
            #     print("Epoch [{}/{}] Step [{}/{}]:"
            #           "d_loss={:.5f} g_loss={:.5f} acc={:.5f}"
            #           .format(epoch + 1,
            #                   params.num_epochs,
            #                   step + 1,
            #                   len_data_loader,
            #                   loss_critic.item(),
            #                   loss_tgt.item(),
            #                   acc.item()))
        #打印每个epoch的loss
        print("Epoch [{}/{}]: d_loss={:.5f} g_loss={:.5f} acc={:.5f}"
              .format(epoch + 1,
                      params.num_epochs,
                      loss_critic.item(),
                      loss_tgt.item(),
                      acc.item()))


        #############################
        # 2.4 save model parameters #
        #############################
        if ((epoch + 1) % params.save_step == 0):  #每100步保存一次模型
            torch.save(critic.state_dict(), os.path.join(
                params.model_root,
                "ADDA-critic-{}.pt".format(epoch + 1)))
            torch.save(tgt_encoder.state_dict(), os.path.join(
                params.model_root,
                "ADDA-target-encoder-{}.pt".format(epoch + 1)))

    torch.save(critic.state_dict(), os.path.join(
        params.model_root,
        "ADDA-critic-final.pt"))
    torch.save(tgt_encoder.state_dict(), os.path.join(
        params.model_root,
        "ADDA-target-encoder-final.pt"))
    return tgt_encoder   #输出经过对抗生成训练的目标域编码器
