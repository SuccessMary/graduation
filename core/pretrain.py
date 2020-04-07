"""Pre-train encoder and classifier for source dataset."""

import torch.nn as nn
import torch.optim as optim

import params
from utils import make_variable, save_model

#对于源域进行空间映射和分类器训练
def train_src(encoder, classifier, data_loader):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    encoder.train()  #train训练模式启用Dropout和BatchNormalization
    classifier.train()

    # setup criterion and optimizer
    # optimizer = optim.Adam(
    #     list(encoder.parameters()) + list(classifier.parameters()),  #可能不能一起训练
    #     lr=params.c_learning_rate,
    #     betas=(params.beta1, params.beta2))
    optimizer = optim.RMSprop(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=params.c_learning_rate,
        alpha=0.9)
    # criterion = nn.CrossEntropyLoss()
    #my
    criterion = nn.MSELoss()

    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs_pre):    #100
        for step, (images, labels) in enumerate(data_loader):
            # make images and labels variable 将图片变成数据
            images = make_variable(images)
            labels = make_variable(labels)  #.squeeze_() labels不需要squeeze（）

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for critic
            preds = classifier(encoder(images))
            # loss = criterion(preds, labels)

            loss1 = criterion(preds[:,0], labels[:,0])
            loss2 = criterion(preds[:,1], labels[:,1])
            loss3 = criterion(preds[:,2], labels[:,2])
            loss = loss1 + loss2 + loss3


            # optimize source classifier
            loss.backward()
            optimizer.step()

            # print step info
            if ((step + 1) % params.log_step_pre == 0):
                print("Epoch [{}/{}] Step [{}/{}]: loss={}"
                      .format(epoch + 1,
                              params.num_epochs_pre,
                              step + 1,
                              len(data_loader),
                              loss.data[0]))

        # eval model on test set
        if ((epoch + 1) % params.eval_step_pre == 0):  #每20个epoch评价一次
            print('[epoch:{}/{}]'.foramt(epoch,params.num_epochs_pre), eval_src(encoder, classifier, data_loader))

        # save model parameters
        if ((epoch + 1) % params.save_step_pre == 0):  #每100个epoch保存一次
            save_model(encoder, "ADDA-source-encoder-{}.pt".format(epoch + 1))
            save_model(classifier, "ADDA-source-classifier-{}.pt".format(epoch + 1))

    # # save final model
    save_model(encoder, "ADDA-source-encoder-final.pt")
    save_model(classifier, "ADDA-source-classifier-final.pt")

    return encoder, classifier


def eval_src(encoder, classifier, data_loader):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    encoder.eval()  #eval验证模式不启用Dropout和BatchNormalization
    classifier.eval()

    # init loss and accuracy
    # loss = 0
    # acc = 0
    loss1 = 0
    loss2 = 0
    loss3 = 0
    acc1 = 0
    acc2 = 0
    acc3 = 0


    #my
    criterion = nn.MSELoss()

    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels)
        # print('标签：',labels)
        # print('标签：',labels.shape)

        preds = classifier(encoder(images))  #.squeeze()
        # print('预测值是：',preds.shape)
        # print('预测值是：',preds)
        # loss += criterion(preds, labels).item()  #data[0]6

        loss1 += criterion(preds[:,0], labels[:,0]).item()
        loss2 += criterion(preds[:,1], labels[:,1]).item()
        loss3 += criterion(preds[:,2], labels[:,2]).item()

        # pred_cls = preds.data.max(1)[1] #返回每一行最大值所在的索引(我的不需要，因为分类器（即我的回归器)直接输出一个结果)
        # acc += pred_cls.eq(labels.data).cpu().sum()
        # acc += ((preds - labels) ** 2).cpu().sum()
        acc1 += ((preds[:,0] - labels[:,0]) ** 2).cpu().sum()
        acc2 += ((preds[:,1] - labels[:,1]) ** 2).cpu().sum()
        acc3 += ((preds[:,2] - labels[:,2]) ** 2).cpu().sum()

    # loss /= len(data_loader)
    # acc /= len(data_loader.dataset)
    loss1 /= len(data_loader)
    loss2 /= len(data_loader)
    loss3 /= len(data_loader)
    acc1 /= len(data_loader.dataset)
    acc2 /= len(data_loader.dataset)
    acc3 /= len(data_loader.dataset)

    # print("Avg Loss = {}, Avg Accuracy = {}".format(loss, acc))
    print('Avg loss1: {}, Avg loss2: {}, Avg loss3: {}'.format(loss1,loss2,loss3))
    print('Avg Acc1: {}, Avg Acc2: {}, Avg Acc3: {}'.format(acc1, acc2, acc3))
