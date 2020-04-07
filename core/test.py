"""Test script to classify target data."""

import torch
import torch.nn as nn

from utils import make_variable
import params


def eval_tgt(encoder, classifier, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder.eval()
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

    # set loss function
    # criterion = nn.CrossEntropyLoss()
    #my
    criterion = nn.MSELoss()

    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)  #为True表示不需要反向传播
        labels = make_variable(labels)  #.squeeze_()
        # print('标签：',labels)

        preds = classifier(encoder(images))

        # print('预测值是：',preds)
        # loss += criterion(preds, labels).item()
        # print('loss:',loss)
        loss1 += criterion(preds[:,0], labels[:,0]).item()
        loss2 += criterion(preds[:,1], labels[:,1]).item()
        loss3 += criterion(preds[:,2], labels[:,2]).item()


        # pred_cls = preds.data.max(1)[1]
        # acc += pred_cls.eq(labels.data).cpu().sum()
        # acc += ((preds - labels) ** 2).cpu().sum()
        # print('acc:',acc)
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


    # print("Avg Loss = {}, Avg Accuracy = {}".format(loss, acc**0.5))
    print('Avg loss1: {}, Avg loss2: {}, Avg loss3: {}'.format(loss1,loss2,loss3))
    print('Avg Acc1: {}, Avg Acc2: {}, Avg Acc3: {}'.format(acc1, acc2, acc3))
