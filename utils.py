"""Utilities for ADDA."""

import os
import random
import pandas as pd
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as Data

import params
# from datasets import get_mnist, get_usps



def make_variable(tensor, volatile=False):  #volatile表示是否需要反向传播，为True表示不需要,适用于推断阶段，不需要反向传播。
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()     #利用GPU计算
    return Variable(tensor, volatile=volatile)


def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def denormalize(x, std, mean):
    """Invert normalization, and then convert array into image."""
    out = x * std + mean
    return out.clamp(0, 1)


def init_weights(layer):
    """Init weights for layers w.r.t. the original paper."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)


def init_random_seed(manual_seed):
    """Init random seed."""
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)   #如果初始值为None，就指定为[1,1000]内的随机数
        # seed = 2020
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)    ##为CPU设置种子用于生成随机数，以使得结果是确定的
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  #如果使用多个GPU，为所有的GPU设置种子
                                          # 用torch.cuda.manual_seed(seed)#为当前GPU设置随机种子


# def get_data_loader(name, train=True):
#     """Get data loader by name."""
    # if name == "MNIST":
    #     return get_mnist(train)
    # elif name == "USPS":
    #     return get_usps(train)


def get_data_loader():
    source_x = torch.from_numpy(np.array(
        (pd.read_csv(r'D:/个人资料/研究生/机器学习/Jupyter/迁移学习练习赛-2/run/data/source_x.csv').iloc[:, 1:]).astype('float32')))
    source_y = torch.from_numpy(np.array(
        (pd.read_csv(r'D:/个人资料/研究生/机器学习/Jupyter/迁移学习练习赛-2/run/data/source_y.csv').iloc[:, 1:]).astype('float32')))
    target_x = torch.from_numpy(np.array(
        (pd.read_csv(r'D:/个人资料/研究生/机器学习/Jupyter/迁移学习练习赛-2/run/data/target_x.csv').iloc[:, 1:]).astype('float32')))
    target_y = torch.from_numpy(np.array(
        (pd.read_csv(r'D:/个人资料/研究生/机器学习/Jupyter/迁移学习练习赛-2/run/data/target_y.csv').iloc[:, 1:]).astype('float32')))
    # test = pd.read_csv(r'data/test.csv')
    #扩大数据维度，原来的【314*9】转变为【314*1*9】
    #x扩充至三维
    # source_x = torch.unsqueeze(source_x, 1)
    source_x = torch.unsqueeze(source_x, 1)
    # target_x = torch.unsqueeze(target_x, 1)
    target_x = torch.unsqueeze(target_x, 1)
    # #y减少至一维
    # source_y = torch.squeeze(source_y, 1)
    # target_y = torch.squeeze(target_y, 1)

    #运用批训练，装进loder里
    source_data = Data.TensorDataset(source_x, source_y)
    target_data = Data.TensorDataset(target_x, target_y)

    source_data_loader = Data.DataLoader(
        dataset=source_data,
        batch_size=params.batch_size,
        shuffle=True
    )
    target_data_loader = Data.DataLoader(
        dataset=target_data,
        batch_size=params.batch_size,
        shuffle=True
    )
    return (source_data_loader,target_data_loader)


def init_model(net, restore):
    """Init models with cuda and weights."""
    # init weights of model初始化权值
    net.apply(init_weights)

    # restore model weights
    if restore is not None and os.path.exists(restore):  #如果已经存在模型，就加载
        net.load_state_dict(torch.load(restore))
        net.restored = True
        print("Restore model from: {}".format(os.path.abspath(restore)))

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()

    return net


def save_model(net, filename):
    """Save trained model."""
    if not os.path.exists(params.model_root):
        os.makedirs(params.model_root)
    torch.save(net.state_dict(),
               os.path.join(params.model_root, filename))
    print("save pretrained model to: {}".format(os.path.join(params.model_root,filename)))
