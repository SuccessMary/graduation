#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# """Params for ADDA."""

# # params for dataset and data loader
# data_root = "data"
# dataset_mean_value = 0.5
# dataset_std_value = 0.5
# dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
# dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)
# batch_size = 50
# image_size = 64

# # params for source dataset 源域数据
# src_dataset = "MNIST"
# src_encoder_restore = "snapshots/ADDA-source-encoder-final.pt"
# src_classifier_restore = "snapshots/ADDA-source-classifier-final.pt"
# src_model_trained = True

# # params for target dataset 目标域数据
# tgt_dataset = "USPS"
# tgt_encoder_restore = "snapshots/ADDA-target-encoder-final.pt"
# tgt_model_trained = True

# # params for setting up models
# model_root = "snapshots"
# d_input_dims = 500
# d_hidden_dims = 500
# d_output_dims = 2
# d_model_restore = "snapshots/ADDA-critic-final.pt"

# # params for training network
# num_gpu = 1
# num_epochs_pre = 100
# log_step_pre = 20
# eval_step_pre = 20
# save_step_pre = 100
# num_epochs = 2000
# log_step = 100
# save_step = 100
# manual_seed = None

# # params for optimizing models
# d_learning_rate = 1e-4
# c_learning_rate = 1e-4
# beta1 = 0.5
# beta2 = 0.9


# 主程序

# In[1]:


import params
from core import eval_src, eval_tgt, train_src, train_tgt
from models import Discriminator, LeNetRegressor, LeNetEncoder
from utils import get_data_loader, init_model, init_random_seed


# 先加载数据

# In[2]:


# init random seed
init_random_seed(params.manual_seed)

# load dataset加载数据
# src_data_loader = get_data_loader(params.src_dataset)
# src_data_loader_eval = get_data_loader(params.src_dataset, train=False)
# tgt_data_loader = get_data_loader(params.tgt_dataset)
# tgt_data_loader_eval = get_data_loader(params.tgt_dataset, train=False)
src_data_loader, tgt_data_loader = get_data_loader()
src_data_loader_eval, tgt_data_loader_eval = get_data_loader()


# 加载各模型

# In[3]:


# load models
src_encoder = init_model(net=LeNetEncoder(),
                             restore=params.src_encoder_restore)
src_classifier = init_model(net=LeNetRegressor(),
                                restore=params.src_classifier_restore)
tgt_encoder = init_model(net=LeNetEncoder(),
                             restore=params.tgt_encoder_restore)
critic = init_model(Discriminator(input_dims=params.d_input_dims,
                                      hidden_dims=params.d_hidden_dims,
                                      output_dims=params.d_output_dims),
                        restore=params.d_model_restore)


# 训练源域回归器

# In[4]:


# train source model
print("=== Training classifier for source domain ===")
print(">>> Source Encoder <<<")
print(src_encoder)
print(">>> Source Classifier <<<")
print(src_classifier)

if not (src_encoder.restored and src_classifier.restored and
            params.src_model_trained):  #如果都没有存储，意为都没有开始训练，所以先训练源域的分类器和编码器
    src_encoder, src_classifier = train_src(
            src_encoder, src_classifier, src_data_loader)

# eval source model 对源域的两个模型做测评
print("=== Evaluating classifier for source domain ===")
eval_src(src_encoder, src_classifier, src_data_loader_eval)


# 进行对抗生成部分的训练
# 
# 输出的是经过GAN思想训练后的目标域编码器

# In[ ]:


# train target encoder by GAN
print("=== Training encoder for target domain ===")
print(">>> Target Encoder <<<")
print(tgt_encoder)
print(">>> Critic <<<")
print(critic)


# In[ ]:


# init weights of target encoder with those of source encoder 共用参数
#如果目标域分类器没训练，就加载已训练的源域分类器的参数到未训练的目标域分类器
if not tgt_encoder.restored:   
    tgt_encoder.load_state_dict(src_encoder.state_dict())

if not (tgt_encoder.restored and critic.restored and   #如果目标域编码器、域分类器都没训练，就开始训练
            params.tgt_model_trained):
    tgt_encoder = train_tgt(src_encoder, tgt_encoder, critic,
                                src_data_loader, tgt_data_loader)


# 测试经过GAN思想训练后的目标域编码器对目标域数据的效果

# In[ ]:


# eval target encoder on test set of target dataset
print("=== Evaluating classifier for encoded target domain ===")
print(">>> source data only <<<")
eval_tgt(src_encoder, src_classifier, tgt_data_loader_eval)
print(">>> domain adaption <<<")
eval_tgt(tgt_encoder, src_classifier, tgt_data_loader_eval)


# In[40]:


import torch

test_data = torch.FloatTensor(2,3)
# 保存数据
torch.save(test_data, "test_data.pkl")

print(test_data)
# 提取数据
print(torch.load("test_data.pkl"))


# In[ ]:




