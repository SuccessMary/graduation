import pandas as pd
class scale_related():
    def __init__(self):
        self.stats = []
        self.input = None

    def norm(self, x):
        self.input = x
        self.stats = self.input.describe().transpose()
        output = (self.input - self.stats['mean']) / (self.stats['max'] - self.stats['min'])
        return output


# def inverse_norm(self,x):
#         print(self.stats)
#         self.input = x
#         out_put = x * (self.stats['max'] - self.stats['min']) + self.stats['mean']
#         return out_put
#     def Normalize(self,y):
#         self.norm_y[0,0] = np.average(y)
#         self.norm_y[0,1] = np.max(y) - np.min(y)
#         return (y - self.norm[0,0]) / self.norm_y[0,1]

import pandas as pd
import numpy as np
import torch
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#jds
def get_target_data_loader():
    x = pd.read_excel('F:/毕业实验/a3/modal/a3_feature.xlsx', header=0).astype('float32')
    x = scale_related().norm(x)
    y = pd.read_excel(r'F:/毕业实验/a3/modal/a3_modal.xlsx', header=0).iloc[:, 1:4].astype('float32')
    # y = scale_related().norm(y)
    trn_x, val_x, trn_y, val_y = train_test_split(x, y, test_size=0.2)
    trn_x = torch.from_numpy(np.array(trn_x))
    trn_y = torch.from_numpy(np.array(trn_y))
    val_x = torch.from_numpy(np.array(val_x))
    val_y = torch.from_numpy(np.array(val_y))

    # x扩充至三维,原来的【314*9】转变为【314*1*9】
    trn_x = torch.unsqueeze(trn_x, 1)
    val_x = torch.unsqueeze(val_x, 1)
    print(trn_x.shape, trn_y.shape, val_x.shape, val_y.shape)
    # #y减少至一维
    # source_y = torch.squeeze(source_y, 1)
    # target_y = torch.squeeze(target_y, 1)

    # 运用批训练，装进loder里
    trn_data = Data.TensorDataset(trn_x, trn_y)
    val_data = Data.TensorDataset(val_x, val_y)

    trn_data_loder = Data.DataLoader(dataset=trn_data,
                                     batch_size=params.batch_size,
                                     shuffle=True
                                     )
    val_data_loder = Data.DataLoader(dataset=val_data,
                                     batch_size=params.batch_size,
                                     shuffle=True
                                     )
    return (trn_data_loder, val_data_loder)


def get_source_data_loader():
    #     x = torch.from_numpy(StandardScaler().fit_transform(np.random.rand(200,9))).float()

    x = pd.read_excel('F:/毕业实验/a1/processed modal/a1_feature.xlsx', header=0).astype('float32')
    x = scale_related().norm(x)
    x = torch.from_numpy(np.array(x))
    x = torch.unsqueeze(x, 1)

    y = pd.read_excel('F:/毕业实验/a1/processed modal/a1_modal_15-23-2.xlsx', header=0).iloc[:, 1:].astype('float32')
    # y = scale_related().norm(y)
    y = torch.from_numpy(np.array(y))

    trn_x, val_x, trn_y, val_y = train_test_split(x, y, test_size=0.2)
    print(trn_x.shape, trn_x.shape, val_x.shape, val_y.shape)

    trn_data = Data.TensorDataset(trn_x, trn_y)
    trn_data_loder = Data.DataLoader(dataset=trn_data,
                                     batch_size=50,
                                     shuffle=True
                                     )
    val_data = Data.TensorDataset(val_x, val_y)
    val_data_loder = Data.DataLoader(dataset=val_data,
                                     batch_size=50,
                                     shuffle=True
                                     )

    return (trn_data_loder, val_data_loder)


# 主程序

import params
from core import eval_src, eval_tgt, train_src, train_tgt
from models import Discriminator, LeNetRegressor, LeNetEncoder
from utils import get_data_loader, init_model, init_random_seed

# 先加载数据

# init random seed
init_random_seed(params.manual_seed)

# load dataset加载数据
# src_data_loader = get_data_loader(params.src_dataset)
# src_data_loader_eval = get_data_loader(params.src_dataset, train=False)
# tgt_data_loader = get_data_loader(params.tgt_dataset)
# tgt_data_loader_eval = get_data_loader(params.tgt_dataset, train=False)
src_data_loader, src_data_loader_eval = get_source_data_loader()
tgt_data_loader, tgt_data_loader_eval = get_target_data_loader()

print(len(tgt_data_loader), len(tgt_data_loader.dataset))

# 加载各模型

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

# 训练源域回归器和源域映射器（也称编码器）

# train source model
print("=== Training classifier for source domain ===")
print(">>> Source Encoder <<<")
print(src_encoder)
print(">>> Source Classifier <<<")
print(src_classifier)

if not (src_encoder.restored and src_classifier.restored and
            params.src_model_trained):  # 如果都没有存储，意为都没有开始训练，所以先训练源域的分类器和编码器
    src_encoder, src_classifier = train_src(
        src_encoder, src_classifier, src_data_loader)

# eval source model 对源域的两个模型做测评
print("=== Evaluating classifier for source domain ===")
eval_src(src_encoder, src_classifier, src_data_loader_eval)

# 进行对抗生成部分的训练
# 输出的是经过GAN思想训练后的目标域编码器

# train target encoder by GAN
print("=== Training encoder for target domain ===")
print(">>> Target Encoder <<<")
print(tgt_encoder)
print(">>> Critic <<<")
print(critic)

# init weights of target encoder with those of source encoder 源域编码器和目标域编码器共用参数
# 如果目标域分类器没训练，就加载已训练的源域分类器的参数到未训练的目标域分类器
if not tgt_encoder.restored:
    tgt_encoder.load_state_dict(src_encoder.state_dict())

if not (tgt_encoder.restored and critic.restored and  # 如果目标域编码器、域分类器都没训练，就开始训练
            params.tgt_model_trained):
    tgt_encoder = train_tgt(src_encoder, tgt_encoder, critic,
                            src_data_loader, tgt_data_loader)

# 测试经过GAN思想训练后的目标域编码器对目标域数据的效果

# eval target encoder on test set of target dataset
print("=== Evaluating classifier for encoded target domain ===")
print(">>> source encoder only <<<")
eval_tgt(src_encoder, src_classifier, tgt_data_loader_eval)
# print(">>> domain adaption with train data of target area<<<")
# eval_tgt(tgt_encoder, src_classifier, tgt_data_loader)
print(">>> domain adaption with valid data of target area<<<")
eval_tgt(tgt_encoder, src_classifier, tgt_data_loader_eval)


from utils import make_variable
print(len(tgt_data_loader_eval.dataset))
predict = []
for (x,y) in tgt_data_loader_eval:
    x = make_variable(x, volatile=True)  # 为True表示不需要反向传播
    a = src_classifier(tgt_encoder(x)).cuda().data.cpu().numpy()
#     print(a)
    b = y.numpy()
#     print(b)
    predict.extend(np.concatenate([np.array(a),np.array(b)],axis=1))
pd.DataFrame(predict).to_csv('aha2.csv',index=False,header=None)