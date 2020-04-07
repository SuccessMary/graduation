"""LeNet model for ADDA."""

import torch.nn.functional as F
from torch import nn


#编码器：映射到相同空间
# class LeNetEncoder(nn.Module):
#     """LeNet encoder model for ADDA."""
#
#     def __init__(self):
#         """Init LeNet encoder."""
#         super(LeNetEncoder, self).__init__()
#
#         self.restored = False
#
#         self.encoder = nn.Sequential(
#             # 1st conv layer
#             # input [1 x 28 x 28]
#             # output [20 x 12 x 12]
#
#             #my input[1*1*n]
#             #my output[20*1/2*n/2]
#
#             nn.Conv2d(1, 20, kernel_size=5),
#             nn.MaxPool2d(kernel_size=2),
#             nn.ReLU(),
#             # 2nd conv layer
#             # input [20 x 12 x 12]
#             # output [50 x 4 x 4]
#
#             nn.Conv2d(20, 50, kernel_size=5),
#             nn.Dropout2d(),
#             nn.MaxPool2d(kernel_size=2),
#             nn.ReLU()
#         )
#             #经过全连接层（即分类层）转换后，encoder编码输出大小为500(500是人为设定的)
#         self.fc1 = nn.Linear(50 * 4 * 4, 500)


    # def forward(self, input):
    #     """Forward the LeNet."""
    #     conv_out = self.encoder(input)
    #     feat = self.fc1(conv_out.view(-1, 50 * 4 * 4))  #经过卷积层后，先把conv_out拉平，再输入全连接层
    #     return feat

#我的编码器
class LeNetEncoder(nn.Module):
    def __init__(self):
        super(LeNetEncoder, self).__init__()
        self.restored = False
        self.encoder = nn.Sequential(
            #input[B, 1, 12]
            #output[B, 20, 5]
            # 第一层1维卷积，卷积核尺寸为3*3，步长为1
            nn.Conv1d(1, 20, kernel_size=3),
            # 池化层，卷积核尺寸为2*2，步长为2（1可否？）
            # nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(20),
            nn.LeakyReLU(),

            # 第二层1维卷积和池化，同上
            #input[B, 20, 5]
            #output[B, 50, 1]
            nn.Conv1d(20, 50, kernel_size=3),
            nn.BatchNorm1d(50),
            # nn.Dropout(),
            # nn.MaxPool1d(kernel_size=2, stride=2),
            nn.LeakyReLU(),

            #第三层卷积
            nn.Conv1d(50,100,3),
            nn.BatchNorm1d(100),
            nn.LeakyReLU()
            )
        self.fc1 = nn.Linear(100 * 2,100)

    def forward(self, input):
        conv_out = self.encoder(input)
        # print(conv_out.shape)
        # 进全链接层之前，要先拉平(相当于tensorflow里的flatten)，第一维是batchsize
        feat = self.fc1(conv_out.view(-1,100 * 2))
        return feat


#分类器：映射后进行分类
# class LeNetClassifier(nn.Module):
#     """LeNet classifier model for ADDA."""
#
#     def __init__(self):
#         """Init LeNet encoder."""
#         super(LeNetClassifier, self).__init__()
#         self.fc2 = nn.Linear(500, 10)
#
#     def forward(self, feat):
#         """Forward the LeNet classifier."""
#         out = F.dropout(F.relu(feat), training=self.training)
#         out = self.fc2(out)
#         return out


#我的回归器
class LeNetRegressor(nn.Module):
    """LeNet classifier model for ADDA."""
    def __init__(self):
        super(LeNetRegressor, self).__init__()
        self.fc2 = nn.Linear(100,3)   #输出格式大小为3

    def forward(self, feat):
        # out = F.dropout(F.relu(feat), training=self.training)
        out = F.relu(feat)
        out = self.fc2(out)
        # out2 = self.fc2(out)
        # out3 = self.fc2(out)
        return out