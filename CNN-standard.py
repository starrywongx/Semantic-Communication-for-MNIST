import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import os
import time

'''
Created on Wen Nov 30 14:40:09 2022
@author: StarryHuang

详细注释版
语义通信在MNIST手写数据集上的识别-AWGN信道
使用CNN+linear分类，网络结构是semantic encoder-> channel encoder-> channel-> channel decoder ->semantic decoder
将每一张图片压缩到16个特征，SNR=20dB
'''
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
torch.manual_seed(1)  # 使用随机化种子使神经网络的初始化每次都相同

# 超参数
EPOCH = 15  # 训练整批数据的次数
BATCH_SIZE = 64
LR = 0.001  # 学习率
M=4
DOWNLOAD_MNIST = True  # 表示还没有下载数据集，如果数据集下载好了就写False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 下载mnist，【训练集】
train_data = torchvision.datasets.MNIST(
    root='./mnist/',  # 保存或提取的位置  会放在当前文件夹中
    train=True,  # true说明是用于训练的数据，false说明是用于测试的数据
    transform=torchvision.transforms.ToTensor(),  # 转换PIL.Image or numpy.ndarray
    # 此处ToTensor()将shape为(H, W, C)的nump.ndarray或img转为shape为(C, H, W)的tensor，其将每一个数值归一化到[0,1]，其归一化方法比较简单，直接除以255即可
    download=DOWNLOAD_MNIST,
)

## load中加入数据集的归一化
# train_data = torchvision.datasets.MNIST(
#     root='./mnist/',
#     train=True,
#     transform=torchvision.transforms.Compose([
#           torchvision.transforms.ToTensor(),
#           torchvision.transforms.Normalize(mean=(0.1307,), std=(0.3081,))]), # 均值，标准差
#     download=DOWNLOAD_MNIST,
# )

# 加载Mnist训练集, Torch中的DataLoader是用来打乱、分配、预处理
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,  # 是否打乱数据，一般都打乱，，常用于进行多批次的模型训练
    drop_last=False  # 设置为True表示当数据集size不能整除batch_size时，则删除最后一个batch_size，否则就不删除
)

# 下载【测试集】并加载，返回值为一个二元组（data，target）
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=torchvision.transforms.ToTensor(),download=DOWNLOAD_MNIST,)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

# # 训练中进行测试
# test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000] / 255
# # torch.unsqueeze(a) 是用来对数据维度进行扩充，这样shape就从(2000,28,28)->(2000,1,28,28)
# # 图像的pixel本来是0到255之间，除以255对图像进行归一化使取值范围在(0,1)
# test_y = test_data.targets[:2000]

# 用class类来建立CNN模型
class CNN(nn.Module):  # 我们建立的CNN继承nn.Module这个模块
    def __init__(self):
        super(CNN, self).__init__()
        # semantic encoding
        self.SE = nn.Sequential(
            # 卷积con2d参数设置：
            # in_channels=1,  # 输入图片通道数，因为minist数据集是灰度图像只有一个通道
            # out_channels=8, # 通道数，
            # kernel_size=3,  # 卷积核的大小
            # stride=1,  # 步长
            # padding=1,  # 想要con2d输出的图片长宽不变，就进行补零操作 padding = (kernel_size-1)/2
            nn.Conv2d(1, 8, 3, 1, 1),  # 输出图像大小(8,28,28)
            nn.BatchNorm2d(8), # 对所有batch的同一个channel上的数据进行归一化
            nn.ReLU(),# 激活函数，非线性操作
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 池化，[(H+2*p-k)/s +1]向下取整，# 输出图像大小(8,14,14)
            # nn.MaxPool2d(kernel_size=(14,28),stride=(14,14),padding=0),  # 高和宽也可以不同，根据需求来设定
            # nn.AvgPool2d(kernel_size=(5,7), stride=(3, 7), padding=0),  # 14-42 比较一下
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 输出图像大小(8,7,7)
        )

        # channel encoding，M是信道编码后每个像素的symbol数，可变
        self.CE = nn.Sequential(
            nn.Conv2d(8, M, 3, 1, 1),
            nn.BatchNorm2d(M),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=2)  # 最后提取出来的特征比较重要，用Average
        )  # 输入图像大小(M,2,2)，特征一共 M* 2 * 2个

        # 加噪，AWGN==============

        # channel decoding
        self.CD = nn.Sequential(
            nn.Conv2d(M, M, 3, 1, 1),
            nn.BatchNorm2d(M),
            nn.ReLU())

        # semantic decoding，建立全卷积连接层分类
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),  # 有一定概率失活，使模型具有一定的泛化性
            nn.Linear(16, 10),  # 输出是10个类。第一个参数要注意：要和上面的输出个数一致 M * h * w
            nn.Softmax()  # 柔性函数，总概率加起来为1
        )

    # 下面定义x的传播路线
    def forward(self, x):
        x = self.SE(x)
        x = self.CE(x)
        x = AWGN_channel(x, 20)  # SNR可以修改
        x = self.CD(x)
        # print('x:', x[0, 0, :, :].size()) # 可以检查网络的输出当前x的height， weight情况。x是四维[batch，通道，height， weight]

        # 把每一个批次的每一个输入都拉成一个维度
        # 因为pytorch里特征的形式是[bs,channel,h,w]，所以x.size(0)就是batchsize
        x = x.view(x.size(0), -1)  # view就是把x弄成batch行个tensor,一行为一图片,-1表示一个不确定的数,网络自己算数，这里是=channel*H*C
        output = self.classifier(x)
        return output

# add noise
def AWGN_channel(x, snr):  # used to simulate additive white gaussian noise channel
    [batch_size, channel, length, len_feature] = x.shape
    # torch.sum(torch.square(x))
    x_power = torch.sum(torch.square(x)) / (batch_size * length * len_feature * channel)
    n_power = x_power / (10 ** (snr / 10.0))
    # print('n_power',n_power)
    # print('n_power', n_power.numpy()**(0.5))
    noise = torch.normal(mean=0, std=n_power.detach().numpy()**(0.5), size=[batch_size, channel, length, len_feature])
    # print('noise',noise[1:])
    # print('noise', x[1:])
    return x + noise

def train():
    for epoch in range(EPOCH):
        for batch_idx, (inputs, targets) in enumerate(train_loader):  # 分配batch data
            output = cnn(inputs)  # 先将数据放到cnn中计算output
            loss = loss_func(output, targets)  # 输出和真实标签的loss，二者位置不可颠倒
            optimizer.zero_grad()  # 清除之前学到的梯度的参数
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 应用梯度
            if batch_idx % 100 == 0:  # 100个batch后测试一下
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())
                train_losses.append(loss.item())  # loss.item() = loss.data.numpy() 等价
                train_counter.append((batch_idx * 64) + (epoch * len(train_loader.dataset))) # len(train_loader.dataset)=60000

    torch.save(cnn, model_path)  # 保存整个模型
    # torch.save(model.state_dict(), "my_model.pth")  # 只保存模型的参数

def test():
    test_ave_acc = []
    net = torch.load(model_path)
    print('model:',net)
    net.eval()  # 设置模型进入预测模式 evaluation
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):  # 分配batch data
            test_output = net(inputs)
            pred_y = torch.max(test_output, 1)[1]  # torch.max(a,1) 返回每一行中最大值的那个元素，且返回其索引。
            # data.numpy()使得tensor变成numpy形式， targets.size(0)= batchsize
            accuracy = float((pred_y.data.numpy() == targets.data.numpy()).astype(int).sum()) / float(targets.size(0))
            test_ave_acc.append(accuracy)
            # print('test accuracy: %.4f' % accuracy)
    print('Test ave acc:%.2f' % (100. * sum(test_ave_acc)/len(test_ave_acc)))
    # print('len(test_ave_acc):',len(test_ave_acc))



cnn = CNN() # print(cnn) # 可以查看网络状态
print('Total params: %.2fw' % (sum(p.numel() for p in cnn.parameters())/10000.0)) # 参数总数
# 优化器选择Adam，lr要比较小。SDG的lr会比较大。
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
# 损失函数
loss_func = nn.CrossEntropyLoss()  # =softmax得到（Y^）->log（Y^）->交叉熵公式（-Ylog（Y^）），目标标签是one-hotted

train_losses = []
train_counter = []
test_losses = []
test_counter = []
model_path = './model_AWGN_test——standard.pth'


if __name__ == '__main__':
    # from torch.optim import lr_scheduler  # 学习率调整器，在训练过程中合理变动学习率
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

    start = time.time()
    # train
    train()
    print('training has done！')

    # test
    test()
    print("Time: " + str((time.time() - start)/60) + 'mins')
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.xlabel('number of training examples')
    plt.ylabel('loss')
    plt.show()
