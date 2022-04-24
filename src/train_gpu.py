# -*- coding: utf-8 -*-
import torch
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

# 定义训练设备
device = torch.device("cuda:0")

# 01 获取数据集
train_data = torchvision.datasets.CIFAR10(root="../dataset", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root="../dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)

# 打印长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的的长度：{}".format(train_data_size))
print("测试数据集的的长度：{}".format(test_data_size))

# 02 DataLoader加载数据集
train_data_loader = DataLoader(train_data, batch_size=64)
test_data_loader = DataLoader(test_data, batch_size=64)

# 03 导入网络模型 from model import *
cifar = Cifar()
cifar = cifar.to(device)

# 04 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 05 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(cifar.parameters(), lr=learning_rate)

# 添加tensorboard
writer = SummaryWriter("../logs")

# 设置网络训练的一些参数
total_train_step = 0
total_test_step = 0
epoch = 10

for i in range(epoch):
    print("--------第{}轮训练开始-------".format(i+1))

    # 训练开始
    cifar.train()
    for data in train_data_loader:
        imgs, targets = data
        imgs = imgs.to(device)
        tagets = targets.to(device)
        outputs = cifar(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器模型优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1

        if total_train_step % 200 == 0:
            print("训练次数：{}, Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试开始
    cifar.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_data:
            imgs, targets = data
            imgs = imgs.to(device)
            tagets = targets.to(device)
            outputs = cifar(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_step + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy / test_data_size))
    writer .add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)

    total_test_step = total_test_step + 1

    torch.save(cifar, "cifar_{}.pth".format(i+1))
    print("模型已保存")

writer.close()


