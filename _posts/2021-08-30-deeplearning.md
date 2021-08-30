---
layout: article
title: Pytorch深度学习——网络搭建、模型训练
tags: Python
---

这个教程是总结前一段时间自己学习并且动手深度学习的过程.
主要包括pytorch的安装、网络模型的搭建、以及训练，并且涉及将代码放在GPU上运行.

<!--more-->

# Pytorch的安装

首先进入[Pytorch官网](https://pytorch.org/)，
然后，我们会看到下面的一个选择框，选择适合的命令进行安装.

<p align="center">
    <img src="/post_image/Deep_learning/Pytorch_install.PNG" width="60%">
</p>

如果没有安装cuda可以选择CPU版本，但是运行速度确实很慢.

如果有cuda的情况下，可以安装相应的版本，我在服务器上安装的时候遇到了空间不足的错误. 可以用下面的方式解决.

{% highlight shell linenos %}
$ cd~
$ mkdir tmp # 已经有可以省略
$ export TMPDIR=$HOME/tmp
$ pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
{% endhighlight %}

# 网络模型的搭建

搭建网络，其实就是创建一个你所定义网络的类，这个类继承自 nn.Module，所以事先import torch.nn

{% highlight python linenos %}
import torch
import torch.nn as nn

class MyConvNet(nn.Module):
    def __init__(self):
        super(MyConvNet,self).__init__() #继承类的开头
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3) ,padding=0)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), padding=0) #定义自己的卷积层
        self.fc1 = nn.Linear(21,1024) #全连接层
        self.dropout = nn.Dropout(0.2) #设置dropout比率
        self.fc2 = nn.Linear(1024,256) #注意全连接层之间的输入核上一层的输出保持一致
        self.fc3 = nn.Sequential(nn.Linear(256,2),nn.Sigmoid())
    def forward(self,x,DCGR_feature):
        shape = x.shape
        out = self.conv1(x)
        out = F.relu(out) #激活函数
        out = self.conv2(out)
        out = F.relu(out)
        out = torch.reshape(out,(shape[0],1,21)) #将张量形状改变
        out = self.fc1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out

model = MyConvNet() #申请一个网络的实例
{% endhighlight %}

着重看一下下面代码中的几个函数

{% highlight python linenos %}
nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3) ,padding=0) # 分别表示卷积层的进入通道数目，输出通道数目，以及卷积核的尺寸，padding是为了给图片补值来保证卷积扫描后图片尺寸不变
{% endhighlight %}

所谓的通道数目，在实际意义上，可以代表比如彩色图片RGB三个值，就是三个通道. 然后后面的通道数目，可以理解为用不同的卷积核来提取特征.

{% highlight python linenos %}
nn.Sequential(nn.Linear(256,2),nn.Sigmoid())
{% endhighlight %}


# 模型训练

# 总结

# 在GPU上运行
