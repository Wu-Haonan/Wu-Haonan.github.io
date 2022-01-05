---
layout: article
title: MLP和CNN实践
aside:
  toc: true
sidebar:
  nav: Deep_learning
---

# Pytorch的安装

Pytorch的安装比较简单，我们可以看到下图的选择栏目，根据你的需要，下面会自动生成合适的命令，把命令copy到command line里，就可以完成安装。

<p align="center">
    <img src="/post_image/Deep_learning/Pytorch_install.PNG" width="60%">
</p>

# 深度学习流程

我先简单的介绍一下深度学习的整个代码框架，我做了下面的一个流程图。大致分为这些模块和步骤。

<p align="center">
    <img src="/post_image/Deep_learning/work_flow.png" width="60%">
</p>

原始的输入数据部分这个比较容易理解，我们一般输入的结构都是特征和标签两部分，当然这个也不一定，依据你们自己设计的模型和框架进行调整。

网络模型部分呢，其实本质上是一个类，从**torch.nn.Module**继承而来。然后我们从torch里面选择合适的函数，来搭建我们自己的网络。所以我们每次使用的时候，就是初始化这个类的一个实例。

然后我们进入到这个代码的主要部分，训练部分，首先我们需要对数据进行处理，例如随机切分训练集、验证集，将数据分成batch。然后，我们进行一些准备工作，例如选择一个合适的优化器，初始化模型参数，定义损失函数等等。

下面就来到了关键部分，我们需要对网络训练若干次，每一个epoch都要经历两个过程，训练过程、验证过程。对于训练过程，就是正向传播，计算损失函数，然后反向传播。验证过程，只需要正向传播，计算损失，不需要反向传播，但是可以根据自己的需要加入一些评价的过程，比如计算准确率，召回率等等。


{% highlight python linenos %}

{% endhighlight %}