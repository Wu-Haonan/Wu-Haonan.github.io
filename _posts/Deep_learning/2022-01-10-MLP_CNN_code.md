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
    <img src="/post_image/Deep_learning/Pytorch_install.PNG" width="80%">
</p>

可以在python shell里检查一下

{% highlight Python linenos %}
>>> import torch
>>> torch.__version__
'1.9.0+cpu'
{% endhighlight %}

# 深度学习流程

我先简单的介绍一下深度学习的整个代码框架，我做了下面的一个流程图。大致分为这些模块和步骤。

<p align="center">
    <img src="/post_image/Deep_learning/work_flow.png" width="80%">
</p>

原始的输入数据部分这个比较容易理解，我们一般输入的结构都是特征和标签两部分，当然这个也不一定，依据你们自己设计的模型和框架进行调整。

网络模型部分呢，其实本质上是一个类，从**torch.nn.Module**继承而来。然后我们从torch里面选择合适的函数，来搭建我们自己的网络。所以我们每次使用的时候，就是初始化这个类的一个实例。

然后我们进入到这个代码的主要部分，训练部分，首先我们需要对数据进行处理，例如随机切分训练集、验证集，将数据分成batch。然后，我们进行一些准备工作，例如选择一个合适的优化器，初始化模型参数，定义损失函数等等。

下面就来到了关键部分，我们需要对网络训练若干次，每一个epoch都要经历两个过程，训练过程、验证过程。对于训练过程，就是正向传播，计算损失函数，然后反向传播。验证过程，只需要正向传播，计算损失，不需要反向传播，但是可以根据自己的需要加入一些评价的过程，比如计算准确率，召回率等等。

# 数据处理

## Tensor

在数学上，一维的数组是向量，二维的数组是矩阵，更高维度的数组就是张量。在深度学习里面我们的数据往往维度很高，举个例子，比如一组图片的数据，在网络学习的时候往往有四个维度${ (B,C,L,W) }$，${ B }$是batch的维度，${ C }$是channel的维度，${ (L,W) }$是图片的长宽。

在pytorch里面，各种计算和优化都是基于Tensor这个数据类型的。但是Tensor的各种操作特别繁多，也不复杂，更重要的是与Python常用的库Numpy的数组操作基本类似，所以这里只介绍简单的一些操作。遇到特殊的情况，可以去torch的官网，寻找你需要的计算函数。还有一个需要注意的地方，在Pytorch的0.4版本之前，Tensor不能直接计算梯度，需要先用Variable类处理一下，才能扔进网络里，但是后面Tensor和Variable合并了，所以这个操作做不做都一样，但是有很多代码还保留了这样的写法，或者仍然习惯这么去写。

### 1.from List

从Python的list生成，注意如果需要求梯度，数据类型一定需要浮点数才可以计算。

{% highlight python linenos %}
>>> torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])
tensor([[ 0.1000,  1.2000],
        [ 2.2000,  3.1000],
        [ 4.9000,  5.2000]])
{% endhighlight %}

反过来和Numpy一样，我们也可以将tensor转化回list

{% highlight python linenos %}
a = torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])
a = a.tolist()
{% endhighlight %}

### 2.from Numpy

从Numpy的数组生成tensor

{% highlight python linenos %}
>>> a = numpy.array([1.,2.,2.])
>>> torch.from_numpy(a)
tensor([1., 2., 2.], dtype=torch.float64)
{% endhighlight %}

同样，我们也可以将tensor转成numpy

{% highlight python linenos %}
>>> a = torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])
>>> a.numpy()
array([[0.1, 1.2],
       [2.2, 3.1],
       [4.9, 5.2]], dtype=float32)
{% endhighlight %}

### 3.reshaping

因为很多Pytorch的函数输入的尺寸要求不同，所以我们往往需要给tensor数据整形，比如你的数据是${ (B,L,W) }$，如果想要输入卷积层，就需要添加一个channel的维度，变成${ (B,1,L,W) }$，或者有时候我们想把矩阵转化成一个向量，扔进全连接层都需要整形。我们下面介绍几个整形的函数

#### view()/reshape()

这两个函数差别不大，把你想整形成的维度输入放在括号里面就可以了

{% highlight python linenos %}
>>> X = torch.tensor([[1,2,3,4],[5,6,7,8],[1,2,5,6]])
>>> X.view(1,12)
tensor([[1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 5, 6]])
>>> X = torch.tensor([[1,2,3,4],[5,6,7,8],[1,2,5,6]])
>>> X.view(3,1,4)
tensor([[[1, 2, 3, 4]],

        [[5, 6, 7, 8]],

        [[1, 2, 5, 6]]])
>>> X.reshape(1,3,4)
tensor([[[1, 2, 3, 4],
         [5, 6, 7, 8],
         [1, 2, 5, 6]]])
>>> X.reshape(2,6)
tensor([[1, 2, 3, 4, 5, 6],
        [7, 8, 1, 2, 5, 6]])
{% endhighlight %}

#### squeeze()

比如我们在处理数据的时候，很有可能会出现有些维度是${ 1 }$，比如图片数据的Channel维度，我们想把这些空维度给压缩掉，就可以简单的使用squeeze()

{% highlight python linenos %}
>>> a = torch.tensor([[[1, 2, 3, 4]],
...
...         [[5, 6, 7, 8]],
...
...         [[1, 2, 5, 6]]])
>>> a.shape
torch.Size([3, 1, 4])
>>> a.squeeze()
tensor([[1, 2, 3, 4],
        [5, 6, 7, 8],
        [1, 2, 5, 6]])
>>> a = a.squeeze()
>>> a.shape
torch.Size([3, 4])
{% endhighlight %}

#### flatten()

flatten()比较容易理解，就是将张量直接拉平。

{% highlight python linenos %}
>>> a = torch.tensor([[[1, 2, 3, 4],
...          [5, 6, 7, 8],
...          [1, 2, 5, 6]]])
>>> a.flatten()
tensor([1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 5, 6])
{% endhighlight %}

## 数据切分、加载

# 网络搭建

## MLP网络

## CNN网络

# 网络训练

## 初始准备

## train epoch

## valid epoch

## run on GPU


{% highlight python linenos %}
{% endhighlight %}

{% highlight python linenos %}
{% endhighlight %}

{% highlight python linenos %}
{% endhighlight %}