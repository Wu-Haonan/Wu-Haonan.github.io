---
layout: article
title: Why and what
tags: Deep-learning
aside:
  toc: true
sidebar:
  nav: Deep_learning
---

# 开场白

从今天开始将跟大家分享一些关于深度学习的内容。其实关于深度学习，网络上有非常多的课程，同时也有很多的书籍供大家参考。咱们数学与统计学院也卧虎藏龙，有很多深度学习领域做的非常优秀的老师和同学。对于我自己而言呢，我其实是个外行，对于深度学习了解的也非常有限，理解的也很浅薄。所以，给大家分享这个深度学习的短课我是诚惶诚恐，不知道能提供给大家什么样的帮助，因为我自己也是一个深度学习的初学者。

介绍一个事物，我比较习惯遵循着为什么，是什么，怎么做来展开。我想了一下，我们这个短课是为了什么呢，想达到什么目的呢。我回忆了一下自己刚接触深度学习时候的感觉，算法大致的思想和原理马马虎虎能够明白，但是怎么拿思想来解决问题呢，怎么动手写代码呢，好像对于新手两眼一抹黑，非常困惑。因此，我想我这个短课更偏重于应用，偏重于实践。我希望大家能够上完这门短课以后，可以简单了解几个深度学习模型，最重要的是能够快速的搭建简单的深度模型，解决自己手头的问题。<!--more-->

在座的各位可能大多都是学数学的，咱们去解决问题的时候，都是奔着效果去的，想要做出一个优秀的算法，希望效果上超越前人。但是，之前我与学计算机的同学以及在互联网企业工作的同学聊天，我发现他们的思维和咱们不太一样，他们有一种工程的思维，他们首先关注怎么去实现。比如，设计一个人脸识别的算法，他们首要关注的在于如何先把这个问题实现，能够先完成一个成品，让你能把人脸输入进去然后返回识别结果，哪怕准确率只有50%，但是没关系，我先有一个端到端的东西出来，然后我再去找更好的方法去优化效果。所以我想，这个课也蕴含了这种思维，让大家不论效果如何，先有个成品，先给问题一个解决的方案，能搭建一个深度学习的模型，至于如果提升效果，那是后话。

因此我给咱们这个短课的题目为“基于Pytorch的神经网络及其应用”，我希望即使是对于深度学习零基础的同学，也能听了短课以后，可以动手去实践。我将会把更多的重点放在代码上，也会给出具体的实践示例，希望喜欢动手的同学，可以每天跟着进度写一写代码。当然，希望同学们可以有一些python的基础，至少了解python中类的概念。相关示例的代码，我也会同步的放在我的[github仓库](https://github.com/Wu-Haonan/Deep_learning_short_course/)。
我的课程讲义，也已经提前发布在了我的[github博客](https://wu-haonan.github.io/2022/01/10/Why_and_what.html)。

# 深度学习与Pytorch

因为这门短课是面向零基础的同学，所以我先花一点时间讲一讲，我认为的学习型的算法大概是怎么回事儿。首先，构建一个深度学习（或者机器学习 ）的算法，我们必须要有数据，数据是由特征和标签组成的。特征就是我们对某个事物刻画，标签就是真实的结果，或者真实的答案。学习型的算法就是希望在大量的数据中，摸索出来特征与标签的一种关系。

比如最经典的一个机器学习问题，鸢尾花的分类。鸢尾花分为三类，山鸢尾、杂色鸢尾、维吉尼亚鸢尾。我们走在学校里面，遇见一株鸢尾花，由于生物知识有限并不知道属于哪一种类别。因此，我们想让计算机来帮我们分类。那么怎么来做呢，我们首先可以采集许许多多的鸢尾花数据样本，对于每一株鸢尾花，我们提取一个特征，比如我们可以测量花萼的长度宽度、花瓣的颜色、株高、叶子的数量等等，它们组成了一个数字向量。同时呢，请一个植物分类学专家，给每一个样本鉴定标签，看看是哪一种鸢尾，可以给三种鸢尾花标记为“0、1、2”。这样我们就获得了一个数据集。

然后我们可以搭建各种各样的算法来解决这个问题，可以是决策树，也可以是支持向量机，也可以神经网络。每一种算法里面都包含了大量参数，我们只有把这些参数确定下来了，才能够最终生成一个完整的模型。就好比最简单的拟合问题，我们可以假设特征向量与标签是线性关系，${ y=ax+b }$，接下来的任务就是想办法求出来${ a,b }$的值。

关于求解的参数的办法，我们从最优化的角度来思考，每当我们给定一组${ a,b }$我们都可以计算一个预测结果${ \hat{y} }$，从而可以计算其与真实值的差距，比如最简单的绝对值${ \lvert y-\hat{y} \rvert }$，所以我们就得到了一种函数关系${ diff(a,b) }$，即给定参数，就能得到预测与真实的差距，我们把这个称为损失函数（loss function）。当然，我们有很多衡量预测值与真实值差距的方法，比如均方根、交叉熵等等，习惯上我们也会把这些函数称为损失函数，我个人认为严格来讲，这些应该称为衡量损失的函数。我们的目的就是优化${ a,b }$使得损失函数的值最小。那么优化的方式有很多，随机梯度下降（SGD），Adam等等。

那么在深度学习的应用中，我们实际的过程是这样的。首先我们将数据分成小组，也就是**batch**，比如${ BATCH=32 }$，这意味着我们每次喂给网络32个样本来优化参数，在第一次的时候，我们将参数随机初始化。然后，将这32个样本扔进现在的模型里面，根据标签计算损失值，这个过程也称为正向传播。依据这个损失值，我们可以求梯度，然后选取一种合适的优化方法（比如随机梯度下降），来优化参数，这个过程也被称为反向传播。如此不断进行，直到所有的样本都被网络学习了一遍。

这样下来，所有的样本被网络见了一遍，但是只优化一次往往很难优化到最优解，或者局部最优解。因此我们设定迭代的次数**epoch**，让网络不断的进行优化，直到损失值收敛，终止优化。在此过程中呢，其实我们还有很多超参数需要去摸索，比如各种优化算法里面的学习率，神经网络里面神经元的个数，网络的层数，Batch、Epoch的数目都是超参数，是需要预先给定的，当然我们也需要不断的调整超参数使得模型的效果最好。

那么，怎么知道我们的模型效果好不好呢，我们手里面训练了一个鸢尾花的分类模型，当然要去校园里找一些新的样本来测试测试啦。所以，我们往往会对数据集进行切分，分为训练集、独立验证集，测试集。训练集很容易理解，就是用来训练参数的，验证集不参与训练，但是我们可以通过其在验证集的效果来选择合适的超参数，一但模型的所有都已经确定，最后放在测试集上测试。网上有个很形象的例子，训练集像课本，教给网络知识，验证集就是作业，实时反馈学的怎么样，然后我们再调整超参数，改变教学策略，最后测试集是期末考试，课本和作业都不能漏题，通过期末考试反应学生的成绩。

最后聊两句Pytorch，这是一个目前比较主流的深度学习框架。我个人认为，用Pytorch做深度学习就是搭积木，里面有各种各样的现成的工具（积木块），无论是损失函数、优化器、线性层、卷积层、池化层、求梯度，都有现成的模块，你可以随心所欲的搭建你想要的东西。按照你的设计，把这些积木组装起来。所以希望大家轻松愉快、像玩游戏一样享受搭建模型的快乐。

# 总结

以上呢，我们就回答了两个问题**"Why"**和**"What"**的问题，即这门课的目的，以及深度学习和pytotch的一个简要介绍。下面，我将介绍一下**"How"**的问题，即课程安排。
