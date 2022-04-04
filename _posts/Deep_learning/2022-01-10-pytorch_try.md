---
layout: article
title: Let's try
tags: Deep learning
aside:
  toc: true
sidebar:
  nav: Deep_learning
---

下面我们一起做一个MLP练习，数据和代码我放在了[仓库](https://github.com/Wu-Haonan/Deep_learning_short_course/tree/main/Drug_Toxicity)中。<!--more-->

在文件夹./Drug_Toxicity/下面，./Drug_Toxicity/Data_file/文件夹中存放了药物的特征（训练集文件train_feature.pkl，测试集文件test_feature.pkl）和相应的hERG(心脏安全性评价的二分类标签)（文件hERG.pkl）[^1]

具体而言，train_feature.pkl文件存放了1974个药物样本，每个样本由729个特征构成，hERG.pkl文件存放了每个药物样本的二分类标签。我们需要用这两个数据进行训练。然后，test_feature.pkl文件存放了测试集的样本，我们需要用生成好的模型来预测其标签。

下面我带着大家用之前学过的内容，串联起来，完成这个任务。

读者可以运行hERG_train.py文件和hERG_test.py文件对数据进行训练和预测，相应的结果以及模型保存在./Drug_Toxicity/hERG/文件夹中

[^1]:[数据来源于华为杯2021数学建模D题](https://cpipc.acge.org.cn//cw/detail/4/2c9080147c73b890017c7779e57e07d2)
