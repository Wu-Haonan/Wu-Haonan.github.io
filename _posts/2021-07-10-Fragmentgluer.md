---
layout: article
title: 文献分享：De Novo Repeat Classification and Fragment Assembly
tags: papers

article_header:
  type: cover
  image: 
    src: /post_image/Fragmentgluer/Fragmentgluer.PNG
---

我们知道基因组拼接中，一个难点就是在Assemble graph中repeats区域的路径选择问题。当然了，这个问题的解决是非常复杂的。这篇文章其实非常早了，是2004年非常著名的Pavel（这个人全名叫Pavel A. Pevzner，在重新整理这篇文章的时候我发现GR网站上里面名字居然是Paul A. Pevzner，而在Pubmed上是我们熟悉的名字，可能是因为这个人是俄罗斯人，后来自己更换了英文对应的音译）发表在Genome Research上的文章。

为什么分享这篇古老的文章呢，因为这篇文章是19年发表的Flye中repeat graph的思想来源，能够解释Flye里面很多处理方法的原理。简单说一下这篇文章的内容，这篇文章分为两个部分，第一个部分是解决了所谓的repeat representation问题（repeat classification），第二部分，借由上面的思想提出了一个组装基因组的方法FragmentGluer。
<!--more-->

# Introduction
我们先来介绍一下repeat的结构，可能我们认为repeat不就是基因组上重复出现的片段嘛，这有什么结构呢？其实不然，在这篇文章中，repeat其实代表了一个区域，这个区域由许多sub-repeats构成，也就是呈现所谓的镶嵌结构。我们来看一个真实的例子，下面的图片是人类Y染色体上repeat区域的一个示意图

![repeat of Chromosome Y](/post_image/Fragmentgluer/repeat_of_Y.PNG)

__Fig.1 Mosaic repeat of human Chromosome Y[^1].__

# Reference

[^1]:图片来源[De Novo Repeat Classification and Fragment Assembly](https://genome.cshlp.org/content/14/9/1786.long)