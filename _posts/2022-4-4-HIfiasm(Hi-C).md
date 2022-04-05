---
layout: article
title: 文献分享：Haplotype-resolved assembly of diploid genomes without parental data
tags: papers

article_header:
  type: cover
  image: 
    src: /post_image/hifiasm(Hi-C)/hifiasm(Hi-C).PNG
---

我们之前以及讨论过Hifiasm这个算法，这是Heng Li发表在Nature Methods上的一篇关于单倍型重构的文章。其实，如果仅仅使用测序数据，我们很难实现telomere-to-telomere的单倍型组装，因为测序长度的限制，所以往往我们需要依靠额外的数据来进行phasing，比如hifiasm就采用了trio数据，即样本父母的二代测序信息。而这篇文章中，采用了Hi-C数据来作为额外的信息，这也是hifiasm(Hi-C)名字的来源。当然了，使用了Hi-C数据，就不再需要trio数据，也就是标题中的所提到的不需要父母数据。
<！--more-->

# Backgroud

那么关于单倍型组装的一些基本知识，我已经在这篇[博客](https://wu-haonan.github.io/2021/05/06/hifiasm.html)中进行了详细介绍。那么这里不再赘述。

hifiasm(Hi-C)与之前发布的hifiasm(trio)所用的assembly graph是相同的，所不同的是序列的划分方式。hifiasm(trio)用的图是unitig graph，即首先构造string graph，节点表示read，边是read之间的overlap，然后进行transitive reduction，即将没有歧义的path收缩成unitig，作为顶点，而边仍然是overlap.hifiasm(Hi-C)其实就是改用Hi-C测序来进行进一步的phase，但是这个想法也不是首次提出了，[FALCON-Phase](https://www.nature.com/articles/s41467-020-20536-y#Sec7)（19年4月挂在biorix上，21年4月发在NC上），我当时也读过这篇文章，他利用Hi-C测序比对到组装的图上之后，大致是使用了类似概率估计的phasing算法，不过之前Heng Li的文章评价，FALCON-Phase没有达到染色体级别的单倍型组装。一个可能的原因是，FALCON没有使用HiFi-read，只是用了PacBio的普通的long read.

<p align="center">
    <img src="/post_image/hifiasm/overview.PNG" width="100%">
    <br />    <small> tittle </small>
</p>

__Fig.1 Overview of hifiasm(trio)[^1].__

文章将unitigs打成k-mer（k=31），然后将Hi-C的short reads测序片段map上去，如果一对儿Hi-C short reads比对上了两个unitigs的杂合子区域，那么它们之间将会连接一条haplotype-specific ‘link’，这就是一种长距离phasing的信息。然后我们可以对unitigs进行二部划分，每一部里面冗余较少，而拥有很多的Hi-C links.我们将unitigs的二部划分问题转化为Max-Cut最大割问题，并用随机算法求解近似解。这个思路也是之前被提出的，文章17年发在Genome Research上，名字叫做[HapCUT2](https://genome.cshlp.org/content/27/5/801.long).
利用Hi-C数据进行二部划分之后，再利用hifiasm(trio)中的binning算法进得到最终的结果。

hifiasm还允许用户仅使用HiFi read，而不使用任何其他信息。在这种模式下，hifiasm假设一个单倍型上的repeat之间的序列差异大于二倍体上杂合子的差异。当然，这样的情况下只能得到primary assemblies，没有完全的phasing，作者把这种组装又起了个名字，叫做dual assembly。这二者其实本质上没啥不同，只不过primary assemblies有一个很长的主链，还有对应的备选，dual assembly相当于输出两个主链，更加连续而已，有利于下游分析。

# Methods

## Overview

对于二倍体而言，前面所说的unitigs graph的节点（unitigs）可以分为两类，来自染色体的同源区段或者杂合子区段。我们可以用read的深度来区别二者，如果我们能够区分杂合区域的unitigs，我们就可以使用之前的hifiasm中graph-binning的算法来得到单倍体。

## Dual assembly

我们先介绍一下，如果没有Hi-C数据，hifiasm是怎么仅使用HiFi read工作的.这里给出本文对于Dual assembly的定义.

>
*Dual assembly is a pair of non-redundant primary assemblies with each assembly representing a complete homologous haplotype.*
>

我们给出两种overlap的定义，<i>cis</i> read  overlaps和<i>trans</i> read overlaps. 其中<i>cis</i> read  overlaps指的是read ${ A,B }$ 被推断为同源的单倍型. 其实这里的 <i>cis</i> 和<i>trans</i>就是之前[博客](https://wu-haonan.github.io/2021/05/06/hifiasm.html)中提到的consistent overlap和inconsistent overlap，hifiasm仅使用<i>cis</i> overlap来组装.

令${ U_st }$来表示杂合关系的unitigs ${ s,t }$之间的<i>trans</i> read overlap的数目. 这可以用来衡量两个unitigs的相似度. （在测序深度一定的前提下，杂合区段的unitigs之间<i>trans</i> overlap越少，二者越相似）. 对于每个杂合unitigs ${ t }$，变量${ \delta_t \in \{1,-1\} }$

# Reference

[^1]:图片来源[Haplotype-resolved de novo assembly using phased assembly graphs with hifiasm](https://www.nature.com/articles/s41592-020-01056-5)