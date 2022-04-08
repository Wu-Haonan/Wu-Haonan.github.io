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
<!--more-->

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

令${ U_{st} }$来表示杂合关系的unitigs ${ s,t }$之间的<i>trans</i> read overlap的数目. 这可以用来衡量两个unitigs的相似度. 对于每个杂合unitigs ${ t }$，变量${ \delta_t \in \\{ 1,-1 \\} }$，所以我们的优化目标是最大化下面的函数

<center>$$
F(\vec{\delta}) = - \sum_{s,t} \delta_s \delta_t U_{st}
$$</center>

这里，${ \vec{\delta} }$代表所有杂合unitigs的分型情况.也就是说，我们要最小化后面的式子${ \sum_{s,t} \delta_s \delta_t U_{st} }$，那也就是我们希望同一个phase中的unitigs之间的${ U_{st} }$尽量小.这里稍微需要大家思考一下，其实虽然${ U_{st} }$代表了inconsistent overlap的数目，但是其实如果有overlap就证明了序列之间有相似度，所以我们当然需要每个phase内部不怎么相似（每个单倍体内部相似序更少），而phase之间杂合区段是相似的. 而且这里我们再回头看前面的假设，hifiasm假设一个单倍型上的repeat之间的序列差异大于二倍体上杂合子的差异，所以，所以即使对于同一phase中的相似序列repeat，我们希望差异更大（${ U_{st} }$越小）的repeat在同一phase内，而杂合区段（${ U_{st} }$更大）的序列放在两个phase中. 所以如果视${ U_{st} }$是权重的话，我们对unitigs的划分就是在寻找最大割. hifiasm使用的优化方法也比较暴力，在最后会提到.

## Max-Cut求近似解

这个部分本来应该最后再讲解，但是因为后面那个优化使用的时候，有点复杂，所以我们先讲解一下这个求解方法.

对于两个优化目标(1)(8)文章用下面方法求解


1. 对于每个unitig ${ t }$，${ \delta_t }$任取${ 1}$或${ -1 }$.

2.任选unitig ${ t }$，如果能提升目标函数，则反转${ \delta_t }$.

3.重复步骤2，直到目标函数不再优化，则达到局部最优解. 如果局部最大值好于历史最大值，则将其设为历史最大值.

4.然后将历史最优解随机反转一部分unitigs（或者反转随机某个unitigs的所有邻居）.  转到步骤2，重新求解局部最优解.

5.重复10000次步骤2-4，返回历史最优解.

对于步骤3，hifiasm会事先探测assembly graph中的‘bubble’区域，将其中的unitigs记录下来，分为两组，我们知道这两组unitigs sets一定来自于两个phase，如果反转到这其中某个unitigs，会将里面的unitigs一起进行操作. 为了防止错误，hifiasm在最后一轮，不使用这个策略.

## Map Hi-C reads

对于大部分的31-mers，在图中至少有两个copy，那些唯一的31-mers很可能就是杂合基因所在的位置. hifiasm标记所有的unique 31-mers在图中的位置. 然后检查Hi-C read pair是否包含两个或者多个不重叠的31-mers，并且丢弃掉Hi-C read中对于提供phasing信息无用的序列. 对于匹配到同源区段的unitigs，也不考虑.

这么做有两个好处，一个是节省了比对的时间，一个是能够尽量的避免错误，相当于我们选取一些非常可信的片段来作为补充信息.

然后根据Hi-C数据的mapping情况，我们可以得到这样两种信息.

1. Hi-C read直接map到杂合子区域的unitigs ${ t }$上，我们称为<i>cis</i> mapping. 这说名这个Hi-C read和unitigs ${ t }$在同一相位.

2. Hi-C read所mapping的位置，存在一个HiFi read，而且这个HiFi read正好和unitigs ${ t' }$存在<i>trans</i> overlap，那么我们就可以推断，Hi-C read与${ t' }$是<i>trans</i> mapping的关系. 那么Hi-C read与${ t' }$的相位是相反关系.

>
*
If there are massive Hi-C cis mappings bridging two heterozygous unitigs <b>s</b> and <b>t</b>, it is likely that s and t originate from the same haplotype. By contrast, large numbers of Hi-C trans mappings bridging <b>s</b> and <b>t</b> indicate that they should be assigned to different haplotypes.
*
>

## Model Hi-C phase

对于一个Hi-C read pair ${ r }$，令${ x_{rt}=1}$表示${ r }$ <i>cis</i> mapping unitig ${ t }$，${ x_{rt} = -1 }$表示 <i>trans</i> mapping unitigs ${ t }$， 其他情况，记${ x_{rt}=0 }$. 同样的，我们还用${ \delta_t }$表示unitig ${ t }$的相位.

对于Hi-C数据，有可能会桥接在两个单倍型上，这就是一种错误. 假设Hi-C read pair ${ r }$，桥接了unitigs ${ s,t }$，那么我们设其发生上述错误的概率是${ \epsilon_r }$. 因为${ r }$已经mapping到了unitigs ${ s,t }$上，所以${ x_rt,x_rs }$只能取${ 1,-1 }$. 由此

<center>$$
\begin{equation}
P(x_{rs},x_{rt}|\delta_s,\delta_t)=
\begin{cases}
(1-\epsilon_r)/2 & \text{if } x_{rt}x_{rt}\delta_s\delta_t=1\\
\epsilon_r/2 & \text{if } x_{rt}x_{rt}\delta_s\delta_t=-1\\
\end{cases}
\end{equation}
$$</center>

我们分析一下这个公式，首先这是一种枚举的写法，也就是说不代表${ x_{rt}x_{rt}\delta_s\delta_t=1 }$所有情况概率的和为${ (1-\epsilon_r)/2 }$，而是一种写法，满足这个条件的等于这个概率（我个人觉得这个写法歧义很大，不是一种很好的写法）. 所以我们其实可以写成下面的样子

<center>$$
\begin{equation}
P(x_{rs},x_{rt}|\delta_s,\delta_t)=
\begin{cases}
(1-\epsilon_r)/2 & \text{if } x_{rt}x_{rt} =1,\delta_s\delta_t=1\\
(1-\epsilon_r)/2 & \text{if } x_{rt}x_{rt} =-1,\delta_s\delta_t=-1\\
\epsilon_r/2 & \text{if } x_{rt}x_{rt}=1,\delta_s\delta_t=-1\\
\epsilon_r/2 & \text{if } x_{rt}x_{rt}=-1,\delta_s\delta_t=1\\
\end{cases}
\end{equation}
$$</center>

分成这么四种情况，其中上面两种，属于Hi-C read正确桥接的情况，加起来的概率应该是${ (1-\epsilon_r) }$，而两种概率等可能，所以分别是${ (1-\epsilon_r)/2 }$，同理下面两种情况也是类似的. 当然了，也可以这么理解，比如第一种情况，${ s,t }$同相的概率是${ 1/2 }$，然后又被正确的测出来了，概率是${ (1-\epsilon_r) }$. 

对于公式(2)，我们可以等价的写成

<center>$$
P(x_{rs},x_{rt}|\delta_s,\delta_t)= \frac{1}{2} \sqrt{\epsilon_r(1-\epsilon_r)} \cdot \left(\frac{1-\epsilon_r}{\epsilon_r}\right)^{\frac{1}{2} x_{rt}x_{rt}\delta_s\delta_t}
$$</center>

所以我们可以写出所有unitigs的似然估计

<center>$$
\begin{align}
\log{\mathcal{L}(\vec{\delta})} &= \sum_r \sum_{s,t} \log{P(x_{rs},x_{rt}|\delta_s,\delta_t)} \\
&= \sum_r \sum_{s,t} \log{\frac{1}{2}} + \frac{1}{2}\log{\epsilon_r(1-\epsilon_r)}+\frac{1}{2}x_{rt}x_{rt}\delta_s\delta_t \log{\frac{1-\epsilon_r}{\epsilon_r}}\\
&= C+\frac{1}{2} \sum_{s,t}\delta_s\delta_t \sum_r x_{rt}x_{rt} \log{\frac{1-\epsilon_r}{\epsilon_r}} \\ 
\end{align}
$$</center>

这里面${ C }$是一个与${ \vec{\delta} }$无关的变量. 我们可以设${ w_r = \log{\frac{1-\epsilon_r}{\epsilon_r}} }$，那么设

<center>$$
W_{st} = \sum_{r\in {r|x_{rs}x_{rt}=1}} w_r
$$</center>

<center>$$
\overline{W}_{st} = \sum_{r\in {r|x_{rs}x_{rt}=-1}} w_r
$$</center>

因此我们的优化目标如下

<center>$$
\log{\mathcal{L}(\vec{\delta})} = C+\frac{1}{2} \sum_{s,t}\delta_s\delta_t (W_st - \overline{W}_{st})
$$</center>

这里有一个问题，${ \epsilon_r }$怎么求呢，hifiasm是这样做的，其认为发生这种错误的概率取决于两个mapping postion的距离，即${ \epsilon_r = \epsilon(d_r) }$，所以我们要去估计这个值. 这里假设每个unitigs是很准确的，都是haplotigs，也就是同源的. 在上面介绍的每轮优化中，根据上一轮优化确定的相位，我们将距离大约是${ d }$的Hi-C mapping聚集，统计里面错误的比例，来作为估计${ \hat{\epsilon}(d) }$. 在第一轮的时候，我们先将Hi-C read的错误率设为${ 0 }$.

# Result

1. HG002

这个图是以trio phasing为groud truth来绘图的，每个点是一个contig，我们统计一下每个contig包含多少父源的31-mer和母源的31-mer，画了这样的图.

<p align="center">
    <img src="/post_image/hifiasm(Hi-C)/Phasing_accuracy_of_HG002_assemblies.PNG" width="80%">
    <br />    <small> tittle </small>
</p>

# Reference

[^1]:图片来源[Haplotype-resolved de novo assembly using phased assembly graphs with hifiasm](https://www.nature.com/articles/s41592-020-01056-5)