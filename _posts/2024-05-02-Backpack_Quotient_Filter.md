---
layout: article
title: The Backpack Quotient Filter： a dynamic and space-efficient data structure for querying k-mers with abundance
tags: papers

---

This paper proposes a novel data structure, Backpack Quotient Filter (BQF), to index $k$-mer. which support querying with efficient space and negligible false positive rate. 

<!--more-->

# Motivation

**Problem Definition: if a sequence is present or absent and, even better their abundance.** 

But it's hard to achieve for a huge database. 

$\Rightarrow$ So we have to do pseudo-alignment (only reporting present or not without coordinates on genome). Like, Kallisto.

$\Rightarrow$ "Threshold" pseudo-alignment: at least a fixed fraction $\tau$ of the $k$-mers are found in that genome. Like, Bifrost.  In other words, these tools break down queried seqs into $k$-mers; and compare them against datasets (organized as colored Bruijn graph). 

But the graph construction is limitation. 

$\Rightarrow$ If allow false-positive, some tools use Approximate Membership Queries (AMQ) data structure. Trade off between size and false-positive rate. 

**Problem Definition: if a $k$-mer is present or absent and, even better their abundance.** 

Data structures covered in these tools. 

1. Bloom filter: insert; but high space usage.

2. XOR: static; better space usage.

3. Quotient Filter: dynamicity; Counting Quotient Filter (CQF) can retrieve the absence and abundance. But requiring a lot of space. 

# Quotient Filter and Counting Quotient Filter

Before diving into the Backpack Quotient Filter (BQF) proposed in this paper, let's first take a look of Quotient filter (QF) and Count Quotient Filter (CQF) [^1]. A QF is a table with $2^q$ slots, each of fixed size $r$, where $q,r$ are defined by user. Given a hash function $h$ that hashes each element to a integer of $q+r$ bits, we can split $h(x)$ to two functions, i.e. $h_0(x)$ of $q$ bits, called "**quotient**"  and $h_1(x)$ of $r$ bits, called "**remainder**". 

1. $h_0(x)$, quotient of fingerprint (hash value): it maps to an address in the table, and the corresponding slot called "**canonical slot**". 
2. $h_1(x)$, remainder of fingerprint: we try to store $h_1(x)$ in some slot of table. 


<p align="center">
    <img src="/post_image/BQF/QF.jpg" width="60%">
</p>
Basically, the idea is if we have an element, say a $k$-mer, 

# Method

## Preliminaries 

### pseudo-alignment

The number of $k$-mers existing in two sequences provides a metric to measure the similarity between them, leading to the so called pseudo-alignment



[^1]: [Prashant Pandey, Michael A. Bender, Rob Johnson, and Rob Patro. A General-Purpose Counting Filter: Making Every Bit Count. *In Proceedings of the 2017 ACM International Conference on Management of Data, SIGMOD ’***17**, pages 775–787, New York, NY, USA, 2017. Association for Computing Machinery.](https://dl.acm.org/doi/abs/10.1145/3035918.3035963).