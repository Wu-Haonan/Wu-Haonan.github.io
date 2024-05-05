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

# Quotient Filter (QF) and Counting QF (CQF)

## Preliminaries

Before diving into the Backpack Quotient Filter (BQF) proposed in this paper, let's first take a look of Quotient filter (QF) and Count Quotient Filter (CQF) [^1]. A QF is a table with $2^q$ slots, each of fixed size $r$, where $q,r$ are defined by user. Given a hash function $h$ that hashes each element to a integer of $q+r$ bits, we can split $h(x)$ to two functions, i.e. $h_0(x)$ of $q$ bits, called "**quotient**"  and $h_1(x)$ of $r$ bits, called "**remainder**". 

1. $h_0(x)$, quotient of fingerprint (hash value): it maps to an address in the table, and the corresponding slot called "**canonical slot**". 
2. $h_1(x)$, remainder of fingerprint: we try to store $h_1(x)$ in some slot of table. 

<p align="center">
    <img src="/post_image/BQF/QF.jpg" width="100%">
</p>

__Fig.1 Quotient Filter[^1].__

Basically, the idea is if we have an element, say a $k$-mer $x$, we calculate $h_0(x)$, i.e. the canonical slot and try to  store $h_1(x)$. If the slot is empty, we are fine. But in most cases, it's occupied. How does this happen? 

1. hard collision: two distinct elements have the same hash value. We avoid this by using a prefect hash function. 
2. soft collision: It is inherent to quotient filters. A soft collision occurs when two distinct elements x and y have different hashes but the same quotient: $h_0(x) = h_0(y)$.

## Insert and Run

To address this collision, we have to shift the reminder into next slots. In other words, elements with same quotient are stored consecutively in the table, which form a "**run**" (showed in a same color in above figure). Inside a run, the QF guarantee the remainders are stored in ascending order. 

### Metadata bits

Thus, above operations lead to another problem, that is even the first element of a run $x_\text{first}$ maybe not locate in $h_0(x_\text{first})$, because upstream runs could occupy this canonical slot. To keep track of the shifting   distance, we assign two bits metadata. 

- The *occupied* bit determines whether a slot is a canonical slot for some inserted element, whose remainder could be shifted and located elsewhere. 
- The *runend* bit indicates whether a slot stores a remainder that is at the end of a run. In this
  case, it is set to 1. Otherwise, it is set to zero, and in this situation, the next slot is either empty
  or is the first one of a different run.

### Rank, Select and Offset

Given a $k$-mer $x$, we first calculate the quotient $i=h_0(x)$, and then we aim to find the end of this run with quotient $i$. Because no matter we do insertion or query, we need to locate at the end of run, and compare the remainder one by one reversely (see Fig. 2). 

The above two metadata bits of all the slots form two binary vector called *occupieds* and *runends*. We define following two operations. 

1. $\textbf{Rank}(occupieds,i)$: counting the number $d$ of runs that present before $i$, i.e. counting the number of $1$ in vector *occupieds* before $i$.  
2. $\textbf{Select}(runends, d)$: find the position of $d^\text{th}$ $1$ in vector *runends*. 

Thus, we can get the end of runs with quotient $i$, $\textbf{Select}(runends, \textbf{Rank}(occupied,i))$. To do it efficiently, we can define this shifting distance as *Offset* and store the information. To pursue a better trade-off between space and speed, QF are divided into blocks of 64 slots. QF uses 64 bits integer to store the offset of the first slot in block as checkpoint. 

<p align="center">
    <img src="/post_image/BQF/QF_offset.png" width="100%">
</p>


__Fig. 2 rank and select.__

Given a slot $j$, and position $i$ as the first slot of corresponding block, we calculate the position of end of the run as 

$$
\begin{equation}
\begin{aligned}
d &= \textbf{Rank}(occupies[i+1:j],j-i+1) \\
t &= \textbf{Select}(runends[i+Offset(i)+1,2^q-1],d) \\
\text{Position}_\text{runend}(j) &= i + Offset(i) + t
\end{aligned}
\end{equation}
$$

The above calculation can be virtualized as below Fig. 3

<p align="center">
    <img src="/post_image/BQF/Block_offset.PNG" width="80%">
</p>


__Fig. 3 Procedure for computing offset $O_j$ given $O_i$[^1].__



# Method

## Preliminaries 

### pseudo-alignment

The number of $k$-mers existing in two sequences provides a metric to measure the similarity between them, leading to the so called pseudo-alignment



[^1]: [Prashant Pandey, Michael A. Bender, Rob Johnson, and Rob Patro. A General-Purpose Counting Filter: Making Every Bit Count. *In Proceedings of the 2017 ACM International Conference on Management of Data, SIGMOD ’***17**, pages 775–787, New York, NY, USA, 2017. Association for Computing Machinery.](https://dl.acm.org/doi/abs/10.1145/3035918.3035963).