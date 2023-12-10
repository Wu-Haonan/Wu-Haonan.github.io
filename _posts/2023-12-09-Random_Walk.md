---
layout: article
title: Expectation of absolute distance in Random walk
tags: Algorithm_Design_and_Analysis, Random_walk, absolute_distance
aside:
  toc: true
---

This blog, I will post my proof about the expectation of absolute distance in Random walk is ${\Theta(\sqrt{i})}$ at ${i}$th step. 

<!--more-->

# Statement of Problem 

Consider a sequence of $n$ flips of an unbiased coin. Let $H_i$ denote the absolute value of the excess of then number of HEADS over the number of TAILS seen in the first $i$ flips. Show that $E[H_i] = \Theta(\sqrt{i})$. 

We can also treat is as a 1D random walk problem.

# Idea of my proof

1. Use the relation between **Expectation** and **Variance** to get the upper bound $\sqrt{i}$.

2. Solve the recursive formula of $E[H_i]$, which can give us a sense of growth speed. We can prove the lower bound greater than ${C\sqrt{i}}$ by induction, where C ranges from $0$ to $1$.



# **Lemma 1:** $\mathbb{E}[H_i]$ satisfies following recursive formula

$$
\begin{equation}
    \mathbb{E}[H_i] = 
    \begin{cases}
        \mathbb{E}[H_{i-1}] + \frac{1}{2^{i-1}} \binom{i-1}{(i-1)/2}, & n \text{ is odd,}\\
        \mathbb{E}[H_{i-1}],& n \text{ is even.}
    \end{cases}\nonumber
\end{equation}
$$

**Proof.** By definition, we have 

$$
\begin{equation}
\begin{aligned}
    \mathbb{E}[H_i] &= \sum_{k=0}^{i} \vert -i + 2k \vert \Pr[k \text{ HEADS in first } i \text{ flips}] \\
    &= \sum_{k=0}^{i} \vert -i + 2k \vert \binom{i}{k} \frac{1}{2^i}
\end{aligned}\nonumber
\end{equation}
$$

* If $i$ is even and $i-1$ is odd. We have

$$
\begin{equation}
\begin{aligned}
    \mathbb{E}[H_i] &= \sum_{k=0}^{i} \vert -i + 2k \vert \binom{i}{k} \frac{1}{2^i} \\
    & = 2 \cdot \sum_{k=i/2}^{i} \vert -i + 2k \vert \binom{i}{k} \frac{1}{2^i} \\
    & =  \vert -i + 2(i/2) \vert \binom{i}{i/2} \frac{1}{2^i} + 2 \cdot \sum_{k=i/2+1}^{i} \vert -i + 2k \vert \binom{i}{k} \frac{1}{2^i} \\
    & = 2 \cdot \sum_{k=i/2}^{i} \vert -i + 2k \vert \binom{i}{k} \frac{1}{2^i} \\
    & = 2 \cdot \sum_{k=i/2}^{i-1} (-i + 2k)  \binom{i}{k} \frac{1}{2^i} + 2 \vert -i + 2i \vert \binom{i}{i} \frac{1}{2^i} \\
    & = 2 \cdot \sum_{k=i/2}^{i-1} (-i + 2k)  \binom{i}{k} \frac{1}{2^i} + \frac{i}{2^{i-1}}
\end{aligned}\nonumber
\end{equation}
$$

And

$$
\begin{equation}
\begin{aligned}
    \mathbb{E}[H_{i-1}] &= \sum_{k=0}^{i-1} \vert -i+1 + 2k \vert \binom{i-1}{k} \frac{1}{2^{i-1}} \\
    & = 2\cdot \sum_{k=i/2}^{i-1} \vert -i+1 + 2k \vert \binom{i-1}{k} \frac{1}{2^{i-1}}\\
    & = 2\cdot \sum_{k=i/2}^{i-1} ( -i+1 + 2k ) \binom{i-1}{k} \frac{1}{2^{i-1}}\\
\end{aligned}\nonumber
\end{equation}
$$

Then, we have 

$$
\begin{equation}
\begin{aligned}
    \mathbb{E}[H_i] - \mathbb{E}[H_{i-1}] &= 2 \cdot \sum_{k=i/2}^{i-1} (-i + 2k)  \binom{i}{k} \frac{1}{2^i} + \frac{i}{2^{i-1}} - 2\cdot \sum_{k=i/2}^{i-1} ( -i+1 + 2k ) \binom{i-1}{k} \frac{1}{2^{i-1}} \\
    & = 2\left(\sum_{k=i/2}^{i-1} (-i + 2k)  \binom{i}{k} \frac{1}{2^i} - ( -i+1 + 2k ) \binom{i-1}{k} \frac{1}{2^{i-1}} \right) + \frac{i}{2^{i-1}}\\
    & = 2\left(\sum_{k=i/2}^{i-1} \frac{(-i + 2k)}{2^i} \left( \binom{i-1}{k} + \binom{i-1}{k-1}\right)-  \binom{i-1}{k} \frac{( -i+1 + 2k )}{2^{i-1}} \right) + \frac{i}{2^{i-1}}\\
    & = 2\left(\sum_{k=i/2}^{i-1} \frac{(-i + 2k)}{2^i} \binom{i-1}{k-1} +  \binom{i-1}{k} \frac{( i -2 - 2k )}{2^{i}} \right) + \frac{i}{2^{i-1}}\\
    & = 2\left(\sum_{k=i/2}^{i-1} \frac{(-i + 2k)}{2^i} \binom{i-1}{k-1} - \sum_{k=i/2}^{i-1} \binom{i-1}{k} \frac{( -i +2 + 2k )}{2^{i}} \right) + \frac{i}{2^{i-1}}\\
    & = 2\left(\sum_{k=i/2}^{i-1} \frac{(-i + 2k)}{2^i} \binom{i-1}{k-1} - \sum_{j=i/2+1}^{i} \binom{i-1}{j-1} \frac{( -i + 2j )}{2^{i}} \right) + \frac{i}{2^{i-1}}\\
    & = 2\left( \frac{(-i + 2(i/2))}{2^i} \binom{i-1}{i/2-1} - \binom{i-1}{i-1} \frac{( -i + 2i )}{2^{i}} \right) + \frac{i}{2^{i-1}} \\
    & = 2\left( 0\cdot \binom{i-1}{i/2-1} - \binom{i-1}{i-1} \frac{i}{2^{i}} \right) + \frac{i}{2^{i-1}} \\
    & = 0
\end{aligned}\nonumber
\end{equation}
$$

* If $i$ is odd and $i-1$ is even. We have

$$
\begin{equation}
\begin{aligned}
    \mathbb{E}[H_{i}] &= \sum_{k=0}^{i} \vert -i + 2k \vert \binom{i}{k} \frac{1}{2^{i}} \\
    & = \sum_{k=0}^{(i-1)/2} \vert -i + 2k \vert \binom{i}{k} \frac{1}{2^{i}} + \sum_{k=(i+1)/2}^{i} \vert -i + 2k \vert \binom{i}{k} \frac{1}{2^{i}} \\
    & = 2\cdot \sum_{k=(i+1)/2}^{i} \vert -i + 2k \vert \binom{i}{k} \frac{1}{2^{i}}\\
    & = 2\cdot \sum_{k=(i+1)/2}^{i-1} ( -i + 2k ) \binom{i}{k} \frac{1}{2^{i}} + 2 \cdot ( -i + 2i ) \binom{i}{i} \frac{1}{2^{i}}  \\
    & = 2\cdot \sum_{k=(i+1)/2}^{i-1} ( -i + 2k ) \binom{i}{k} \frac{1}{2^{i}} +   \frac{i}{2^{i-1}}  \\
\end{aligned}\nonumber
\end{equation}
$$

And

$$
\begin{equation}
\begin{aligned}
    \mathbb{E}[H_{i-1}] &= \sum_{k=0}^{i-1} \vert -i+1 + 2k \vert \binom{i-1}{k} \frac{1}{2^{i-1}} \\
    & = 2 \cdot \sum_{k=(i+1)/2}^{i-1} (-i+1 + 2k) \binom{i-1}{k} \frac{1}{2^{i-1}}
\end{aligned}\nonumber
\end{equation}
$$

Then, we have 

$$
\begin{equation}
\begin{aligned}
    \mathbb{E}[H_i] - \mathbb{E}[H_{i-1}] &= 2\cdot \sum_{k=(i+1)/2}^{i-1} ( -i + 2k ) \binom{i}{k} \frac{1}{2^{i}} +  \frac{i}{2^{i-1}} - 2 \cdot \sum_{k=(i+1)/2}^{i-1} (-i+1 + 2k)  \binom{i-1}{k} \frac{1}{2^{i-1}} \\
    & = 2 \cdot \left(\sum_{k=(i+1)/2}^{i-1} ( -i + 2k ) \binom{i}{k} \frac{1}{2^{i}} - (-i+1 + 2k) \binom{i-1}{k} \frac{1}{2^{i-1}} \right) +  \frac{i}{2^{i-1}}\\
    & = 2 \cdot \left(\sum_{k=(i+1)/2}^{i-1} \frac{( -i + 2k )}{2^{i}} \left(\binom{i-1}{k} + \binom{i-1}{k-1}\right) -\binom{i-1}{k} \frac{ (-i+1 + 2k) }{2^{i-1}} \right) +  \frac{i}{2^{i-1}}\\
    & = 2 \cdot \left(\sum_{k=(i+1)/2}^{i-1} \frac{( -i + 2k )}{2^{i}} \binom{i-1}{k-1}+\binom{i-1}{k} \frac{ (i-2 - 2k) }{2^{i}} \right) +  \frac{i}{2^{i-1}}\\
    & = 2 \cdot \left(\sum_{k=(i+1)/2}^{i-1} \frac{( -i + 2k )}{2^{i}} \binom{i-1}{k-1} - \sum_{k=(i+1)/2}^{i-1} \binom{i-1}{k} \frac{ (-i+2 + 2k) }{2^{i}} \right) +  \frac{i}{2^{i-1}}\\
    & = 2 \cdot \left(\sum_{k=(i+1)/2}^{i-1} \frac{( -i + 2k )}{2^{i}} \binom{i-1}{k-1} - \sum_{j=(i+3)/2}^{i} \binom{i-1}{j-1} \frac{ (-i+2 j) }{2^{i}} \right) +  \frac{i}{2^{i-1}}\\
    & = 2 \cdot \left( \frac{( -i + 2((i+1)/2) )}{2^{i}} \binom{i-1}{(i+1)/2-1} -  \binom{i-1}{i-1} \frac{ (-i+2 i) }{2^{i}} \right) +  \frac{i}{2^{i-1}}\\
    & = \frac{1}{2^{i-1}} \binom{i-1}{(i-1)/2}
\end{aligned}\nonumber
\end{equation}
$$

Hence, we prove the recursive formula

$$
\begin{equation}
    \mathbb{E}[H_i] = 
    \begin{cases}
        \mathbb{E}[H_{i-1}] + \frac{1}{2^{i-1}} \binom{i-1}{(i-1)/2}, & n \text{ is odd,}\\
        \mathbb{E}[H_{i-1}],& n \text{ is even.}
    \end{cases}\nonumber   \quad \quad \quad \square
\end{equation} 
$$

# **Theorem 1**: $\mathbb{E}[H_i] = \Theta(\sqrt{i})$.

**Proof.** 

1. First, Let's show $\mathbb{E}[H_i] \leq \sqrt{i}$
Denote the random variable $X_j, j = 1,2,\cdots, i$ as below

$$
\begin{equation}
    X_j = 
    \begin{cases}
    1 & \text{HEAD is seen in } i \text{th flip}\\
    -1 & \text{TAIL is seen in } i \text{th flip}\\
    \end{cases}\nonumber
\end{equation}
$$

It's easy to have $H_i = \vert \sum_{j=1}^i X_i \vert $, $\mathbb{E}[X_i]=0$ and we know $X_1,X_2,\cdots,X_i$ are pairwise independent. By the definition of variance $\mathbb{D}[H_i]$, we have $\mathbb{D}[H_i] = \mathbb{E}[H_i^2] - (\mathbb{E}[H_i])^2$. Then we have 

$$
\begin{equation}
\begin{aligned}
    (\mathbb{E}[H_i])^2 = \mathbb{E}[H_i^2] - \mathbb{D}[H_i]
\end{aligned}\nonumber
\end{equation}
$$

Because $X_1,X_2,\cdots,X_i$ are pairwise independent, we have           

$$
\begin{equation}
\begin{aligned}
    \mathbb{E}[H_i^2] & = \mathbb{E}\left[\left(\sum_{j=1}^i X_j\right)^2\right] \\
    & = \sum_{j=1}^i \mathbb{E}[X_j^2] + 2 \sum_{j<k} \mathbb{E}[X_j X_k]\\
    & = \sum_{j=1}^i \mathbb{E}[X_j^2] + 2 \sum_{j<k} \mathbb{E}[X_j] \mathbb{E}[X_k]\\
    & = i 
\end{aligned}\nonumber
\end{equation}
$$

$(\mathbb{E}[H_i])^2 = \mathbb{E}[H_i^2] - \mathbb{D}[H_i]$ that means $(\mathbb{E}[H_i])^2 \leq \mathbb{E}[H_i^2] = i$. Thus $\mathbb{E}[H_i] \leq \sqrt{i}$.


2. We will show $\mathbb{E}[H_i] > C \sqrt{i}, C= e^{-1} \sqrt{\frac{2}{\pi}}$ by induction (It's clear $0<C<1$)

Base case: For $i=1$, we know $\mathbb{E}[H_1] = \vert 1 \vert \cdot \frac{1}{2}+\vert -1 \vert \cdot \frac{1}{2} = 1 > C \sqrt{1}$. And for $i=2$, we know  $\mathbb{E}[H_2] = \vert 2 \vert \cdot \frac{1}{4} +\vert -2 \vert \cdot \frac{1}{4} + 0 \cdot \frac{1}{2} = 1 > \frac{\sqrt{2}}{e} \sqrt{\frac{2}{\pi}} = C \sqrt{2} $.\\

Induction Hypothesis: $\mathbb{E}[H_j] > \frac{\sqrt{j}}{2}, \forall j \leq k$

We will show ${\mathbb{E}[H_{k+1}] > \frac{\sqrt{k+1}}{2}}$ in the following 

* If $k+1$ is even: By Lemma 1, $\mathbb{E}[H_{k+1}] = \mathbb{E}[H_{k}] = \mathbb{E}[H_{k-1}] + \frac{1}{2^{k-1}} \binom{k-1}{(k-1)/2} $. Then, we have

$$
\begin{equation}
\begin{aligned}
    \frac{1}{2^{k-1}} \binom{k-1}{(k-1)/2} & = \frac{1}{2^{k-1}} \frac{(k-1)!}{\left(\left(\frac{k-1}{2}\right)!\right)^2}
\end{aligned}\nonumber
\end{equation}
$$

By Stirling formula, we have

$$
\begin{equation}
\begin{aligned}
    \frac{1}{2^{k-1}} \binom{k-1}{(k-1)/2} & = \frac{1}{2^{k-1}} \frac{(k-1)!}{\left(\left(\frac{k-1}{2}\right)!\right)^2} \\
    & > \frac{1}{2^{k-1}} \frac{\sqrt{2 \pi (k-1)} \left(\frac{k-1}{e}\right)^{k-1}}{\pi (k-1) \left(\frac{k-1}{2e}\right)^{k-1} e^\frac{1}{12(k-1)}}\\
    & = e^{-\frac{1}{12(k-1)}} \sqrt{\frac{2}{\pi}} \frac{1}{\sqrt{k-1}}\\
    & > e^{-1} \sqrt{\frac{2}{\pi}} \frac{1}{\sqrt{k-1}}\\
    & = e^{-1} \sqrt{\frac{2}{\pi}} \frac{2}{2\sqrt{k-1}}> C \frac{2}{\sqrt{k+1} + \sqrt{k-1}}  = C (\sqrt{k+1} - \sqrt{k-1})
\end{aligned}\nonumber
\end{equation}
$$

By induction hypothesis, $\mathbb{E}[H_{k-1}] > C\sqrt{k-1}$, that means

$$
\begin{equation}
\begin{aligned}
    \mathbb{E}[H_{k+1}] &= \mathbb{E}[H_{k}] \\
    &= \mathbb{E}[H_{k-1}] + \frac{1}{2^{k-1}} \binom{k-1}{(k-1)/2}\\
    & > C\sqrt{k-1} + C (\sqrt{k+1} - \sqrt{k-1}) \\
    & = C \sqrt{k+1}
\end{aligned}\nonumber
\end{equation}
$$

* If $k+1$ is odd: By Lemma 1, $\mathbb{E}[H_{k+1}] = \mathbb{E}[H_{k}] + \frac{1}{2^{k}} \binom{k}{k/2} $. Then, we have

$$
\begin{equation}
\begin{aligned}
    \frac{1}{2^{k}} \binom{k}{k/2} & = \frac{1}{2^{k}} \frac{k!}{\left(\left(\frac{k}{2}\right)!\right)^2}
\end{aligned}\nonumber
\end{equation}
$$

By Stirling formula, we have

$$
\begin{equation}
\begin{aligned}
    \frac{1}{2^{k}} \binom{k}{k/2} & > \frac{1}{2^{k}} \frac{\sqrt{2 \pi k} \left(\frac{k}{e}\right)^{k}}{\pi k \left(\frac{k}{2e}\right)^{k} e^\frac{1}{12k}}\\
    & > e^{-1} \sqrt{\frac{2}{\pi}} \frac{1}{\sqrt{k}}\\
    & > e^{-1} \sqrt{\frac{2}{\pi}} \frac{1}{2\sqrt{k}}> C \frac{1}{\sqrt{k+1} + \sqrt{k}}  = C (\sqrt{k+1} - \sqrt{k})
\end{aligned}\nonumber
\end{equation}
$$

By induction hypothesis, $\mathbb{E}[H_{k}] > C\sqrt{k}$, that means

$$
\begin{equation}
\begin{aligned}
    \mathbb{E}[H_{k+1}] &= \mathbb{E}[H_{k}] + \frac{1}{2^{k}} \binom{k}{k/2}\\
    & > C\sqrt{k} + C (\sqrt{k+1} - \sqrt{k}) \\
    & = C \sqrt{k+1}
\end{aligned}\nonumber
\end{equation}
$$

Hence, we show $\mathbb{E}[H_i] > C \sqrt{i}, C= e^{-1} \sqrt{\frac{2}{\pi}}$, and $\mathbb{E}[H_i] \leq \sqrt{i}$, which imply $\mathbb{E}[H_i] = \Theta(\sqrt{i})$. $\square$
