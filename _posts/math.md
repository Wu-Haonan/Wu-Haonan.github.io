---
title: Math
tags: 数学
article_header:
  type: cover
sidebar:
  nav: math
---



$$\sum_{i=1}^{k} a_i = y$$

[张林](http://blog.sciencenet.cn/blog-3423233-1270140.html)

\begin{equation}
    f(x)=1
\end{equation}

{% highlight python linenos %}
def Hanota_step (n,s,t,r):
    global i
    if n == 1:
        print("Step-%03d: #%d %c -> %c"%(i,n,s,t))
        i = i + 1
    else:
        Hanota_step(n-1,s,r,t)
        print("Step-%03d: #%d %c -> %c"%(i,n,s,t))
        i = i + 1
        Hanota_step(n-1,r,t,s)
Hanota_step(4,'A','C','B')
{% endhighlight %}
