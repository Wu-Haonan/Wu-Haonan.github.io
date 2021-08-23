---
layout: article
title: Python Chapter3
tags: Python
---

<!--more-->

# matplotlib.pyplot

## 将图片转化成数组，然后再把数组转化为图片

{% highlight python linenos %}
import matplotlib.pyplot as plt 
image1= plt.imread('C:/Users/lenovo/Desktop/1.jpg')   #返回数组，数组里的数据是图片内容
plt.imshow(image1)   #将numpy数组进行可视化
plt.show()           #展示图片
{% endhighlight %}