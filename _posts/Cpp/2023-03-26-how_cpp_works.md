---
layout: article
title: How C++ Works
tags: C++
aside:
  toc: true
sidebar:
  nav: Cherno_Cpp
---

I will follow the video of Cherno to learn C++. The notes of the video will be organized into this blog series. The Cherno's video is in [here](https://www.youtube.com/watch?v=18c3MTX0PK0&list=PLlrATfBNZ98dudnM48yfGUldqGD0S4FFb).

This blog is going to introduce how C++ works. How C++ process a source file (a text file) to an executable binary file. The workflow of writting a C++ program is: 

series C++ source files --Compiler--> binary file

Let's check it on Visual Studio (VS).
<!--more-->

{% highlight C++ linenos %}
#include <iostream>

int main()
{
	std::cout << "Hello World" << std::endl;
	std::cin.get();
}
{% endhighlight %}
