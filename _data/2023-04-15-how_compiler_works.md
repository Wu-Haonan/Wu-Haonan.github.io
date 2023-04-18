---
layout: article
title: How C++ Compiler Works
tags: C++
aside:
  toc: true
sidebar:
  nav: Cherno_Cpp
---

This blog is going to introduce how C++ Compiler works. In fact, the C++ source files are just some text files, and we need some way to transform them into some executable binary files. We have two operations here, one of them called <b>compiling</b> and one of them called <b>linking</b> (we will talk about it in next blog). Actually, what the compiler does is taking our source files and convert them into an object files (which is a kind of intermediate format). 

Let's go ahead and see what compiler does!

<!--more-->

In the above blog, we have the following two cpp files, Log.cpp file and 

{% highlight C++ linenos %}
// Log.cpp

#include <iostream>

void Log(const char* message)


{
	std::cout << message << std::endl;
}

{% endhighlight %}

main.cpp file.

{% highlight C++ linenos %}
//main.cpp

#include <iostream>

void Log(const char* message);
// OR ignore the variable, like this:
// void Log(const char*); 

int main()
{
	Log("Hello World!");
	std::cin.get();
}
{% endhighlight %}

After we debug our program, will find there are two obj files, Log.obj and main.obj, in our debug folder. So, what compiler have done is to generate obj file for each C++ file. 

And these cpp files are called <b>transition units</b>. In fact, C++ doesn't care about files, which are just a way to feed source code for compiler. We just need to tell the compiler, "Hey! This is a cpp file" by adding the extension ".cpp". If we create a file with extension as ".h", compiler will treat the ".h" file as a head file. These are default convention, if you want to compile a file like "XX.random" as a Cpp file, you just need to tell the compiler, please compile it as a cpp file. To sum up, files are just files, which only provide source code.

Each translation unit will generate an obj file. If we creat a cpp file which includes other cpp files, actually compiler will treat it as one transilation unit and creat only one obj file.

Here we give another more simple exmaple. We creat a new cpp file called "Math.cpp" shown in below

{% highlight C++ linenos %}
//Math.cpp
int Multiply(int a, int b) 
{
	int result = a * b;
	return result;
}
{% endhighlight %}

After we use "Ctrl+F7" to build up this file, we can get Math.obj file in our folder. <font color=purple> Before we look what exactly is in obj file, let's first talk about the first stage of compilation <b>"Pre-processing"</b>. </font>

# Pre-processing

During the pre-processing stage, the compiler will basically go through all of our pre-processing statements. The common statements we use are <b>include</b>, <b>define</b>, <b>if</b>, <b>if def</b> and <b>pragma</b>. 

## \#include

First, we take a look at one of the most common preprocessor statement -- <span style="background-color: #0d0d0d"><font face="Monaco" color='#87CEFA'> &thinsp; # include &thinsp;</font></span>.

<span style="background-color: #0d0d0d"><font face="Monaco" color='#87CEFA'> &thinsp; # include &thinsp;</font></span> is exactly really simple, the preprocessor will just open the file that we include, <b>read</b> all its contents and <b>paste</b> it into the file where we wrote <span style="background-color: #0d0d0d"><font face="Monaco" color='#87CEFA'> &thinsp; # include &thinsp;</font></span>.

To prove that, we can take an example. 

We add a header file called "EndBrace.h" in our project, and only type a "\}" in it. 

{% highlight C++ linenos %}
//EndBrace.h
}
{% endhighlight %}

And, then go back to our Math.cpp file, and replace the closing curly bracket with <span style="background-color: #0d0d0d"><font face="Monaco" color='#87CEFA'> &thinsp; # include "EndBrace.h" &thinsp;</font></span>. 

{% highlight C++ linenos %}
//Math.cpp

int Multiply(int a, int b) 
{
	int result = a * b;
	return result;
#include "EndBrace.h"
{% endhighlight %}

We use "Ctrl+F7" to compile it, and it compile successfully. So, what <span style="background-color: #0d0d0d"><font face="Monaco" color='#87CEFA'> &thinsp; # include &thinsp;</font></span> does is just to copy and paste all the contents in our specified file.

## \#define

We can try another example, use <span style="background-color: #0d0d0d"><font face="Monaco" color='#87CEFA'> &thinsp; # define &thinsp;</font></span> to replace "INTGER" with "int". Actually, what <span style="background-color: #0d0d0d"><font face="Monaco" color='#87CEFA'> &thinsp; # define &thinsp;</font></span> does it just read all the code in our file and replace all the first "word" with following "word".

{% highlight C++ linenos %}
//Math.cpp

#define INTEGER int

INTEGER Multiply(int a, int b)
{
	INTEGER result = a * b;
	return result;
}
{% endhighlight %}

Here we can change our property and get the preprocessing file. Like the following figure,

<p align="center">
    <img src="/post_image/cpp/Preprocess_to_file.png" width="70%">
</p>

And open the "Math.i" in our folder, we will find all the "INTGER"s are replaced with "int".

<p align="center">
    <img src="/post_image/cpp/Math_i.png" width="80%">
</p>