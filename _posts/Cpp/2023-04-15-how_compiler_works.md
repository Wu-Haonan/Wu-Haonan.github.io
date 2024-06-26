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

First, we take a look at one of the most common preprocessor statement -- <span style="background-color: #0d0d0d"><font face="Monaco" color='#87CEFA'> &thinsp; #include &thinsp;</font></span>.

<span style="background-color: #0d0d0d"><font face="Monaco" color='#87CEFA'> &thinsp; #include &thinsp;</font></span> is exactly really simple, the preprocessor will just open the file that we include, <b>read</b> all its contents and <b>paste</b> it into the file where we wrote <span style="background-color: #0d0d0d"><font face="Monaco" color='#87CEFA'> &thinsp; #include &thinsp;</font></span>.

To prove that, we can take an example. 

We add a header file called "EndBrace.h" in our project, and only type a "\}" in it. 

{% highlight C++ linenos %}
//EndBrace.h
}
{% endhighlight %}

And, then go back to our Math.cpp file, and replace the closing curly bracket with <span style="background-color: #0d0d0d"><font face="Monaco" color='#87CEFA'> &thinsp; #include "EndBrace.h" &thinsp;</font></span>. 

{% highlight C++ linenos %}
//Math.cpp

int Multiply(int a, int b) 
{
	int result = a * b;
	return result;
#include "EndBrace.h"
{% endhighlight %}

We use "Ctrl+F7" to compile it, and it compile successfully. So, what <span style="background-color: #0d0d0d"><font face="Monaco" color='#87CEFA'> &thinsp; #include &thinsp;</font></span> does is just to copy and paste all the contents in our specified file.

## \#define

We can try another example, use <span style="background-color: #0d0d0d"><font face="Monaco" color='#87CEFA'> &thinsp; #define &thinsp;</font></span> to replace "INTGER" with "int". Actually, what <span style="background-color: #0d0d0d"><font face="Monaco" color='#87CEFA'> &thinsp; #define &thinsp;</font></span> does it just read all the code in our file and replace all the first "word" with following "word".

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
    <img src="/post_image/cpp/Preprocess_to_file.png" width="60%">
</p>

And open the "Math.i" in our folder, we will find all the "INTGER"s are replaced with "int".

<p align="center">
    <img src="/post_image/cpp/Math_i.png" width="80%">
</p>

# \#if

The preprocessor <span style="background-color: #0d0d0d"><font face="Monaco" color='#87CEFA'> &thinsp; #If &thinsp;</font></span> can let us exclude or include code based on a give condition. If we write <span style="background-color: #0d0d0d"><font face="Monaco" color='#87CEFA'> &thinsp; #If 1 &thinsp;</font></span>, that means the condition is always true. 

{% highlight C++ linenos %}
//Math.cpp

#if 1

int Multiply(int a, int b)
{
	int result = a * b;
	return result;
}

#endif
{% endhighlight %}

After compilation, we can check our preprocessor file, which looks excatly like the result without <span style="background-color: #0d0d0d"><font face="Monaco" color='#87CEFA'> &thinsp; #If 1 &thinsp;</font></span> statement.

If we turn off here with <span style="background-color: #0d0d0d"><font face="Monaco" color='#87CEFA'> &thinsp; #If 0 &thinsp;</font></span>

{% highlight C++ linenos %}
//Math.cpp

#if 0

int Multiply(int a, int b)
{
	int result = a * b;
	return result;
}

#endif
{% endhighlight %}

The preprocessor file "Math.i" is like this, and our code was disabled.

<p align="center">
    <img src="/post_image/cpp/Math_i_hash_if.PNG" width="70%">
</p>

Ok, that's all about preprocessor, if intersted, you can check the preprocessor file after adding "\#include \<iostream\>". You will find the preprocessor file will become so large, that's because, "\<iostream\>" has a lot of content and also include other files.

# Obj file 

Now, let's change back the setting of "property". And compile our cpp file again, we will get "Math.obj" in the folder. Let's check what's inside in our obj file. But <b>unfortunately</b>, it's a binary file, and we can not understand it directly.

{% highlight C++ linenos %}
//Math.cpp

int Multiply(int a, int b)
{
	int result = a * b;
	return result;
}
{% endhighlight %}

So, let's convert it to a more readable form. We can also hit "Property" here and set the "Assembler Output" to "Assembly-Only Listing (/FA)"

<p align="center">
    <img src="/post_image/cpp/Assembler_Output.png" width="70%">
</p>

And, then you will find "Math.asm" file in our "Debug" folder. Which is basically a readable result if what the object file contains. Let's check the critical part of this file.

<p align="center">
    <img src="/post_image/cpp/Asm_multiply.png" width="60%">
</p>

[-> 1]. Move variable <b>a</b> to "eax"

[-> 2]. Let "eax" multiply variable <b>b</b>. 

[-> 3]. Then move "eax" to variable <b>result</b>

[-> 4]. Move <b>result</b> back to "eax" to return it.

We find Step 3 and 4 actully are redundant. This is also a example why we need optimization during compilation. If we change our code as

{% highlight C++ linenos %}
//Math.cpp

int Multiply(int a, int b)
{
	return a * b;
}
{% endhighlight %}

After we compile it, "Math.asm" will also change as below, which only need two steps.

<p align="center">
    <img src="/post_image/cpp/Aem_multiply_optim.png" width="70%">
</p>

# Optimization

We will find, there are a lot of code in our asm file. That's because we don't use optimization in debug mode. Here we can temporarily change our "property" as follow to use optimization during compilation.

First, we set "Maximum Optimization (Favor Speed) (/O2)" in "Optimization" under "Debug" configuration shown as below

<p align="center">
    <img src="/post_image/cpp/optimization_max_speed.png" width="60%">
</p>

If we just compile it now, we will have error, so we have to change another place. We need to set "Default" in "Basic Runtime Checks" at "Code Generation" item.

<p align="center">
    <img src="/post_image/cpp/Basic_Runtime_Checks.png" width="60%">
</p>

Then if we compile it, we will find the "Math.asm" file is quite simple than before.

Here we will take another different example, here the funtion don't take any input and just return 5*2. 

{% highlight C++ linenos %}
//Math.cpp

int Multiply()
{
	return 5 * 2;
}
{% endhighlight %}

Let's compile it and check the "Math.asm" file. It just moves 10 to our "eax" <b>register</b>, which store our return value. So, the opimization just simplified our "5*2" as "10".

<p align="center">
    <img src="/post_image/cpp/return_10.PNG" width="70%">
</p>

Let's take another example, we add "Log" funtion here.

{% highlight C++ linenos %}
//Math.cpp

const char* Log(const char* message) 
{
	return message;
}

int Multiply(int a, int b)
{
	Log("Multiply");
	return a * b;
}
{% endhighlight %}

After compilation, we find the <span style="background-color: #0d0d0d"><font face="Monaco" color='#87CEFA'> &thinsp; Log &thinsp;</font></span> function is just move "message" to eax register.

And move to <span style="background-color: #0d0d0d"><font face="Monaco" color='#87CEFA'> &thinsp; Multiply &thinsp;</font></span> funtion, we find what it do, is just to <b>call</b> <span style="background-color: #0d0d0d"><font face="Monaco" color='#87CEFA'> &thinsp; Log &thinsp;</font></span> function before multiplication.

<p align="center">
    <img src="/post_image/cpp/Log_Multiply.png" width="60%">
</p>

Here, we notice the "Log" function in asm file is decorated with a string of complex charcters and signs. That is actually the <b>function signature</b>, which is used to uniquely define our funtion, we will talk more about it in the following blog about "Linking".