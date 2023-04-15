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

<b>series C++ source files</b> <i>--Compiler--></i> <b>binary file</b>

Let's check it on Visual Studio (VS).

<!--more-->

Here is a very basic C++ program.

# Source file

The content of source file as follow:

{% highlight C++ linenos %}
#include <iostream>

int main()
{
	std::cout << "Hello World!" << std::endl;
	std::cin.get();
}
{% endhighlight %}

## Preprocessor Statement

"\#include \<iostream\>" called <b>Preprocessor Statement</b>. And the first thing the compiler does for the source file is preprocessing all the "Preporcessor statements" before compilation. 

<b>What "include" do</b> is find a file called "iostream" and take all of the contents of that file and just paste it into current file. And these file we include are typically called "header files".

Why we need to "include" the file called "iostream" is because we need a <b>"declaration"</b> for a function called "cout", which let us print stuff to our <b>"console"</b>. 

## main function

<b>Main function</b> is very important because every C++ program has something like that. The main function is the <b>entry point</b> for our application. When we run our application, the computer starts executing code begins in this funtion.

Actually, the return types of main is "int". However, we don't return a integer. That is because the main funtion is a special case. We don't have to return any kind of value from the main funtion. In fact it will assume the returning is 0 (only applies to main funtion).

## cout & get()

We can treat the operator "\<<" as a funtion. Here, what we do is pushing this "Hello World" string into this <span style="background-color: #0d0d0d"><font face="Monaco" color=#87CEFA>cout</font></span>, which will print the string to the console. Then we're pushing this "<font face="Monaco" color=green>endl</font>" (end line), which tell the console to advance to the next line.

The function "<font face="Monaco" color=green>cin.get()</font>" is just wait until we press ENTER before advancing to next code (here is nothing).

# Compilation

Again, the computer will just pcopy all the contents from the header files (mentioned in preprocessor statement) and paste it in current file. Once the preprocessor statement have been evaluated, our file gets compiled. Our compiler will transform all of this C++ code into machine code.

## Configuration

There are several important settings that determine how compilation happens. The following screen shot shows two important "drop down" menus, the left one's called "solution configuration" and the right one's called "solution platform". 

<p align="center">
    <img src="/post_image/cpp/debug_x86.png" width="80%">
</p>

A <b>configuration</b> is a set of rules which applies to the building of a project. And the "platform" is what platform we're targeting with our current compilation. (Note: x86 is exactly same to win32). We can right-click our projrct and then hit "Properties" to get the "Property Page" shown at the right part in following figure. In fact, the settings in the "Property Page" defines all the rules during compilation. The default settings of visual studio is pretty good, so we don't need to do anything.

<p align="center">
    <img src="/post_image/cpp/configuration_properties.png" width="80%">
</p>

We will notice that the "Debug" mode turns off the optimization, so it's slower than "release" mode.

## Obj & exe file

Each C++ file will get compiled, but header files do not get compiled. Because, header files get included via preprocessor statement. And every C++ file will get <b>compiled individually</b> into a <b>"Object file"</b> (named XX.obj in visual studio). Then the "Linker" will stitch (glue) these obj files together into one exe file.

We can hit "ctrl + F7" to compile these C++ files. OR customize a button called "Compile". If we "compile" the C++ file we will get an obj file in "Debug" folder, and if we "build" our prject, we will get an exe file in the same folder. And if we run exe file, we can also get the "Hello World!" in console.

<p align="center">
    <img src="/post_image/cpp/build_compile.png" width="80%">
    <br />    <small> tittle </small>
</p>

## Multiple C++ files

We will see what happen when we have multiple C++ file. Here we will give a simple example. In the following code, we use "<font face="Monaco" color=green>Log</font>" function to replace the "cout" funtion, in another words, "<font face="Monaco" color=green>Log</font>" function wrap the "cout" funtion. 

{% highlight C++ linenos %}
#include <iostream>

void Log(const char* message)
{
	std::cout << message << std::endl;
}

int main()
{
	Log("Hello World!");
	std::cin.get();
}
{% endhighlight %}

The "<font face="Monaco" color=green>Log</font>" funtion take a string (called message) and print the string to the console. Here we can simply treat the "<font face="Monaco" color=green>const char*</font>" as a kind of type that can hold a string of text. Then we want to move the "<font face="Monaco" color=green>Log</font>" funtion to a new C++ file (called Log.cpp). 

So the Log.cpp file like this

{% highlight C++ linenos %}
// Log.cpp

void Log(const char* message)


{
	std::cout << message << std::endl;
}

{% endhighlight %}

And the main.cpp file like this

{% highlight C++ linenos %}
//main.cpp

#include <iostream>

int main()
{
	Log("Hello World!");
	std::cin.get();
}
{% endhighlight %}

Actually, we will get a lot of errors after we compile it, because we don't write "declaration" in Log.cpp to declarate "<font face="Monaco" color=green>cout</font>" funtion.

If we copy the "\#include <iostream>" into Log.cpp file we will also get error "Log is not found". Why?? That's because compiler will compile each C++ file separately. So, when compiling the main.cpp file, the compiler didn't recognize what's 'Log' is?

There we need <b>"Declaration"</b> which can declare somthing called 'Log' exists. That is like a kind of promise, but actually the compiler doesn't care about where the <b>definition</b> of 'Log' is. TO sum up, the "Declaration" is tell compiler there exists a function called 'Log', and "Definition" is the function body. Then we rewrite the Log.cpp file...

{% highlight C++ linenos %}
// Log.cpp

#include <iostream>

void Log(const char* message)


{
	std::cout << message << std::endl;
}

{% endhighlight %}

and the main.cpp file!

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

Congratulations! We compile the code successfully! BUT!! WHY??

How compiler know there is a 'Log' funtion? The "Declaration" told him.

How compiler run the right code? The answer is the <b>Linker</b>! After we compile the two C++ file, the Linker will find the definition of 'Log' and wire it up to the main.cpp. If it can not find the definition, we will get a linking error!

In the end, we will find two obj files (Log.obj and main.cpp in "Debug" folder) and one exe in that folder!