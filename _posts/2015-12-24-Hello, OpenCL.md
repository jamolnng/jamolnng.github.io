---
layout: post
title: Hello, OpenCL
desc: 
---

This is the beginning of what I plan to be a tutorial/introductory series to OpenCL using the C++ bindings.

What is OpenCL?
---------------

OpenCL (Open Computing Language) is an "open standard for parallel programming of heterogeneous systems." Itself is not the code to do parallel programming but the set of rules on how such code should be implemented on standard complient devices. These devices can be your standard CPUs and GPUs or be something like a FPGA or other hardware you might not think of. As long as they are standard complient they are guaranteed to run your OpenCL code with certain error bounds. OpenCL is maintained by the [Khronos Group](https://www.khronos.org/) who also maintains OpenGL, WebGL, WebCL (the web version of OpenCL), and many other standards like Vulkan, the upcoming replacement to OpenGL.

Why use OpenCL?
---------------

OpenCL is fast, especially using it on specifically designed hardware like GPUs. It can run massively parallel processes in fractions of the time that your normal CPU can, even high end ones. The stipulation is that these processes can have little or no shared memory between them otherwise if you choose to do this it can drastically slow down the processing speed waiting for other processes to complete. Some examples of using OpenCL or OpenCL like things, such as NVIDIA's CUDA, are image processing, self-learning systems and even some video games use OpenCL for things like cloth and smoke physics.


If you want to use OpenCL you need to make sure that the algorithm(s) that you have are inherently able to be parallel, if not then you will not see much, if any, performance gain from OpenCL.

<br/><br/>
<br/><br/>

Tutorial 1: Hello, OpenCL
-------------------------

This is going to be my version of an introductory application that utilizes OpenCL. The application is going to take two arrays of numbers and add them together and output them to another array.

Before I continue I should tell you what system I am using and how I am compiling my code. I have an Intel i7-4790k processor and a Geforce GTX 980ti. I will be using the GPU on my 980ti to run all of the OpenCL code. I am compiling and testing my code with both Microsoft Visual Studio 2015 Community and MinGW's g++.

{% highligh cpp %}
int main(int narg, char* argv[])
{
	return 0;
}
{% endhighlight %}

{% include twitter_plug.html %}