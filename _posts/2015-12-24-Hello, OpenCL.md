---
layout: post
title: Hello, OpenCL
tags: [opencl, tutorials, cpp]
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

Before I continue I should tell you what system I am using and how I am compiling my code. I have an Intel i7-4790k processor and a Geforce GTX 980ti. I will be using the GPU on my 980ti to run all of the OpenCL code. I am compiling and testing my code with both Microsoft Visual Studio 2015 Community and MinGW's g++. I am also assuming that you have a basic knowledge of C++. All of the headers and library files for OpenCL can be retrieved from your GPU/CPU vendor.

First we need to take care of our includes. The main one being the OpenCL C++ header CL/cl.hpp. The rest are just for utility.

{% highlight cpp %}
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <iterator>
{% endhighlight %}

As you may have noticed I also defined something above the OpenCL header. __CL_ENABLE_EXCEPTIONS is used to produce C++ execptions for OpenCL. This is useful because it will usually crash your program when you do something stupid instead of just having a log that you may or may not look at.

<br/><br/>

Now before we get started I am going to create a function to read a file to a string so that we can load OpenCL programs from files. This will come in handy when you want to easily have more than one OpenCL kernel.

{% highlight cpp %}
std::string readFile(std::string fileName)
{
	std::ifstream t(fileName);
	std::string str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
	return str;
}
{% endhighlight %}

Now on to our main code.

First we need to create our 3 arrays. The two we are going to add together and then the one the output will be written to.

{% highlight cpp %}
int main(int arg, char* args[])
{
	const int size = 10;
	int A[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	int B[] = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 0 };
	int C[size];
	return 0;
}
{% endhighlight %}

Now we need to create an OpenCL context to run our programs in. This is a straight forward and simple way to do it.

{% highlight cpp %}
//stl vector to store all of the available platforms
std::vector<cl::Platform> platforms;
//get all available platforms
cl::Platform::get(&platforms);

if (platforms.size() == 0)
{
	std::cout << "No OpenCL platforms found" << std::endl;//This means you do not have an OpenCL compatible platform on your system.
	exit(1);
}

//Create a stl vector to store all of the availbe devices to use from the first platform.
std::vector<cl::Device> devices;
//Get the available devices from the platform. For me the platform for my 980ti is actually th e second in the platform list but for simplicity we will use the first one.
platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
//Set the device to the first device in the platform. You can have more than one device associated with a single platform, for instance if you had two of the same GPUs on your system in SLI or CrossFire.
cl::Device device = devices[0];

//This is just helpful to see what device and platform you are using.
std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
std::cout << "Using platform: " << platforms[0].getInfo<CL_PLATFORM_NAME>() << std::endl;

//Finally create the OpenCL context from the device you have chosen.
cl::Context context(device);
{% endhighlight %}

Next you need to create OpenCL buffers to store your arrays that you want to add on your OpenCL device. This code creates a buffer for each array you are adding and the output array.

{% highlight cpp %}
cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * size);
cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * size);
cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int) * size);
{% endhighlight %}

With your context and buffers created you now need to create your OpenCL program. The source for the program itself is simmple, it just takes two input buffers and adds each index together with the other buffer and stores it to an output.

{% highlight cpp %}
//A source object for your program
cl::Program::Sources sources;
std::string kernel_code = readFile("simple_add.cl");
//Add your program source
sources.push_back({ kernel_code.c_str(),kernel_code.length() });

//Create your OpenCL program and build it.
cl::Program program(context, sources);
if (program.build({ device }) != CL_SUCCESS)
{
	std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;//print the build log to find any issues with your source
	exit(1);//Quit if your program doesn't compile
}
{% endhighlight %}

The source for our OpenCL program is:

{% highlight cpp %}
//simple_add.cl
void kernel simple_add(global const int* A, global const int* B, global int* C)
{
	C[get_global_id(0)] = A[get_global_id(0)] + B[get_global_id(0)];
}
{% endhighlight %}

After we compile out program we need to create a command queue to queue our task to run on the device we have selected. A command queue is pretty much exactly how it sounds, it sequentially executes commands on the OpenCL device. If you need multiple programs to run at one time you can have multiple command queues. For this we only need one.

{% highlight cpp %}
//Create command queue using our OpenCL context and device
cl::CommandQueue queue(context, device, NULL, NULL);

//Write our buffers that we are adding to our OpenCL device
queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int) * size, A);
queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int) * size, B);

//Create our Kernel (basically what is the starting point for our OpenCL program)
cl::Kernel simple_add(program, "simple_add");
//Set our arguements for the kernel
simple_add.setArg(0, buffer_A);
simple_add.setArg(1, buffer_B);
simple_add.setArg(2, buffer_C);

//Make sure that our queue is done with all of its tasks before continuing
queue.finish();
{% endhighlight %}

Finally we need to run our code and get the output

{% highlight cpp %}
//Create an event that we can use to wait for our program to finish running
cl::Event e;
//This runs our program, the ranges here are the offset, global, local ranges that our code runs in.
queue.enqueueNDRangeKernel(simple_add, cl::NullRange, cl::NDRange(size), cl::NullRange, 0, &e);

//Waits for our program to finish
e.wait();
//Reads the output written to our buffer into our final array
queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int) * size, C);

//prints the array
std::cout << "Result:" << std::endl;
for (int i = 0; i < size; i++)
{
	std::cout << A[i] << " + " << B[i] << " = " << C[i] << std::endl;
}
{% endhighlight %}

This is your basic "Hello, World!" program for OpenCL. I tried to explain as much as I can but I am also just starting with OpenCL.

Sources for all of the tutorials can be found at the projects github [https://github.com/jamolnng/OpenCL-CUDA-Tutorials](https://github.com/jamolnng/OpenCL-CUDA-Tutorials)