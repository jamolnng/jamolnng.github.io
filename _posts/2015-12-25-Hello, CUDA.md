---
layout: post
title: Hello, CUDA
desc: 
---

This is the beginning of what I plan to be a tutorial/introductory series to CUDA

What is CUDA?
---------------

CUDA (Compute Unified Device Architecture) is a platform and API for parallel programming. Unlike OpenCL, CUDA was created by and is currently maintained by NVIDIA. CUDA is only available on NVIDIA devices. These devices are NVIDIA's line of GPUs, such as the GeForce, Quadro, and Tesla series. CUDA is designed to work directly with C, C++, and Fortran.

Why use CUDA?
---------------

CUDA is fast, especially using it on high end NVIDIA GPUs. It can run massively parallel processes in fractions of the time that your normal CPU can, even high end ones. The stipulation is that these processes can have little or no shared memory between them otherwise if you choose to do this it can drastically slow down the processing speed waiting for other processes to complete. Some examples of using CUDA or CUDA like things, such as OpenCL, are image processing, self-learning systems and even some video games use CUDA for things like cloth and smoke physics.


If you want to use CUDA you need to make sure that the algorithm(s) that you have are inherently able to be parallel, if not then you will not see much, if any, performance gain from CUDA.

<br/><br/>
<br/><br/>

Tutorial 1: Hello, CUDA
-------------------------

This is going to be my version of an introductory application that utilizes CUDA. The application is going to take two arrays of numbers and add them together and output them to another array.

Before I continue I should tell you what system I am using and how I am compiling my code. I have an Intel i7-4790k processor and a Geforce GTX 980ti. I will be using the GPU on my 980ti to run all of the CUDA code. I am compiling and testing my code with NVIDIA's CUDA version 7.5. I am also assuming that you have a basic knowledge of C++. All of the headers and library files for CUDA can be retrieved from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads).

First we need to take care of our includes. The main one being the CUDA headers. The rest are just for utility.

{% highlight cpp %}
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
{% endhighlight %}

Unlike in OpenCL, CUDA programs can be written directly into your code as C++ code. This makes it so you don't have to do as much file handling.

So first we need to create our 3 arrays. The two we are going to add together and then the one the output will be written to.

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

Next we need to get the device that our CUDA program is going to run on. Unlike OpenCL there are no CUDA contexts that you have to worry about.

{% highlight cpp %}
// Choose which GPU to run on, change this on a multi-GPU system.
cudaStatus = cudaSetDevice(0);
if (cudaStatus != cudaSuccess)
{
	std::cout << "No CUDA devices found!" << std::endl;
	exit(1);
}
{% endhighlight %}

Now we need to create buffer pointers to store our arrays on our device. Also we create a cudaError_t to do help with a little error checking.

{% highlight cpp %}
int *buffer_A = 0;
int *buffer_B = 0;
int *buffer_C = 0;
cudaError_t cudaStatus;
{% endhighlight %}

Now because I want to have this code be as similar as possible to my OpenCL tutorials we get the CUDA device properties and print the name of the device we are using

{% highlight cpp %}
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

std::cout << "Using device: " << prop.name << std::endl;
{% endhighlight %}

Next we allocate the buffers on the GPU and then copy the data from our two input buffers into the input buffers on the GPU.

{% highlight cpp %}
// Allocate GPU buffers for three vectors (two input, one output).
cudaMalloc((void**)&buffer_A, size * sizeof(int));
cudaMalloc((void**)&buffer_B, size * sizeof(int));
cudaMalloc((void**)&buffer_C, size * sizeof(int));

// Copy input vectors from host memory to GPU buffers.
cudaMemcpy(buffer_A, A, size * sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(buffer_B, B, size * sizeof(int), cudaMemcpyHostToDevice);
{% endhighlight %}

We are finally ready to launch our kernel. The kernel code is very similar to that of my first OpenCL tutorial. One major difference is that the kernel code is actually included in the source file and looks pretty much like a normal function. I have my kernel declared above my main function as so:

{% highlight cpp %}
__global__ void simple_add(const int *A, const int *B, int *C)
{
	C[threadIdx.x] = A[threadIdx.x] + B[threadIdx.x];
}

int main(int arg, char* args[])
{
	...
{% endhighlight %}

Now we are ready to launch our kernel. It is very straight forward and looks like just calling your standard C++ function. The major noticable difference is that you have to supply the pointers to you device buffers and the <<<...>>> before the function parameters. Between the <<<...>>> are the number of blocks and the number of threads in each block. We only have one block and we want a thread for each index of our arrays.

After we laucnh our kernel we check for errors in the launch. If there are errors we free the GPU memory and quit the program.

{% highlight cpp %}
// Launch a kernel on the GPU with one thread for each element.
simple_add<<<1, size>>>(buffer_A, buffer_B, buffer_C);

// Check for any errors launching the kernel
cudaStatus = cudaGetLastError();
if (cudaStatus != cudaSuccess)
{
	std::cout << "Kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
	cudaFree(buffer_A);
	cudaFree(buffer_B);
	cudaFree(buffer_C);
	exit(1);
}
{% endhighlight %}

Now we need to synchronize the GPU so we can get the final result of the kernel. If there are errors with synching we free the GPU memory and quit the program.

{% highlight cpp %}
// cudaDeviceSynchronize waits for the kernel to finish, and returns
// any errors encountered during the launch.
cudaStatus = cudaDeviceSynchronize();
if (cudaStatus != cudaSuccess)
{
	std::cout << "Could not synchronize device!" << std::endl;
	cudaFree(buffer_A);
	cudaFree(buffer_B);
	cudaFree(buffer_C);
	exit(1);
}
{% endhighlight %}

Finally we retrieve the output from our kernel. After we do that we free the GPU memory and then print the results.

{% highlight cpp %}
cudaStatus = cudaMemcpy(C, buffer_C, size * sizeof(int), cudaMemcpyDeviceToHost);
cudaFree(buffer_A);
cudaFree(buffer_B);
cudaFree(buffer_C);

if(cudaStatus != cudaSuccess)
{
	std::cout << "Could not copy buffer memory to host!" << std::endl;
	exit(1);
}

//Prints the array
std::cout << "Result:" << std::endl;
for (int i = 0; i < size; i++)
{
	std::cout << A[i] << " + " << B[i] << " = " << C[i] << std::endl;
}
{% endhighlight %}

One last thing we need to do before we are done is reset our device.

{% highlight cpp %}
// cudaDeviceReset must be called before exiting in order for profiling and
// tracing tools such as Nsight and Visual Profiler to show complete traces.
cudaStatus = cudaDeviceReset();
if (cudaStatus != cudaSuccess)
{
	std::cout << "Device reset failed!" << std::endl;
	exit(1);
}
{% endhighlight %}

This is your basic "Hello, World!" program for CUDA. I tried to explain as much as I can but I am also just starting with CUDA.

Sources for all of the tutorials can be found at the projects github [https://github.com/jamolnng/OpenCL-CUDA-Tutorials](https://github.com/jamolnng/OpenCL-CUDA-Tutorials)

{% include twitter_plug.html %}