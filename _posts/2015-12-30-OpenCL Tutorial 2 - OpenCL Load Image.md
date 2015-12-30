---
layout: post
title: OpenCL Tutorial 2 - OpenCL Load Image
tags: [opencl, tutorials, cpp]
---

In this tutorial we are going to copy an image using OpenCL.

Before we start make sure you download the PNG.h header available in the include and includes on the project's [GitHub page](https://github.com/jamolnng/OpenCL-CUDA-Tutorials) it in your project. We use this to load and save PNGs.

1) We need to create the OpenCL context. This is explained a bit more in depth in the first tutorial.

{% highlight cpp %}
int main(int arg, char* args[])
{
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0)
	{
		std::cout << "No OpenCL platforms found" << std::endl;//This means you do not have an OpenCL compatible platform on your system.
		exit(1);
	}
	std::vector<cl::Device> devices;
	platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
	cl::Device device = devices[0];
	std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
	std::cout << "Using platform: " << platforms[0].getInfo<CL_PLATFORM_NAME>() << std::endl;
	cl::Context context(device);
	//...
{% endhighlight %}

2) Now we need to load our image from file and into OpenCL

Loading our image is pretty simple. Lenna.png is available in the project files on the project's GitHub page.

The image we will be using is
![Lenna](https://raw.githubusercontent.com/jamolnng/OpenCL-CUDA-Tutorials/master/OpenCL/Tutorial%202%20-%20OpenCL%20load%20image/Lenna.png)

{% highlight cpp %}
PNG inPng("Lenna.png");

//store width and height so we can use them for our output image later
const unsigned int w = inPng.w;
const unsigned int h = inPng.h;
{% endhighlight %}

Now we have to create the OpenCL image from our loaded image.

{% highlight cpp %}
//input image
const cl::ImageFormat format(CL_RGBA, CL_UNSIGNED_INT8);
cl::Image2D in(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, format, w, h, 0, &inPng.data[0]);

//we are done with the image so free up its memory
inPng.Free();
{% endhighlight %}

format describes how the image is formatted in memory
CL_MEM_READ_ONLY says that the memory will not be modified by the OpenCL implementation
CL_MEM_COPY_HOST_PTR is set so OpenCL copies the image data from the host to the OpenCL implementation

3) Create out output image

This is similar to creating the input image except we do not need to send any data to the OpenCL implementation

{% highlight cpp %}
//output image
cl::Image2D out(context, CL_MEM_WRITE_ONLY, format, w, h, 0, NULL);
{% endhighlight %}

4) Create out kernel, this is explained in the first tutorial

{% highlight cpp %}
cl::Program::Sources sources;
std::string kernel_code = readFile("cl_tutorial_2_copy.cl");
//Add your program source
sources.push_back({ kernel_code.c_str(),kernel_code.length() });

//Create your OpenCL program and build it.
cl::Program program(context, sources);
if (program.build({ device }) != CL_SUCCESS)
{
	std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;//print the build log to find any issues with your source
	exit(1);//Quit if your program doesn't compile
}

//set the kernel arguments
cl::Kernel kernelCopy(program, "copy");
kernelCopy.setArg(0, in);
kernelCopy.setArg(1, out);
{% endhighlight %}

5) Create command queue and execute our kernel

{% highlight cpp %}
//create command queue
cl::CommandQueue queue(context, device, 0, NULL);

//execute kernel
//have a two dimensional global range of the width and height of our image so we can go through all of the pixels of the image
queue.enqueueNDRangeKernel(kernelCopy, cl::NullRange, cl::NDRange(w, h), cl::NullRange);

//wait for kernel to finish
queue.finish();
{% endhighlight %}

The kernel code for this is pretty simple, we just take the colors at each pixel and copy it to the output image

{% highlight cpp %}
const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

void kernel copy(__read_only image2d_t in, __write_only image2d_t out)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int2 pos = (int2)(x, y);
	uint4 pixel = read_imageui(in, smp, pos);
	write_imageui(out, pos, pixel);
}
{% endhighlight %}

6) Read our image back from the OpenCL implementation

{% highlight cpp %}
//start and end coordinates for reading our image (I really do not like how the c++ wrapper does this)
cl::size_t<3> origin;
cl::size_t<3> size;
origin[0] = 0;
origin[1] = 0;
origin[2] = 0;
size[0] = w;
size[1] = h;
size[2] = 1;

//output png
PNG outPng;
//create the image with the same width and height as original
outPng.Create(w, h);

//temporary array to store the result from opencl
auto tmp = new unsigned char[w * h * 4];
//CL_TRUE means that it waits for the entire image to be copied before continuing
queue.enqueueReadImage(out, CL_TRUE, origin, size, 0, 0, tmp);

//copy the data from the temp array to the png
std::copy(&tmp[0], &tmp[w * h * 4], std::back_inserter(outPng.data));

//write the image to file
outPng.Save("cl_tutorial_2.png");
//free the iamge's resources since we are done with it
outPng.Free();

//free the temp array
delete[] tmp;
{% endhighlight %}


















Sources for all of the tutorials can be found at the projects github [https://github.com/jamolnng/OpenCL-CUDA-Tutorials](https://github.com/jamolnng/OpenCL-CUDA-Tutorials)