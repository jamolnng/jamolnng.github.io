---
layout: post
title: CUDA Tutorial 2 - CUDA Load Image
tags: [cuda, tutorials, cpp]
---

In this tutorial we are going to copy an image using CUDA.

Before we start make sure you download the PNG.h header available in the include and includes on the project's [GitHub page](https://github.com/jamolnng/OpenCL-CUDA-Tutorials) it in your project. We use this to load and save PNGs.
<br/><br/>
<h4>1) Create our input and output images</h4>

Loading our image is pretty simple. Lenna.png is available in the project files on the project's GitHub page.

The image we will be using is
![Lenna](https://raw.githubusercontent.com/jamolnng/OpenCL-CUDA-Tutorials/master/OpenCL/Tutorial%202%20-%20OpenCL%20load%20image/Lenna.png)

{% highlight cpp %}
PNG inPng("Lenna.png");
PNG outPng;
outPng.Create(inPng.w, inPng.h);

//store width and height so we can use them for our output image later
const unsigned int w = inPng.w;
const unsigned int h = inPng.h;
//4 because there are 4 color channels R, G, B, and A
int size = w * h * 4;
{% endhighlight %}
<br/><br/>
<h4>2) Create our CUDA buffers for the input and output images</h4>

{% highlight cpp %}
unsigned char *in = 0;
unsigned char *out = 0;
// Allocate GPU buffers for the images
cudaMalloc((void**)&in, size * sizeof(unsigned char));
cudaMalloc((void**)&out, size * sizeof(unsigned char));
	
// Copy image data from host memory to GPU buffers.
cudaMemcpy(in, &inPng.data[0], size * sizeof(unsigned char), cudaMemcpyHostToDevice);

//free the input image because we do not need it anymore
inPng.Free();
{% endhighlight %}
<br/><br/>
<h4>3) Run our kernel</h4>

{% highlight cpp %}
// Launch a kernel on the GPU with one thread for each element.
copy<<<w, h>>>(in, out);
//wait for the kernel to finish
cudaDeviceSynchronize();
{% endhighlight %}

The kernel code for this is not complex but may be confusing for if you haven't worked with images before

{% highlight cpp %}
__global__ void copy(const unsigned char* in, unsigned char* out)
{
	int x = blockIdx.x;
	int y = threadIdx.x;
	int width = blockDim.x;
	int index = (x + y * width) * 4;

	//copy each color channel
	out[index] = in[index];
	out[index + 1] = in[index + 1];
	out[index + 2] = in[index + 2];
	out[index + 3] = in[index + 3];
}
{% endhighlight %}
<br/><br/>
<h4>4) Copy the output data into the output png and save it to file</h4>

{% highlight cpp %}
//temporary array to store the result from opencl
auto tmp = new unsigned char[w * h * 4];
// Copy output vector from GPU buffer to host memory.
cudaStatus = cudaMemcpy(tmp, out, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
cudaFree(in);
cudaFree(out);

//copy the data from the temp array to the png
std::copy(&tmp[0], &tmp[w * h * 4], std::back_inserter(outPng.data));

//write the image to file
outPng.Save("cuda_tutorial_2.png");
//free the iamge's resources since we are done with it
outPng.Free();

//free the temp array
delete[] tmp;
{% endhighlight %}

Sources for all of the tutorials can be found at the projects github [https://github.com/jamolnng/OpenCL-CUDA-Tutorials](https://github.com/jamolnng/OpenCL-CUDA-Tutorials)