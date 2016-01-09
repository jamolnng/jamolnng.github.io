---
layout: post
title: OpenCL image filtering
tags: [opencl, tutorials, cpp]
---

Load and filter an image using a simple box filter.

This builds on the last tutorial so I am only going to go over the kernel.

<h4>What is a box filter?</h4>

A box filter is a linear filter where it averages the value of a pixel with itself and it's neighboring pixels. Typically it is a 3x3 box where the pixel being modified is the one in the middle.

The method I implemented using the sliding window algorithm but there are other ways to speed it up. If you are interested here is [a presentation pdf](http://web.archive.org/web/20060718054020/http://www.acm.uiuc.edu/siggraph/workshops/wjarosz_convolution_2001.pdf) on it.

Here is an example using a 11x11 filter
![11x11 filter](http://jlaning.com/public/assets/OpenCL image filtering/filter.png)

To implement the filter we need to iterate over every pixel that is going to be read
{% highlight cpp %}
for(int i = -halfBoxWidth; i <= halfBoxWidth; i++)
{
	for(int j = -halfBoxHeight; j <= halfBoxHeight; j++)
	{
	}
}
{% endhighlight %}

Then we can just simply get the value of the pixel at each offset using built in functions

{% highlight cpp %}
int2 coord = pos + (int2)(i, j);
if(coord.x >= 0 && coord.y >= 0 && coord.x < imageWidth && coord.y < imageHeight)//This keeps the pixel inside the image, the image sampler is actually set up the handle this but this keeps it as close as possible to the CUDA version of this tutorial
{
	total += read_imageui(in, smp, pos + (int2)(i, j));
	count++;
}
{% endhighlight %}

You can see that there is also another variable named count in there as well. This counts the number of pixels used for the filter (it changes on the edges). It is used when writing the final pixel value.

Finally we write the total value of each pixel divided by the number of them sampled to the output image.

{% highlight cpp %}
write_imageui(out, pos, total / count);
{% endhighlight %}

Here is the full kernel I am using:

{% highlight cpp %}
const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

void kernel boxFilter(__read_only image2d_t in, __write_only image2d_t out, const int imageWidth, const int imageHeight, const int halfBoxWidth, const int halfBoxHeight)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int2 pos = (int2)(x, y);

	uint4 total = {0, 0, 0, 0};

	int count = 0;

	for(int i = -halfBoxWidth; i <= halfBoxWidth; i++)
	{
		for(int j = -halfBoxHeight; j <= halfBoxHeight; j++)
		{
			int2 coord = pos + (int2)(i, j);
			if(coord.x >= 0 && coord.y >= 0 && coord.x < imageWidth && coord.y < imageHeight)
			{
				total += read_imageui(in, smp, pos + (int2)(i, j));
				count++;
			}
		}
	}
	write_imageui(out, pos, total / count);
}
{% endhighlight %}

Sources for all of the tutorials can be found at the projects github [https://github.com/jamolnng/OpenCL-CUDA-Tutorials](https://github.com/jamolnng/OpenCL-CUDA-Tutorials)