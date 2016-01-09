---
layout: post
title: CUDA image filtering
tags: [cuda, tutorials, cpp]
---

Load and filter an image using a simple box filter.

This builds on the last tutorial so I am only going to go over the kernel.

<h4>What is a box filter?</h4>

A box filter is a linear filter where it averages the value of a pixel with itself and it's neighboring pixels. Typically it is a 3x3 box where the pixel being modified is the one in the middle.

The method I implemented is very naive but there are other ways to speed it up. If you are interested here is [a presentation pdf](http://web.archive.org/web/20060718054020/http://www.acm.uiuc.edu/siggraph/workshops/wjarosz_convolution_2001.pdf) on it.

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

Then we can just simply get the value of the pixel at each offset and add that to the total

{% highlight cpp %}
int cx = x + i;
int cy = y + j;
if (cx >= 0 && cy >= 0 && cx < imageWidth && cy < imageHeight)//keeps the pixel in the image
{
	int adjIndex = (cx + cy * imageWidth) * 4;
	for (int c = 0; c < 4; c++)
	{
		total[c] += static_cast<unsigned int>(in[adjIndex + c]);
	}
	count++;
}
{% endhighlight %}

You can see that there is also another variable named count in there as well. This counts the number of pixels used for the filter (it changes on the edges). It is used when writing the final pixel value.

Finally we write the total value of each pixel divided by the number of them sampled to the output image.

{% highlight cpp %}
out[index]     = static_cast<unsigned char>(total[0] / count);
out[index + 1] = static_cast<unsigned char>(total[1] / count);
out[index + 2] = static_cast<unsigned char>(total[2] / count);
out[index + 3] = static_cast<unsigned char>(total[3] / count);
{% endhighlight %}

Here is the full kernel I am using:

{% highlight cpp %}
__global__ void boxFilter(const unsigned char* in, unsigned char* out, const int imageWidth, const int imageHeight, const int halfBoxWidth, const int halfBoxHeight)
{
	int x = blockIdx.x;
	int y = blockIdx.y;

	int count = 0;

	int index = (x + y * imageWidth) * 4;

	unsigned int total[4] = { 0, 0, 0, 0 };

	for (int i = -halfBoxWidth; i <= halfBoxWidth; i++)
	{
		for (int j = -halfBoxHeight; j <= halfBoxHeight; j++)
		{
			int cx = x + i;
			int cy = y + j;
			if (cx >= 0 && cy >= 0 && cx < imageWidth && cy < imageHeight)
			{
				int adjIndex = (cx + cy * imageWidth) * 4;
				for (int c = 0; c < 4; c++)
				{
					total[c] += static_cast<unsigned int>(in[adjIndex + c]);
				}
				count++;
			}
		}
	}

	out[index]     = static_cast<unsigned char>(total[0] / count);
	out[index + 1] = static_cast<unsigned char>(total[1] / count);
	out[index + 2] = static_cast<unsigned char>(total[2] / count);
	out[index + 3] = static_cast<unsigned char>(total[3] / count);
}
{% endhighlight %}

Sources for all of the tutorials can be found at the projects github [https://github.com/jamolnng/OpenCL-CUDA-Tutorials](https://github.com/jamolnng/OpenCL-CUDA-Tutorials)