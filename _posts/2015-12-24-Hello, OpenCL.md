---
layout: post
title: Hello, OpenCL
desc: 
---

This is the beginning of what I plan to be a tutorial/introductory series to OpenCL using the C++ bindings.

What is OpenCL?
---------------

OpenCL (Open Computing Language) is an "open standard for parallel programming of heterogeneous systems." Itself is not the code to do parallel programming but the set of rules on how such code should be implemented on standard complient devices. These devices can be your standard CPUs and GPUs or be something like a FPGA or other hardware you might not think of. As long as they are standard complient they are guaranteed to run your OpenCL code with certain error bounds. OpenCL is maintained by the [Khronos Group](https://www.khronos.org/) who also maintains OpenGL, WebGL, WebCL (the web version of OpenCL), and many other standards like Vulkan, the upcoming replacement to OpenGL.

{% include twitter_plug.html %}