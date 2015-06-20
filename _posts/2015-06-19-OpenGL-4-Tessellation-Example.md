---
layout: post
title: OpenGL 4 Tessellation Example
---

With a new, and not piece of shit, computer I decided it would be a good time to start moving into modern OpenGL. I bought the [OpenGL SuperBible Sixth Edition](http://www.openglsuperbible.com/) last winter and hadn't really had a chance to read it until I got my new computer. I am basically throwing everything I know about OpenGL out the window because all of it relates to OpenGL 2.1 which is pretty outdated at this point. One cool feature I found, and why I decided to write this blog about me actually doing something, is the tessellation engine that comes with OpenGL 4. I decided to start learning shaders and came up with this cool example of the OpenGL tessellation engine.

I am nowhere near an expert with OpenGL so here is an armature's take on tessellation.

testasdjklaskj;lasfdklasdkl
{% highlight glsl %}
#version 430 core

layout (location = 0) in vec3 vertex;
layout (location = 1) in vec4 col;
uniform mat4 mvp;
uniform float time;
uniform float uc;

out VS_OUT
{
	vec4 colors;
	float time;
	mat4 mvp;
} vs_out;

void main(void)
{
	vs_out.time = time;
	vs_out.colors = col;
	vs_out.mvp = mvp;
	gl_Position = vec4(vertex, 1);
	if(gl_VertexID < 3) vs_out.colors = vec4(1, 0, 1, 1);
	else if(gl_VertexID < 6) vs_out.colors = vec4(1, 1, 0, 1);
	else if(gl_VertexID < 9 && gl_VertexID > 5) vs_out.colors = vec4(1, 0, 0, 1);
	else if(gl_VertexID < 12) vs_out.colors = vec4(1, 1, 0, 1);
	else if(gl_VertexID < 15) vs_out.colors = vec4(1, 0, 1, 1);
	else if(gl_VertexID < 21 && gl_VertexID > 17) vs_out.colors = vec4(0, 1, 0, 1);
	else if(gl_VertexID < 21 && gl_VertexID > 14) vs_out.colors = vec4(1, 0, 0, 1);
	else if(gl_VertexID < 27) vs_out.colors = vec4(0, 1, 1, 1);
	else if(gl_VertexID < 33) vs_out.colors = vec4(1, 1, 1, 1);
	else vs_out.colors = vec4(0, 1, 0, 1);
	
	if(uc < 1.0)
	{
		vs_out.colors = vec4(0, 0, 0, 1);
	}
}
{% endhighlight %}

[Video](https://youtu.be/sOfyvtNvlJ8)

<iframe width="560" height="315" src="https://www.youtube.com/embed/sOfyvtNvlJ8" frameborder="0" allowfullscreen></iframe>

I cannot recommend the [OpenGL SuperBible Sixth Edition](http://www.openglsuperbible.com/) enough for someone who has a lot of programming knowledge but has no idea how to get started into computer graphics. And while Googling may help a bit most of the stuff you will find is outdated or just flat out wrong. Also the OpenGL SuperBible was written by people who have worked with OpenGL and the hardware it runs on.

{% include twitter_plug.html %}