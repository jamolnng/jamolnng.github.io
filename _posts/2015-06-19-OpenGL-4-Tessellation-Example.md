---
layout: post
title: OpenGL 4 Tessellation Example
tags: [opengl]
---

With a new, and not piece of shit, computer I decided it would be a good time to start moving into modern OpenGL. I bought the [OpenGL SuperBible Sixth Edition](http://www.openglsuperbible.com/) last winter and hadn't really had a chance to read it until I got my new computer. I am basically throwing everything I know about OpenGL out the window because all of it relates to OpenGL 2.1 which is pretty outdated at this point. One cool feature I found, and why I decided to write this blog about me actually doing something, is the tessellation engine that comes with OpenGL 4. I decided to start learning shaders and came up with this cool example of the OpenGL tessellation engine.

I am nowhere near an expert with OpenGL so here is an armature's take on tessellation. This is most likely not a good tutorial, I am just showing the basic shader code.

### So what is tessellation?
Tessellation is a stage in the OpenGL rendering pipeline that takes patches of vertex data and subdivides it into smaller primitives. Two shaders govern tessellation. The tessellation control shader governs how much tessellation to do while the tessellation evaluation shader takes the output of the tessellation control shader and computes the values for each vertex. Tessellation can be used for, but not limited to, level of detail or LOD in objects displayed on the screen. This means while a mountain in the background may not be tessellated at all, the closer you get the more tessellation that might be done so it doesn't look all jagged and rough.

In this shader code it uses an angle to rotate each vertex output by the tessellation engine to provide a smooth edge and keeps faces from overlapping

### Vertex Shader:
{% highlight glsl %}
#version 430 core

layout (location = 0) in vec3 vertex;
layout (location = 1) in vec4 col;
uniform mat4 mvp;
uniform float angle;

out VS_OUT
{
	vec4 colors;
	float angle;
	mat4 mvp;
} vs_out;

void main(void)
{
	vs_out.angle = angle;
	vs_out.colors = col;
	vs_out.mvp = mvp;
	gl_Position = vec4(vertex, 1);
}
{% endhighlight %}

### Tessellation Control Shader:
{% highlight glsl %}
#version 430 core

layout (vertices = 3) out;

in VS_OUT
{
	vec4 colors;
	float angle;
	mat4 mvp;
} cs_in[];
out CS_OUT
{
	vec4 colors;
	float angle;
	mat4 mvp;
} cs_out[];

void main(void)
{
	cs_out[gl_InvocationID].mvp = cs_in[gl_InvocationID].mvp;
	cs_out[gl_InvocationID].angle = cs_in[gl_InvocationID].angle;
	cs_out[gl_InvocationID].colors = cs_in[gl_InvocationID].colors;
	if (gl_InvocationID == 0)
	{
		//lets create a lot of vertices, this is essentially your LOD
		gl_TessLevelInner[0] = 1000.0;
		gl_TessLevelOuter[0] = 1000.0;
		gl_TessLevelOuter[1] = 1000.0;
		gl_TessLevelOuter[2] = 1000.0;
	}
	gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
}
{% endhighlight %}

### Tessellation Evaluation Shader:
{% highlight glsl %}
#version 430 core

layout (triangles, equal_spacing, ccw) in;

in CS_OUT
{
	vec4 colors;
	float angle;
	mat4 mvp;
} es_in;
out ES_OUT
{
	vec4 colors;
} es_out;

void main(void)
{
	es_out.colors = es_in.colors;
	gl_Position = (gl_TessCoord.x * gl_in[0].gl_Position +
				   gl_TessCoord.y * gl_in[1].gl_Position + 
				   gl_TessCoord.z * gl_in[2].gl_Position);
	vec3 v = gl_Position.xyz;
	//rotate each vertex based on its y coordinate, the closer to 0 the less rotation
	float s = sin(es_in.angle * v.y);
	float c = cos(es_in.angle * v.y);
	float xnew = v.x * c - v.z * s;
	float znew = v.x * s + v.z * c;
	//We apply the model view projection matrix here so we can rotate each
	//vertex output by the tessellation engine based on its y coordinate
	gl_Position = vec4(xnew, v.y, znew, 1) * es_in.mvp;
}
{% endhighlight %}

### Fragment Shader:
{% highlight glsl %}#version 430 core

out vec4 color;
in ES_OUT
{
	vec4 colors;
} fs_in;

void main(void)
{
	color = fs_in.colors;
}
{% endhighlight %}

[Video](https://youtu.be/sOfyvtNvlJ8)

<iframe width="560" height="315" src="https://www.youtube.com/embed/sOfyvtNvlJ8" frameborder="0" allowfullscreen></iframe>

I cannot recommend the [OpenGL SuperBible Sixth Edition](http://www.openglsuperbible.com/) enough for someone who has a lot of programming knowledge but has no idea how to get started into computer graphics. And while Googling may help a bit most of the stuff you will find is outdated or just flat out wrong. Also the OpenGL SuperBible was written by people who have worked with OpenGL and the hardware it runs on.

{% include twitter_plug.html %}