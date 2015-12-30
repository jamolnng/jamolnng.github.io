---
layout: page
title: OpenCL
---

{% if site.tags.OpenCL != empty %}
<div class="posts">
  {% for post in site.tags.OpenCL %}
  <article class="post">
    <h1 class="post-title">
      <a href="{{ site.baseurl }}{{ post.url }}">
        {{ post.title }}
      </a>
    </h1>

    <time datetime="{{ post.date | date_to_xmlschema }}" class="post-date">{{ post.date | date_to_string }}</time>

    {{ post.content }}
  </article>
  {% endfor %}
</div>
{% endif %}