---
layout: page
title: OpenCL
---

{% if site.tags.{{ page.title }} != empty %}
<div class="posts">
  {% for post in site.categories["OpenCL"] %}
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