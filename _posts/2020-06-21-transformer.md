---
layout: post
title:  Transformer
date:   2020-06-21 00:00:00 +0800
categories: 自然语言处理
tag: Transformer
---

* content
{:toc}


<h2 align="center">宏观理解</h2>

Transformer是当今nlp领域最伟大的发明了，几乎成了如今自然语言处理的默认模型和代名词。后来的bert系列, GPT系列等都是依赖于transformer的架构拼接而成的。算得上是nlp领域的一次里程碑了。
transformer的主旨在抛弃RNN，只用attention可以让模型更加有效率也更容易被拓展 - attention is all you need。

<h2 align="center">微观分析</h2>

transformer也是沿用了encoder - decoder的架构：

<p align="center"> 
  <img src="/imgs/transformer/1.png">
</p>

我们可以发现整个结构新颖的地方就是一个叫self attention的block，这就是大佬们推出这个模型的关键点，搞明白了self attention也就搞明白了transformer。
但是在self attention中还隐藏着两个重要的trick，一个是multi-head，一个是mask。人们叫做multi-head self attention和masked multi-head self attention。
接下来分别看一看。

<h4>self attention</h4>

首先我们把所有的输入向量给到模型，比如图中的x1和x2，那么对应的也就有了两个输出y1和y2。像之前说的attention一样，我们的每一个y，是对所有x的一个加权平均。可以表示为
![](https://latex.codecogs.com/gif.latex?y_i%20%3D%20%5Csum_%7Bj%7D%5E%7B%7Dw_%7Bij%7Dx_j)其中
















