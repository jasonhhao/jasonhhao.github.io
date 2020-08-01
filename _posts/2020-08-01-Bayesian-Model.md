---
layout: post
title:  贝叶斯模型Bayesian Model
date:   2020-08-01 00:00:00 +0800
categories: 自然语言处理
tag: 贝叶斯模型
---

* content
{:toc}


<h1 align="center">宏观理解</h1>

为什么贝叶斯模型在各种领域里地位很重要？

1. 因为LDA模型在主题模型里面起到非常重要的作用。LDA模型是无监督学习模型，可以当作是Mix-membership models的一种，作用是如果给定一篇文章属于多个主题，它可以生成每个主题的概率
分布，而如果用朴素贝叶斯则只能生成一个主题类的预测，也可以叫做Uni-membership model。

2. 用于小数据量的学习问题。因为小数据量很容易造成过拟合，那解决的方法可以是集成模型，而贝叶斯模型本身也是一个集成模型。而和传统集成模型比如随机森林集成有限个模型不同的是，它集成的是无限多个模型。
但是在数据量比较大的时候贝叶斯模型学起来会很慢。

3. 把不确定性融合在模型本身。

4. 把先验融合到模型中。

5. 模型压缩。



<h1 align="center">微观分析</h1>

贝叶斯是一个很大的领域，所以我们按以下的顺序进行剖析：
1. 什么是贝叶斯？


<h3>1. 什么是贝叶斯？</h3>

说到什么是贝叶斯，我们就要理清楚MLE、MAP和Bayesian的区别。这三个都是为了构造不同的目标函数的。当一个模型分别用这三种方式构造的话，形成的模型也是不同的：

<p align="center"> 
  <img src="/imgs/bayesianModel/1.png">
</p>