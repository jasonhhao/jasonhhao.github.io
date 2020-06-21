---
layout: post
title:  Attention机制
date:   2020-06-20 00:00:00 +0800
categories: 自然语言处理
tag: Attention机制
---

* content
{:toc}


<h2 align="center">宏观理解</h2>

在RNN或者在seq2seq模型中，有一个中间向量context来描述encoding的所有含义，然后再通过decoding解码。这样会产生两个问题，我们叫bottleneck problem：
1. 中间向量的生成如果一旦有问题，那么整个后面的解码过程都完蛋了。
2. 无论输入是10个词还是100个词，我们没办法保证context vector都可以学到一个很好的描述。

那么我们能不能按照人类的思维模式来改进？比如当我们翻译today这个词的时候，我们不会去过度关注后面的所有词，把更多的精力放在today这个词就好 -- attention is all we need


<h2 align="center">微观分析</h2>

网上找到这个图可以很好的展示我们要做的事情：

<p align="center"> 
  <img src="/imgs/attention/1.png">
</p>

从<start>开始我们计算它的隐向量和输入的每个隐向量之间的内积（相似度）。得到一个attention weight之后要做一个之和为1的归一化。然后通过attention weight
得到一个只服务于当前的context verctor。例如C1 = 0.5*h1 + 0.3*h2 + 0.1*h3 + 0.1*h4。随后我们就可以通过C1和ht得到第一个答案y1 = f([C1, ht])，再通过一个softmax来选择可能性最大的词“我”。
之后进行第二轮把“我”和前一个hidden unit ht来计算生成ht+1，再和每个输入进行一个attention weight计算。



















