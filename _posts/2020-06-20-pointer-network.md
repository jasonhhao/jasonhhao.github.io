---
layout: post
title:  指针网络Pointer Network
date:   2020-06-20 00:00:00 +0800
categories: 自然语言处理
tag: 指针网络
---

* content
{:toc}


<h2 align="center">宏观理解</h2>

在seq2seq模型中，我们在时刻t的输出结果，比如机器翻译中的某个单词，是从词库中｜v｜个单词中产生的，完全不依赖于输入，所以seq2seq的输出是一种已知给定的fixed输出，
致使它不能被应用在一些组合优化问题上，例如凸包问题，旅行商问题这种输出要完全依赖于输入的。
例如凸包问题可以被描述为：一个平面上有n个点，我们怎么连接某些点使所有的其余点都被包含到？

<p align="center"> 
  <img src="/imgs/pointernetwork/1.png">
</p>

我们想要的输出答案不可能会超过输入的选项，指针网络就是为了解决这种组合优化的问题。使输出的范围=输入的范围，输出完全依赖于输入，使输出有一个动态的变化。


<h2 align="center">宏观理解</h2>

指针网络很适合于做文本总结，因为文本总结做的事情其实可以理解为选择问题。在生成新的单词的时候，我们可以用seq2seq；但是在生成特有的名词或者某个行业内的专有名词的时候，词库里是未必会有的，这个
时候就需要pointer network来替我们选择这些行业内的名词。

我们之前的attention中在计算完attention weight之后还要过一个归一化，但是指针网络这里是完全没必要的，我们只是要找一个当前最大的可能就好，所以指针网络很简单。


<p align="center"> 
  <img src="/imgs/pointernetwork/2.png">
</p>

如果我们需要按照之前提到的让它和seq2seq模型做一个融合，可以参考这个流程


<p align="center"> 
  <img src="/imgs/pointernetwork/3.png">
</p>


指针网络和其他的模型各生成一个概率分布，然后通过加权平均整合成一个final distrubution。就有了最终的答案。











