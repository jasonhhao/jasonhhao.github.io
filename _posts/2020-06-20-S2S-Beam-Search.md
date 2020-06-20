---
layout: post
title:  Seq2Seq模型和Beam Search
date:   2020-06-20 00:00:00 +0800
categories: 自然语言处理
tag: Seq2Seq
---

* content
{:toc}


<h2 align="center">宏观理解</h2>

Seq2Seq由前后两个RNN拼接而成，一个叫encoder一个叫decoder。encoder负责把输入输进模型之后生成一个context向量，包含所有文本信息。decoder负责把context向量解码成我们想要的结果。
好处是它解决了输入定长的问题，我们不需要再纠结于输入和输出的长度限定，最好的例子就是机器翻译，中文的token个数可未必和翻译出来的英文token个数一致。

<p align="center"> 
  <img src="/imgs/seq2seq/1.png">
</p>













