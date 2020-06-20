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


<h2 align="center">微观分析</h2>

例如我们下面的这个例子中，decoder根据context信息经过一个softmax生成了第一个结果today‘s，然后再把today‘s和hidden weight喂给时刻2的decoder，经过softmax生成了weather。但是现在问题来了，我们每次经过softmax的时候都是找到概率最大的那个，假如有一种情况是在时刻2经过softmax后is的概率比weather高一点点，那么模型就会认定输出为is。这种情况显然不是我们想要的。

![image](./imgs/seq2seq/1.png)

<p align="center"> 
  <img src="/imgs/seq2seq/1.png">
</p>


这种算法叫Greedy decoding，它每次都贪心的选取当前时刻最优的选择。但是我们怎么去避免这种情况发送呢？



<h4>Exhaustic Search</h4>

最容易想到的就是暴力解决，也就是每次在softmax之后我们选择全部。假设在时刻1的时候有10种选择，那么我们再把每种选择都预测下一个再生成一共100个选择，。。。这样继续下去复杂度就是可选择个数v的指数级。虽然这样可以拿到全局最优解，但是太过于复杂笨重。

随后人们就想有没有这两种方法的折中呢？我们不光考虑一个，也不要去考虑所有，那么我们可以每次考虑k个。


<h4>Beam Search</h4>







