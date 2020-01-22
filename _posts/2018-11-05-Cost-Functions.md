---
layout: post
title:  Cost Functions
date:   2018-11-05 00:00:00 +0800
categories: 机器学习
tag: 损失函数
---

* content
{:toc}


### 宏观理解

cost function 顾名思义就是用来计算模型的cost，cost指的是模型在真实值y和预测值y_hat之间的差异大小，也可以说是距离大小。
这种计算差异大小的作用就是让模型在训练的时候有一个可以优化的方向，也就是我们都希望这个差异越小、预测值越贴近真实值越好。

### 微观分析

现在我们来把数学表达对应上，真实值我们依然用y来表示，预测值我们可以用hθ(x)来表示，这里的x就是数据集中的所有features，θ是我们模型在拟合过程中所学习到的参数，这两个值在几乎所有情况下都是一个矩阵matrix。通过一个函数h()可以得到我们最后的预测值。

再来说cost function, 它的通用表达是J(θ)，J是怎么来的这个传说是因为Jacobian Matrix（所有方向上的求导）而来的，who knows。我们希望最小化这个cost function，那就在前面加上一个min，合起来就是min J(θ)。

所以把之前所有提到的变量全部整合在一起，就有了这样的公式：min J(θ) = min(hθ(x) – y)。

这个公式还是很naive的，这里只是为了表达J, hθ(x)和y的关系写的很省略，在后面的很多模型中我们会慢慢的在这个基础上添加计算项。所以换句话来说，不同的模型的cost function也许是不同的。

我们说到了cost function是用来计算差异的，那么通过什么步骤来计算，会在后面的梯度下降文章中介绍。

最后扔一点比较常见的cost function，这些以后都会再次介绍：

均方误差（MSE，也叫L2 loss）：<a name='img1'>![](/imgs/costfunction/1.png)</a>
交叉熵函数（cross entropy）：![Alt text](/imgs/costfunction/2.png)
