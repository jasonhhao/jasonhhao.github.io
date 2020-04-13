---
layout: post
title:  Faster R-CNN
date:   2018-12-14 00:00:00 +0800
categories: 计算机视觉
tag: 卷积神经网络
---

* content
{:toc}


### 宏观理解

我们学完R-CNN后，想想它有什么地方可以改进的？有什么地方是拖累了整个算法的速度？在2015年同一个作者提出了Fast R-CNN算法，它是基于R-CNN在选取候选框时大量的重叠导致同一图片的特征被多次提取而改进的算法。它使用一个叫RoI（regions of interest） pooling层来将抠出来的候选框resize到统一的大小。

在提出Fast R-CNN后，又是同样的作者，在2015年提出来又一个新的改进叫Faster R-CNN 。我们在R-CNN选取候选框的时候用的是selective search，但是它无法在GPU上进行加速，所以为了实现算法的速度更快，作者终于把整个算法都用一个CNN来实现，借助CNN来生成候选框。

### 微观分析

我们先看看faster R-CNN的整个流水线：

<p align="center"> 
  <img src="/imgs/fasterrcnn/1.png">
</p>

对于每一个图片，我们先经过conv layers来提取图片特征，生成一张feature map。然后通过一个叫做Region Proposal Network的网络结构来代替selective search生成候选框。然后把生成的proposal和feature map给到RoI pooling层来得到统一大小的候选框提取特征。最后给一个分类器来分类并且精修bounding box的位置。

再来看一个更加细节的图示：

<p align="center"> 
  <img src="/imgs/fasterrcnn/2.jpeg">
</p>

大家可以对照上面的两个图对应一下每个网络的位置。

<p align="center"> 
  <img src="/imgs/fasterrcnn/3.jpeg">
</p>

我们分成3个部分来学习整个pipeline。

对于蓝色框
对于我们的每个卷积层，作者设置kernel size = 3，padding = 1，stride = 1 。这意味着什么？意味着一个M*N的原图经过卷积后的feature map还是M*N，卷积层并没有改变图片的大小。

对于我们的每个max池化层，作者设置kernel size = 2，padding = 0，stride = 2。这样经过池化后feature map的长宽会缩小1/2，所以在通过4个池化层后得到的feature map就会变成size = (M / 16) * (N / 16) * 256的大小。

对于红色框
随后我们得到feature map后，先把它送给一个叫RPN（Region Proposal Network）的网络来生成候选框。在选候选框的时候，作者预定义了三种形状并且三种大小的Anchors：

  <p align="center"> 
  <img src="/imgs/fasterrcnn/4.png">
</p>

<p align="center"> 
  <img src="/imgs/fasterrcnn/5.png">
</p>

作者期望几乎所有的前景都可以通过这几种anchors来框出来。那么怎么框呢？我们之前在卷积层中把原图映射到了(M / 16) * (N / 16) 的大小，所以在feature map中的每个点是不是就映射着原图中的16*16的区域。然后我们把anchors和16*16的中心点对应，也就是在原图中画出这9中anchor。

在流程图中我们首先经过一个3*3的卷积，它的目的是让每一个点可以融合周围8个点的特征。随后分成了两条支路。我们先走上面的支路。

<p align="center"> 
  <img src="/imgs/fasterrcnn/6.png">
</p>

通过1*1*18的卷积使得(M / 16) * (N / 16) * 256的feature map变成了 (M / 16) * (N / 16) * 18的feature map。然后经过reshape成(M / 16) * (N / 16) * 9 * 2 。为什么呢？因为作者想要这9 * 2的大小再加上后面的一个softmax层来对应到9个anchors分别为前景还是背景的概率。最后再把之前reshape的数据再reshape回去。

<p align="center"> 
  <img src="/imgs/fasterrcnn/7.png">
</p>

下面的这个pipeline通过一个1*1*36的卷积来把输入的(M / 16) * (N / 16) * 256的feature map变成了 (M / 16) * (N / 16) * 36的feature map。这个36对应的就是4*9个anchor的位置变化x, y, h, w。

现在整个RPN就做完了，我们知道了结构还需要知道损失函数是什么。

<p align="center"> 
  <img src="/imgs/fasterrcnn/8.png">
</p>

这个loss把两项相加，前面是分类的loss，后面是位置移动的loss，所以加一起就变成了整个RPN的loss。这里不展开这个loss function了。

RPN过后我们就从anchor找到了经过修正过的proposals。因为这时候的proposal还是形态各异，所以我们可以先手动的调整一下超过图片边界或者大小阈值的proposal，最后交给NMS来生成最终的proposals。

对于粉色框
我们做完RPN之后的结果和上面传下来的卷积后的结果先会传给RoI pooling层。它做的就是把大小不一的proposal给resize成大小一样的。这个过程很简单，无论我们之前传过来的proposals的大小为多少，我们都把这个图分成n*n的宫格（下图中n = 2），然后在这个宫格中每个格做一次max pooling，所以最终无论什么形状都会变成n*n的proposal。

<p align="center"> 
  <img src="/imgs/fasterrcnn/9.gif">
</p>

做完RoI pooling后将这些图做几层的全连接，然后预测bounding box和classification，这两个操作和之前的RPN一样，唯一不一样的是分类的类别从之前的2类根据自己的项目分成n类。
