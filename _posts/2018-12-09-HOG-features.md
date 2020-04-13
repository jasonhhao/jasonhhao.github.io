---
layout: post
title:  HOG features
date:   -- 00:00:00 +0800
categories: 计算机视觉
tag: HOG特征
---

* content
{:toc}


### 宏观理解
在计算机视觉里有很多方法去处理图片搞分类和物体检测，比如现如今的CNN, RNN，R-CNN一类，当然在他们被证实更有效果之前，还有很多处理特征的方法，其中一个鼻祖就是HOG。HOG特征的全名叫histogram of oriented gradient。可以从名字看出来他是从图片的梯度入手来做图片物体检测的，在行人检测中获得了极大的成功。

这些传统的物体检测方法一般都有相似的流程：

1. 选取候选窗口
2. 提取候选窗口的图像特征
3. 对特征进行分类

HOG特征的主要思想就是既然我们人类观察一个物体进行物体检测和分类是从物体的轮廓开始的，那么它也可以。那么怎么来通过物体的轮廓来锁定物体呢，那就是物体边缘的颜色呗，一片绿色的草原上插着一朵红色的花，是不是很容易发现？那么怎么看边缘颜色呢？梯度呗。所以HOG在做的正是利用图片的梯度来检测物体的位置，是不是有点像夜间的红外线探测仪，你看到一个颜色深的轮廓你能直觉出可能那是个物体。

##### 背景知识介绍

之前提到了物体检测的第一步就是选取候选窗口，那么这一步怎么做呢？首先能想起来的就是暴力的方法，把截取的窗口的坐标循环一遍。但是显而易见这种方法时间复杂度太大，更别说做成实时的模型了。所以前人提出了一种叫做图像金字塔的方法，如下图。我们把每张图片缩放到不同的几种大小，然后仅仅用一种固定大小的框，像CNN中的kernel一样去扫一遍这个图片，然后把每次扫到的片段进行特征提取和分类。特征提取就是HOG的事情了。

<p align="center"> 
  <img src="/imgs/hogfeatures/1.png">
</p>

### 微观分析
用梯度来检测物体就很简单，听起来就不是什么复杂的算法，无非就是提取图片所有像素点梯度，然后把有规则的归在一起，再用分类器进行分类呗。总结一下HOG处理过程：

<p align="center"> 
  <img src="/imgs/hogfeatures/2.png">
</p>

1. 对输入的图片进行预处理，比如裁剪或者resize。

<p align="center"> 
  <img src="/imgs/hogfeatures/3.png">
</p>

2. 对预处理完的图片进行灰度处理
因为HOG特征注意的是边缘特征而不是颜色，可以用opencv库来转化：sample_image = cv2.cvtColor(sample_image,cv2.COLOR_RGB2GRAY)

<p align="center"> 
  <img src="/imgs/hogfeatures/4.png">
</p>

3. 归一化图片
用Gamma校正法对输入图像进行颜色空间归一化，意义在于去除噪声，降低图片光度造成的影响。所谓的gamma校正就是在原图中开根号即可。

<p align="center"> 
  <img src="/imgs/hogfeatures/5.png">
</p>

4. 计算出梯度图像
我们怎么得到梯度呢，需要用两个卷积核对原图进行卷积。

<p align="center"> 
  <img src="/imgs/hogfeatures/6.png">
</p>

通过卷积之后我们可以得到两个梯度，分别是GX和GY：

- GX = translate(sample_image,1,0) - translate(sample_image,-1,0)
- GY = translate(sample_image,0,1) - translate(sample_image,0,-1)

<p align="center"> 
  <img src="/imgs/hogfeatures/7.png">
</p>

然后通过GX，GY计算出幅值delta_G和方向angle得到梯度图：

- delta_G = np.sqrt(GX ** 2 + GY ** 2)
- angle = np.arctan(GY / GX) / np.pi * 180

<p align="center"> 
  <img src="/imgs/hogfeatures/8.png">
</p>

5. 在n*n网格中计算梯度直方图

<p align="center"> 
  <img src="/imgs/hogfeatures/9.png">
</p>

我们把图像分成若干个小cell，每个cell为6*6个像素。我们采用9个bin的直方图来统计每个cell上的梯度信息，也就是把上一步求出来的方向angle分成9份，比如0-20，20-40，40-60。。。然后如果求出来的方向在20-40内，我们就把第二个bin的值加上它的幅值delta_G。

<p align="center"> 
  <img src="/imgs/hogfeatures/10.png">
</p>

6. 归一化
归一化就是要避免光照的影响。我们把每个小网格周围的3个网格进行拼接concate，然后进行l2归一化，最后的效果是这样的：

<p align="center"> 
  <img src="/imgs/hogfeatures/11.png">
</p>

接着我们用svm来做分类，最后得出分类结果即可。这大概就是CNN的前身。
