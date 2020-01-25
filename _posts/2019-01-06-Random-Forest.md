---
layout: post
title:  Random Forest
date:   2019-01-06 00:00:00 +0800
categories: 机器学习
tag: 随机森林
---

* content
{:toc}


### 宏观理解
随机森林是集成学习中bagging的一种。是把很多决策树放到一起进行预测，但是要clear的一点是随机森林的每个决策树都长得不一样，如果我们仅仅把不同的样本给到不同的决策树中训练，这只能叫bagging of decision tree。所以仅仅说随机森林就是很多的决策树是不严谨的。随机森林的目的是在保证不升高bias的情况下降低模型variance。

### 微观分析
我们说了如果仅仅是把样本有放回的分成好几份来用决策树训练，然后再预测，这只能叫做bagging of decision tree，用的是全部特征。

如果我们在把样本有放回的分成几份之后，还要在每个样本子集中随机抽取一次特征，那么这种方法叫做Fake Random Forest。

如果我们在把样本有放回的分成几份之后，还要在每个样本子集中抽取特征，但是在抽取特征时，我们不止抽取一次，而是在每次抽取的n个特征中选择一个最佳特征来构建决策树直到决策树的终止条件触发，这种方法叫Random Forest。

用一句话区分Fake Random Forest和Random Forest是Fake Random Forest每次是从已经采集的n个特征中挑选最佳特征，而Random Forest在每一次划分的时候都要重新采集n个特征然后在中间选取最佳特征。

我们可以用伪代码来理解一下这三者之间的区别：

- 对于bagging of decision tree

<p align="center"> 
  <img src="/imgs/randomforest/1.png">
</p>

- 对于Fake Random Forest

<p align="center"> 
  <img src="/imgs/randomforest/2.png">
</p>

- 对于Random Forest

<p align="center"> 
  <img src="/imgs/randomforest/3.png">
</p>

注：关于特征选取的数量n，对于分类问题 n = floor(sqrt(total_features)); 对于回归问题 n = floor(total_features/ 3)
