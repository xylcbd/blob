---
title: 我所理解的深度学习技术栈
date: 2017-02-17 21:19:03
categories:
 - 深度学习
tags:
 - 深度学习
---

深度学习，自2012年以来，一年比一年火。相比80年代的人工智能热，这一次，基于深度学习的各种智能服务确实带来了可见的惊喜。  
深度学习在图像领域的各层面都取得了骄人的成绩，包括分割、识别等高级任务，以及去噪、二值化、超分辨率等低级任务。  
在语音以及自然语言处理等方面也在快速发展。  

介绍一下我所认为的深度学习技术栈。  

### 数据收集与扩充
数据是深度学习成功的重要因素，并且随着各种深度学习框架的普及，数据越来越成为最重要的门槛。  
数据一般有2种来源，其一是来自于业务场景的真实数据（包括人工标注的数据），其二是模拟真实场景仿造的数据。  
真实数据昂贵不易获得，很多时候就需要产生大量的模拟数据以满足深度学习的要求。模拟数据因尽可能的接近真实数据的分布情况。    
在图像领域，产生模拟数据的手段包括但不限于如下技术：
* 旋转
* 剪切
* 翻转
* 缩放
* 加噪声
* 有损压缩（JPEG、Mpeg等）
* 水印

### 基础深度学习模型
应该理解并能编程实现基础的深度学习模型（可参考：[EasyCNN](https://github.com/xylcbd/EasyCNN),[EasyTF](https://github.com/xylcbd/EasyTF)），包括如下基础模型：
* MLP
* CNN
* LSTM

### 高级深度学习模型
需要理解高级深度学习模型，并能够通过通用框架（如TensorFlow、Keras等）实现之。
* AlexNet
* VGG
* GoogleNet (Inception v1-v4, Xception)
* ResNet
* Network-in-network
* SqueezeNet
* E-Net
* R-CNN、SSD、YOLO
* Stack RNN/LSTM/GRU
* Seq2Seq
* CRNN

### 各种trick
在训练深度学习模型的时候，有时候各种trick能起到很好的效果。应该了解其大致作用和原理。
* Dropout
* Batch Normalization
* Local Response Normalization
* L1/L2正则化

### 各种Operator
在现代深度学习框架中，普遍都有Operator这样的抽象概念，需要了解一些重要的OP的原理和推导。
* Conv
* Pool
* RNN/LSTM
* Element-wise op
* Softmax
* CTC

### 参数初始化
参数初始化，了解其作用即可。
* xavier
* constant
* gaussian
* uniform
* bilinear

### 激活函数
激活函数，了解其作用即可。
* Sigmoid
* Tanh
* RELU
* Leaky ReLU
* ELU
* Maxout

### 损失函数
不同的任务选用不同的损失函数。
* Absolute loss
* Square loss
* Log loss
* Cross-Entroy loss
* Zeor-one loss
* Hinge loss
* Perceptron loss
* Exponential loss
* CTC loss

### 优化方法
优化方法影响模型精度和模型收敛速度，一般选择SGD+Momentum。
* SGD
* Momentum
* Nesterov
* Adagrad
* Adadelta
* RMSprop
* Adam
* Adamax
* Nadam

### 框架实现
框架用多了，自然就了解框架的大概模块和原理了。也可以尝试自己实现一遍框架，这时候不只是考量深度学习基本知识，还考验工程架构能力。
* 框架设计
* 计算图与其优化
* 数据并行与模型并行
* 分布式

### 理论与实际
深度学习发展非常快，隔不了几个月就有新概念，一般关注一下相关的新闻和最新的论文即可。一些实际问题在踩坑和交流的过程中会有答案。
* Zero-center
* 梯度消失
* 梯度爆炸
* 如何调参
* Finetune
* 增强学习
* 模型压缩
* 速度优化（low bit）

### 应用
各种深度学习的应用，“没有做不到，只有想不到”。
* OCR
* 语音识别
* 自然语言处理
* 风格转移
* 人脸定位 & 人脸识别
* 手势识别
* 物体追踪
* 图像检索
* 超分辨率重建
* 去噪&去马赛克

个人推崇带着问题去学习，在实际项目中学习。因此这里大部分只是关键字，并不尝试去解释每个词的意思。
