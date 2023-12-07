---
title: 开源2D图形引擎对比
date: 2016-11-17 20:37:09
categories:
 - 开源
tags:
 - 开源
 - 图形引擎
 - 2D
---

# 说明
2D图形引擎，是指专门针对2D绘图作优化的引擎。一般会提供如下功能：
* 绘制基本形状：点、线、矩形、圆、曲线等
* 抗锯齿
* 改变线条颜色和粗细
* 填充形状内部
* 带alpha通道的绘制
* 从image、PDF、window等加载background
* 绘制目标可以是image、PDF、window等
* etc.

# 选型
## 要求
需要的2D引擎必须拥有如下特点：
* 跨平台：windows、linux、android、iOS
* 无显见bug，如内存泄露等问题
* 绘制效率高
* 抗锯齿、亚像素平滑、贝塞尔曲线支持、线与线之间接口圆润

## 引擎列表
### AGG
AGG([http://www.antigrain.com](http://www.antigrain.com))，是一个开源(GPL)跨平台(windows,linux)的高质量2D图形引擎。引擎采用c++编写，大量应用模板，文档也很少，比较难用，最后更新时间是2006年，并且据说有不少bug。因此，暂不考虑该引擎。

### Skia
Skia([https://skia.org/](https://skia.org/))，是一个开源(BSD)跨平台(windows,linux,android,chrome,firefox)的2D图形引擎，目前由Google维护，chrome和android内部均使用该引擎。引擎采用c++编写，文档丰富，效果很不错，不过需要c++11编译支持，目前我们服务器不支持c++11编译。因此暂时搁置。

### Cairo
Cairo([https://www.cairographics.org/](https://www.cairographics.org/))，是一个开源(LGPL)的2D图形引擎，可支持将输出定向到X11 window，Win32 window、PDF、SVG、buffer等，firefox浏览器采用该引擎作为底层图形绘制引擎。考虑中。

### fog
fog([https://code.google.com/archive/p/fog/](https://code.google.com/archive/p/fog/))，基于AGG引擎优化，速度比较快。已经被合并到[blend2d](https://blend2d.com/)，blend2d尚未发布，之后可能开源(Zlib)发布。

### Azure
Auzre([https://wiki.mozilla.org/Platform/GFX/Moz2D](https://wiki.mozilla.org/Platform/GFX/Moz2D))是Mozilla Firefox的新一代图形引擎，用于替换Cairo。需要C++11的编译支持。因此暂时搁置。

### Qt
Qt([https://www.qt.io/](https://www.qt.io/))是一个全能型的C++框架，其中包含2D图形绘制功能。过于庞大，暂不考虑。

# 结果
最后选用了Cairo，使用挺方便，效果也不错，不过接口全是c的，都有点不习惯了。。。
