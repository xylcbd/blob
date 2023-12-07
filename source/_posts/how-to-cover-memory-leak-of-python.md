---
title: 解决python训练数据内存泄露的一个非正常方法
date: 2017-06-28 12:09:00
categories:
 - python
tags:
 - python
 - shell
 - 内存泄露
---

## 内存泄露
Python越来越广泛的应用于机器学习/深度学习领域，是目前最火的该领域编程语言，各大深度学习框架基本都支持python接口。

在TensorFlow训练模型的过程中，一般数据加载分情景各有不同。  

1. 当数据量能直接全部载入内存时，当然最是方便，直接全部载入内存，然后训练即可。
2. 数据无法全部载入内存时，有多种方法。介绍其中2种用的多的，其一是边训练边读取（可以用多线程进行优化），其二是随机打乱所有训练数据的索引然后随机选择部分数据进行训练测试。

第2类情景中，有可能由于对Python的GC机制理解不深（Me...），出现内存泄露。比如下面的第12行的列表就无法释放，导致内存泄露，而11行则不会出现内存泄露。

```python
import numpy as np
import gc
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss/(1024*1024)

print 'before load, memory: %d MB' % get_memory_usage()
#dataset = [np.zeros((6000,6000),dtype=np.float32) for _ in range(30000)]
dataset = [(np.zeros((6000,6000),dtype=np.float32),np.zeros((200,200),dtype=np.int32)) for _ in range(10000)]
print 'after load, memory: %d MB' % get_memory_usage()

print 'before release, memory: %d MB' % get_memory_usage()
del dataset
dataset = None
gc.collect()
print 'after release, memory: %d MB' % get_memory_usage()
```

第11行内存释放结果：  
```shell
before load, memory: 25 MB
after load, memory: 220 MB
before release, memory: 220 MB
after release, memory: 27 MB
```

第12行内存释放结果：  
```shell
before load, memory: 23 MB
after load, memory: 1406 MB
before release, memory: 1406 MB
after release, memory: 1170 MB
```

可以看到第12行内存泄露严重。出现这种内存泄露时，模型训练基本只能训练几个Epoch就会出现OOM。

## 大力出奇迹
这种问题当然最优的是从根源解决，即研究清楚Python的GC机制，解决内存泄露。  

在时间比较紧或者问题比较麻烦时，有时粗暴快速的方法也不失为一种选择。  

训练模型的目标是根据训练数据以及模型结构优化模型参数，因此只要达成目的，过程不重要。  

在这个例子中，可以直接每次只训练1个Epoch，然后重启Python进程读取模型进行fine-tune。与直接训练若干个Epoch效果接近。

```shell
#训练50个epoch
seq 50|xargs -i python train.py --model-path=./model
```