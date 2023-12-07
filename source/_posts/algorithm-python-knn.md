---
title: Python算法系列：KNN（K近邻算法）
date: 2018-03-04 16:29:00
categories:
 - 算法
tags:
 - 算法
 - 机器学习
 - KNN
---

### 简介
K-Nearest-Neighbor(KNN，K近邻算法)是最基础最常用的分类算法之一，具体介绍可以参考：[K-nearest-neighbors-algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)。

使用Python实现KNN算法，简单的流程如下：
* 加载数据，切分为训练和测试，占比为7:3
* 实现KNN算法
* 用KNN算法去对测试数据进行分类，并计算准确率

### 数据
共有1000条数据，每条数据有3个属性，类别是1/2/3，数据之间以tab作为分割。
如下：
```
40920	8.326976	0.953952	3
14488	7.153469	1.673904	2
26052	1.441871	0.805124	1
75136	13.147394	0.428964	1
...
```
全部数据下载：[dataset.txt](/2018/03/04/algorithm-python-knn/dataset.txt)


### 代码
代码就不多解释了，直接粘贴出来。
```python
#coding: utf-8
import numpy as np
import sys

def min_list(lists):
    assert len(lists) >= 1
    mins = lists[0][:]
    for i in range(1, len(lists)):
        row = lists[i]
        for j in range(len(mins)):
            mins[j] = min(mins[j], row[j])
    return mins

def max_list(lists):
    assert len(lists) >= 1
    maxs = lists[0][:]
    for i in range(1, len(lists)):
        row = lists[i]
        for j in range(len(maxs)):
            maxs[j] = max(maxs[j], row[j])
    return maxs

def load_dataset(file_path):
    lines = open(file_path).readlines()
    inputs = []
    outputs = []
    min_params = []
    max_params = []
    for line in lines:
        line = line.strip()
        parts = line.split('\t')
        params = parts[:-1]
        label = parts[-1]
        params = [float(param) for param in params]
        label = int(label)
        inputs.append(params)
        outputs.append(label)
        if len(min_params) == 0:            
            min_params = params[:]
            max_params = params[:]
        else:
            min_params = min_list([min_params, params])
            max_params = max_list([max_params, params])
    #normalize
    norm_inputs = []
    for params in inputs:
        norm_params = []
        for i in range(len(params)):
            norm_params.append((params[i]-min_params[i]) / max((max_params[i]-min_params[i]), 1e-6))
        norm_inputs.append(norm_params)
    inputs = norm_inputs

    #split to train & test
    train_rate = 0.7
    train_cnt = int(train_rate * len(inputs))
    train_inputs = inputs[:train_cnt]
    train_outputs = outputs[:train_cnt]
    test_inputs = inputs[train_cnt:]
    test_outputs = outputs[train_cnt:]
    return train_inputs, train_outputs, test_inputs, test_outputs

def distance(lhs, rhs):
    dist = 0.0
    for i in range(len(lhs)):
        lx = lhs[i]
        rx = rhs[i]
        dist += (lx-rx)*(lx-rx)
    return dist

def classify(train_inputs, train_outputs, x, K):
    scores = []
    for train_x, train_y in zip(train_inputs, train_outputs):        
        dist = distance(x, train_x)
        score = -1.0 * dist
        scores.append((score, train_y))
    scores = sorted(scores, key=lambda x: x[0], reverse=True)
    labels = {}
    for i in range(min(K, len(scores))):
        label = scores[i][1]
        labels.setdefault(label, 0)
        labels[label] += 1
    best_label = -1
    best_cnt = 0
    for label, count in labels.items():
        if best_cnt <= count:
            best_label = label
    return best_label

def test(train_inputs, train_outputs, test_inputs, test_outputs, K):
    total = 0
    correct = 0
    for x,y in zip(test_inputs, test_outputs):
        pd = classify(train_inputs, train_outputs, x, K)        
        if pd == y:
            correct += 1
        total += 1
    return float(correct)/float(total)

def main():
    train_inputs, train_outputs, test_inputs, test_outputs = load_dataset('dataset.txt')
    K = 5
    accuracy = test(train_inputs, train_outputs, test_inputs, test_outputs, K)
    print('KNN accuracy: %.4f' % (accuracy*100.0))

if __name__ == '__main__':
    main()
```

### 结果
准确率为：92.33%。
KNN是一种非常简单有效的分类算法。
