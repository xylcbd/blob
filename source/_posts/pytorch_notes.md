---
title: PyTorch学习笔记
date: 2018-03-15 19:19:03
categories:
 - 深度学习
tags:
 - PyTorch
 - 深度学习
 - 笔记
---

最近学习了下PyTorch的使用，感觉这种“所见即所得”的网络构建运行方式确实方便，比TensorFlow之类的先定义静态图然后再run的方式要灵活很多，调试也方便不少。

学习过程中也有一些疑惑，记录下来备忘。

### PyTorch零散的知识

* 默认只有叶子节点才保持梯度，如：
```python
A = Variable(torch.ones(2), requires_grad = True)
```

* 在中间节点可以设置保持梯度，如：
```python
A = Variable(torch.ones(2), requires_grad = True)
B = A*2 
B.retain_grad()
C = B.norm()
C.backward()
print B.grad

#outputs
Variable containing:
0.7071
0.7071
[torch.FloatTensor of size 2]
```

  也可以设置hook输出梯度：
```python
A = Variable(torch.ones(2), requires_grad = True)
B = A*2 
def show_grad(grad):
    print(grad)
B.register_hook(show_grad)
C = B.norm()
C.backward()
#...
#grad show be auto displayed after grad of B is generated
#outputs
Variable containing:
0.7071
0.7071
[torch.FloatTensor of size 2]
```

* 每一个非叶子节点都会保存其创建，如：
```python
# not B.creator now, ref: https://github.com/pytorch/tutorials/pull/91/files
print B.grad_fn 
<MulBackward0 object at 0x7f3b0536f710>
```

* 早期PyTorch的执行模型参见：[function.py](https://github.com/pytorch/pytorch/blob/v0.1.1/torch/autograd/function.py)、[engine.py](https://github.com/pytorch/pytorch/blob/v0.1.1/torch/autograd/engine.py)  

* backward只能由结果是标量的operator执行，比如：nn.CrossEntropyLoss。原因暂不明。

* Variable包含一些Inplcae操作（其requires_grad不能为True），均以下划线“_”结尾，如：
```python
A = Variable(torch.ones(2), requires_grad = False)
A.fill_(3.0)

#outputs
Variable containing:
 3
 3
[torch.FloatTensor of size 2]
```

* Storage与Tensor的概念：Storage是实际内存（1维），Tensor是对Storage的信息描述。

* 自动求导原理： [autograd](http://pytorch.org/docs/0.3.1/notes/autograd.html)

* 设置CUDA同步执行，设置环境变量：CUDA_LAUNCH_BLOCKING=1，或者使用copy_强制同步

* PyTorch多进程作用和使用

* PyTorch的模型存取，2种方式：
 1.读取参数
```python
#save model
torch.save(the_model.state_dict(), PATH)
 
#load model
the_model = TheModelClass(*args, **kwargs)
the_model.load_state_dict(torch.load(PATH))
```

 2.读取模型
```python
#save model
torch.save(the_model, PATH)

#load model
the_model = torch.load(PATH)
```

* CrossEntropyLoss与NLLLoss不同。前者是Softmax与CrossEntropy的组合，在Loss之前不要再加Softmax；后者只是CrossEntropy，需要在Loss之前加Softmax。

* 默认为training状态，在test时需要设置为module.eval()或者module.train(False)
  > * 在训练每个batch之前记得加model.train()，训练完若干个iteration之后在验证前记得加model.eval()。否则会影响dropout和BN。  
  > * 用F.dropout()时一定要手动设参数self.training，正确用法：F.dropout(x, 0.2, self.training)。  
  > * reference: https://www.zhihu.com/question/67209417/answer/303214329

* Tensor.unsqueeze与Tensor.view作用类似，在某个地方插入一个维度（1）

* Tensor.contiguous将Tensor内可能是不同的内存块整理为一块连续的内存，如果本来就是连续的则不作操作。

