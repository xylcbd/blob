---
title: PyTorch之warp-ctc binding编译问题
date: 2018-04-09 19:24:03
categories:
 - 深度学习
tags:
 - PyTorch
 - 深度学习
 - 笔记
---

最近在用PyTorch实现一些类CRNN的网络，需要使用到百度的warp-ctc，找了下资料，发现已经有现成的PyTorch binding。然而按照官方的文档，编译失败了，折腾了点时间搞定，记录一下以备忘。

### 问题描述
warp-ctc的pytorch binding(https://github.com/SeanNaren/warp-ctc)编译失败。 
```shell
git clone https://github.com/SeanNaren/warp-ctc.git
cd warp-ctc
mkdir build; cd build
cmake ..
make
```
在此处编译warp-ctc本体时OK，接下来进入pytorch\_binding编译pytorch的binding时出错。  
注：WARP\_CTC\_PATH和CUDA\_HOME均设置正确。
```
cd pytorch_binding
python setup.py install
```
上面的命令出错，出错信息：
```python
Traceback (most recent call last):
  File "setup.py", line 52, in <module>
    extra_compile_args=extra_compile_args)
  File "~/.conda/envs/dl/lib/python2.7/site-packages/torch/utils/ffi/__init__.py", line 162, in create_extension
    ffi.cdef(_typedefs + all_headers_source)
  File "~/.conda/envs/dl/lib/python2.7/site-packages/cffi/api.py", line 107, in cdef
    self._cdef(csource, override=override, packed=packed)
  File "~/.conda/envs/dl/lib/python2.7/site-packages/cffi/api.py", line 121, in _cdef
    self._parser.parse(csource, override=override, **options)
  File "~/.conda/envs/dl/lib/python2.7/site-packages/cffi/cparser.py", line 315, in parse
    self._internal_parse(csource)
  File "~/.conda/envs/dl/lib/python2.7/site-packages/cffi/cparser.py", line 320, in _internal_parse
    ast, macros, csource = self._parse(csource)
  File "~/.conda/envs/dl/lib/python2.7/site-packages/cffi/cparser.py", line 278, in _parse
    self.convert_pycparser_error(e, csource)
  File "~/.conda/envs/dl/lib/python2.7/site-packages/cffi/cparser.py", line 307, in convert_pycparser_error
    raise CDefError(msg)
cffi.error.CDefError: cannot parse "int cpu_ctc(THFloatTensor *probs,"
<cdef source string>:29:34: Illegal character '\r'
```

### 出错猜想
看起来是该行内容（下行）后面包含“\r”字符，而python的cffi模块不允许在parser的输入内容中包含该字符以至于出现错误。
```c++
int cpu_ctc(THFloatTensor *probs,
```
该内容所在文件为src/cpu\_binding.h，该文件内容如下：
```c++
int cpu_ctc(THFloatTensor *probs,
                        THFloatTensor *grads,
                        THIntTensor *labels_ptr,
                        THIntTensor *label_sizes_ptr,
                        THIntTensor *sizes,
                        int minibatch_size,
                        THFloatTensor *costs);
```

### 尝试方法1
可以通过pytorch_binding/setup.py的headers变为空数组，可以编译通过，但是运行时会出现错误：  
```python
Traceback (most recent call last):
  File "main.py", line 222, in <module>
    main()
  File "main.py", line 203, in main
    avg_loss, avg_acc = train(model, ctc, optimizer, train_dataset_loader, label_map, train_total_batches)
  File "main.py", line 41, in train
    loss = ctc.loss(predict_Ys, preds_size, batch_merge_Labels, batch_label_lens)
  File "~/workspace/ocr/model.py", line 109, in loss
    rs = self.criterion(preds, label, preds_length, label_length) / batch_size
  File "~/.conda/envs/dl/lib/python2.7/site-packages/torch/nn/modules/module.py", line 357, in __call__
    result = self.forward(*input, **kwargs)
  File "build/bdist.linux-x86_64/egg/warpctc_pytorch/__init__.py", line 76, in forward
    
  File "build/bdist.linux-x86_64/egg/warpctc_pytorch/__init__.py", line 17, in forward
    __version__ = PILLOW_VERSION
AttributeError: 'module' object has no attribute 'gpu_ctc'
```
**方法1失败！**

### 尝试方法2

阅读出错信息，查看出错部分代码，发现主要在于读取headers参数中的文件内容后拼接产生的all\_headers\_source中包含了“\r”字符，因此可以尝试修改代码把该字符去除。  

修改~/.conda/envs/dl/lib/python2.7/site-packages/torch/utils/ffi/__init__.py（地址自行修改） +162处。
原来：
```python
for header in headers:
        with open(os.path.join(base_path, header), 'r') as f:
            all_headers_source += f.read() + '\n\n'
```
修改后：
```python
for header in headers:
        with open(os.path.join(base_path, header), 'r') as f:
            all_headers_source += f.read() + '\n\n'
all_headers_source = all_headers_source.replace('\r','')
```
**方法2成功！**

### 后记
原因暂时未知，看起来是读取header文件（src/cpu\_binding.h）时读入了“\r”字符，而cffi模块不允许该字符出现在parser中，但是手动打开该文件（src/cpu\_binding.h）并且查找“\r”字符并没有找到。因此尚不知晓为何读入了“\r”字符。

暂时通过方法2修改源代码解决此问题。

注：CTCLoss.forward的GPU版本的4个参数要求是CUDATensor, IntTensor, IntTensor, IntTensor
```
* acts: Tensor of (seqLength x batch x outputDim) containing output from network
* labels: 1 dimensional Tensor containing all the targets of the batch in one sequence
* act_lens: Tensor of size (batch) containing size of each output sequence from the network
* label_lens: Tensor of (batch) containing label length of each example
```
