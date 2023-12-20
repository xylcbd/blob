---
title: 从零开始实现attention/transformer/bert/gpt模型
date: 2023-12-13 11:13:06
categories:
 - 技术
tags:
 - 技术
 - 从零开始
 - transformer
 - bert
 - gpt
---
---

本文代码可从 [我的github](https://github.com/xylcbd/blogs_code) 下载。

### 1. scaled_dot_product_attention

> * 点乘注意力可以换成加性注意力（更多参考：[深度学习之注意力机制attention](https://blog.csdn.net/m0_37327467/article/details/89307750)）；
> * scaled是点乘会导致注意力矩阵方差变大（为dk），所以要除以sqrt(dk)；
> * mask是外部希望把先验信息注入到注意力矩阵中；

架构图：

<img src="sdpa.png" alt="scaled_dot_product_attention" width="200px" height="400px">

代码:

```python
# coding: utf-8
# scaled_dot_product_attention.py

import math
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(q, k, v, atten_mask=None):
    # param check
    assert q.shape[0] == k.shape[0] and q.shape[0] == v.shape[0] and q.shape[-1] == k.shape[-1] and q.shape[-1] == v.shape[-1]
    b, l_q, d = q.shape
    b, l_kv, d = k.shape
    # bld -> bdl
    kt = k.permute(0, 2, 1)
    logits = torch.matmul(q, kt) / math.sqrt(d)
    bias = torch.zeros(logits.shape, dtype=q.dtype, device=logits.device)
    if atten_mask is not None:
        bias.masked_fill_(atten_mask.expand_as(logits).logical_not(), float("-inf"))
    logits += bias
    score = F.softmax(logits, dim = -1)
    res = torch.matmul(score, v)
    return res, score

if __name__ == '__main__':
    b, l_q, l_kv, d = 8, 32, 64, 128
    q = torch.randn((b, l_q, d), dtype=torch.float32)
    k = torch.randn((b, l_kv, d), dtype=torch.float32)
    v = torch.randn((b, l_kv, d), dtype=torch.float32)
    atten_mask = torch.ones(l_q, l_kv, dtype=torch.bool).tril(diagonal=0)
    res, score = scaled_dot_product_attention(q, k, v, atten_mask=atten_mask)

    print(f'q shape: {q.shape}')
    print(f'k shape: {q.shape}')
    print(f'v shape: {q.shape}')
    print(f'score shape: {score.shape}')
    print(f'res shape: {res.shape}')

    # 与标准pytorch实现进行比对
    _res = F.scaled_dot_product_attention(q, k, v, attn_mask=atten_mask)
    print(res[0, 0, :16])
    print(_res[0, 0, :16])
```

输出:

```txt
q shape: torch.Size([8, 512, 128])
k shape: torch.Size([8, 512, 128])
v shape: torch.Size([8, 512, 128])
score shape: torch.Size([8, 512, 512])
res shape: torch.Size([8, 512, 128])
tensor([-0.1030,  0.0694,  0.0937, -0.0749,  0.0407, -0.0132,  0.0624,  0.0617,
         0.1486,  0.0573, -0.0625, -0.1060,  0.1770,  0.0497, -0.0901,  0.0338])
tensor([-0.1030,  0.0694,  0.0937, -0.0749,  0.0407, -0.0132,  0.0624,  0.0617,
         0.1486,  0.0573, -0.0625, -0.1060,  0.1770,  0.0497, -0.0901,  0.0338])
```

### 2. multi_head_scaled_dot_product_attention

> * 多头注意力相比单头注意力可以类比卷积和全连接，多头每个头提取的特征各有倾向，自由性更高，即使多头和单头设置同样大小的参数量；
> * 多头注意力的实现上把多头各自的处理以batch形式统一处理了，效率相比循环处理每个头会更高一些；
> * 这里的实现假定了batch是维度1，与pytorch假定batch是维度2不同；
> * 这里的实现假定了attention按照多头均值来计算，实际中可以获取每个头对应的注意力；
> * 仅有key_padding_mask而没有query_padding_mask，是因为虽然query中允许有padding但是这些padding产生的attention以及logits并不会在最终结果中产生loss，因此可忽略query_padding_mask；

架构图：

<img src="mhsa.png" alt="multi_head_scaled_dot_product_attention" width="400px" height="600px">

代码：

```python
# coding: utf-8
# multi_head_scaled_dot_product_attention.py

import math
import random
import torch
import torch.nn.functional as F

from scaled_dot_product_attention import scaled_dot_product_attention

def multi_head_scaled_dot_product_attention(q, k, v, n_heads, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias, out_weight, out_bias, key_padding_mask=None, atten_mask=None):
    # param check
    b, l_q, d = q.shape
    b, l_kv, d = k.shape
    assert q_weight.shape[0] % n_heads == 0
    assert k_weight.shape[0] % n_heads == 0
    assert v_weight.shape[0] % n_heads == 0
    d_header = d // n_heads

    # linear projection
    def _linear(x, w, b):
        r = torch.matmul(x, w.T)
        if b is not None:
            r += b
        return r
    _q = _linear(q, q_weight, q_bias)
    _k = _linear(k, k_weight, k_bias)
    _v = _linear(v, v_weight, v_bias)

    # b, l, d => b, l, n_heads, d_header => b, n_heads, l, d_header => b * n_heads, l, d_header
    _q = _q.view(b, l_q, n_heads, d_header).permute(0, 2, 1, 3).reshape(-1, l_q, d_header)
    _k = _k.view(b, l_kv, n_heads, d_header).permute(0, 2, 1, 3).reshape(-1, l_kv, d_header)
    _v = _v.view(b, l_kv, n_heads, d_header).permute(0, 2, 1, 3).reshape(-1, l_kv, d_header)

    if key_padding_mask is not None:
        assert key_padding_mask.shape[0] == b and key_padding_mask.shape[1] == l_kv
        key_padding_mask = key_padding_mask.view(b, 1, 1, l_kv).expand(-1, n_heads, -1, -1).reshape(b * n_heads, 1, l_kv)
        if atten_mask is None:
            atten_mask = key_padding_mask
        else:
            atten_mask = torch.logical_and(atten_mask, key_padding_mask)

    # scaled dot product attention
    res, score = scaled_dot_product_attention(_q, _k, _v, atten_mask = atten_mask)

    # b * n_heads, l, d_header => b, n_heads, l, d_header => b, l, n_heads, d_header => b, l_q, d
    res = res.view(b, n_heads, l_q, d_header).permute(0, 2, 1, 3).reshape(b, l_q, d)
    res = _linear(res, out_weight, out_bias)

    # b * n_heads, l_q, l_kv => b, n_heads, l_q, l_kv => b, l_q, l_kv
    score = score.view(b, n_heads, l_q, l_kv).mean(dim = 1)

    return res, score

if __name__ == '__main__':
    num_heads = 8
    b, l_q, l_kv, d = 8, 32, 64, 128
    assert d % num_heads == 0
    q = torch.randn((b, l_q, d), dtype=torch.float32)
    k = torch.randn((b, l_kv, d), dtype=torch.float32)
    v = torch.randn((b, l_kv, d), dtype=torch.float32)

    bool_atten_mask = torch.ones(l_q, l_kv, dtype=torch.bool).tril(diagonal=0)
    atten_mask = torch.zeros(bool_atten_mask.shape, dtype=q.dtype)
    atten_mask.masked_fill_(bool_atten_mask.logical_not(), float("-inf"))
    
    bool_key_padding_mask = torch.ones(b, l_kv, dtype=torch.bool)
    for i in range(b):
        pad_len = random.randint(0, l_kv//2)
        bool_key_padding_mask[i, -pad_len:] = False
    key_padding_mask = torch.zeros(bool_key_padding_mask.shape, dtype=q.dtype)
    key_padding_mask.masked_fill_(bool_key_padding_mask.logical_not(), float("-inf"))

    q_weight = torch.randn((d, d), dtype=torch.float32)
    k_weight = torch.randn((d, d), dtype=torch.float32)
    v_weight = torch.randn((d, d), dtype=torch.float32)
    out_weight = torch.randn((d, d), dtype=torch.float32)
    q_bias = torch.randn((d), dtype=torch.float32)
    k_bias = torch.randn((d), dtype=torch.float32)
    v_bias = torch.randn((d), dtype=torch.float32)
    out_bias = torch.randn((d), dtype=torch.float32)

    res, score = multi_head_scaled_dot_product_attention(q, k, v, num_heads, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias, out_weight, out_bias, key_padding_mask = bool_key_padding_mask, atten_mask = bool_atten_mask)

    print(f'num_heads: {num_heads}')
    print(f'q shape: {q.shape}')
    print(f'k shape: {q.shape}')
    print(f'v shape: {q.shape}')
    print(f'q_weight shape: {q_weight.shape}')
    print(f'k_weight shape: {k_weight.shape}')
    print(f'v_weight shape: {v_weight.shape}')
    print(f'out_weight shape: {out_weight.shape}')
    print(f'q_bias shape: {q_bias.shape}')
    print(f'k_bias shape: {k_bias.shape}')
    print(f'v_bias shape: {v_bias.shape}')
    print(f'out_bias shape: {out_bias.shape}')

    print(f'score shape: {score.shape}')
    print(f'res shape: {res.shape}')

    # 与标准pytorch实现进行比对
    _q, _k, _v = (x.transpose(1, 0) for x in (q, k, v))
    _res, _score = F.multi_head_attention_forward(_q, _k, _v, d, num_heads, q_proj_weight=q_weight, k_proj_weight=k_weight, v_proj_weight=v_weight, in_proj_bias=torch.concat([q_bias, k_bias, v_bias], dim=-1), out_proj_weight=out_weight, out_proj_bias=out_bias, key_padding_mask=key_padding_mask, attn_mask=atten_mask, use_separate_proj_weight=True, in_proj_weight=None, bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0.0, training=False)
    _res = _res.transpose(1, 0)

    print(res[0, 0, :16])
    print(_res[0, 0, :16])
```

输出：

```txt
num_heads: 8
q shape: torch.Size([4, 512, 128])
k shape: torch.Size([4, 512, 128])
v shape: torch.Size([4, 512, 128])
q_weight shape: torch.Size([128, 128])
k_weight shape: torch.Size([128, 128])
v_weight shape: torch.Size([128, 128])
out_weight shape: torch.Size([128, 128])
q_bias shape: torch.Size([128])
k_bias shape: torch.Size([128])
v_bias shape: torch.Size([128])
out_bias shape: torch.Size([128])
score shape: torch.Size([4, 512, 512])
res shape: torch.Size([4, 512, 128])
tensor([  84.0602,  -16.5560,    0.3141,  -38.8174,  148.4963,  -17.2330,
        -101.3793,   86.8671,  -13.9499, -188.4724,    4.6761,   89.3186,
          -7.1403,   92.3565,    1.9944,  138.3780])
tensor([  84.0602,  -16.5560,    0.3141,  -38.8174,  148.4963,  -17.2330,
        -101.3793,   86.8671,  -13.9499, -188.4724,    4.6761,   89.3186,
          -7.1403,   92.3565,    1.9944,  138.3780])
```

### 3. transformer

> * 标准transformer是一个encoder-decoder架构，这里没有在predict阶段显示自回归解码，但实际上它就是自回归解码的（因为应用了output atten mask）；
> * 在验证时，最开始没有加MultiheadAttention._reset_parameters去特意初始化里面的参数，仅仅是简单的empty或者randn进行初始化，结果一直出现nan或者loss不收敛的情况，加了之后就正常收敛了。在现代深度学习里面，CV领域一般都会加BN等操作，几乎不会再注意权重初始化这个问题，没想到还是被现实教育了一顿；

架构图：
<img src="transformer.png" alt="transformer" width="400px" height="600px">

模型代码：

```python
# coding: utf-8
# transformer.py

import numpy as np
import math
import copy
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.init import xavier_normal_, xavier_uniform_

from multi_head_scaled_dot_product_attention import multi_head_scaled_dot_product_attention

class PositionEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super(PositionEmbedding, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.pe = torch.zeros(self.max_len, self.d_model, dtype = torch.float32)
        pos = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0)/self.d_model))
        self.pe[:, 0::2] = torch.sin(pos * div_term)
        self.pe[:, 1::2] = torch.cos(pos * div_term)

    def forward(self, x):
        b, l, d = x.shape
        assert d == self.d_model
        assert l <= self.max_len
        return x + self.pe[:l, :].to(x.device).expand_as(x).clone().detach()

class MultiheadAttention(nn.Module):
    def __init__(self, n_heads, d_model):
        super(MultiheadAttention, self).__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.q_weight = nn.Parameter(torch.empty((d_model, d_model), dtype=torch.float32), requires_grad=True)
        self.k_weight = nn.Parameter(torch.empty((d_model, d_model), dtype=torch.float32), requires_grad=True)
        self.v_weight = nn.Parameter(torch.empty((d_model, d_model), dtype=torch.float32), requires_grad=True)
        self.out_weight = nn.Parameter(torch.empty((d_model, d_model), dtype=torch.float32), requires_grad=True)
        self.q_bias = nn.Parameter(torch.empty((1, 1, d_model), dtype=torch.float32), requires_grad=True)
        self.k_bias = nn.Parameter(torch.empty((1, 1, d_model), dtype=torch.float32), requires_grad=True)
        self.v_bias = nn.Parameter(torch.empty((1, 1, d_model), dtype=torch.float32), requires_grad=True)
        self.out_bias = nn.Parameter(torch.empty((1, 1, d_model), dtype=torch.float32), requires_grad=True)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.q_weight)
        xavier_uniform_(self.k_weight)
        xavier_uniform_(self.v_weight)
        xavier_uniform_(self.out_weight)
        xavier_normal_(self.q_bias)
        xavier_normal_(self.k_bias)
        xavier_normal_(self.v_bias)
        xavier_normal_(self.out_bias)

    def forward(self, q, k, v, key_padding_mask = None, atten_mask = None):
        res, score = multi_head_scaled_dot_product_attention(q, k, v, self.n_heads, self.q_weight, self.q_bias, self.k_weight, self.k_bias, self.v_weight, self.v_bias, self.out_weight, self.out_bias, key_padding_mask=key_padding_mask, atten_mask=atten_mask)
        return res, score

class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_fc):
        super(EncoderLayer, self).__init__()
        self.self_mhsa = MultiheadAttention(n_heads, d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_fc, bias=False),
            nn.ReLU(),
            nn.Linear(d_fc, d_model, bias=False)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask = None, atten_mask = None):
        # post norm
        res, score = self.self_mhsa(x, x, x, key_padding_mask = key_padding_mask, atten_mask = atten_mask)
        res = self.norm1(x + res)
        res = self.norm2(x + self.fc(res))
        return res, score

class DecoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_fc):
        super().__init__()
        self.n_heads = n_heads
        self.self_atten = MultiheadAttention(n_heads, d_model)
        self.cross_atten = MultiheadAttention(n_heads, d_model)

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_fc, bias=False),
            nn.ReLU(),
            nn.Linear(d_fc, d_model, bias=False)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, y, memory, y_key_padding_mask=None, self_atten_mask=None, memory_key_padding_mask=None, cross_atten_mask=None):
        res1, self_score = self.self_atten(y, y, y, key_padding_mask = y_key_padding_mask, atten_mask = self_atten_mask)
        res1 = self.norm1(y + res1)

        res2, cross_score = self.cross_atten(res1, memory, memory, key_padding_mask = memory_key_padding_mask, atten_mask = cross_atten_mask)
        res2 = self.norm2(res1 + res2)

        res3 = self.norm3(res2 + self.fc(res2))
        
        return res3, self_score, cross_score

class Encoder(nn.Module):
    def __init__(self, n_layers, encoder_layer):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(n_layers)])

    def forward(self, x, key_padding_mask=None, atten_mask=None):
        scores = []
        for layer in self.layers:
            x, score = layer(x, key_padding_mask=key_padding_mask, atten_mask=atten_mask)
            scores.append(score)
        return x, scores

class Decoder(nn.Module):
    def __init__(self, n_layers, decoder_layer):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(n_layers)])

    def forward(self, y, memory, key_padding_mask=None, self_atten_mask=None, memory_key_padding_mask=None, cross_atten_mask=None):
        self_scores = []
        cross_scores = []
        for layer in self.layers:
            y, self_score, cross_score = layer(y, memory, key_padding_mask, self_atten_mask, memory_key_padding_mask, cross_atten_mask)
            self_scores.append(self_score)
            cross_scores.append(cross_score)
        return y, self_scores, cross_scores

class Transformer(nn.Module):
    def __init__(self, d_model, d_fc, n_heads, n_encoder_layers, n_decoder_layers):
        super(Transformer, self).__init__()
        encoder_layer = EncoderLayer(n_heads, d_model, d_fc)
        self.encoder = Encoder(n_encoder_layers, encoder_layer)
        decoder_layer = DecoderLayer(n_heads, d_model, d_fc)
        self.decoder = Decoder(n_decoder_layers, decoder_layer)

    def forward(self, x, y, x_key_padding_mask=None, x_self_atten_mask=None, y_key_padding_mask=None, y_self_atten_mask=None, y_mem_key_padding_mask=None, y_cross_atten_mask=None):
        memory, x_self_scores = self.encoder(x, x_key_padding_mask, x_self_atten_mask)
        
        y, y_self_scores, y_cross_scores = self.decoder(y, memory, y_key_padding_mask, y_self_atten_mask, y_mem_key_padding_mask, y_cross_atten_mask)
        
        return memory, y, [x_self_scores, y_self_scores, y_cross_scores]
```

训练代码：

```python
# transformer_exp.py
################################################## exp ##################################################
class MyModel(nn.Module):
    def __init__(self, max_len, x_vocab, y_vocab, d_model, d_fc, n_heads, n_encoder_layers, n_decoder_layers) -> None:
        super(MyModel, self).__init__()
        self.x_embedding = nn.Embedding(x_vocab, d_model)
        self.y_embedding = nn.Embedding(y_vocab, d_model)
        self.transformer = Transformer(d_model, d_fc, n_heads, n_encoder_layers, n_decoder_layers)
        self.pe = PositionEmbedding(max_len, d_model)
        self.fc = nn.Linear(d_model, y_vocab)

    def forward(self, x, y, x_key_padding_mask=None, y_key_padding_mask=None, y_self_atten_mask=None, y_mem_key_padding_mask=None):
        x = self.x_embedding(x)
        x = self.pe(x)

        y = self.y_embedding(y)
        y = self.pe(y)

        x, y, attens = self.transformer(x, y, x_key_padding_mask, None, y_key_padding_mask, y_self_atten_mask, y_mem_key_padding_mask, None)

        y = self.fc(y)

        return y

def make_pair_data(nums, max_length):
    length = random.randint(1, max_length)
    x = np.random.choice(nums, length)
    y = np.zeros(x.shape, dtype=x.dtype)
    for i in range(len(x)):
        new_order = 0
        cur = x[i]
        for j in range(i):
            if cur >= x[j]:
                new_order += 1
        y[i] = new_order
    return x, y

class MyDataSet(Dataset):
    def __init__(self, num_data, max_val, max_length, pad_id, bos_id, eos_id):
        super(MyDataSet, self).__init__()
        self.num_data = num_data
        self.max_val = max_val
        self.max_length = max_length
        self.nums = range(max_val)
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        # norm sampling
        x, y = make_pair_data(self.nums, self.max_length)
        
        # add offset: pad/bos/eos
        x += 3
        y += 3

        # append pad/bos/eos
        x = torch.LongTensor(x.tolist() + [self.pad_id] * (self.max_length - len(x)))
        y_inp = torch.LongTensor([self.bos_id] + y.tolist() + [self.pad_id] * (self.max_length - len(y)))
        y_out = torch.LongTensor(y.tolist() + [self.eos_id] + [self.pad_id] * (self.max_length - len(y)))

        x_key_padding_mask = x.not_equal(self.pad_id)
        y_key_padding_mask = y_inp.not_equal(self.pad_id)
        y_length = y_inp.shape[0]
        y_self_atten_mask = torch.ones(y_length, y_length, dtype=torch.bool).tril(diagonal=0)
        y_mem_key_padding_mask = x.not_equal(self.pad_id)

        return x, y_inp, y_out, x_key_padding_mask, y_key_padding_mask, y_self_atten_mask, y_mem_key_padding_mask

if __name__ == '__main__':
    # model configs
    d_model = 256
    d_fc = d_model * 4
    n_heads = 8
    n_encoder_layers = 6
    n_decoder_layers = 6
    max_length = 6
    x_vocab = 100
    y_vocab = max_length

    # data configs
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2

    num_data = 100000
    batch_size = 320
    dataset = MyDataSet(num_data, x_vocab, max_length, PAD_ID, BOS_ID, EOS_ID)
    data_loader = DataLoader(dataset, batch_size = batch_size, shuffle=True, num_workers=0)

    # train configs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 5
    model = MyModel(max_length + 1, x_vocab + 3, y_vocab + 3, d_model, d_fc, n_heads, n_encoder_layers, n_decoder_layers)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    # train
    model.train()
    for epoch in range(epochs):
        for batch, (x, y_inp, y_out, x_key_padding_mask, y_key_padding_mask, y_self_atten_mask, y_mem_key_padding_mask) in enumerate(data_loader):
            x = x.to(device)
            y_inp = y_inp.to(device)
            y_out = y_out.to(device)
            x_key_padding_mask = x_key_padding_mask.to(device)
            y_key_padding_mask = y_key_padding_mask.to(device)
            y_self_atten_mask = y_self_atten_mask.to(device)[0]
            y_mem_key_padding_mask = y_mem_key_padding_mask.to(device)
            yp = model(x, y_inp, x_key_padding_mask, y_key_padding_mask, y_self_atten_mask, y_mem_key_padding_mask)

            loss = criterion(yp.view(-1, y_vocab + 3), y_out.view(-1))
            print(f'epoch: {(epoch + 1)}, batch: {(batch + 1)}, lr: {lr_scheduler.get_last_lr()[0]:.7f}, loss: {loss:.6f}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_scheduler.step()

    # val
    model.eval()
    with torch.no_grad():
        (x, y_inp, y_out, x_key_padding_mask, y_key_padding_mask, y_self_atten_mask, y_mem_key_padding_mask) = next(iter(data_loader))
        x = x.to(device)
        y_inp = y_inp.to(device)
        y_out = y_out.to(device)
        x_key_padding_mask = x_key_padding_mask.to(device)
        y_key_padding_mask = y_key_padding_mask.to(device)
        y_self_atten_mask = y_self_atten_mask.to(device)[0]
        y_mem_key_padding_mask = y_mem_key_padding_mask.to(device)
        yp = model(x, y_inp, x_key_padding_mask, y_key_padding_mask, y_self_atten_mask, y_mem_key_padding_mask)
        yp = F.softmax(yp, dim = -1)
        ypg = torch.argmax(yp, dim = -1)
        ypg[y_out == PAD_ID] = PAD_ID
        
        print(f'x: {x[0]}')
        print(f'y_inp: {y_inp[0]}')
        print(f'y_out: {y_out[0]}')
        print(f'ypg: {ypg[0]}')
```

输出：

```txt
epoch: 1, batch: 1, lr: 0.0001000, loss: 2.147824
epoch: 1, batch: 2, lr: 0.0001000, loss: 1.683637
epoch: 1, batch: 3, lr: 0.0001000, loss: 1.619123
epoch: 1, batch: 4, lr: 0.0001000, loss: 1.590135
...
epoch: 5, batch: 310, lr: 0.0000096, loss: 0.022584
epoch: 5, batch: 311, lr: 0.0000096, loss: 0.015408
epoch: 5, batch: 312, lr: 0.0000096, loss: 0.019144
epoch: 5, batch: 313, lr: 0.0000096, loss: 0.023588
x: tensor([76, 63, 90, 32, 18, 50], device='cuda:0')
y_inp: tensor([1, 3, 3, 5, 3, 3, 5], device='cuda:0')
y_out: tensor([3, 3, 5, 3, 3, 5, 2], device='cuda:0')
ypg: tensor([3, 3, 5, 3, 3, 5, 2], device='cuda:0')
```

### 4. BERT

> * BERT是一个encoder only的模型；
> * 目前BERT主流使用MLM进行预训练，然后再在下游进行token classify或者doc classify，本例中演示了同时进行MLM以及doc分类两个任务；

架构图：
<img src="bert1.png" alt="transformer" width="1000px" height="400px">
<img src="bert2.png" alt="transformer" width="1000px" height="300px">

代码：

```python
# coding: utf-8
# bert.py

import numpy as np
import copy
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformer import EncoderLayer, PositionEmbedding

class BERT(nn.Module):
    def __init__(self, d_model, d_fc, n_heads, n_layers):
        super(BERT, self).__init__()
        encoder_layer = EncoderLayer(n_heads, d_model, d_fc)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(n_layers)])

    def forward(self, x, key_padding_mask=None):
        for layer in self.layers:
            x, scores = layer(x, key_padding_mask)
        return x
    
################################################## exp ##################################################
class MyModel(nn.Module):
    def __init__(self, max_len, mlm_cls_num, doc_cls_num, d_model, d_fc, n_heads, n_layers) -> None:
        super(MyModel, self).__init__()
        self.token_embedding = nn.Embedding(mlm_cls_num, d_model)
        self.pos_embedding = PositionEmbedding(max_len, d_model)
        self.bert = BERT(d_model, d_fc, n_heads, n_layers)
        self.mlm_fc = nn.Linear(d_model, mlm_cls_num)
        self.cls_fc = nn.Linear(d_model, doc_cls_num)

    def forward(self, x, key_padding_mask=None):
        x = self.token_embedding(x)
        x = self.pos_embedding(x)
        x = self.bert(x, key_padding_mask)
        yp_mlm = self.mlm_fc(x)
        yp_cls = self.cls_fc(x[:, 0])
        return yp_mlm, yp_cls

def make_pair_data(max_val, max_length, mask_rate=0.2):
    length = random.randint(1, max_length)
    beg = random.randint(0, max_val - length)
    end = beg + length
    x = np.array(list(range(beg, end)))
    y = copy.deepcopy(x)
    for i in range(len(x)):
        if random.random() < mask_rate:
            x[i] = -1
    mask_idxes = (x == -1)
    # 均值的类型：0, 小于中值；1, 大于中值。
    doc_cls = 0 if y.mean() < max_val//2 else 1
    return x, y, mask_idxes, doc_cls

class MyDataSet(Dataset):
    def __init__(self, num_data, max_val, max_length, pad_id=0, mask_id=1, cls_id=2):
        super(MyDataSet, self).__init__()
        self.num_data = num_data
        self.max_val = max_val
        self.max_length = max_length
        self.pad_id = pad_id
        self.mask_id = mask_id
        self.cls_id = cls_id

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        # norm sampling
        x, y, mask_idxes, doc_cls = make_pair_data(self.max_val, self.max_length)

        # add offset: pad/mask/cls
        x += 3
        y += 3

        # reset mask
        x[mask_idxes] = self.mask_id

        # append pad/cls
        x = torch.LongTensor([self.cls_id] + x.tolist() + [self.pad_id] * (self.max_length - len(x)))
        y_mlm = torch.LongTensor([self.pad_id] + y.tolist() + [self.pad_id] * (self.max_length - len(y)))
        key_padding_mask = x.not_equal(self.pad_id)

        y_cls = torch.LongTensor([doc_cls])
        
        return x, y_mlm, key_padding_mask, y_cls
    
if __name__ == '__main__':
    # model configs
    d_model = 256
    d_fc = d_model * 4
    n_heads = 8
    n_layers = 6
    max_length = 15
    vocab = 100
    mlm_cls_num = vocab + 3
    doc_cls_num = 2

    # data configs
    PAD_ID = 0
    MASK_ID = 1
    CLS_ID = 2

    num_data = 100000
    batch_size = 320
    dataset = MyDataSet(num_data, vocab, max_length, PAD_ID, MASK_ID, CLS_ID)
    data_loader = DataLoader(dataset, batch_size = batch_size, shuffle=True, num_workers=0)

    # train configs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 5
    model = MyModel(max_length + 1, mlm_cls_num, doc_cls_num, d_model, d_fc, n_heads, n_layers)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    # train
    model.train()
    for epoch in range(epochs):
        for batch, (x, y_mlm, key_padding_mask, y_cls) in enumerate(data_loader):
            x = x.to(device)
            y_mlm = y_mlm.to(device)
            key_padding_mask = key_padding_mask.to(device)
            y_cls = y_cls.to(device)

            yp_mlm, yp_cls = model(x, key_padding_mask)

            loss = criterion(yp_mlm.view(-1, mlm_cls_num), y_mlm.view(-1)) + criterion(yp_cls.view(-1, doc_cls_num), y_cls.view(-1))
            print(f'epoch: {(epoch + 1)}, batch: {(batch + 1)}, lr: {lr_scheduler.get_last_lr()[0]:.7f}, loss: {loss:.6f}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_scheduler.step()

    # val
    model.eval()
    with torch.no_grad():
        (x, y_mlm, key_padding_mask, y_cls) = next(iter(data_loader))
        x = x.to(device)
        y_mlm = y_mlm.to(device)
        key_padding_mask = key_padding_mask.to(device)
        y_cls = y_cls.to(device)
        
        yp_mlm, yp_cls = model(x, key_padding_mask)

        yp_mlm = F.softmax(yp_mlm, dim = -1)
        yp_mlm = torch.argmax(yp_mlm, dim = -1)
        yp_mlm[y_mlm == PAD_ID] = PAD_ID

        yp_cls = F.softmax(yp_cls, dim = -1)
        yp_cls = torch.argmax(yp_cls, dim = -1)

        print(f'x: {x[0]}')
        print(f'y_mlm: {y_mlm[0]}')
        print(f'yp_mlm: {yp_mlm[0]}')
        print(f'y_cls: {y_cls[0]}')
        print(f'yp_cls: {yp_cls[0]}')
```

输出：

```txt
epoch: 1, batch: 1, lr: 0.0001000, loss: 5.913435
epoch: 1, batch: 2, lr: 0.0001000, loss: 5.010942
epoch: 1, batch: 3, lr: 0.0001000, loss: 4.733379
epoch: 1, batch: 4, lr: 0.0001000, loss: 4.635901
epoch: 1, batch: 5, lr: 0.0001000, loss: 4.531482
epoch: 1, batch: 6, lr: 0.0001000, loss: 4.476653
epoch: 1, batch: 7, lr: 0.0001000, loss: 4.427347
...
epoch: 5, batch: 310, lr: 0.0000096, loss: 0.018381
epoch: 5, batch: 311, lr: 0.0000096, loss: 0.015477
epoch: 5, batch: 312, lr: 0.0000096, loss: 0.032741
epoch: 5, batch: 313, lr: 0.0000096, loss: 0.026694
x: tensor([ 2, 94, 95, 1, 97,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       device='cuda:0')
y_mlm: tensor([ 0, 94, 95, 96, 97,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       device='cuda:0')
yp_mlm: tensor([ 0, 94, 95, 96, 97,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       device='cuda:0')
y_cls: tensor([1], device='cuda:0')
yp_cls: 1
```

### 5. GPT

架构图：
<img src="gpt.png" alt="transformer" width="700px" height="800px">

```python
# coding: utf-8
# gpt.py

import numpy as np
import copy
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformer import EncoderLayer, PositionEmbedding

class GPT(nn.Module):
    def __init__(self, d_model, d_fc, n_heads, n_layers, max_len, vocab_size):
        super(GPT, self).__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = PositionEmbedding(max_len, d_model)

        layer = EncoderLayer(n_heads, d_model, d_fc)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, key_padding_mask=None, atten_mask=None):
        x = self.token_embedding(x)
        x = self.pos_embedding(x)

        for layer in self.layers:
            x, scores = layer(x, key_padding_mask, atten_mask)
        
        x = self.fc(x)

        return x
    
################################################## exp ##################################################
def generate(gpt_model, start_number, max_seq_size, bos_id, eos_id):
    seq = [bos_id, start_number + 3]

    for i in range(max_seq_size):
        # as batch and as tensor
        inp_seq = torch.LongTensor([seq])
        inp_seq = inp_seq.to(device)
        out_seq = gpt_model(inp_seq)
        out_seq = F.softmax(out_seq, dim = -1)
        out_seq = torch.argmax(out_seq, dim = -1)
        g_id = out_seq[0][-1].item()
        seq.append(out_seq[0][-1].item())
        if g_id == eos_id:
            break
    return seq

def make_pair_data(max_val, max_length):
    length = random.randint(1, max_length)
    beg = random.randint(0, max_val - length)
    end = beg + length
    seq = np.array(list(range(beg, end)))
    return seq

class MyDataSet(Dataset):
    def __init__(self, num_data, max_val, max_length, magic_number=42, pad_id=0, bos_id=1, eos_id=2):
        super(MyDataSet, self).__init__()
        self.num_data = num_data
        self.max_val = max_val
        self.max_length = max_length
        self.magic_number = magic_number
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        # 给定一个起始数字，如果该数字大于magic_number，则输出EOS结束，否则输出这个数字的下一个数，直到当前数大于阈值
        assert self.max_length > 1

        # norm sampling
        inp_seq = make_pair_data(self.max_val, self.max_length - 1)

        # add offset: pad/bos/eos
        inp_seq += 3

        # add bos and shift
        out_seq = inp_seq.tolist() + [inp_seq[-1]+1]
        inp_seq = [self.bos_id] + inp_seq.tolist()

        # update eos/pad
        thre = (self.magic_number + 3)

        eos_idx = None
        for i in range(len(inp_seq)):
            if eos_idx is None and inp_seq[i] >= thre:
                eos_idx = i
            if eos_idx is not None:
                if i == eos_idx:
                    out_seq[i] = self.eos_id
                    continue
                else:
                    inp_seq[i] = self.pad_id
                    out_seq[i] = self.pad_id

        assert len(inp_seq) == len(out_seq)
        inp_seq = torch.LongTensor(inp_seq + [self.pad_id] * (self.max_length - len(inp_seq)))
        out_seq = torch.LongTensor(out_seq + [self.pad_id] * (self.max_length - len(out_seq)))
        
        key_padding_mask = inp_seq.not_equal(self.pad_id)
        length = inp_seq.shape[0]
        atten_mask = torch.ones(length, length, dtype=torch.bool).tril(diagonal=0)

        return inp_seq, out_seq, key_padding_mask, atten_mask
    
if __name__ == '__main__':
    # model configs
    d_model = 256
    d_fc = d_model * 4
    n_heads = 8
    n_layers = 6
    max_length = 16
    vocab_size = 100
    cls_num = vocab_size + 3

    # data configs
    MAGIC_NUMBER = 42
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2

    num_data = 100000
    batch_size = 320
    dataset = MyDataSet(num_data, vocab_size, max_length, MAGIC_NUMBER, PAD_ID, BOS_ID, EOS_ID)
    data_loader = DataLoader(dataset, batch_size = batch_size, shuffle=True, num_workers=0)

    # train configs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 3
    model = GPT(d_model, d_fc, n_heads, n_layers, max_length, cls_num)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    # train
    model.train()
    for epoch in range(epochs):
        for batch, (inp_seq, out_seq, key_padding_mask, atten_mask) in enumerate(data_loader):
            inp_seq = inp_seq.to(device)
            out_seq = out_seq.to(device)
            key_padding_mask = key_padding_mask.to(device)
            atten_mask = atten_mask.to(device)

            pout_seq = model(inp_seq, key_padding_mask, atten_mask)

            # ignore first token
            loss = criterion(pout_seq[:, 1:].reshape(-1, cls_num), out_seq[:, 1:].reshape(-1))
            print(f'epoch: {(epoch + 1)}, batch: {(batch + 1)}, lr: {lr_scheduler.get_last_lr()[0]:.7f}, loss: {loss:.6f}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_scheduler.step()

    # val
    model.eval()
    with torch.no_grad():
        (inp_seq, out_seq, key_padding_mask, atten_mask) = next(iter(data_loader))
        inp_seq = inp_seq.to(device)
        out_seq = out_seq.to(device)
        key_padding_mask = key_padding_mask.to(device)
        atten_mask = atten_mask.to(device)

        pout_seq = model(inp_seq, key_padding_mask, atten_mask)

        pout_seq = F.softmax(pout_seq, dim = -1)
        pout_seq = torch.argmax(pout_seq, dim = -1)
        pout_seq[out_seq == PAD_ID] = PAD_ID

        # ignore first token
        pout_seq[:, 0] = PAD_ID

        print(f'inp_seq: {inp_seq[0]}')
        print(f'out_seq: {out_seq[0]}')
        print(f'pout_seq: {pout_seq[0]}')

    # test
    model.eval()
    with torch.no_grad():
        while True:
            start_number = int(input('input first number: '))
            print(f'start_number: {start_number}')
            assert start_number >= 0 and start_number < vocab_size
            g_seq = generate(model, start_number, max_length-1, BOS_ID, EOS_ID)
            g_seq = [(id-3) for id in g_seq[1:]]
            print(f'generated sequence: {g_seq}')
```

输出：

```txt
epoch: 1, batch: 1, lr: 0.0001000, loss: 4.714213
epoch: 1, batch: 2, lr: 0.0001000, loss: 4.447280
epoch: 1, batch: 3, lr: 0.0001000, loss: 4.202495
epoch: 1, batch: 4, lr: 0.0001000, loss: 3.998471
epoch: 1, batch: 5, lr: 0.0001000, loss: 3.801060
epoch: 1, batch: 6, lr: 0.0001000, loss: 3.595473
epoch: 1, batch: 7, lr: 0.0001000, loss: 3.411419
...
epoch: 3, batch: 311, lr: 0.0000096, loss: 0.002071
epoch: 3, batch: 312, lr: 0.0000096, loss: 0.002044
epoch: 3, batch: 313, lr: 0.0000096, loss: 0.002077
inp_seq: tensor([ 1, 48,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       device='cuda:0')
out_seq: tensor([48,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       device='cuda:0')
pout_seq: tensor([0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0')

input first number: 23
start_number: 23
generated sequence: [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]

input first number: 34
start_number: 34
generated sequence: [34, 35, 36, 37, 38, 39, 40, 41, 42, -1]

input first number: 45
start_number: 45
generated sequence: [45, -1]

input first number: 40
start_number: 40
generated sequence: [40, 41, 42, -1]
```
