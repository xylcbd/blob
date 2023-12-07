---
title: 使用tensorflow构造RNN/SRU网络识别mnist数字
date: 2017-10-13 20:09:00
categories:
 - deep learning
tags:
 - tensorflow
 - deep learning
 - LSTM
 - SRU
---

### SRU

SRU(Simple Recurrent Unit)是近期一个新提出来的类似LSTM和GRU的处理单元结构。

论文：[https://arxiv.org/abs/1709.02755](https://arxiv.org/abs/1709.02755)

官方代码：[https://github.com/taolei87/sru](https://github.com/taolei87/sru)

本文代码：[https://github.com/xylcbd/tensorflow_mnist_sru](https://github.com/xylcbd/tensorflow_mnist_sru)

### LSTM vs SRU

依据论文解释，与LSTM相比，SRU具有最大的优势在于其运算速度。

SRU主要将最耗时的时间维上的矩阵乘法运算变成了Hadamard乘积运算（即element-wise product），因此速度极具优势，速度快10倍以上。

### SRU for mnist

sru.py

```python
#coding: utf-8
import math
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell

class SRUCell(RNNCell):
    def __init__(self, num_units, using_highway=True):        
        super(SRUCell, self).__init__()
        self._num_units = num_units
        self._using_highway = using_highway

    @property
    def state_size(self):
        return (self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, x, state, scope=None):
        if self._using_highway:
            return self.call_with_highway(x, state, scope)
        else:
            return self.call_without_highway(x, state, scope)

    def call_without_highway(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):                        
            c, _ = state
            x_size = x.get_shape().as_list()[1]
            
            W_u = tf.get_variable('W_u', [x_size, 3 * self.output_size])
            b_f = tf.get_variable('b_f', [self._num_units])
            b_r = tf.get_variable('b_r', [self._num_units])

            xh = tf.matmul(x, W_u)
            z, f, r = tf.split(xh, 3, 1)            

            f = tf.sigmoid(f + b_f)
            r = tf.sigmoid(r + b_r)            

            new_c = f * c + (1 - f) * z
            new_h = r * tf.tanh(new_c)

            return new_h, (new_c, new_h)

    def call_with_highway(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):                        
            c, _ = state
            x_size = x.get_shape().as_list()[1]
            
            W_u = tf.get_variable('W_u', [x_size, 4 * self.output_size])
            b_f = tf.get_variable('b_f', [self._num_units])
            b_r = tf.get_variable('b_r', [self._num_units])

            xh = tf.matmul(x, W_u)
            z, f, r, x = tf.split(xh, 4, 1)            

            f = tf.sigmoid(f + b_f)
            r = tf.sigmoid(r + b_r)            

            new_c = f * c + (1 - f) * z
            new_h = r * tf.tanh(new_c) + (1 - r) * x

            return new_h, (new_c, new_h)
```

train.py

```python
#coding: utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sru
import time
import os
import sys
import numpy as np

######################################
#tools
class Tools(object):
    @staticmethod
    def currect_time():
        return time.strftime("%H:%M:%S", time.localtime()) + '.%03d' % (time.time() % 1 * 1000)

    @staticmethod
    def log_print(content):
        print("[" + Tools.currect_time() + "] " + content)

######################################
#setting
class Setting(object):
    @staticmethod
    def checkpoint_dir():
        return "model"

######################################
#network
class Network(object):
    def __init__(self):
        # Network Parameters
        self.num_input = 28 # MNIST data input (img shape: 28*28)
        self.timesteps = 28 # timesteps
        self.num_hidden = 128 # hidden layer num of features
        self.num_classes = 10 # MNIST total classes (0-9 digits)
        self.lstm_layers = 2
        self.using_sru = sys.argv[1] == "SRU"
	print("Using SRU" if self.using_sru else "Using LSTM")

        # tf Graph input
        self.X = tf.placeholder("float", [None, self.timesteps, self.num_input])
        self.Y = tf.placeholder("float", [None, self.num_classes])

        # Define weights
        weights = {
            'out': tf.Variable(tf.random_normal([self.num_hidden, self.num_classes]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([self.num_classes]))
        }

        x = tf.unstack(self.X, self.timesteps, 1)

        # Define a lstm cell with tensorflow    
        if self.using_sru:
            rnn_cell = lambda: sru.SRUCell(self.num_hidden, False)
        else:
            rnn_cell = lambda: tf.nn.rnn_cell.LSTMCell(self.num_hidden, forget_bias=1.0)    

        cell_stack = tf.nn.rnn_cell.MultiRNNCell([rnn_cell() for _ in range(self.lstm_layers)], state_is_tuple=True)

        # Get lstm cell output
        outputs, _ = tf.nn.static_rnn(cell_stack, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        self.logits = tf.matmul(outputs[-1], weights['out']) + biases['out']

        self.prediction = tf.nn.softmax(self.logits)
    
    def get_input_ops(self):
        return self.X, self.Y

    def get_input_shape(self):
        return self.timesteps, self.num_input

    def get_output_ops(self):
        return self.logits, self.prediction

#####################################
#main route
def save_model(saver, sess, model_path):    
    Tools.log_print('save model to {0}.'.format(model_path))
    saver.save(sess, model_path)

def load_model(saver, sess, model_path):
    Tools.log_print('try to load model from {0}.'.format(model_path))    
    saver.restore(sess, model_path)
    Tools.log_print('load model success')
    return True

def train():
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    model_dir = 'model'
    model_path = os.path.join(model_dir, 'mnist_nn')

    network = Network()
    X, Y = network.get_input_ops()    
    timesteps, num_input = network.get_input_shape()
    logits, prediction = network.get_output_ops()

    # Training Parameters
    learning_rate = 1e-2
    display_step = 100
    train_epochs = 3
    train_batchsize = 128    
    test_batchsize = 128

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits= logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    grads = optimizer.compute_gradients(loss_op)
    max_grad_norm = 1.0
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss_op, tvars), max_grad_norm)
    train_op = optimizer.apply_gradients(zip(grads, tvars))


    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        saver = tf.train.Saver(tf.global_variables())
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        for epoch in range(1, train_epochs+1):
            train_steps = len(mnist.train.labels) / train_batchsize   
            for step in range(1, train_steps+1):
                batch_x, batch_y = mnist.train.next_batch(train_batchsize)
                batch_x = batch_x.reshape((train_batchsize, timesteps, num_input))
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
                if step % display_step == 0 or step == 1:
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
                    Tools.log_print("Epoch[%d/%d] Step[%d/%d] Train Minibatch Loss= %.4f, Training Accuracy= %.4f" % (epoch, train_epochs, step, train_steps, loss, acc))

        Tools.log_print("Optimization Finished!")
        
        save_model(saver, sess, model_path)

        if load_model(saver, sess, model_path):                     
            test_steps = len(mnist.test.labels) / test_batchsize   
            acc_list = []
            for step in range(1, test_steps+1):
                batch_x, batch_y = mnist.test.next_batch(test_batchsize)
                batch_x = batch_x.reshape((test_batchsize, timesteps, num_input))
                batch_acc = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y})
                acc_list.append(batch_acc)
            acc = np.mean(acc_list)
            Tools.log_print("Testing Accuracy: {0}".format(acc))


if __name__ == '__main__':
    train()
```

运行：

```shell
python train.py sru
```

或则会

```shell
python train.py lstm
```

结果日志：

```
--------------
LSTM network

[21:29:29.047] Epoch[1/3] Step[1/429] Train Minibatch Loss= 6.7037, Training Accuracy= 0.1484
[21:29:32.328] Epoch[1/3] Step[100/429] Train Minibatch Loss= 0.1882, Training Accuracy= 0.9453
[21:29:35.656] Epoch[1/3] Step[200/429] Train Minibatch Loss= 0.1174, Training Accuracy= 0.9688
[21:29:39.065] Epoch[1/3] Step[300/429] Train Minibatch Loss= 0.0988, Training Accuracy= 0.9609
[21:29:42.432] Epoch[1/3] Step[400/429] Train Minibatch Loss= 0.0790, Training Accuracy= 0.9766
[21:29:43.556] Epoch[2/3] Step[1/429] Train Minibatch Loss= 0.0739, Training Accuracy= 0.9766
[21:29:46.850] Epoch[2/3] Step[100/429] Train Minibatch Loss= 0.1305, Training Accuracy= 0.9609
[21:29:50.155] Epoch[2/3] Step[200/429] Train Minibatch Loss= 0.0396, Training Accuracy= 0.9922
[21:29:53.420] Epoch[2/3] Step[300/429] Train Minibatch Loss= 0.0611, Training Accuracy= 0.9688
[21:29:56.757] Epoch[2/3] Step[400/429] Train Minibatch Loss= 0.0499, Training Accuracy= 0.9766
[21:29:57.765] Epoch[3/3] Step[1/429] Train Minibatch Loss= 0.0275, Training Accuracy= 0.9922
[21:30:01.142] Epoch[3/3] Step[100/429] Train Minibatch Loss= 0.0119, Training Accuracy= 1.0000
[21:30:04.452] Epoch[3/3] Step[200/429] Train Minibatch Loss= 0.0285, Training Accuracy= 1.0000
[21:30:07.776] Epoch[3/3] Step[300/429] Train Minibatch Loss= 0.0105, Training Accuracy= 1.0000
[21:30:11.112] Epoch[3/3] Step[400/429] Train Minibatch Loss= 0.0546, Training Accuracy= 0.9766
[21:30:12.083] Optimization Finished!
[21:30:12.083] save model to model/mnist_nn.
[21:30:12.816] try to load model from model/mnist_nn.
[21:30:12.871] load model success
[21:30:13.947] Testing Accuracy: 0.980368614197

-------------
SRU network

[21:28:22.461] Epoch[1/3] Step[1/429] Train Minibatch Loss= 2.2775, Training Accuracy= 0.2344
[21:28:25.181] Epoch[1/3] Step[100/429] Train Minibatch Loss= 0.5900, Training Accuracy= 0.7812
[21:28:27.940] Epoch[1/3] Step[200/429] Train Minibatch Loss= 0.3459, Training Accuracy= 0.8906
[21:28:30.782] Epoch[1/3] Step[300/429] Train Minibatch Loss= 0.1854, Training Accuracy= 0.9609
[21:28:33.651] Epoch[1/3] Step[400/429] Train Minibatch Loss= 0.1230, Training Accuracy= 0.9531
[21:28:34.636] Epoch[2/3] Step[1/429] Train Minibatch Loss= 0.1979, Training Accuracy= 0.9453
[21:28:37.388] Epoch[2/3] Step[100/429] Train Minibatch Loss= 0.1033, Training Accuracy= 0.9688
[21:28:40.225] Epoch[2/3] Step[200/429] Train Minibatch Loss= 0.1192, Training Accuracy= 0.9609
[21:28:43.135] Epoch[2/3] Step[300/429] Train Minibatch Loss= 0.0294, Training Accuracy= 0.9922
[21:28:46.066] Epoch[2/3] Step[400/429] Train Minibatch Loss= 0.0988, Training Accuracy= 0.9766
[21:28:46.929] Epoch[3/3] Step[1/429] Train Minibatch Loss= 0.0817, Training Accuracy= 0.9922
[21:28:49.824] Epoch[3/3] Step[100/429] Train Minibatch Loss= 0.0673, Training Accuracy= 0.9922
[21:28:52.707] Epoch[3/3] Step[200/429] Train Minibatch Loss= 0.0836, Training Accuracy= 0.9766
[21:28:55.718] Epoch[3/3] Step[300/429] Train Minibatch Loss= 0.0565, Training Accuracy= 0.9844
[21:28:58.637] Epoch[3/3] Step[400/429] Train Minibatch Loss= 0.0340, Training Accuracy= 0.9922
[21:28:59.511] Optimization Finished!
[21:28:59.511] save model to model/mnist_nn.
[21:29:00.364] try to load model from model/mnist_nn.
[21:29:00.428] load model success
[21:29:01.153] Testing Accuracy: 0.97185498476
```



### 总结

目前测试下来SRU确实比LSTM快很多（当然，这段mnist的代码体现不出优势，因为并没有重构整个RNN网络）。但是相比LSTM，SRU可能收敛更慢一点，精度也些逊一点。如果非常在意速度的话，可以考虑使用SRU网络替换LSTM网络。