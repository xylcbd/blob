---
title: TensorFlow多任务多标签分类
date: 2017-12-23 15:09:00
categories:
 - TensorFlow
tags:
 - TensorFlow
 - MultiLabel
---



本文是应用TensorFlow进行多任务&多标签分类的一个示例(TensorFlow Multi-Task Multi-Label Example)。

以Mnist数据集为燃料，以简单的MLP为引擎，代码走起。

本文代码见：[https://github.com/xylcbd/blogs_code/blob/master/tensorflow-multitask-multilabel/main.py](https://github.com/xylcbd/blogs_code/blob/master/tensorflow-multitask-multilabel/main.py)

### 任务概述

任务共有2个，分别如下；

* 第1个任务是识别数字（即10类分类问题）
* 第2个任务是识别属性（图片中的数字是否是奇数、图片中的数字是否大于5）

可以看到第2个任务是有多个属性，即多标签任务。

* [0, 0]代表[不是奇数, 不大于5]
* [1, 0]代表[是奇数, 不大于5]
* [0, 1]代表[不是奇数, 大于5]
* [1, 1]代表[是奇数, 大于5]

属性可以再增加，不是固定为2个属性。

### 模型概述

简单的解释一下。

这里模型采用的是一个3层MLP模型，前2层共用，最后一层为2个任务各自所有。

* 对于预测数字的任务，采用的是tf.losses.softmax_cross_entropy
* 对于预测属性的任务，采用的是tf.losses.sigmoid_cross_entropy



### 示例代码

```python
#coding: utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
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
        self.num_class = 10 # MNIST total class (0-9 digits)
        self.num_attrs = 2 #MNIST number attrs, is_odd_number & is_larger_than_5
        self.num_hidden = 128

        #inputs <-> goldens
        self.inputs = tf.placeholder("float", [None, self.num_input * self.num_input])
        self.golden_class = tf.placeholder("float", [None, self.num_class])
        self.golden_attrs = tf.placeholder("float", [None, self.num_attrs])

        out = self.fc('fc1', self.inputs, self.num_input*self.num_input, self.num_hidden, True)
        out = self.fc('fc2', out, self.num_hidden, self.num_hidden, True)

        #last tail layer
        self.logits_class = self.fc('fc_class', out, self.num_hidden, self.num_class, False)
        self.logits_attrs = self.fc('fc_attrs', out, self.num_hidden, self.num_attrs, False)

    def get_weight_varible(self, name, shape):
        return tf.get_variable(name, shape=shape,
                       initializer=tf.contrib.layers.xavier_initializer())

    def get_bias_varible(self, name, shape):
        return tf.get_variable(name, shape=shape,
                           initializer=tf.contrib.layers.xavier_initializer())
    
    def fc(self, layer_name, x, inp_dim, out_dim, with_act):
        with tf.variable_scope(layer_name):
            y = tf.reshape(x, shape=[-1, inp_dim])
            w = self.get_weight_varible('w', [inp_dim, out_dim])
            b = self.get_bias_varible('b', [out_dim])
            y = tf.add(tf.matmul(y, w), b)
            if with_act:
                y = tf.nn.relu(y)
        return y

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

def custom_lables(batch_class):
    batch_attrs = []
    for label in batch_class:
        idx = np.argmax(label)
        attrs = [float(idx % 2), float(idx > 5)]
        batch_attrs.append(attrs)
    return batch_class, batch_attrs 

def train():
    Tools.log_print('loading dataset...')
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    Tools.log_print('loade dataset success.\n')

    Tools.log_print('building network...')
    model_dir = 'model'
    model_path = os.path.join(model_dir, 'mnist_MLP')
    network = Network()
    #training parameters
    num_input = 28
    learning_rate = 0.01
    display_step = 100
    train_epochs = 10
    train_batchsize = 64
    test_batchsize = 64

    #loss of class
    loss_class_op = tf.reduce_mean(tf.losses.softmax_cross_entropy(
        logits= network.logits_class, onehot_labels=network.golden_class))
    #loss of attrs
    loss_attrs_op = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
        logits= network.logits_attrs, multi_class_labels=network.golden_attrs))
    #total loss
    loss_op = loss_class_op + loss_attrs_op

    #optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #train operator
    train_op = optimizer.minimize(loss_op)

    #accuracy of class
    correct_class_op = tf.equal(tf.argmax(network.logits_class, 1), tf.argmax(network.golden_class, 1))
    accuracy_class_op = tf.reduce_mean(tf.cast(correct_class_op, tf.float32))
    #accuracy of attrs
    correct_attrs_op = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(network.logits_attrs), 0.5), tf.int32), tf.cast(network.golden_attrs, tf.int32))
    accuracy_attrs_op = tf.reduce_mean(tf.reduce_min(tf.cast(correct_attrs_op, tf.float32), 1))

    #initialize the variables (i.e. assign their default value)
    init_op = tf.global_variables_initializer()
    Tools.log_print('build network success.\n')

    #start training
    with tf.Session() as sess:
        #run the initializer
        sess.run(init_op)
        
        #model dir
        saver = tf.train.Saver(tf.global_variables())
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        #run epochs
        Tools.log_print('start training...')
        for epoch in range(1, train_epochs+1):
            train_steps = len(mnist.train.labels) / train_batchsize   
            total_loss = 0.0
            total_acc_class = 0.0
            total_acc_attrs = 0.0
            for step in range(1, train_steps+1):
                batch_x, batch_y = mnist.train.next_batch(train_batchsize)
                batch_class, batch_attrs = custom_lables(batch_y)
                batch_x = batch_x.reshape((train_batchsize, num_input * num_input))
                _, batch_loss, batch_acc_class, batch_acc_attrs = sess.run([train_op, loss_op, accuracy_class_op, accuracy_attrs_op], feed_dict={network.inputs: batch_x, network.golden_class: batch_class, network.golden_attrs: batch_attrs})
                total_loss += batch_loss
                total_acc_class += batch_acc_class
                total_acc_attrs += batch_acc_attrs
                if step % display_step == 0:
                    avg_loss = total_loss / display_step
                    avg_acc_class = total_acc_class / display_step
                    avg_acc_attrs = total_acc_attrs / display_step
                    Tools.log_print("Epoch[%d/%d] Step[%d/%d] Train Minibatch Loss= %.4f, Class Accuracy= %.4f, Attrs Accuracy= %.4f" % (epoch, train_epochs, step, train_steps, avg_loss, avg_acc_class, avg_acc_attrs))
                    total_loss = 0.0
                    total_acc_class = 0.0
                    total_acc_attrs = 0.0
        Tools.log_print("finished training!")
        
        #save model
        save_model(saver, sess, model_path)

        #load model and test
        Tools.log_print('start testing...')
        if load_model(saver, sess, model_path):                     
            test_steps = len(mnist.test.labels) / test_batchsize   
            acc_class_list = []
            acc_attrs_list = []
            for step in range(1, test_steps+1):
                batch_x, batch_y = mnist.test.next_batch(test_batchsize)                
                batch_x = batch_x.reshape((test_batchsize, num_input * num_input))
                batch_class, batch_attrs = custom_lables(batch_y)
                batch_acc_class, batch_acc_attrs = sess.run([accuracy_class_op, accuracy_attrs_op], feed_dict={network.inputs: batch_x, network.golden_class: batch_class, network.golden_attrs: batch_attrs})
                acc_class_list.append(batch_acc_class)
                acc_attrs_list.append(batch_acc_attrs)
            acc_class = np.mean(acc_class_list)
            acc_attrs = np.mean(acc_attrs_list)
            Tools.log_print("Testing Class Accuracy: %.4f, Attrs Accuracy: %.4f" % (acc_class, acc_attrs))
        Tools.log_print('finished testing...')

if __name__ == '__main__':
    train()

```



### 模型效果

```
[17:46:51.972] Epoch[10/10] Step[100/859] Train Minibatch Loss= 0.1084, Class Accuracy= 0.9791, Attrs Accuracy= 0.9848
[17:46:52.442] Epoch[10/10] Step[200/859] Train Minibatch Loss= 0.1188, Class Accuracy= 0.9802, Attrs Accuracy= 0.9852
[17:46:52.905] Epoch[10/10] Step[300/859] Train Minibatch Loss= 0.1122, Class Accuracy= 0.9797, Attrs Accuracy= 0.9859
[17:46:53.357] Epoch[10/10] Step[400/859] Train Minibatch Loss= 0.1019, Class Accuracy= 0.9814, Attrs Accuracy= 0.9875
[17:46:53.833] Epoch[10/10] Step[500/859] Train Minibatch Loss= 0.0833, Class Accuracy= 0.9841, Attrs Accuracy= 0.9886
[17:46:54.297] Epoch[10/10] Step[600/859] Train Minibatch Loss= 0.1323, Class Accuracy= 0.9745, Attrs Accuracy= 0.9828
[17:46:54.759] Epoch[10/10] Step[700/859] Train Minibatch Loss= 0.1333, Class Accuracy= 0.9752, Attrs Accuracy= 0.9808
[17:46:55.224] Epoch[10/10] Step[800/859] Train Minibatch Loss= 0.1227, Class Accuracy= 0.9767, Attrs Accuracy= 0.9838
[17:46:55.493] finished training!
[17:46:55.493] save model to model/mnist_MLP.
[17:46:55.584] start testing...
[17:46:55.584] try to load model from model/mnist_MLP.
[17:46:55.636] load model success
[17:46:56.014] Testing Class Accuracy: 0.9637, Attrs Accuracy: 0.9722
[17:46:56.014] finished testing...
```

