---
title: TensorFlow踩坑笔记
date: 2017-06-28 21:09:00
categories:
 - TensorFlow
tags:
 - TensorFlow
 - 踩坑
---

古有夜航船，今有踩坑集。

## 踩坑集

### tf.train.Saver().restore(sess, model_path)与sess.run(tf.global_variables_initializer())孰先孰后
两者并存时，sess.run(tf.global_variables_initializer())为先，否则加载的模型参数被重新初始化了，即白加载了。
