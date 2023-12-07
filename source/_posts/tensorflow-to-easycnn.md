---
title: TensorFlow模型转换为EasyCNN模型
date: 2017-06-24 14:09:00
categories:
 - 算法
tags:
 - 算法
 - 深度学习
 - EasyCNN
 - TensorFlow
---

[EasyCNN](https://github.com/xylcbd/EasyCNN)是一个轻量级的CNN框架，纯C++11编写，不依赖于任何库，可跨平台应用于Linux/Windows/Android/iOS等平台。  

[TensorFlow](https://www.tensorflow.org/)是Google开发的深度学习框架，由专业工程人员与算法开发人员合作开发而成，是目前最火的开源深度学习框架。  

EasyCNN具有极轻量级的优点，很容易port到任何使用场景，而TensorFlow设计训练模型非常方便高效。  

因此本文结合TensorFlow与EasyCNN进行模型的训练与部署：
1. 采用TensorFlow设计训练模型；
2. 将TensorFlow模型转换为EasyCNN模型；
3. 部署EasyCNN模型到实际业务中。

TensorFlow模型代码：

```python
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
	
def build_cnn(x,test=False):
	x_image = tf.reshape(x, [-1,std_height,std_width,1])
	
	#conv-pool 1x48x48 -> 6x24x24
	W_conv1 = weight_variable([3, 3, 1, 6])      
	b_conv1 = bias_variable([6])       
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool(h_conv1)

	#conv-pool 6x24x24 -> 12x12x12
	W_conv2 = weight_variable([3, 3, 6, 12])
	b_conv2 = bias_variable([12])	
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool(h_conv2)
	
	#conv-pool 12x12x12 -> 24x6x6
	W_conv3 = weight_variable([3, 3, 12, 24])
	b_conv3 = bias_variable([24])
	h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
	h_pool3 = max_pool(h_conv3)

	#conv-pool 24x6x6 -> 36x3x3
	W_conv4 = weight_variable([3, 3, 24, 36])
	b_conv4 = bias_variable([36])
	h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
	h_pool4 = max_pool(h_conv4)
	
	h_trans = tf.transpose(h_pool4, [0,3,1,2])
	h_pool_flat = tf.reshape(h_trans, [-1, 3 * 3 * 36])
	
	W_fc1 = weight_variable([3 * 3 * 36, 512])
	b_fc1 = bias_variable([512])		
	h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

	if test:
		h_fc1_drop = h_fc1
	else:
		h_fc1_drop = tf.nn.dropout(h_fc1, 0.5)

	W_fc2 = weight_variable([512, classes])
	b_fc2 = bias_variable([classes])
	y_predict=tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	
	probs = tf.nn.softmax(y_predict)
	
	params = (W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_conv4,b_conv4,W_fc1,b_fc1,W_fc2,b_fc2)
	return y_predict,params
```

对应的EasyCNN模型代码：

```c++
static EasyCNN::NetWork buildConvNet(const size_t batch, const size_t channels, const size_t width, const size_t height)
{
	const int classes = label_map.size();

	EasyCNN::NetWork network;
	network.setInputSize(EasyCNN::DataSize(batch, channels, width, height));
	//input data layer 0
	std::shared_ptr<EasyCNN::InputLayer> _0_inputLayer(std::make_shared<EasyCNN::InputLayer>());
	network.addayer(_0_inputLayer);

	//convolution layer 1
	std::shared_ptr<EasyCNN::ConvolutionLayer> _1_convLayer(std::make_shared<EasyCNN::ConvolutionLayer>());
	_1_convLayer->setParamaters(EasyCNN::ParamSize(32, 1, 3, 3), 1, 1, true,EasyCNN::ConvolutionLayer::SAME);
	network.addayer(_1_convLayer);	
	//pooling layer 1
	std::shared_ptr<EasyCNN::PoolingLayer> _1_poolingLayer(std::make_shared<EasyCNN::PoolingLayer>());
	_1_poolingLayer->setParamaters(EasyCNN::PoolingLayer::PoolingType::MaxPooling, EasyCNN::ParamSize(1, 32, 2, 2), 2, 2,EasyCNN::PoolingLayer::SAME);
	network.addayer(_1_poolingLayer);
	network.addayer(std::make_shared<EasyCNN::ReluLayer>());

	//convolution layer 2
	std::shared_ptr<EasyCNN::ConvolutionLayer> _2_convLayer(std::make_shared<EasyCNN::ConvolutionLayer>());
	_2_convLayer->setParamaters(EasyCNN::ParamSize(64, 32, 3, 3), 1, 1, true, EasyCNN::ConvolutionLayer::SAME);
	network.addayer(_2_convLayer);
	//pooling layer 2
	std::shared_ptr<EasyCNN::PoolingLayer> _2_pooingLayer(std::make_shared<EasyCNN::PoolingLayer>());
	_2_pooingLayer->setParamaters(EasyCNN::PoolingLayer::PoolingType::MaxPooling, EasyCNN::ParamSize(1, 64, 2, 2), 2, 2, EasyCNN::PoolingLayer::SAME);
	network.addayer(_2_pooingLayer);
	network.addayer(std::make_shared<EasyCNN::ReluLayer>());

	//convolution layer 3
	std::shared_ptr<EasyCNN::ConvolutionLayer> _3_convLayer(std::make_shared<EasyCNN::ConvolutionLayer>());
	_3_convLayer->setParamaters(EasyCNN::ParamSize(64, 64, 3, 3), 1, 1, true, EasyCNN::ConvolutionLayer::SAME);
	network.addayer(_3_convLayer);	
	//pooling layer 3
	std::shared_ptr<EasyCNN::PoolingLayer> _3_pooingLayer(std::make_shared<EasyCNN::PoolingLayer>());
	_3_pooingLayer->setParamaters(EasyCNN::PoolingLayer::PoolingType::MaxPooling, EasyCNN::ParamSize(1, 32, 2, 2), 2, 2, EasyCNN::PoolingLayer::SAME);
	network.addayer(_3_pooingLayer);
	network.addayer(std::make_shared<EasyCNN::ReluLayer>());

	//convolution layer 4
	std::shared_ptr<EasyCNN::ConvolutionLayer> _4_convLayer(std::make_shared<EasyCNN::ConvolutionLayer>());
	_4_convLayer->setParamaters(EasyCNN::ParamSize(64, 64, 3, 3), 1, 1, true, EasyCNN::ConvolutionLayer::SAME);
	network.addayer(_4_convLayer);
	//pooling layer 4
	std::shared_ptr<EasyCNN::PoolingLayer> _4_pooingLayer(std::make_shared<EasyCNN::PoolingLayer>());
	_4_pooingLayer->setParamaters(EasyCNN::PoolingLayer::PoolingType::MaxPooling, EasyCNN::ParamSize(1, 64, 2, 2), 2, 2, EasyCNN::PoolingLayer::SAME);
	network.addayer(_4_pooingLayer);
	network.addayer(std::make_shared<EasyCNN::ReluLayer>());

	//full connect layer 6
	std::shared_ptr<EasyCNN::FullconnectLayer> _6_fcLayer(std::make_shared<EasyCNN::FullconnectLayer>());
	_6_fcLayer->setParamaters(EasyCNN::ParamSize(1, 512, 1, 1), true);
	network.addayer(_6_fcLayer);
	network.addayer(std::make_shared<EasyCNN::ReluLayer>());

	//full connect layer 7
	std::shared_ptr<EasyCNN::FullconnectLayer> _7_fcLayer(std::make_shared<EasyCNN::FullconnectLayer>());
	_7_fcLayer->setParamaters(EasyCNN::ParamSize(1, classes, 1, 1), true);
	network.addayer(_7_fcLayer);

	//soft max layer 6
	std::shared_ptr<EasyCNN::SoftmaxLayer> _6_softmaxLayer(std::make_shared<EasyCNN::SoftmaxLayer>());
	network.addayer(_6_softmaxLayer);

	return network;
}
```

抽取TensorFlow模型参数转换为EasyCNN模型参数。TensorFlow模型参数存储在tf.Variable中，是一种类Numpy的Tensor。EasyCNN模型参数是多维数组，非常简单的结构。 
 
下面是模型转换代码：

```python 
def export_input(f,channel,width,height):
	f.write('InputLayer %d %d %d\n' % (channel,width,height))
	
def export_layer(f,name):
	f.write(name + '\n')
	
def export_conv(f,conv_weight,conv_bias,padding_type):
	#ConvolutionLayer 64 64 3 3 1 1 1 -0.0533507
	print('export conv layer.')
	print(conv_weight.shape)			
	print(conv_bias.shape)	
	#hwcn -> nchw
	#3x3x1x32 -> 32x1x3x3
	conv_weight = np.transpose(conv_weight,[3,2,0,1])
	oc,ic,kw,kh,sw,sh,bias = conv_weight.shape[0],conv_weight.shape[1],conv_weight.shape[2],conv_weight.shape[3],1,1,1
	f.write('ConvolutionLayer %d %d %d %d %d %d %d %d ' % (oc,ic,kw,kh,sw,sh,bias,padding_type))				
	f.write(' '.join(map(str,conv_weight.flatten().tolist())) + ' ')		
	f.write(' '.join(map(str,conv_bias.flatten().tolist())) + ' ')
	f.write('\n')
	
def export_pool(f,channel):
	#PoolingLayer [pool_type] 1 32 2 2 2 2 
	f.write('PoolingLayer 0 1 %d 2 2 2 2\n' % (channel))
	
def export_fc(f,fc_weight,fc_bias):
	#FullconnectLayer 1 512 1 1 1 0.139041
	print(fc_weight.shape)			
	print(fc_bias.shape)
	f.write('FullconnectLayer 1 %d 1 1 1 ' % fc_bias.shape[0])
	fc_weight = np.transpose(fc_weight,[1,0])
	f.write(' '.join(map(str,fc_weight.flatten().tolist())) + ' ')		
	f.write(' '.join(map(str,fc_bias.flatten().tolist())) + ' ')
	f.write('\n')
	
def export_model(tf_model_path,easycnn_model_path):	
	x = tf.placeholder(tf.float32, [None,std_width*std_height])
	predict,params = build_cnn(x)
	y = tf.placeholder(tf.float32, [None,classes])

	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess,tf_model_path)
		
		f = open(easycnn_model_path,'w')	
		#input
		export_input(f,1,std_width,std_height)
		#conv1
		conv_weight = params[0].eval()
		conv_bias = params[1].eval()		
		export_conv(f,conv_weight,conv_bias,1)		
		export_layer(f,'ReluLayer')
		export_pool(f,conv_weight.shape[3])
		
		#conv2
		conv_weight = params[2].eval()
		conv_bias = params[3].eval()		
		export_conv(f,conv_weight,conv_bias,1)		
		export_layer(f,'ReluLayer')
		export_pool(f,conv_weight.shape[3])
		
		#conv3
		conv_weight = params[4].eval()
		conv_bias = params[5].eval()		
		export_conv(f,conv_weight,conv_bias,1)		
		export_layer(f,'ReluLayer')
		export_pool(f,conv_weight.shape[3])
		
		#conv4
		conv_weight = params[6].eval()
		conv_bias = params[7].eval()		
		export_conv(f,conv_weight,conv_bias,1)		
		export_layer(f,'ReluLayer')
		export_pool(f,conv_weight.shape[3])
		
		#fc1
		fc_weight = params[8].eval()
		fc_bias = params[9].eval()		
		export_fc(f,fc_weight,fc_bias)
		export_layer(f,'ReluLayer')
		
		#fc2
		fc_weight = params[10].eval()
		fc_bias = params[11].eval()		
		export_fc(f,fc_weight,fc_bias)		
		export_layer(f,'SoftmaxLayer')
```

这种方法能很好的利用TensorFlow进行高效的GPU模型训练，同时方便的部署到任何环境。
