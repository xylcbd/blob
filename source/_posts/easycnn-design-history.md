---
title: EasyCNN的设计实现
date: 2016-11-12 20:37:09
categories:
 - 软件架构
 - 深度学习
tags:
 - 软件架构
 - 深度学习
---

# EasyCNN介绍
[EasyCNN](https://github.com/xylcbd/EasyCNN)是一个练手之作，权当熟悉CNN（[Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network))。  

当然这个项目其实也可以用于实际业务场景，如果对性能要求不高的话:) 。  

EasyCNN完全由跨平台c++11代码构成，目前可运行在windows和android系统下，linux的Makefile还没写，不过应该很容易port到其他平台。  

EasyCNN提供了CNN的一些基本功能，包括：  
* 基本层：data，convolution，pooling，full-connect,softmax
* 激活函数：ReLU，Tanh，Sigmoid
* Loss函数：MSE，Cross Entropy
* 优化方法：SGD
* 训练、测试、保存、载入CNN模型

非常的toy，:)，不过，对于理解CNN或者一些简单的应用场景，差不多是够了。

未来可能会继续优化它，包括卷积的Winograd优化，使用成熟BLAS库，多线程优化等。

# EasyCNN实现
有云：麻雀虽小，五脏俱全。虽然是个小项目，但是它已经基本拥有了现代软件中该有的基本构件，下面逐条来说。

## License
这里用的是听起来就很厉害的[WTFPL](http://www.wtfpl.net/)(Do What the Fuck You Want to Public License)，直译就是“你想干嘛就干嘛的许可证”。  

商业项目就不说了，用不着License。开源项目基本都需要一个License，权责要清晰，不然别人还真不敢用（当然，国内另说，管你什么License拿过来就用）。主要的License可见[此处](http://choosealicense.com/licenses/)，包括：GPL/LGPL、MIT、Apache、BSD等。  

这里随便选了个WTFPL，hhhhha...

## 目录
c/c++项目的目录组织一般包括：  
* ReadMe：/README.md，主要是项目的一些介绍
* License：/WTFPL.LICENSE，许可证
* 头文件：/header/EasyCNN，SDK发布时的头文件
* 源文件：/src，实现部分
* 文档：/doc，文档，这里文档还是比较少，基本都放在/README.md中了
* 构件脚本：/jni、/msvc，相关平台下的构件脚本
* 例子：/examples，示例程序
* 资源：/res，示例程序或者其他需要用到的资源文件

基本看多了github就知道一个项目目录大概应该长什么样子了，依着具体项目需要增减目录。

## 文档
文档由很多部分组成，一般包括：项目介绍、API文档、示例程序解释、设计实现文档等。 

这里暂时只有项目介绍 :( 。

## 测试
测试一般包含2部分：单元测试和功能测试。  

这里只做了人肉功能测试，大家不要学我，有时间的话还是要写好单元测试。单元测试的好处多多：
* 结构清晰
> 结构不清晰的根本没法做单元测试hhhha...

* 重构方便
> 有了单元测试，就可以放心大胆的做重构了，不用怕弄成一团麻遗漏什么方向的测试。

当然，单元测试也有个粒度问题。粒度太细，可能需要写很多单元测试代码，比较麻烦，而且重构的时候不少测试例程用不着了需要删掉；粒度太粗，单元测试不好写，容易遗漏。这个还是要自己把握一下。

## 架构
咦，终于说回EasyCNN了。  

EasyCNN是一个类caffe的第一代深度学习框架，即框架以层（Layer）作为基础组成部分，网络由层堆叠而成。至于第二代深度学习框架，其实已经不只是深度学习了，应该叫做数值计算框架，如tensorflow等，我也写过一个类似的小型的，以后有机会的话可以贴出来看看。  

深度学习框架需要抽象出来的几个概念，在EasyCNN中分别有实现，下面细说。

* 数据  
> 数据包含2部分，一部分是输入数据（如图像数据）和回传梯度等，另一部分是网络参数等。  
> 这里将这两部分数据分别抽象出数据结构（代码有精简）。  
> 图像数据的抽象：DataBucket  
```c++
namespace EasyCNN
{
	struct DataSize
	{
	public:
		size_t number = 0;
		size_t channels = 0;
		size_t width = 0;
		size_t height = 0;
	};
	class DataBucket
	{
	public:
		DataSize size;
		std::shared_ptr<float> data;
	};
}
```
> 网络参数的抽象：ParamBucket
```c++
namespace EasyCNN
{
	struct ParamSize
	{
	public:
		size_t number = 0;
		size_t channels = 0;
		size_t width = 0;
		size_t height = 0;
	};
	class ParamBucket
	{
	public:
		ParamSize size;
		std::shared_ptr<float> data;
	};
}
```
> 可以看出两者其实是一样的结构，主要是因为CNN网络中的卷积核也需要number/channels/width/height等元信息，为了包容统一，所以ParamBucket和DataBucket差不多。在最新的一些框架中这些元信息已经被统一包含在Shape抽象结构中了，Shape一般是一个int数组。由层自己定义参数Shape和数据Shape，运行时层自己依据Shape取出处理，所谓如人饮水冷暖自知。

* 层（Layer）  
> 层（Layer），第一代深度学习框架的基本概念之一。一般每个层包含forward和backward，分别对应前向和后向的数据流。  
> EasyCNN中模仿caffe，把激活函数也单独抽取出来作为层了。
> EasyCNN中层的抽象是这样的：  
```c++
namespace EasyCNN
{
#define DECLARE_LAYER_TYPE static const std::string layerType;
#define DEFINE_LAYER_TYPE(class_type,type_string) const std::string class_type::layerType = type_string;
#define FRIEND_WITH_NETWORK friend class NetWork;
	enum class Phase
	{
		Train,
		Test
	};
	class Layer
	{
		FRIEND_WITH_NETWORK
	protected:
		virtual std::string getLayerType() const = 0;
		virtual std::string serializeToString() const{ return getLayerType(); };
		virtual void serializeFromString(const std::string content){/*nop*/};
		//phase
		inline void setPhase(Phase phase) { this->phase = phase; }
		inline Phase getPhase() const{ return phase; }
		//learning rate
		inline void setLearningRate(const float learningRate){ this->learningRate = learningRate; }
		inline float getLearningRate() const{ return learningRate; }
		//size
		inline void setInputBucketSize(const DataSize size){ inputSize = size; }
		inline DataSize getInputBucketSize() const{ return inputSize; }
		inline void setOutpuBuckerSize(const DataSize size){ outputSize = size; }
		inline DataSize getOutputBucketSize() const{ return outputSize; }
		//solve params
		virtual void solveInnerParams(){ outputSize = inputSize; }
		//data flow		
		virtual void forward(const std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket) = 0;
		virtual void backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket, std::shared_ptr<DataBucket>& nextDiffBucket) = 0;
	private:
		Phase phase = Phase::Train;
		DataSize inputSize;
		DataSize outputSize;
		float learningRate = 0.1f;
	};
}
```

* 损失函数（Loss）  
>  损失函数，是CNN中反向传播的起始点，将CNN的残差往前传递。
> EasyCNN目前包含MSE和Cross Entropy这2种损失函数，当然，也很容易添加其他Loss函数。  
> 损失函数的抽象结构如下：  
```c++
namespace EasyCNN
{
	class LossFunctor
	{
	public:
		virtual float getLoss(const std::shared_ptr<EasyCNN::DataBucket> labelDataBucket,
			const std::shared_ptr<EasyCNN::DataBucket> outputDataBucket) = 0;
		virtual std::shared_ptr<EasyCNN::DataBucket> getDiff(const std::shared_ptr<EasyCNN::DataBucket> labelDataBucket,
			const std::shared_ptr<EasyCNN::DataBucket> outputDataBucket) = 0;
	};
}
```

* 优化方法（参数更新）  
> 优化方法，EasyCNN这里偷了个懒，没有单独抽象出来。

* 核心运行时
> 核心运行时一般包括：配置（config）、日志（log）、错误处理（except、assert等）、工具函数（性能检测、字符串处理等）、基础数据处理函数（gemm、convolution等）  
> EasyCNN也内置了一些基础运行时：
> * 配置： Configure.h
> * 日志：EasyLogger.h
> * 错误处理： EasyAssert.h
> * 工具函数：CommonTools.h
> * 基础数据处理函数：暂无

* 网络
> 网络将层组织起来，并控制数据流的运转，是整个框架逻辑上最复杂的一部分。并且包括模型的存储加载等。
> 下面是EasyCNN的网络抽象：
```c++
namespace EasyCNN
{
	class NetWork
	{
	public:
		NetWork();
		virtual ~NetWork();
	public:
		//common
		void setPhase(Phase phase);
		Phase getPhase() const;
		//test only!
		bool loadModel(const std::string& modelFile);
		std::shared_ptr<EasyCNN::DataBucket> testBatch(const std::shared_ptr<DataBucket> inputDataBucket);
		//train only!
		void setInputSize(const DataSize size);
		void setLossFunctor(std::shared_ptr<LossFunctor> lossFunctor);
		void addayer(std::shared_ptr<Layer> layer);
		float trainBatch(const std::shared_ptr<DataBucket> inputDataBucket,
			const std::shared_ptr<DataBucket> labelDataBucket, float learningRate);
		bool saveModel(const std::string& modelFile);
	private:
		//common
		std::shared_ptr<EasyCNN::DataBucket> forward(const std::shared_ptr<DataBucket> inputDataBucket);
		float backward(const std::shared_ptr<DataBucket> labelDataBucket, float learningRate);
		std::string serializeToString() const;
		std::vector<std::shared_ptr<EasyCNN::Layer>> serializeFromString(const std::string content);
		std::shared_ptr<EasyCNN::Layer> createLayerByType(const std::string layerType);
	private:
		Phase phase = Phase::Train;
		std::vector<std::shared_ptr<Layer>> layers;
		std::vector<std::shared_ptr<DataBucket>> dataBuckets;
		std::shared_ptr<LossFunctor> lossFunctor;
	};
}
```

# 后记
EasyCNN是熟悉CNN的练手之作，不足之处非常明显，就不多说了。这里就大概以EasyCNN为例讲了讲现代软件设计的一个基本构型。  

具体CNN的实现原理可结合其他文章和这个代码，边调试边理解。

有什么疑惑请留言或者在[github](https://github.com/xylcbd/EasyCNN)上给我发[issue](https://github.com/xylcbd/EasyCNN/issues)。

:)
