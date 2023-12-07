---
title: ARM NEON 编程系列1 - 导论
date: 2016-05-13 21:18:47
categories: 
 - 性能优化
tags: 
 - arm
 - neon
 - 性能优化
---

## 前言
本系列博文用于介绍ARM CPU下NEON指令优化。  
* 博文github地址：[github](https://github.com/xylcbd/blogs "博文")  
* 相关代码github地址：[github](https://github.com/xylcbd/blogs_code "博文相关demo")

## NEON历史
ARM处理器的历史可以阅读文献[2]，本文假设读者已有基本的ARM CPU下编程的经验，本文面向需要了解ARM平台下通过NEON进行算法优化的场景。  
ARM CPU最开始只有普通的寄存器，可以进行基本数据类型的基本运算。自ARMv5开始引入了VFP（Vector Floating Point）指令，该指令用于向量化加速浮点运算。自ARMv7开始正式引入NEON指令，NEON性能远超VFP，因此VFP指令被废弃。

## NEON用途
类似于Intel CPU下的MMX/SSE/AVX/FMA指令，ARM CPU的NEON指令同样是通过向量化计算来进行速度优化，通常应用于图像处理、音视频处理等等需要大量计算的场景。  

## Hello world
下面给一个最基本的例子来说明NEON的作用：
>注意：  
1. 代码采用C++11编写，后续博客代码均以C++11编写，不再重述） 
2. 此系列博客采用[neon2sse.h](https://software.intel.com/sites/default/files/managed/cf/f6/NEONvsSSE.h "neon2sse.h")将NEON指令翻译成SSE指令以使得代码可以在x86/x64 CPU上运行。本文所有代码均在windows vs2013以及android-ndk-r11c下编译测试通过。

** 完整代码地址：[基本NEON优化示例代码](https://github.com/xylcbd/blogs_code/tree/master/2016_04_16_10_56_arm_neon_introduction "基本NEON优化示例") **  
```c++
//填充随机数
static void fill_random_value(std::vector<float>& vec_data)
{
	std::uniform_real_distribution<float> distribution(
		std::numeric_limits<float>::min(),
		std::numeric_limits<float>::max());
	std::default_random_engine generator;

	std::generate(vec_data.begin(), vec_data.end(), [&]() { return distribution(generator); });
}
//判断两个vector是否相等
static bool is_equals_vector(const std::vector<float>& vec_a,
  const std::vector<float>& vec_b)
{
	if (vec_a.size() != vec_b.size())
	{
		return false;
	}
	for (size_t i = 0; i < vec_a.size(); i++)
	{
		if (vec_a[i] != vec_b[i])
		{
			return false;
		}
	}
	return true;
}
//正常的vector相乘 （注意：需要关闭编译器的自动向量化优化）
static void normal_vector_mul(const std::vector<float>& vec_a,
  const std::vector<float>& vec_b,
  std::vector<float>& vec_result)
{
	assert(vec_a.size() == vec_b.size());
	assert(vec_a.size() == vec_result.size());
	//compiler may optimized auto tree vectorize (test this diabled -ftree-vectorize)
	for (size_t i = 0; i < vec_result.size();i++)
	{
		vec_result[i] = vec_a[i] * vec_b[i];
	}
}
//NRON优化的vector相乘
static void neon_vector_mul(const std::vector<float>& vec_a,
  const std::vector<float>& vec_b,
  std::vector<float>& vec_result)
{
	assert(vec_a.size() == vec_b.size());
	assert(vec_a.size() == vec_result.size());
	int i = 0;
	//neon process
	for (; i < (int)vec_result.size() - 3 ; i+=4)
	{
		const auto data_a = vld1q_f32(&vec_a[i]);
		const auto data_b = vld1q_f32(&vec_b[i]);
		float* dst_ptr = &vec_result[i];
		const auto data_res = vmulq_f32(data_a, data_b);
		vst1q_f32(dst_ptr, data_res);
	}
	//normal process
	for (; i < (int)vec_result.size(); i++)
	{
		vec_result[i] = vec_a[i] * vec_b[i];
	}
}
//测试函数
//FuncCostTimeHelper是一个计算时间消耗的helper类
static int test_neon()
{
	const int test_round = 1000;
	const int data_len = 10000;
	std::vector<float> vec_a(data_len);
	std::vector<float> vec_b(data_len);
	std::vector<float> vec_result(data_len);
	std::vector<float> vec_result2(data_len);
	//fill random value in vecA & vecB
	fill_random_value(vec_a);
	fill_random_value(vec_b);
	//check the result is same
	{
		normal_vector_mul(vec_a, vec_b, vec_result);
		neon_vector_mul(vec_a, vec_b, vec_result2);
		if (!is_equals_vector(vec_result,vec_result2))
		{
			std::cerr << "result vector is not equals!" << std::endl;
			return -1;
		}
	}
	//test normal_vector_mul
	{
		FuncCostTimeHelper time_helper("normal_vector_mul");
		for (int i = 0; i < test_round;i++)
		{
			normal_vector_mul(vec_a, vec_b, vec_result);
		}
	}
	//test neon_vector_mul
	{
		FuncCostTimeHelper time_helper("neon_vector_mul");
		for (int i = 0; i < test_round; i++)
		{
			neon_vector_mul(vec_a, vec_b, vec_result2);
		}
	}
	return 0;
}

int main(int, char*[])
{
	return test_neon();
}
```

说明：  
> 这段代码在关闭编译器的自动向量化优化之后，neon_vector_mul大约比normal_vector_mul速度快3倍左右。  
这段代码中使用了3条NEON指令：vld1q_f32，vmulq_f32，vst1q_f32。具体指令的作用会在后续博文中说明。此处仅作演示。  



## 参考：
1. [DEN0018A_neon_programmers_guide](http://pan.baidu.com/s/1c14agog "DEN0018A_neon_programmers_guide")
2. [DDI0487A_f_armv8_arm](http://pan.baidu.com/s/1qXYdMOW,"DDI0487A_f_armv8_arm")
3. [DEN0013D_cortex_a_series_PG](http://pan.baidu.com/s/1jIPcMSe "DEN0013D_cortex_a_series_PG")
