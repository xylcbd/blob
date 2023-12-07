---
title: ARM NEON 编程系列2 - 基本指令集
date: 2016-05-13 21:34:43
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

## NEON指令集
主流支持目标平台为ARM CPU的编译器基本都支持NEON指令。可以通过在代码中嵌入NEON汇编来使用NEON，但是更加常见的方式是通过类似C函数的NEON Instrinsic来编写NEON代码。就如同[NEON hello world](2016_04_16_10_56_arm_neon_introduction.md "上一篇博文中的hello world代码")一样。NEON Instrinsic是编译器支持的一种buildin类型和函数的集合，基本涵盖NEON的所有指令，通常这些Instrinsic包含在arm_neon.h头文件中。  
本文以android-ndk-r11c中armv7的arm_neon.h为例，讲解NEON的指令类型。  

## 寄存器
ARMV7架构包含：  
* 16个通用寄存器（32bit），R0-R15
* 16个NEON寄存器（128bit），Q0-Q15（同时也可以被视为32个64bit的寄存器，D0-D31）
* 16个VFP寄存器（32bit），S0-S15  
> NEON和VFP的区别在于VFP是加速浮点计算的硬件不具备数据并行能力，同时VFP更尽兴双精度浮点数（double）的计算，NEON只有单精度浮点计算能力。更多请参考[stackoverflow:neon vs vfp](http://stackoverflow.com/questions/4097034/arm-cortex-a8-whats-the-difference-between-vfp-and-neon "neon vs vfp")

## 基本数据类型
* 64bit数据类型，映射至寄存器即为D0-D31  
相应的c/c++语言类型（stdint.h或者csdtint头文件中类型）在注释中说明。
```c++
//typedef int8_t[8] int8x8_t;
typedef __builtin_neon_qi int8x8_t	__attribute__ ((__vector_size__ (8)));
//typedef int16_t[4] int16x4_t;
typedef __builtin_neon_hi int16x4_t	__attribute__ ((__vector_size__ (8)));
//typedef int32_t[2] int32x2_t;
typedef __builtin_neon_si int32x2_t	__attribute__ ((__vector_size__ (8)));
//typedef int64_t[1] int64x1_t;
typedef __builtin_neon_di int64x1_t;
//typedef float16_t[4] float16x4_t;
//（注：该类型为半精度，在部分新的CPU上支持，c/c++语言标注中尚无此基本数据类型）
typedef __builtin_neon_hf float16x4_t	__attribute__ ((__vector_size__ (8)));
//typedef float32_t[2] float32x2_t;
typedef __builtin_neon_sf float32x2_t	__attribute__ ((__vector_size__ (8)));
//poly8以及poly16类型在常用算法中基本不会使用
//详细解释见：
//http://stackoverflow.com/questions/22224282/arm-neon-and-poly8-t-and-poly16-t
typedef __builtin_neon_poly8 poly8x8_t	__attribute__ ((__vector_size__ (8)));
typedef __builtin_neon_poly16 poly16x4_t	__attribute__ ((__vector_size__ (8)));
#ifdef __ARM_FEATURE_CRYPTO
typedef __builtin_neon_poly64 poly64x1_t;
#endif
//typedef uint8_t[8] uint8x8_t;
typedef __builtin_neon_uqi uint8x8_t	__attribute__ ((__vector_size__ (8)));
//typedef uint16_t[4] uint16x4_t;
typedef __builtin_neon_uhi uint16x4_t	__attribute__ ((__vector_size__ (8)));
//typedef uint32_t[2] uint32x2_t;
typedef __builtin_neon_usi uint32x2_t	__attribute__ ((__vector_size__ (8)));
//typedef uint64_t[1] uint64x1_t;
typedef __builtin_neon_udi uint64x1_t;
```
* 128bit数据类型，映射至寄存器即为Q0-Q15  
相应的c/c++语言类型（stdint.h或者csdtint头文件中类型）在注释中说明。
```c++
//typedef int8_t[16] int8x16_t;
typedef __builtin_neon_qi int8x16_t	__attribute__ ((__vector_size__ (16)));
//typedef int16_t[8] int16x8_t;
typedef __builtin_neon_hi int16x8_t	__attribute__ ((__vector_size__ (16)));
//typedef int32_t[4] int32x4_t;
typedef __builtin_neon_si int32x4_t	__attribute__ ((__vector_size__ (16)));
//typedef int64_t[2] int64x2_t;
typedef __builtin_neon_di int64x2_t	__attribute__ ((__vector_size__ (16)));
//typedef float32_t[4] float32x4_t;
typedef __builtin_neon_sf float32x4_t	__attribute__ ((__vector_size__ (16)));
//poly8以及poly16类型在常用算法中基本不会使用
//详细解释见：
//http://stackoverflow.com/questions/22224282/arm-neon-and-poly8-t-and-poly16-t
typedef __builtin_neon_poly8 poly8x16_t	__attribute__ ((__vector_size__ (16)));
typedef __builtin_neon_poly16 poly16x8_t	__attribute__ ((__vector_size__ (16)));
#ifdef __ARM_FEATURE_CRYPTO
typedef __builtin_neon_poly64 poly64x2_t	__attribute__ ((__vector_size__ (16)));
#endif
//typedef uint8_t[16] uint8x16_t;
typedef __builtin_neon_uqi uint8x16_t	__attribute__ ((__vector_size__ (16)));
//typedef uint16_t[8] uint16x8_t;
typedef __builtin_neon_uhi uint16x8_t	__attribute__ ((__vector_size__ (16)));
//typedef uint32_t[4] uint32x4_t;
typedef __builtin_neon_usi uint32x4_t	__attribute__ ((__vector_size__ (16)));
//typedef uint64_t[2] uint64x2_t;
typedef __builtin_neon_udi uint64x2_t	__attribute__ ((__vector_size__ (16)));
typedef float float32_t;
typedef __builtin_neon_poly8 poly8_t;
typedef __builtin_neon_poly16 poly16_t;
#ifdef __ARM_FEATURE_CRYPTO
typedef __builtin_neon_poly64 poly64_t;
typedef __builtin_neon_poly128 poly128_t;
#endif
```

## 结构化数据类型
下面这些数据类型是上述基本数据类型的组合而成的结构化数据类型，通常为被映射到多个寄存器中。
```c++
typedef struct int8x8x2_t
{
  int8x8_t val[2];
} int8x8x2_t;
...
//省略...
...
#ifdef __ARM_FEATURE_CRYPTO
typedef struct poly64x2x4_t
{
  poly64x2_t val[4];
} poly64x2x4_t;
#endif
```

## 基本指令集
NEON指令按照操作数类型可以分为正常指令、宽指令、窄指令、饱和指令、长指令。  
> * 正常指令：生成大小相同且类型通常与操作数向量相同到结果向量。  
> * 长指令：对双字向量操作数执行运算，生产四字向量到结果。所生成的元素一般是操作数元素宽度到两倍，并属于同一类型。L标记，如VMOVL。  
> * 宽指令：一个双字向量操作数和一个四字向量操作数执行运算，生成四字向量结果。W标记，如VADDW。  
> * 窄指令：四字向量操作数执行运算，并生成双字向量结果，所生成的元素一般是操作数元素宽度的一半。N标记，如VMOVN。  
> * 饱和指令：当超过数据类型指定到范围则自动限制在该范围内。Q标记，如VQSHRUN  

NEON指令按照作用可以分为：加载数据、存储数据、加减乘除运算、逻辑AND/OR/XOR运算、比较大小运算等，具体信息参考资料[1]中附录C和附录D部分。  

常用的指令集包括：

* 初始化寄存器
寄存器的每个lane（通道）都赋值为一个值N
```c++
Result_t vcreate_type(Scalar_t N)
Result_t vdup_type(Scalar_t N)
Result_t vmov_type(Scalar_t N)
```
> lane（通道）在下面有说明。

* 加载内存数据进寄存器  
间隔为x，加载数据进NEON寄存器
```c++
Result_t vld[x]_type(Scalar_t* N)
Result_t vld[x]q_type(Scalar_t* N)
```
间隔为x，加载数据进NEON寄存器的相关lane（通道），其他lane（通道）的数据不改变
```c++
Result_t vld[x]_lane_type(Scalar_t* N,Vector_t M,int n)
Result_t vld[x]q_lane_type(Scalar_t* N,Vector_t M,int n)
```
从N中加载x条数据，分别duplicate（复制）数据到寄存器0-(x-1)的所有通道
```c++
Result_t vld[x]_dup_type(Scalar_t* N)
Result_t vld[x]q_dup_type(Scalar_t* N)
```
> * lane（通道）：比如一个float32x4_t的NEON寄存器，它具有4个lane（通道），每个lane（通道）有一个float32的值，因此 ```c++ float32x4_t dst = vld1q_lane_f32(float32_t* ptr,float32x4_t src,int n=2) ``` 的意思就是先将src寄存器的值复制到dst寄存器中，然后从ptr这个内存地址中加载第3个（lane的index从0开始）float到dst寄存器的第3个lane（通道中）。最后dst的值为：{src[0],src[1],ptr[2],src[3]}。  
> * 间隔：交叉存取，是ARM NEON特有的指令，比如 ```c++ float32x4x3_t = vld3q_f32(float32_t* ptr) ``` ，此处间隔为3，即交叉读取12个float32进3个NEON寄存器中。3个寄存器的值分别为：{ptr[0],ptr[3],ptr[6],ptr[9]}，{ptr[1],ptr[4],ptr[7],ptr[10]}，{ptr[2],ptr[5],ptr[8],ptr[11]}。

* 存储寄存器数据到内存  
间隔为x，存储NEON寄存器的数据到内存中
```c++
void vstx_type(Scalar_t* N)
void vstxq_type(Scalar_t* N)
```
间隔为x，存储NEON寄存器的相关lane（通道）到内存中
```c++
Result_t vst[x]_lane_type(Scalar_t* N,Vector_t M,int n)
Result_t vst[x]q_lane_type(Scalar_t* N,Vector_t M,int n)
```

* 读取/修改寄存器数据
读取寄存器第n个通道的数据
```c++
Result_t vget_lane_type(Vector_t M,int n)
```
读取寄存器的高/低部分到新的寄存器中，数据变窄（长度减半）。
```c++
Result_t vget_low_type(Vector_t M)
Result_t vget_high_type(Vector_t M)
```
返回在复制M的基础上设置通道n为N的寄存器数据
```c++
Result_t vset_lane_type(Scalar N,Vector_t M,int n)
```

* 寄存器数据重排
从寄存器M中取出后n个通道的数据置于低位，再从寄存器N中取出x-n个通道的数据置于高位，组成一个新的寄存器数据。
```c++
Result_t vext_type(Vector_t N,Vector_t M,int n)
Result_t vextq_type(Vector_t N,Vector_t M,int n)
```
其他数据重排指令还有：
> vtbl_tyoe,vrev_type,vtrn_type,vzip_type,vunzip_type,vcombine ...  
> 等以后有时间一一讲解。

* 类型转换指令
强制重新解释寄存器的值类型，从SrcType转化为DstType，其内部实际值不变且总的字节数不变，举例：vreinterpret_f32_s32(int32x2_t)，从int32x2_t转化为float32x2_t。
```c++
vreinterpret_DstType_SrcType(Vector_t N)
```

* 算数运算指令
[普通指令] 普通加法运算 res = M+N
```c++
Result_t vadd_type(Vector_t M,Vector_t N)
Result_t vaddq_type(Vector_t M,Vector_t N)
```
[长指令] 变长加法运算 res = M+N，为了防止溢出，一种做法是使用如下指令，加法结果存储到长度x2的寄存器中，如：vuint16x8_t res = vaddl_u8(uint8x8_t M,uint8x8_t N)。
```c++
Result_t vaddl_type(Vector_t M,Vector_t N)
```
[宽指令] 加法运算 res = M+N，第一个参数M宽度大于第二个参数N。
```c++
Result_t vaddw_type(Vector_t M,Vector_t N)
```
[普通指令] 加法运算 res = trunct(M+N)（溢出则截断）之后向右平移1位，即计算M和N的平均值
```c++
Result_t vhadd_type(Vector_t M,Vector_t N)
```
[普通指令] 加法运算 res = round(M+N)（溢出则循环）之后向右平移1位，即计算M和N的平均值
```c++
Result_t vrhadd_type(Vector_t M,Vector_t N)
```
[饱和指令] 饱和加法运算 res = st(M+N)，如：vuint8x8_t res = vqadd_u8(uint8x8_t M,uint8x8_t N)，res超出int8_t的表示范围（0，255），比如256，则设为255.
```c++
Result_t vqadd_type(Vector_t M,Vector_t N)
```
[窄指令] 加法运算 res = M+N，结果比参数M/N的长度小一半，如 uint8x8_t res = vaddhn_u16(uint16x8_t M,uint16x8_t N)
```c++
Result_t vaddhn_type(Vector_t M,Vector_t N)
```
[普通指令] 减法运算 res = M-N
```c++
Result_t vsub_type(Vector_t M,Vector_t N)
```
[普通指令] 乘法运算 res = M*N
```c++
Result_t vmul_type(Vector_t M,Vector_t N)
Result_t vmulq_type(Vector_t M,Vector_t N)
```
[普通指令] 乘&加法运算 res = M+N*P
```c++
Result_t vmla_type(Vector_t M,Vector_t N,Vector_t P)
Result_t vmlaq_type(Vector_t M,Vector_t N,Vector_t P)
```
[普通指令] 乘&减法运算 res = M-N*P
```c++
Result_t vmls_type(Vector_t M,Vector_t N,Vector_t P)
Result_t vmlsq_type(Vector_t M,Vector_t N,Vector_t P)
```
> 类似加法运算，减法和乘法运算也有一系列变种...

* 数据处理指令
[普通指令] 计算绝对值 res=abs(M)
```c++
Result_t vabs_type(Vector_t M)
```
[普通指令] 计算负值 res=-M
```c++
Result_t vneg_type(Vector_t M)
```
[普通指令] 计算最大值 res=max(M,N)
```c++
Result_t vmax_type(Vector_t M,Vector_t N)
```
[普通指令] 计算最小值 res=min(M,N)
```c++
Result_t vmin_type(Vector_t M,Vector_t N)
```
> ...

* 比较指令
[普通指令] 比较是否相等 res=mask(M == N)
```c++
Result_t vceg_type(Vector_t M,Vector_t N)
```
[普通指令] 比较是否大于或等于 res=mask(M >= N)
```c++
Result_t vcge_type(Vector_t M,Vector_t N)
```
[普通指令] 比较是否大于 res=mask(M > N)
```c++
Result_t vcgt_type(Vector_t M,Vector_t N)
```
[普通指令] 比较是否小于或等于 res=mask(M <= N)
```c++
Result_t vcle_type(Vector_t M,Vector_t N)
```
[普通指令] 比较是否小于 res=mask(M < N)
```c++
Result_t vclt_type(Vector_t M,Vector_t N)
```
> ...

* 归约指令
[普通指令] 归约加法，M和N内部的元素各自相加，最后组成一个新的结果
```c++
Result_t vpadd_type(Vector_t M,Vector_t N)
```
[普通指令] 归约最大比较，M和N内部的元素比较得出最大值，最后组成一个新的结果
```c++
Result_t vpmax_type(Vector_t M,Vector_t N)
```
[普通指令] 归约最小比较，M和N内部的元素比较得出最小值，最后组成一个新的结果
```c++
Result_t vpmin_type(Vector_t M,Vector_t N)
```

...

## 参考
1. [DEN0018A_neon_programmers_guide](http://pan.baidu.com/s/1c14agog "DEN0018A_neon_programmers_guide")
2. [DDI0487A_f_armv8_arm](http://pan.baidu.com/s/1qXYdMOW,"DDI0487A_f_armv8_arm")
3. [DEN0013D_cortex_a_series_PG](http://pan.baidu.com/s/1jIPcMSe "DEN0013D_cortex_a_series_PG")
4. [Coding for NEON - Part 1: Load and Stores](https://community.arm.com/groups/processors/blog/2010/03/17/coding-for-neon--part-1-load-and-stores "Coding for NEON - Part 1: Load and Stores")
5. [Coding for NEON - Part 2: Dealing With Leftovers](https://community.arm.com/groups/processors/blog/2010/05/10/coding-for-neon--part-2-dealing-with-leftovers "Coding for NEON - Part 2: Dealing With Leftovers")
6. [Coding for NEON - Part 3: Matrix Multiplication](https://community.arm.com/groups/processors/blog/2010/06/28/coding-for-neon--part-3-matrix-multiplication "Coding for NEON - Part 3: Matrix Multiplication")
7. [Coding for NEON - Part 4: Shifting Left and Right](https://community.arm.com/groups/processors/blog/2010/09/01/coding-for-neon--part-4-shifting-left-and-right "Coding for NEON - Part 4: Shifting Left and Right")
8. [Coding for NEON - Part 5: Rearranging Vectors](https://community.arm.com/groups/processors/blog/2012/03/13/coding-for-neon--part-5-rearranging-vectors "Coding for NEON - Part 5: Rearranging Vectors")
