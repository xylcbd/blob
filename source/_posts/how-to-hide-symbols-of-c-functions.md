---
title: 如何隐藏C/C++编译生成的函数符号
date: 2016-06-24 20:37:09
categories:
 - 软件安全 
tags:
 - 函数符号
 - 安全
 - 软件安全
---

如何隐藏C/C++编译生成的二进制文件中的函数符号以及字符串，减少软件暴露出来的信息。

通常，在二进制文件（静态库、动态库、可执行文件等）中包含了程序运行所需要的汇编指令、字符串、变量、导入导出的函数符号，以及一些其他的资源。

其中包含的函数符号和字符串，我们希望能对外隐藏这部分信息，使得竞争对手或者其他人无法分析出我们的程序中采用了哪些技术，以及防止其他人能直接调用我们的函数。

首先，需要理解什么是函数符号，以及字符串为什么会在目标二进制文件中存在。函数符号是编译器在编译（compile）过程中产生的一种函数的别名，这个别名也用于库之间的链接（link），使得一个应用程序可以将代码拆分成不同的库。

在C语言中，一般函数符号的命名方式为函数原名，或者在函数之前加一个下划线（\_func ）。C语言不支持重载，因此这种命名方式很简单也很方便。
比如人，在gcc中，下列函数符号为add_int。
```c
int add_int(int a,int b)
{
  return a+b;
}
```
而在C++语言中，由于C++支持函数的重载，因此上述C语言的命名方式就不再适用了。
在C++语言中，函数的重载的定义是：在同一范围内（相同编译单元相同类内部）函数名相同而参数不同。因此C++编译器一般讲函数符号命名为：前缀+函数名+参数编码。
比如，在g++中，下面两个函数符号分别是_Z3addii和_Z3addff。
```c++
int add(int a,int b)
{
  return a+b;
}
float add(float a,float b)
{
  return a+b;
}
```
二进制文件中包含了代码中的字符串，在运行中这些字符串将被加载到内存中，被程序所采用。这些字符串如果不做特殊处理，那么通过一些反编译工具（如IDA Pro等）能全部看得到。

回到函数符号的隐藏，一般可以在gcc编译选项中加入如下编译选项：
```makefile
CFLAGS = -ffunction-sections -fdata-sections -fvisibility=hidden
LDFLAGS += -Wl,--gc-sections
```
在xcode中，需要修改如下编译选项（加粗部分）：  
```makefile
Strip Stype -> Non-Global Symbols  
Use Separate Strip -> Yes
Other Linker Flags -> -Xlinker -x
Debug Information Level -> Line Tables only
Generate Debug Symbols -> No
Symbols Hidden by Default -> Yes
```

通过上述设置，库内部的函数符号就可以隐藏了。而我们需要一些导出函数给其他人用的话，可以在需要导出的函数前面加上：  
```c++
__attribute__((visibility("default")))
```

