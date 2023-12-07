---
title: Python调用C++模块时发生crash
date: 2017-04-19 22:09:00
categories:
 - 踩坑
tags:
 - 踩坑
 - python
 - c++
 - 技术
---

现在写python代码越来越多，实在太方便了。

有一次在python代码中通过ctypes调用c++模块，一直发生glibc free invalidate pointer错误。类似如下代码：  
```c++
struct Foo
{
  std::string content;
};

extern "C" void bar(const char* src)
{
  Foo foo;
  foo.content = src;
  //do something
}
```

一直在```c++ foo.content=src; ```这一行发生crash，且替换成```c++ const std::string content=src;``就不会发生crash。并且直接在c++ main函数中调用也没有问题。    

一开始怀疑是前面部分代码内存越界，经二分法排查，不是内存越界的原因。  

然后怀疑是多次释放，换成char[]仍然有问题。  

然后怀疑人生，怀疑在extern "C"函数内部不能直接写c++代码，于是把所有代码拎出来单独放在一个c++ wrap函数中，仍然异常。

最后检查编译选项，设置为-O0 -g，仍然异常。看到编译选项中有-fexceptions -fnon-call-exceptions（编译android ndk库时设置的，直接copy过来了），去掉这2个选项，一切OK。

经查这2个选项分别是：
* -fexceptions
> Enable exception handling. Generates extra code needed to propagate exceptions. For some targets, this implies GCC generates frame unwind information for all functions, which can produce significant data size overhead, although it does not affect execution. If you do not specify this option, GCC enables it by default for languages like C++ that normally require exception handling, and disables it for languages like C that do not normally require it. However, you may need to enable this option when compiling C code that needs to interoperate properly with exception handlers written in C++. You may also wish to disable this option if you are compiling older C++ programs that don’t use exception handling.

* -fnon-call-exceptions
> Generate code that allows trapping instructions to throw exceptions. Note that this requires platform-specific runtime support that does not exist everywhere. Moreover, it only allows trapping instructions to throw exceptions, i.e. memory references or floating-point instructions. It does not allow exceptions to be thrown from arbitrary signal handlers such as SIGALRM.

应该是与python ctypes内部的一些设置相关，没有再深入研究下去。记录一下，以便自己与其他人踩坑时搜索。
