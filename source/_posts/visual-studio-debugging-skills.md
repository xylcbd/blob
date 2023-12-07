---
title: visual studio调试技巧
date: 2017-11-28 20:09:00
categories:
 - skill
tags:
 - visual studio
 - tools
 - skill
---

### 调试图像

安装ImageWatch插件，可以在断点时查看图像。

下载地址：[ImageWatch](https://marketplace.visualstudio.com/items?itemName=WolfKienzle.ImageWatch)

![](./ImageWatch.jpg)

### 查看utf8编码的字符串

在watch窗口中输入：

```c++
content.c_str(), s8
```



### 调试服务器代码
采用[Poco](pocoproject.org/index.html)或者[libevent](http://libevent.org/)等编写跨平台服务器端代码。  
服务器输入和输出均采用JSON的形式，HTTP POST。  

整个project分成3大部分，分别是：
* function core
* server core
* test core

其中，function core提供核心功能并稳定API，server core提供多并发网络输入输出，test core提供单元测试和功能调试。

调试时function core，如果需要模拟网络环境进行本地调试，则可以将参数通过本地资源进行访问。



### 立即窗口

立即窗口（Immediate window），可以用来输出当前上下文中的变量，并做一些简单的计算操作。

![](./ImmediateWindow.jpg)



### 条件断点

条件断点可以用来帮助定位问题，以免一遍遍的断点查看。

![](./breakpoint_condition.jpg)



### 数据断点

与条件断点一样，数据断点也能帮助快速定位问题（尤其是一些莫名其妙的问题，如越界等）。数据断点需要事先获取数据地址，然后监视数据变化。

注意：数据断点有字节数限制，不能监控大段内存，仅能监控一些最基本的数据类型长度。

![](./data_breakpoint_1.jpg)

![](./data_breakpoint_2.jpg)



### 性能热点分析调试

visual studio 2013以后自带性能热点分析工具，非常好用。

![](./performance_profiling_1.jpg)

![](./performance_profiling_2.jpg)



### Error Lookup

在TOOLS菜单下有若干工具（可自行配置工具），有些工具很实用，如：Create GUID，Error Lookup，Spy++等。

Error Lookup工具将windows内部错误代码翻译成可供工程师查看参考的自然语言，以辅助调试问题。

![](./error_lookup.jpg)



### Spy++

Spy++工具可用于辅助调试Windows窗口相关的问题，如某窗口的父窗口子窗口的Handle分别是什么等。

![](./spyxx.jpg)



### depends

哈哈，这个其实不属于Visual Studio的范畴，但是真的太好用了，不能不提。调试各种缺少动态库提示之类的问题不要太好用。（谁用谁知道）

![](./depends.jpg)

---

to be continued...
