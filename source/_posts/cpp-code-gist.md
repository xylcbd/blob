---
title: c++实用代码片段
date: 2016-06-07 22:19:03
categories:
 - 代码片段
tags:
 - c++
 - code gist
---

这篇文章介绍了我在工程应用中经常会用到的c++工具代码。

## 编译期的todo list
下面这些宏能产生编译期的TODO/WARN/FIXME等信息，有助于编写代码。
```c++
//don't use these macros
#define WTF_YOU_DONT_WANT_TOOLS_STRINGSIZE( L ) #L
#define WTF_YOU_DONT_WANT_TOOLS_MAKESTRING_IMP( M, L ) M(L)
#define WTF_YOU_DONT_WANT_TOOLS_MAKESTRING(x) WTF_YOU_DONT_WANT_TOOLS_MAKESTRING_IMP(WTF_YOU_DONT_WANT_TOOLS_STRINGSIZE,x)
#define WTF_YOU_DONT_WANT_TOOLS_MESSAGE_LINE "\n====================================\n"
#define WTF_YOU_DONT_WANT_TOOLS_MESSAGE_POSITION __FILE__ "(" WTF_YOU_DONT_WANT_TOOLS_MAKESTRING(__LINE__) ")\n"
#define WTF_YOU_DONT_WANT(type,content) message (WTF_YOU_DONT_WANT_TOOLS_MESSAGE_LINE WTF_YOU_DONT_WANT_TOOLS_MESSAGE_POSITION type content WTF_YOU_DONT_WANT_TOOLS_MESSAGE_LINE)
//use these macros
#define TODO(content) WTF_YOU_DONT_WANT("TODO : ",content)
#define WARN(content) WTF_YOU_DONT_WANT("WARN : ",content)
#define FIXME(content) WTF_YOU_DONT_WANT("FIXME : ",content)
```
例子：
```c++
//test
#pragma TODO("this function is not implemented!")
void not_implement_function()
{

}
#pragma WARN("this function is implemented by a hack way!")
float fast_inv_sqrt(float x)
{
	float xhalf = 0.5f*x;
	int i = *(int*)&x;
	i = 0x5f3759df - (i >> 1);
	x = *(float*)&i;
	x = x*(1.5f - xhalf*x*x);
	return x;
}
#pragma FIXME("buggy ! !")
float custom_add(float a, float b)
{
	return a - b;
}
int main(int argc, char* argv[])
{
	return 0;
}
```
vs2013输出结果：
```c++
1>  ====================================
1>  d:\workspace\local\neontest\neontest.cpp(19)
1>  TODO : this function is not implemented!
1>  ====================================
1>  
1>  
1>  ====================================
1>  d:\workspace\local\neontest\neontest.cpp(24)
1>  WARN : this function is implemented by a hack way!
1>  ====================================
1>  
1>  
1>  ====================================
1>  d:\workspace\local\neontest\neontest.cpp(34)
1>  FIXME : buggy ! !
1>  ====================================
```

## ScopeExit Tool
下面这个代码片段利用了c++11中的lambda表达式以及C++的RAII（[Resource Acquisition Is Initialization](https://en.wikipedia.org/wiki/Resource_Acquisition_Is_Initialization "Resource Acquisition Is Initialization")）特性，使得代码及其优雅，在不用写wrapper class的情况下充分利用RAII的好处。能够降低编码的心智负担，在哪里申请资源就在哪里释放，保证必须成对出现的操作一定会成对执行。总之，好顶赞！

```c++
namespace{
	template <class Lambda> class AtScopeExit {
		Lambda& m_lambda;
	public:
		AtScopeExit(Lambda& action) : m_lambda(action) {}
		~AtScopeExit() { m_lambda(); }
	};
};
#define WTF_TOKEN_PASTEx(x, y) x ## y
#define WTF_TOKEN_PASTE(x, y) WTF_TOKEN_PASTEx(x, y)
#define WTF_INTERNAL1(lname, aname, ...) \
    auto lname = [&]() { __VA_ARGS__; }; \
    ::AtScopeExit<decltype(lname)> aname(lname);
#define WTF_INTERNAL2(ctr, ...) \
    WTF_INTERNAL1(WTF_TOKEN_PASTE(anonymity_func_, ctr), \
                   WTF_TOKEN_PASTE(anonymity_instance_, ctr), __VA_ARGS__)
#define SCOPE_EXIT(...) WTF_INTERNAL2(__COUNTER__, __VA_ARGS__)
```
例子：
```c++
void dump_data(const char* filename, const char* data, const int data_len)
{
	//close file after dump data
	FILE* fp = fopen(filename, "wb");
	SCOPE_EXIT(if (fp){ fclose(fp); });

	//delete data after process and dump
	int* processed_data = new int[data_len];
	SCOPE_EXIT(delete[] processed_data;);
	for (int i = 0; i < data_len;i++)
	{
		processed_data[i] = data[i] + 100;
	}
	fwrite(processed_data, sizeof(processed_data[0]), data_len, fp);
}
```

## 未完待续

