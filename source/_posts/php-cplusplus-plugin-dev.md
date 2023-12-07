---
title: 如何使用C/C++开发PHP插件
date: 2017-10-13 20:09:00
categories:
 - c++
tags:
 - c++
 - php
 - plugin
---

### PHP插件

php作为web时代最常用的编程语言之一，其广泛应用于web page和web service的开发。

可以很方便的通过c/c++为php开发插件，以方便部署服务。这篇文章以一个hello world程序作为示例解释如何使用c/c++为php开发插件。

该代码支持php5和php7。

本文代码：[https://github.com/xylcbd/php_cpp_plugin](https://github.com/xylcbd/php_cpp_plugin)

### 基本框架

插件名为allen，提供say函数。say函数输入一个名字，输出"hello $name"。插件的用法如下：

```php
<?php
$name = "Jim Green";
echo say($name) . "\n";
?>
```

在开发php插件之前请先安装php5或者php7，以及php5-dev或者php7-dev。

然后在工作目录下创建config.m4，内容如下：

```makefile
PHP_ARG_ENABLE(allen, whether to enable say support,
[ --enable-allen Enable say support])

if test "$PHP_ALLEN" = "yes"; then
    CXXFLAGS="-std=c++0x -I./src -I/usr/local/include/"
    LDFLAGS="-L/path/to/link -lstdc++"
    PHP_REQUIRE_CXX()
    PHP_SUBST(VEHICLE_SHARED_LIBADD)
    PHP_ADD_LIBRARY(stdc++, 1, VEHICLE_SHARED_LIBADD)
    AC_DEFINE(HAVE_ALLEN, 1, [Whether you have say])
    PHP_NEW_EXTENSION(allen, php_allen.cpp src/allen.cpp, $ext_shared)
fi
```

注意：config.m4中可以指定头文件目录和库文件目录等，具体设置请查阅php官方网站。

接着在工作目录下创建php_allen.h，内容如下：

```c++
#ifndef PHP_ALLEN_H
    #define PHP_ALLEN_H 1
    #define PHP_HW_BEAUTY_VERSION "1.0"
    #define PHP_HW_BEAUTY_EXTNAME "allen"
    PHP_FUNCTION(say);
    extern zend_module_entry allen_module_entry;
    #define phpext_allen_ptr &allen_module_entry
#endif
```

以及php_allen.cpp，内容如下：

```c++
#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif 
#include "php.h"
#include "php_allen.h"
#include "./src/allen.h"
static zend_function_entry allen_functions[] = {
    PHP_FE(say, NULL)
    {NULL, NULL, NULL}
};
zend_module_entry allen_module_entry = {
#if ZEND_MODULE_API_NO >= 20010901
    STANDARD_MODULE_HEADER,
#endif
    PHP_HW_BEAUTY_EXTNAME,
    allen_functions,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
#if ZEND_MODULE_API_NO >= 20010901
    PHP_HW_BEAUTY_VERSION,
#endif
    STANDARD_MODULE_PROPERTIES
};
#ifdef COMPILE_DL_ALLEN
    ZEND_GET_MODULE(allen)
#endif

#if PHP_MAJOR_VERSION < 7
typedef int string_size_t;
#else
typedef size_t string_size_t;
#endif

PHP_FUNCTION(say)
{
    char* name = NULL;
    string_size_t name_len = 0;
    int argc = ZEND_NUM_ARGS();
    if (zend_parse_parameters(argc TSRMLS_CC, "s", &name, &name_len) == FAILURE)
    {
        php_error(E_WARNING, "say: parameter invalidate!");
        RETURN_NULL();
    }
    char* imgData = allen_say_to(name);
    char* result = estrdup(imgData);
    release_memory(imgData);
#if PHP_MAJOR_VERSION < 7
    RETURN_STRING(result, 0);
#else
    RETURN_STRING(result);
#endif
}
```

要注意其中一个宏PHP_MAJOR_VERSION，这个宏下的内容表示了php5和php7的差异，请细体会。

### 编译设置

准备好上述文件之后，在工作目录下执行命令；

```
phpize
./configure --enable-allen
make
make test
```

编译成功的话，将生成的so复制到/usr/local/lib下或者在原地。然后编辑php.ini文件，在php.ini文件最后加入如下设置（php.ini一般在/etc/php5/cli/php.ini或者/etc/php5/fpm/php.ini）：

```
extension=/usr/local/lib/allen.so
```

然后重启php-fpm服务，通过web或者命令行执行say函数，执行无误则表示插件安装完毕。

### 总结

php大法好，web分分钟。