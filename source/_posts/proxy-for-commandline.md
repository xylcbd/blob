---
title: 命令行使用http代理
date: 2016-11-11 10:37:09
categories:
 - 工具使用
tags:
 - 工具使用
 - 代理
---

在windows中命令行设置http代理方法如下：
``` shell
#http代理
SET http_proxy=ip:port
#https代理
SET https_proxy=ip:port
#如果需要身份认证，还需要加上以下两行
set http_proxy_user=user
set http_proxy_pass=pwd
```

---

在linux中命令行设置http代理方法如下：
``` shell
#http代理
export http_proxy=”http://USERNAME:PASSWORD@<proxyserver>:<proxyport>
#https代理
export https_proxy=”http://USERNAME:PASSWORD@<proxyserver>:<proxyport>
```