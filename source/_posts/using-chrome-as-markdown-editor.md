---
title: 使用chrome浏览器作为Markdown写作工具
date: 2017-03-08 20:49:03
categories:
 - 工具使用
tags:
 - Markdown
---

# Markdown


Markdown是一种非常简约的「标记语言」，它语法简单，可编辑出一定的格式。  
因此，被越来越多的人们所广泛使用。包括各大博客、stackoverflow、github等平台均支持Markdown格式。

Markdown语法规则见：[Markdown-Cheatsheet](https://github.com/adam-p/Markdown-here/wiki/Markdown-Cheatsheet)

---

# Markdown编辑器

Markdown是一种纯文本格式，任何文本编辑器都可以进行Markdown的编辑。  
然而，一个带即时渲染的Markdown编辑器可以有效帮助进行Markdown的格式调整。  
常用的带即时渲染的Markdown编辑器有：[Atom](https://atom.io/)，[Sublime Text](https://www.sublimetext.com/)，等。  
在试用过这些编辑器作为Markdown编辑器之后，均感觉效果不是很好，要么渲染很卡，要么渲染效果差。  
直到找到一个神器...

---

# Chrome As Markdown Previewer

使用Chrome浏览器作为Markdown渲染器，使用notepad++作为编辑器。

效果：
![chrome-markdown](./chrome-markdown.png)

安装步骤：

1. 安装Chrome插件[Markdown Preview Plus](https://chrome.google.com/webstore/detail/markdown-preview-plus/febilkbfcbhebfnokafefeacimjdckgl)
2. 在本地使用任意编辑器编辑Markdown文件，假设为test.md
3. 使用Chrome打开test.md，并且设置Markdown Preview Plus关联md格式，每秒reload，如果需要甚至还可以开启公式latex的渲染。

---

Notepad++与chrome Markdown Preview Plus组合，编辑Markdown文件so easy！

$$
\begin{equation} \label{euler} e^{\pi i} + 1 = 0 \end{equation}
$$
