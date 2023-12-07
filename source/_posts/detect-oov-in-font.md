---
title: 如何检测字体文件中是否包含某字符
date: 2019-09-23 21:19:03
categories:
 - 工具使用
tags:
 - 工具使用
---


### 简介
字体文件为.ttf或者.otf等。

在使用字体渲染某些字符时，有可能渲染出空白或者“口”字形，原因在于该字体文件中不包含该字符的字形。

### 分析
可能的原因包括：
1. 字符不在字体的cmap表中（cmap表是字体文件声明的支持字符表）；
2. 字符不在字体的glyf表中（glyf表是字体文件中实际包含的字形，类似svg描述）；
3. 未知原因，需要人工进行查看，有些字体比较坑，缺失的字体就随便填个glyf。

第一种原因的检测方法：
```
from fontTools.ttLib import TTFont

ch = '你'

font_path = 'test.ttf'
font = TTFont(font_path)

found = False
for table in font['cmap'].tables:
    if ord(ch) in table.cmap.keys():
        found = True
        break
```

第二种原因的检测方法：
```
from fontTools.ttLib import TTFont

ch = '你'

font_path = 'test.ttf'
font = TTFont(font_path)
glyph_name = None
for table in font['cmap'].tables:
    glyph_name = table.cmap.get(ord(ch))
    if glyph_name is not None:
        break
if glyph_name is not None:
    glyf = font['glyf']
    found = glyf.has_key(glyph_name) and glyf[glyph_name].numberOfContours > 0
else:
    found = False
```

安装[fonttools](https://github.com/fonttools/fonttools)后，将ttf转换为xml格式可以使用文本化的形式查看字体：
```
ttx test.ttf
```

查看后发现该字符没有字形数据：
![](./detect-oov-in-font.png)

### 参考
1. https://stackoverflow.com/questions/47948821/python-imagefont-and-imagedraw-check-font-for-character-support
2. https://zhuanlan.zhihu.com/p/54379490
3. https://github.com/fonttools/fonttools/blob/master/Lib/fontTools/ttLib/tables/_g_l_y_f.py
4. https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6glyf.html
5. https://github.com/fonttools/fonttools/blob/master/Lib/fontTools/ttLib/tables/_c_m_a_p.py
6. https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6cmap.html
