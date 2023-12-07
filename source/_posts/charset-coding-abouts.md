---
title: 字符编码问题全解释
date: 2016-12-07 19:46:21
categories:
 -  编码问题
tags:
 - unicode
 - 字符编码
---

## 前言
不管在什么编程语言中，编码问题总是一个经常会碰到的问题。如果不弄清楚它，就不得不每次都要头疼医头脚疼医脚。然后经常出现下面的问题：  
> “c++ 读取文件 乱码”  
> “python 写入文件 乱码”  
> “php print 乱码”  
> “java utf8 gbk”  
> ...

对于这类问题，首先要弄清楚字符编码的概念。  

## 字符编码概念
下面介绍常会碰到的字符编码中的概念。

#### 字符编码
字符需要被计算机处理，因此需要用内存字节去表示字符。字符编码就是一种规定，告诉计算机这一块内存中的字节是xx编码，你给我按照xx编码来解释这块内存，我让你打印到终端的时候，你也给我画出相应的字符图形出来。

#### ASCII
计算机最早出现在美国，而美国佬用的是英文字符，英文字符27个字符加上一些标点符号，为了节省内存空间，因此美国佬就用7个bit（表示范围：2^7 = 128，正好够用）来表示所有的字符。0表示字符结束，1表示标题开始，2表示正文开始，...（更多可见[ASCII码表](https://en.wikipedia.org/wiki/ASCII)）。

#### ASCII扩展
美国人搞出计算机来，老亲戚欧洲大陆也想用这东东，发现我靠，字符编码的范围全被美国人占了，于是搞了个ASCII的8bit扩展，128-255之间于是又产生了新的编码。

#### ANSI
全世界陆陆续续开始使用计算机，同样的发现字符编码范围被占了，而且对于类似中文、日文这种字符巨多的语种，8bit根本不够用。于是各个国家开始自己造自己的轮子。比如说中国人民就制定了2种自己的规范，分别是gb2312和gbk，GB18030。

#### gb2312
一个小于127的字符的意义与原来相同（ASCII），但两个大于127的字符连在一起时，就表示一个汉字，前面的一个字节（他称之为高字节）从0xA1用到 0xF7，后面一个字节（低字节）从0xA1到0xFE，这样我们就可以组合出大约7000多个简体汉字了。在这些编码里，我们还把数学符号、罗马希腊的字母、日文的假名们都编进去了，连在 ASCII 里本来就有的数字、标点、字母都统统重新编了两个字节长的编码，这就是常说的"全角"字符，而原来在127号以下的那些就叫"半角"字符了。[1]

#### gbk
后来发现gb2312还是不够用，于是干脆不再要求低字节一定是127号之后的内码，只要第一个字节是大于127就固定表示这是一个汉字的开始，不管后面跟的是不是扩展字符集里的内容。结果扩展之后的编码方案被称为 GBK 标准，GBK 包括了 GB2312 的所有内容，同时又增加了近20000个新的汉字（包括繁体字）和符号。[1]

#### gb18030
后来少数民族也要用电脑了，于是我们再扩展，又加了几千个新的少数民族的字，GBK 扩成了 GB18030。从此之后，中华民族的文化就可以在计算机时代中传承了。[1]

#### big5
这个是台湾香港同胞按照类似于的方法制定的一个编码规范，通行于台湾香港地区。

#### unicode
可以看到即使是海峡两岸都搞了2种编码规则，互相不兼容，更不用说全世界那么多地区自己搞的规范了。因此大家觉得这么搞下去吃枣药丸，大家得搞个通用的标准出来，不然这个编码问题就成现代版本的巴别塔了。于是国际标准组织就牵头一起搞了个unicode标准，规定2个字节表示1个字符，称为UCS-2标准，如果不够用的话呢，就4个字节表示1个字符，称为UCS-4标准。unicode标准制定好了，但是这个标准并不规定数据怎么存怎么传输。就好比说上帝规定人每天都要吃饭，至于你吃什么饭上帝就不管了。所以围绕这个unicode又产生好几种编码实现，包括：utf-8，utf-16，utf-32。

#### utf-8
以美国为首的英语，拉丁语系国家，觉得unicode规范是好的，但是我用了那么久的ascii，要切换到多字节表示来会很麻烦，我得搞个兼容我ascii编码的unicode实现来，于是就搞出来一个utf-8编码。这个编码规定单字节用于表示ascii，多字节表示其他的字符。

#### utf-16
utf-16采用固定的2个字节来表示unicode编码空间，不够用的话可以扩展（一般用不到，一般可以任务utf-16的字符就是个unsigned short，16bit）。

#### utf-32
utf-32采用固定的4个字节来表示unicode编码，基本就是utf-16前面多加2个为0的字节。对目前来说可以这么理解，后面如果有外星人请求将他们的字符并入unicode的话，utf-32的前2字节就可以拿出来用了。

#### unicode之bom
UTF-8以字节为编码单元，没有字节序的问题。UTF-16以两个字节为编码单元，在解释一个UTF-16文本前，首先要弄清楚每个编码单元的字节序。例如“奎”的Unicode编码是594E，“乙”的Unicode编码是4E59。如果我们收到UTF-16字节流“594E”，那么这是“奎”还是“乙”？
Unicode规范中推荐的标记字节顺序的方法是BOM。BOM不是“Bill Of Material”的BOM表，而是Byte Order Mark。BOM是一个有点小聪明的想法：
在UCS编码中有一个叫做"ZERO WIDTH NO-BREAK SPACE"的字符，它的编码是FEFF。而FFFE在UCS中是不存在的字符，所以不应该出现在实际传输中。UCS规范建议我们在传输字节流前，先传输字符"ZERO WIDTH NO-BREAK SPACE"。
这样如果接收者收到FEFF，就表明这个字节流是Big-Endian的；如果收到FFFE，就表明这个字节流是Little-Endian的。因此字符"ZERO WIDTH NO-BREAK SPACE"又被称作BOM。
UTF-8不需要BOM来表明字节顺序，但可以用BOM来表明编码方式。字符"ZERO WIDTH NO-BREAK SPACE"的UTF-8编码是EF BB BF（读者可以用我们前面介绍的编码方法验证一下）。所以如果接收者收到以EF BB BF开头的字节流，就知道这是UTF-8编码了。
Windows就是使用BOM来标记文本文件的编码方式的。[2]

## Python中的字符编码问题
搞清楚了字符编码的原理，现在以python为例常会碰到的解释字符编码。
python内部采用utf-16或者utf-32作为默认编码，可以通过下面的代码来改变默认编码（下面改为默认utf-8编码）：
```python
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
```
除了python内部的编码，还有另外几个问题需要弄清楚。python代码文件本身的编码是什么，操作系统默认的字符编码是什么。
以几个例子说明：
本机是windows，默认编码是ANSI(gbk)。

#### 例1
下面的代码以gbk编码保存为test.py，然后执行python test.py。
```python
# coding:utf-8
print '你好'
```
输出：
> 你好

#### 例2
下面的代码以utf-8编码保存为test.py，然后执行python test.py。
```python
# coding:utf-8
print '你好'
```
输出：
> 浣犲ソ

#### 例3
下面的代码以utf-8编码保存为test.py，然后执行python test.py。
```python
# coding:utf-8
print '你好'.decode('utf-8').encode('gbk')
```
输出：
> 你好


#### 例4
下面的代码以gbk编码保存为test.py，然后执行python test.py。
```python
# coding:utf-8
print '你好'.decode('utf-8').encode('gbk')
```
输出：
> Traceback (most recent call last):
  File "x.py", line 2, in <module>
    print '你好'.decode('utf-8').encode('gbk')
  File "C:\Users\allen\Anaconda2\lib\encodings\utf_8.py", line 16, in decode
    return codecs.utf_8_decode(input, errors, True)
UnicodeDecodeError: 'utf8' codec can't decode byte 0xc4 in position 0: invalid c
ontinuation byte

#### 例5
下面的代码以gbk编码保存为test.py，然后执行python test.py。
```python
# coding:utf-8
print '你好'.decode('gbk').encode('gbk')
```
输出：
> 你好

对比例1、例2、例3、例4、例5，可以发现python源代码的保存格式决定了源代码中字符串的格式，而python第一行主动声明的# coding:utf-8并没有起作用。而print需要的字符串必须与本机默认编码一致，否则print出来的是乱码。而字符编码的转换都需要经过中间的unicode，即需要decode再encode。

#### 例6
将“你好”2个字写到data.txt，以gbk编码。然后执行下面的代码：
```python
# coding:utf-8
print open('data.txt').read()
```
输出：
> 你好

#### 例7
将“你好”2个字写到data.txt，以utf-8编码。然后执行下面的代码：
```python
# coding:utf-8
print open('data.txt').read()
```
输出：
> 浣犲ソ

#### 例8
将“你好”2个字写到data.txt，以utf-8编码。然后执行下面的代码：
```python
# coding:utf-8
print open('data.txt').read().decode('utf-8').encode('gbk')
```
输出：
> 你好

#### 例9
将“你好”2个字写到 你好.txt ，以utf-8编码。然后执行下面的代码：
```python
# coding:utf-8
print open('你好.txt').read().decode('utf-8').encode('gbk')
```
输出：
> Traceback (most recent call last):
  File "x.py", line 2, in <module>
    print open('浣犲ソ.txt').read().decode('utf-8').encode('gbk')
IOError: [Errno 2] No such file or directory: '\xe4\xbd\xa0\xe5\xa5\xbd.txt'

#### 例10
将“你好”2个字写到 你好.txt，以utf-8编码。然后执行下面的代码：
```python
# coding:utf-8
print open('你好.txt'.decode('utf-8')).read().decode('utf-8').encode('gbk')
```
输出：
> 你好

对比例6、例7、例8、例9、例10，可以发现python读取文件时，输入的文件名必须是python内部默认编码格式，如果不是则需要转换成默认编码，python获取到的文件内容是如果需要print出来不乱吗，均需要做转换。

## 参考
1. [字符编解码的故事（ascii，ansi，unicode，utf-8区别）](http://www.imkevinyang.com/2009/02/%E5%AD%97%E7%AC%A6%E7%BC%96%E8%A7%A3%E7%A0%81%E7%9A%84%E6%95%85%E4%BA%8B%EF%BC%88ascii%EF%BC%8Cansi%EF%BC%8Cunicode%EF%BC%8Cutf-8%E5%8C%BA%E5%88%AB%EF%BC%89.html)
2. [谈谈Unicode编码，简要解释UCS、UTF、BMP、BOM等名词](http://www.fmddlmyy.cn/text6.html)
3. [python编码问题](http://www.cnblogs.com/huxi/articles/1897271.html)
