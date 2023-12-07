---
title: 调试一个顺手的ubuntu
date: 2019-09-23 21:19:03
categories:
 - 工具使用
tags:
 - 工具使用
---

### home目录设置

```bash
sudo su
# 下面命令将无法执行，因为登录session仍在运行。并会告知session对应的pid
usermod -d /home/user502home -m user502
# 4220是登录session的pid
nohup kill 4220; sleep 2; usermod -d /home/user502home -m user502 &
```

### 创建常用目录

```bash
mkdir dataset
mkdir downloads
mkdir workspace
mkdir opensource
```

### console颜色设置

```bash
在.bashrc的最后面添加：
PS1='${debian_chroot:+($debian_chroot)}\[\033[01;35;40m\]\u\[\033[00;00;40m\]@\[\033[01;35;40m\]\h\[\033[00;31;40m\]:\[\033[00;00;40m\]\w \[\033[01;32;40m\]\$ \[\033[01;36;40m\]'
```

### 设置apt源

vim /etc/apt/sources.list

```
# 阿里源
deb http://mirrors.aliyun.com/ubuntu/ vivid main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ vivid-security main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ vivid-updates main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ vivid-proposed main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ vivid-backports main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ vivid main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ vivid-security main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ vivid-updates main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ vivid-proposed main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ vivid-backports main restricted universe multiverse
# 官方源
deb http://archive.ubuntu.com/ubuntu/ trusty main restricted universe multiverse
deb http://archive.ubuntu.com/ubuntu/ trusty-security main restricted universe multiverse
deb http://archive.ubuntu.com/ubuntu/ trusty-updates main restricted universe multiverse
deb http://archive.ubuntu.com/ubuntu/ trusty-proposed main restricted universe multiverse
deb http://archive.ubuntu.com/ubuntu/ trusty-backports main restricted universe multiverse
```

### 设置pip源

> vim ~/.pip/pip.conf

```
[global]
index-url = http://mirrors.aliyun.com/pypi/simple/
[install]
trusted-host=mirrors.aliyun.com
```

### 安装常用库

```
anaconda: https://www.anaconda.com/
sudo apt upgrade
sudo apt update
sudo apt install build-essential cmake

pip install numpy scipy pandas matplotlib requests lmdb Pillow torch torchvision  tensorboard tensorboardX opencv-python gunicorn flask editdistance
```

### 设置vim

设置~/.vimrc为如下：

```
set encoding=utf-8
let g:indentLine_char='┆'
let g:indentLine_enabled = 1
set nocompatible
set number
set noic
set guioptions-=r
set guioptions-=L
set guioptions-=b
set showtabline=0
set guifont=Monaco:h13
syntax on
set background=dark
set nowrap
set fileformat=unix
set cindent
set tabstop=4
set shiftwidth=4
set showmatch
set scrolloff=5
set laststatus=2
set fenc=utf-8
set backspace=2
set mouse=a
set selection=exclusive
set selectmode=mouse,key
set matchtime=5
set ignorecase
set incsearch
set hlsearch
set expandtab
set whichwrap+=<,>,h,l
set autoread
set cursorline
filetype off
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()
Plugin 'VundleVim/Vundle.vim'
Plugin 'Lokaltog/vim-powerline'
Plugin 'scrooloose/nerdtree'
Plugin 'Yggdroot/indentLine'
Plugin 'jiangmiao/auto-pairs'
Plugin 'tell-k/vim-autopep8'
Plugin 'scrooloose/nerdcommenter'
Plugin 'solarnz/thrift.vim'
Plugin 'zxqfl/tabnine-vim'
call vundle#end()
filetype plugin indent on
```

安装vim插件：

```
cd ~/
mkdir -p ~/.vim/bundle
git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim
# 打开vimrc进行插件安装
vim ~/.vimrc
:PluginInstall
```
