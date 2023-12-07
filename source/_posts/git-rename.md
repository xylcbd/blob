---
title: git如何修改历史commit的用户名和邮箱
date: 2016-12-03 13:49:03
categories:
 - 工具使用
tags:
 - git
---

进入目标git仓库，将以下脚本中的YOUR_NAME和YOUR_EMAIL替换成自己的用户名和邮箱，保存为脚本执行。  

之后本地的git提交历史中所有的用户名和邮箱将被替换成你所需要的。  

注：master可能无法直接push到远端仓库，可将远端仓库删除重建。

```shell
#!/bin/sh

git filter-branch -f --env-filter '

CORRECT_NAME="YOUR_NAME"
CORRECT_EMAIL="YOUR_EMAIL"

export GIT_COMMITTER_NAME="$CORRECT_NAME"
export GIT_COMMITTER_EMAIL="$CORRECT_EMAIL"

export GIT_AUTHOR_NAME="$CORRECT_NAME"
export GIT_AUTHOR_EMAIL="$CORRECT_EMAIL"

' --tag-name-filter cat -- --branches --tags
```

或者如果是多人参与的项目，只是想改一部分用户名的话，可以用下面的脚本。替换YOUR_OLD_EMAIL、YOUR_NAME、YOUR_EMAIL。 

```shell
#!/bin/sh

git filter-branch --env-filter '

OLD_EMAIL="YOUR_OLD_EMAIL"
CORRECT_NAME="YOUR_NAME"
CORRECT_EMAIL="YOUR_EMAIL"

if [ "$GIT_COMMITTER_EMAIL" = "$OLD_EMAIL" ]
then
    export GIT_COMMITTER_NAME="$CORRECT_NAME"
    export GIT_COMMITTER_EMAIL="$CORRECT_EMAIL"
fi
if [ "$GIT_AUTHOR_EMAIL" = "$OLD_EMAIL" ]
then
    export GIT_AUTHOR_NAME="$CORRECT_NAME"
    export GIT_AUTHOR_EMAIL="$CORRECT_EMAIL"
fi
' --tag-name-filter cat -- --branches --tags
```
