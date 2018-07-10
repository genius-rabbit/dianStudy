#!/bin/bash
# echo "usage:file_size_get.sh [-n N] [-d DIR]"
# echo "show top N largest files/directories"
# 获得指定或默认目录下的前N个文件或者所有文件

# 存在参数
if [ -n "$1" -o -n "$3" ];then
    # 有两个参数
    if [ -n "$1" -a -n "$3" ];then
        du -ah $4 | sort -nr | head -$2
    # 只有参数-n
    elif [ "-n" = "$1" ];then
        du -ah | sort -nr | head -$2
    # 只有参数-d
    else
        du -ah $2 | sort -nr
    fi
# 不存在参数
else
    du -ah | sort -nr
fi