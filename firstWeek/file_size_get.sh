#!/bin/bash
# echo "usage:file_size_get.sh [-n N] [-d DIR]"
# echo "show top N largest files/directories"

if [ -n "$1" -o -n "$3" ];then
    if [ -n "$1" -a -n "$3" ];then
        du -ah $4 | sort -nr | head -$2
    elif [ "-n" = "$1" ];then
        du -ah | sort -nr | head -$2
    else
        du -ah $2 | sort -nr
    fi
else
    du -ah | sort -nr
fi