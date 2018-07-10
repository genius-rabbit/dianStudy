#!/bin/bash
# 计算阶乘
if [ -n "$1" ]; then
    sum=1
    i=$1
    # 循环计算
    while((i>0))
    do
        let sum=sum*i
        let i=i-1
    done
    echo result:$sum
else
    echo "usage: factorial.sh [n]"
    echo "calculates a number's factorial"
fi