#!/bin/bash
# 计算阶乘
# 由于$?无法返回大数字,使用全局变量存储返回的值

let result=0
let localreturn=0

funFactorial(){
    let num1=1
    if [ $1 -gt $num1 ]; then
        let numSub=$1-1
        funFactorial $numSub
        let result=$localreturn*$1
        let localreturn=$result
    else
        let result=1
        let localreturn=1;
    fi
}
if [ -n "$1" ]; then
    funFactorial $1
    echo result:$result
else
    echo "usage: factorial.sh [n]"
    echo "calculates a number's factorial"
fi