#!/bin/bash

if [ -n "$1" ]; then
    sum=1
    i=$1
    while((i>0))
    do
        let sum=sum*i
        let i=i-1
    done
    echo $sum
else
    echo "usage: factorial.sh [n]"
    echo "calculates a number's factorial"
fi