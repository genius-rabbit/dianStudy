#!/bin/bash

# usage: self_compression.sh [--list] or [source compressd file] [destination path]
# self compression accroding to file name suffix
if [ "" = "$1" ]; then
    echo "usage: self_compression.sh [--list] or [source compressd file] [destination path]"
    echo "self compression accroding to file name suffix"
else 
    if [ "--list" = "$1" ]; then
        echo "Support file types:zip tar tar.gz tar.bz2"
    else
        # get type
        file=$1
        type=${file##*.}
        echo $type

        if [ "$type" = "zip" ]; then
            unzip $1 -d $2
        elif [ "$type" = "tar" ]; then
            tar -xf $1 -C $2
        elif [ "$type" = "gz" ]; then
            tar -xzvf $1 -C $2
        elif [ "$type" = "bz2" ]; then
            tar -xjvf $1 -C $2
        else
            echo "$type Not Suport!!"
        fi
    fi
fi