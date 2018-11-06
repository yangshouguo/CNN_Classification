#!/bin/bash

# extract code section from elf binary

if [ "$#" -ne 1 ];then
    echo "Example: $0 bin_path"
    exit
fi

bin_path=$1

readelf -x .text $bin_path |tail -n +4 |cut -c 8-|xxd -r
#hexdump $bin_path |tail -n +4|cut -c 8-