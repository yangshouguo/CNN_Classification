#!/bin/bash

# extract code section from elf binary

if [ "$#" -ne 1 ];then
    echo "Example: $0 bin_path"
    exit
fi

bin_path=$1

#using in linux
# "tail -n +3" for executables
# "tail -n +4" for object file

#readelf -x .text $bin_path |tail -n +3 |cut -c 8-|xxd -r


# using in mac os
greadelf -x .text $bin_path |tail -n +4 |cut -c 8-|xxd -r
