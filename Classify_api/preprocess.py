#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/24 上午11:09
# @Author  : 杨寿国
# @mail    : 891584158@qq.com
# @File    : preprocess.py
# @intro:  调用脚本从二进制文件中提取4k大小的片段返回

import subprocess as sp

def get_text_sec_binary(filepath):
    script = './extract_text_section.sh'
    cmd_str = 'ls -l {}'.format(filepath)
    cmd_list = cmd_str.split(' ')
    print(cmd_list)
    out = sp.check_output(cmd_list)

    out_byte = list(bytearray(out))

    arr = []

    for i in range(40):
        arr.append(out_byte[i*4:(i+1)*4])
        for j in range(4):
            print(hex(arr[i][j]), end=' ')
        print()



    #取前40行数据






if __name__ == '__main__':
    get_text_sec_binary('./')
