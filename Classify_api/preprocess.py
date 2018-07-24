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

    lines = out.split(b'\n')
    arr = []

    #取前40行数据
    n = 40
    for i in range(n):
        arr.append(list(bytearray(lines[i])))

    return arr





if __name__ == '__main__':
    get_text_sec_binary('./')
