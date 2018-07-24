#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/24 上午11:09
# @Author  : 杨寿国
# @mail    : 891584158@qq.com
# @File    : preprocess.py
# @intro:  调用脚本从二进制文件中提取4k大小的片段返回

import subprocess as sp

def get_text_sec_binary(filepath):
    cmd_str = './extract_text_section.sh {}'.format(filepath)
    print(cmd_str)
    s,o = sp.getstatusoutput(cmd_str)
    print(s , o)



if __name__ == '__main__':
    get_text_sec_binary('./ps')
