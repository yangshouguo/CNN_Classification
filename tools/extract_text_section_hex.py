#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/27 下午2:01
# @Author  : 杨寿国
# @mail    : 891584158@qq.com
# @File    : extract_text_section_hex.py
# @intro: 提取数据集中 obj 文件的 .text 段的所有数据，并以16进制形式存入另一个文件

import os, sys
script_path = ''

destinationdir = ''

def getFileNameFromDir(source_dir):

    for parent, dirnames, filenames in os.walk(source_dir, followlinks=True):

        for filename in filenames:
            execScript(os.path.join(os.path.split(parent)[-1], filename))

    pass


def execScript(targetfile):
    target = os.path.join(destinationdir, targetfile)
    if not os.path.exists(target):
        os.mkdir(target)

    print('do {} to target file {}'.format(targetfile, target))
    pass


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print('python xxx.py sourcedir targetdir')
        exit(0)

    source_dir = sys.argv[1]
    destinationdir = sys.argv[2]

    getFileNameFromDir(source_dir)
