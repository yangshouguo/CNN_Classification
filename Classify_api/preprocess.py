#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/27 下午2:01
# @Author  : 杨寿国
# @mail    : 891584158@qq.com
# @File    : extract_text_section_hex.py
# @intro: 提取数据集中 obj 文件的 .text 段的所有数据，并以16进制形式存入另一个文件
# python extract_hex_and_save.py ../raw-data/ARM/O0/ ../hex_dataset_ysg/ARM/O0/
import os, sys
import subprocess
script_path = 'extract_text_section.sh'

destinationdir = ''

def getFileNameFromDir(source_dir):

    for parent, dirnames, filenames in os.walk(source_dir, followlinks=True):

        for filename in filenames:
            execScript(os.path.join(parent, filename) ,os.path.split(parent)[-1], filename)


def execScript(sourcefile ,targetdir, targetfile):
    target = os.path.join(destinationdir, targetdir)
    n_targetfile = os.path.splitext(targetfile)
    if not os.path.exists(target):
        os.mkdir(target)
    cmd = "./{} {} > {}".format(script_path, sourcefile,os.path.join(target, n_targetfile[0]+'.hex'))
    print (cmd)
    s,o = subprocess.getstatusoutput(cmd)


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print('python xxx.py sourcedir targetdir')
        exit(0)

    source_dir = sys.argv[1]
    destinationdir = sys.argv[2]

    getFileNameFromDir(source_dir)