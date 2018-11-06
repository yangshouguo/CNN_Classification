#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/6 11:07 AM
# @Author  : 杨寿国
# @mail    : 891584158@qq.com
# @File    : test_optimization_recog.py
# @intro:  将一个 大于16K的文件分成4X4K的文件，分别判断每份的编译优化选项
import tensorflow as tf
import numpy as np
import os
import argparse

#设置tf的log日志等级
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append('..')

class Classifier(object):

    def __init__(self, model_dir):
        # self._model = ''
        self._model_pth = model_dir
        # self._saver = tf.train.Saver()
        self._sess = tf.Session()
        self.load_model(self._model_pth)
        self._loaded = False
        self._inputx = ''
        self.load_model(model_dir)
        pass

    #加载模型
    def load_model(self, path):

        #恢复图和权重
        saver = tf.train.import_meta_graph(path)
        saver.restore(self._sess, tf.train.latest_checkpoint(os.path.dirname(path)))

        #恢复占位符
        graph = tf.get_default_graph()
        self._inputx = graph.get_tensor_by_name("inputx:0")
        self._dropout = graph.get_tensor_by_name("dropout_keep_prob:0")

        # for op in graph.get_operations():
        #     print(op)
        # self.output = graph.get_tensor_by_name("predictions:0")
        self._output = graph.get_tensor_by_name("output/predictions:0")

        self._loaded = True

    def predict(self, inputx):

        if not self._loaded:
            print('you need load model first!')
            return 0

        # self._sess.run()

        result = self._sess.run(self._output, feed_dict={self._inputx:inputx, self._dropout:1.0})

        result = result.tolist()

        #取众数
        count = np.bincount(result)
        pre = np.argmax(count)
        #计算众数所占比例
        rate = count[pre] / len(result)
        print('rate : {}'.format(rate))

        if rate > 0.5:
            return 1

        print('model predict output:' + str(pre))

        print('该文件的编译优化选项为',)
        if pre == 0:
            print(' -O0 ')
        elif pre == 1:
            print(' -O1 ')
        elif pre == 2:
            print(' -O2 ')
        elif pre == 3:
            print(' -O3 ')
        else:
            print(' -Os ')

        return 0


def GenerateBinaryData(fromfile, tofile):
    cmd = './extract_text_section.sh {} > {}'.format(fromfile, tofile)
    os.system(cmd)


if __name__ == '__main__':
    #参数
    ArgParse = argparse.ArgumentParser(description="本脚本加载训练好的模型，用于5分类编译优化选项识别")
    ArgParse.add_argument('-f','--file',help="指定待识别编译优化选项的二进制文件的路径")
    args = ArgParse.parse_args()

    from data_helper import DataHelper
    datahelper = DataHelper()

    if args.file:

        file_path = args.file
        print(file_path)
        cla = Classifier('./checkpoints/model-30.meta')

        result = []
        with open(file_path,'r') as f:
            file_lines = f.readlines()
            for file_line in file_lines:
                inputx = datahelper.read_binary_from_file(file_line.strip())
                result.append(cla.predict(inputx.reshape(-1, 1024, 4)))

        print(np.mean(result))
        print(len(result))





