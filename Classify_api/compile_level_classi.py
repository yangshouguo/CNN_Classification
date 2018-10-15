#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/24 下午2:50
# @Author  : 杨寿国
# @mail    : 891584158@qq.com
# @File    : compile_level_classi.py
# @intro:  加载之前训练好的分类模型，对输入的数据进行预测分类
import tensorflow as tf
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
            return

        # self._sess.run()

        result = self._sess.run(self._output, feed_dict={self._inputx:inputx, self._dropout:1.0})

        print('model predict output:' + str(result[0]))

        print('该文件的编译优化选项为',)
        if result[0] == 0:
            print(' -O0 ')
        elif result[0] == 1:
            print(' -O1 ')
        elif result[0] == 2:
            print(' -O2 ')
        elif result[0] == 3:
            print(' -O3 ')
        else:
            print(' -Os ')


        pass

if __name__ == '__main__':
    #参数
    ArgParse = argparse.ArgumentParser(description="本脚本加载训练好的模型，用于5分类编译优化选项识别")
    ArgParse.add_argument('-f','--file',help="指定待识别编译优化选项的二进制文件的路径")
    args = ArgParse.parse_args()

    cla = Classifier('./checkpoints/model-30.meta')
    from data_helper import DataHelper
    datahelper = DataHelper()



    if args.file:

        file_path = args.file
        print(file_path)
        inputx = datahelper.read_binary_from_file(file_path)
        cla.predict(inputx.reshape(-1,1024,4))


