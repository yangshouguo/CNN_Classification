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
import numpy as np
import matplotlib.pyplot as plt
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

        #恢复图和权
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

        self._maxpool= graph.get_tensor_by_name("conv-maxpool-4/pool_argmax:0")

        self.filer_W = graph.get_tensor_by_name("conv-maxpool-4/W:0")

        self._loaded = True

    def predict(self, inputx):

        if not self._loaded:
            print('you need load model first!')
            return

        # self._sess.run()

        result,maxpool = self._sess.run([self._output, self._maxpool], feed_dict={self._inputx: inputx, self._dropout: 1.0})

        result = result.tolist()

        def analyse_pooled(seg = 0):

            maxpooled = maxpool[seg].flatten()
            # pos_zero = np.where(maxpooled>5)[0]
            # maxpooled[pos_zero] = 0
            pos_max = np.where(maxpooled>4)[0]



            #"""Turn the interactive mode on."""
            plt.ion()
            plt.figure(0)
            plt.ylabel('score')
            plt.xlabel('instruction position')
            plt.title('zsh_exec_O2')
            plt.scatter(pos_max, maxpooled[pos_max])
            for pos in pos_max:
                # print(inputx[0][pos])
                plt.annotate("({},{:.2f})".format(hex(pos*4), maxpooled[pos]), (pos, maxpooled[pos]), xycoords='data'
                             )
                print("{}:{}, mark=''".format(pos, maxpooled[pos]))
                print(hex(pos*4))
            plt.show()


        analyse_pooled()
        # channel_0 = np.reshape(conv_out, [-1,1])
        # get channel avg and filter
        # plt.figure(0)
        # # plt.subplot(2,4,i+1)
        # plt.scatter(range(len(segment_i)),segment_i)
        # for pos in annotate_pos:
        #     print(inputx[0][pos])
        #     plt.annotate("({},{:.2f})".format(pos,round(segment_i[pos],3)), (pos, segment_i[pos]))
        # plt.show()

        # 取众数

        count = np.bincount(result)

        pre = np.argmax(count)

        print('model predict output:' + str(pre))

        print('该文件的编译优化选项为', )
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

        return pre


def GenerateBinaryData(fromfile, tofile):
    cmd = './extract_text_section.sh {} > {}'.format(fromfile, tofile)
    os.system(cmd)


if __name__ == '__main__':
    #参数
    ArgParse = argparse.ArgumentParser(description="本脚本加载训练好的模型，用于5分类编译优化选项识别")
    ArgParse.add_argument('-f','--file',help="指定待识别编译优化选项的二进制文件的路径")
    args = ArgParse.parse_args()

    from Classify_api.data_helper_test import DataHelper
    datahelper = DataHelper()


    if args.file:

        file_path = args.file
        print(file_path)

        #加载数据集
        tmpfile = file_path+'.bin'
        GenerateBinaryData(file_path, tmpfile)
        inputx = datahelper.read_binary_from_file(tmpfile)
        if os.path.exists(tmpfile):
            os.remove(tmpfile)
        cla = Classifier('./checkpoints/model-67500.meta')
        cla.predict(inputx.reshape(-1,1024,4))


