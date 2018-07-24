#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/24 下午2:50
# @Author  : 杨寿国
# @mail    : 891584158@qq.com
# @File    : compile_level_classi.py
# @intro:  加载之前训练好的分类模型，对输入的数据进行预测分类
import tensorflow as tf
import os

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

        print('predict:' + str(result[0]))

        pass

if __name__ == '__main__':
    cla = Classifier('./checkpoints/model-30.meta')
    from data_helper import DataHelper
    datahelper = DataHelper()
    inputx,y = datahelper.load_data_and_labels('../../dataset/ARM')
    cla.predict(inputx[0].reshape(-1,1024,4))


