#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/18 2:48 PM
# @Author  : 杨寿国
# @mail    : 891584158@qq.com
# @File    : roc_curve_plot.py.py
# @intro:  只是单纯为了刻画模型的roc曲线   ,  代码是在 compiler_option_level_classif 的基础上进行修改


import tensorflow as tf
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
print(tf.__version__)
print(tf.__path__)
import os

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
        self._output = graph.get_tensor_by_name("output/scores:0")

        self._maxpool= graph.get_tensor_by_name("conv-maxpool-4/pool_argmax:0")

        self.filer_W = graph.get_tensor_by_name("conv-maxpool-4/W:0")

        self._loaded = True

    def predict(self, inputx):

        if not self._loaded:
            print('you need load model first!')
            return

        # self._sess.run()

        result,maxpool = self._sess.run([self._output, self._maxpool], feed_dict={self._inputx: inputx, self._dropout: 1.0})
        return result






from train import preprocess

print((dir(tf.flags.FLAGS)))

if 'dev_sample_percentage' in tf.flags.FLAGS:
    tf.flags.FLAGS.__delattr__('dev_sample_percentage')
if 'data_file' in tf.flags.FLAGS:
    tf.flags.FLAGS.__delattr__('data_file')
tf.flags.DEFINE_float("dev_sample_percentage", .8, "Percentage of the training data to use for validation (default is 0.2)")
tf.flags.DEFINE_string("data_file", "../dataset_single_obj/",
                       "Data source for the data.")

x_train, y_train, x_dev, y_dev = preprocess()

cla = Classifier('Classify_api/checkpoints/model-67500.meta')

# result = []

batch = 1
pos = 1

result = cla.predict(x_dev[0:1])

while batch+pos <= len(x_dev):
    result = np.concatenate((result, cla.predict(x_dev[pos:pos+batch])), axis=0)
    pos+=batch

if pos < len(x_dev)-1:
    result = np.concatenate((result, cla.predict(x_dev[pos:])), axis=0)


auc = metrics.roc_auc_score(y_dev, result)
# print('function auc : ',)

fpr, tpr, threshold = metrics.roc_curve(y_dev.ravel(), result.ravel())

plt.plot(fpr, tpr, c = 'r', lw=2, alpha = 0.7, label = u'AUC=%.3f' % auc)

plt.plot((0, 1), (0, 1), c = '#808080', lw = 1, ls = '--', alpha = 0.7)

plt.xlim((-0.01, 1.02))
plt.ylim((-0.01, 1.02))
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlabel('False Positive Rate', fontsize=13)
plt.ylabel('True Positive Rate', fontsize=13)
plt.grid(b=True, ls=':')
plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
plt.title(u'Compiler Option Classification ROC&AUC', fontsize=17)
plt.savefig('roc.pdf')
