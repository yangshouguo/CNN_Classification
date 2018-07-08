#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2018/7/8 上午10:33
# @Author  : 杨寿国
# @mail    : 891584158@qq.com
# @File    : data_helper.py
# @intro: 读取各个优化级别的二进制文件，并且自动附加标签，从O0 - Os 依次对应 0~5
import os
import numpy as np


class DataHelper(object):
    def __init__(self):
        self.rowbyte = 4
        pass

    def get_all_data(self, directory='./dataset/ARM'):

        all_result = []
        print('loading data {} from {}'.format("O0", directory))
        all_result += self.get_certain_compile_level_with_label(os.path.join(directory, 'O0'), label=0)
        print('loading data {} from {}'.format("O1", directory))
        all_result += self.get_certain_compile_level_with_label(os.path.join(directory, 'O1'), label=1)
        print('loading data {} from {}'.format("O2", directory))
        all_result += self.get_certain_compile_level_with_label(os.path.join(directory, 'O2'), label=2)
        print('loading data {} from {}'.format("O3", directory))
        all_result += self.get_certain_compile_level_with_label(os.path.join(directory, 'O3'), label=3)
        print('loading data {} from {}'.format("Os", directory))
        all_result += self.get_certain_compile_level_with_label(os.path.join(directory, 'Os'), label=4)

        return np.array(all_result)

    def get_certain_compile_level_with_label(self, directory, label):
        # directory: 某一优化级别的二进制文件所在目录
        # label:该类二进制文件对应标签

        result = []
        # 迭代访问目录下每个文件
        for file in os.listdir(directory):
            result.append([self.read_binary_from_file(os.path.join(directory, file)), label])

        return result

    def load_data_and_labels(self, dir):
        data = self.get_all_data(dir)
        train_data = np.array(list(data[:, 0]), dtype=np.float32)
        return train_data, data[:, 1]

    def read_binary_from_file(self, file_path, rowbyte=4, filesize=4096):
        # file_path: 二进制文件所在目录
        # rowbyte: 每一行的字节数
        # filesize : 文件大小
        # 返回一个矩阵 N*32矩阵 N = 4096/32 = 128

        allbytevalue = []
        with open(file_path, 'rb') as f:
            allbytevalue = list(bytearray(f.read(filesize)))

        byte_array = np.array(allbytevalue)
        return byte_array.reshape(-1, rowbyte)

    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]


def test():
    dh = DataHelper()
    data = dh.get_all_data("./dataset/ARM/")
    input = data[:, 0]
    labels = data[:, 1]
    single_in = input[0]
    print(labels)


if __name__ == '__main__':
    test()
