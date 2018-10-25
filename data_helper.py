#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2018/7/8 上午10:33
# @Author  : 杨寿国
# @mail    : 891584158@qq.com
# @File    : data_helper.py
# @intro: 读取各个优化级别的二进制文件，并且自动附加标签，从O0 - Os 依次对应 0~5
import os
import numpy as np
import re


class DataHelper(object):
    def __init__(self, class_num=4):
        self.rowbyte = 4
        self._class_num = class_num

        pass

    def get_all_data(self, directory='./dataset/ARM'):

        all_result = []
        label = 0

        print('loading data {} from {} : label {}'.format("O0", directory, label))
        all_result += self.get_certain_compile_level_with_label(os.path.join(directory, 'O0'), label=label)
        label+=1

        print('loading data {} from {} : label {}'.format("O1", directory, label))
        all_result += self.get_certain_compile_level_with_label(os.path.join(directory, 'O1'), label=label)
        label += 1
        print('loading data {} from {} : label {}'.format("O2", directory, label))
        all_result += self.get_certain_compile_level_with_label(os.path.join(directory, 'O2'), label=label)
        if self._class_num ==5 :
            label+=1

        print('loading data {} from {} : label {}'.format("O3", directory, label))
        all_result += self.get_certain_compile_level_with_label(os.path.join(directory, 'O3'), label=label)
        label+=1
        print('loading data {} from {} : label {}'.format("Os", directory, label))

        all_result += self.get_certain_compile_level_with_label(os.path.join(directory, 'Os'), label=label)

        return np.array(all_result)

    def get_certain_compile_level_with_label(self, directory, label):
        # directory: 某一优化级别的二进制文件所在目录
        # label:该类二进制文件对应标签

        result = []
        # 迭代访问目录下每个文件
        for parent, dirs, files in os.walk(directory):
            # result.append([self.read_binary_from_file(filez), label])
            for file in files:
                filedatas = self.read_binary_from_file(os.path.join(parent, file))
                for hexfiledata in filedatas:
                    if hexfiledata is not None:
                        result.append([hexfiledata, label])

        return result

    def load_data_and_labels(self, dir):
        data = self.get_all_data(dir)
        train_data = np.array(list(data[:, 0]), dtype=np.float32)
        print(data.shape)
        # 归一化
        # train_data = np.true_divide(train_data, 1)
        return train_data, data[:, 1]

    def read_binary_from_file(self, file_path, rowbyte=4, min_row=100, rows=1024):
        # file_path: 二进制文件的路径
        # rowbyte: 每一行的字节数
        # min_row : 最小字节行数 ( min_row * 4 )
        # rows : 从二进制文件中提取的字节行数 ( rows * 4 )
        # 返回一个矩阵 N*32矩阵 N = 4096/32 = 128

        allbytevalue = []

        #读取所有字节内容
        # 将所有字节内容分成多个 rows * rowbyte 的numpy数组并返回
        with open(file_path, 'rb') as f:
            allbytevalue = list(bytearray(f.read()))

        seg_length = rowbyte*rows
        segs = int(len(allbytevalue)/seg_length)

        byte_arrays = []
        for i in range(segs):
            byte_arrays.append(np.array(allbytevalue[i*seg_length:(i+1)*seg_length]).reshape(rows, rowbyte))

        byte_array = np.array(allbytevalue[segs*seg_length:])
        if byte_array.size < rowbyte * rows:
            byte_array = np.pad(byte_array, (0, rows * rowbyte - byte_array.size), 'constant')
        byte_arrays.append(byte_array.reshape(rows, rowbyte))
        return byte_arrays

    def read_hex_from_file(self, file_path, rows=1024, min_len=100, fromMid=True):
        # file_path: 二进制文件的路径
        # rows : 最多读取文件的行数，即二进制文件的指令长度
        # min_len ： 文件包含的最少的指令长度，如果少于这个长度，丢弃该文件
        # fromMid : 从二进制文件的中间部分开始读，如果文件长度大于1024行

        data_4byte = []
        readed_lines = rows
        byte_len = 4  # 每一行4个字节
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if len(lines) < min_len:
                return None

            if rows > len(lines):
                readed_lines = len(lines)

            for index in range(readed_lines):
                hexdata = lines[index].strip()

                if len(hexdata) == 0:
                    continue

                split_data = re.findall(r'.{2}', hexdata)

                split_integer = []
                for byte_hex in split_data:
                    split_integer.append(int('0x' + byte_hex, base=16))

                if len(split_integer) < byte_len:
                    n_split_integer = np.zeros(byte_len)
                    n_split_integer[0:len(split_integer)] = split_integer
                    split_integer = n_split_integer.tolist()

                data_4byte.append(split_integer)

            if rows > len(data_4byte):
                fill_data = np.zeros((rows, byte_len))
                fill_data[0:len(data_4byte)] = data_4byte
                return fill_data

        return np.array(data_4byte).reshape((-1, byte_len))

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
    dh = DataHelper(5)
    data = dh.load_data_and_labels('../dataset_singlefunc/')

    print({0:[1,2,3]})


if __name__ == '__main__':
    test()
