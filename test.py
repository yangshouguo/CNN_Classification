#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/24 下午2:36
# @Author  : 杨寿国
# @mail    : 891584158@qq.com
# @File    : test.py.py
# @intro:
import numpy as np
import tensorflow as tf

def position_encoding(sentence_size, embedding_size):
    """
    Position Encoding described in section 4.1 of
    End-To-End Memory Networks (https://arxiv.org/abs/1503.08895).

    Args:
      sentence_size: length of the sentence
      embedding_size: dimensionality of the embeddings

    Returns:
      A numpy array of shape [sentence_size, embedding_size] containing
      the fixed position encodings for each sentence position.
    """
    encoding = np.ones((sentence_size, embedding_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_size + 1
    for k in range(1, le):
        for j in range(1, ls):
            encoding[j - 1, k - 1] = (1.0 - j / float(ls)) - (
                                                                 k / float(le)) * (1. - 2. * j / float(ls))

            print("({},{}) = (1.0 - {}/{}) - ( {} / {} ) * (1 - 2 * {} / {}) ".format(j-1,k-1, j, ls , k, le, j ,ls))
    return encoding


pe = position_encoding(1024, 4)


