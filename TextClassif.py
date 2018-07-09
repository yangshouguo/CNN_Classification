#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2018/7/8 上午11:49
# @Author  : 杨寿国
# @mail    : 891584158@qq.com
# @File    : data_helper.py
# @intro: 神经网络模板
import tensorflow as tf


class TextCNN(object):
    def __init__(self
                 , seq_len, seq_width, num_class, vocabsize, embedding_size, filter_sizes, num_filters):
        pass

        self.input_x = tf.placeholder(tf.float32, [None, seq_len, seq_width], name='inputx')
        self.input_y = tf.placeholder(tf.float32, [None, num_class], name='inputy')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self._hidden_size = 500

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            # W = tf.Variable(
            #     tf.random_uniform([vocabsize, embedding_size], -1.0, 1.0), name='W'
            # )
            #
            # # The result of the embedding operation is a 3-dimensional tensor of shape [None, sequence_length, embedding_size]
            # self.embedding_chars = tf.nn.embedding_lookup(W, self.input_x)
            # self.embedding_chars_expanded = tf.expand_dims(self.embedding_chars, -1)
            self.embedding_chars_expanded = tf.expand_dims(self.input_x, -1)

        # Convolutional and max-pooling layers
        # use filters of differents sizes
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                # Conv layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                conv = tf.nn.conv2d(self.embedding_chars_expanded,
                                    W,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    name='conv')

                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                pooled = tf.nn.max_pool(h, ksize=[1, seq_len - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding="VALID",
                                        name='pool')
                pooled_outputs.append(pooled)
        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(axis=3, values=pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        with tf.name_scope('dense_layer'):
            W_d = tf.Variable(tf.truncated_normal([num_filters_total, self._hidden_size], stddev=0.1), name="W_d")
            b_d = tf.Variable(tf.constant(0.2, shape=[self._hidden_size]),name='b_d')
            self.hidden_out = tf.sigmoid(tf.nn.xw_plus_b(self.h_drop, W_d, b_d))

        with tf.name_scope('output'):
            W = tf.Variable(tf.truncated_normal([self._hidden_size, num_class], stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[num_class]), name="b")
            self.scores = tf.nn.xw_plus_b(self.hidden_out, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
