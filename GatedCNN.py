#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/2 下午4:04
# @Author  : 杨寿国
# @mail    : 891584158@qq.com
# @File    : GatedCNN.py
# @intro:

import tensorflow as tf
import numpy as np


class TextCNN(object):
    def __init__(self
                 , seq_len, seq_width, num_class, hidden_size, embedding_size, filter_sizes, num_filters,
                 vocabsize=pow(2, 8), position_embedding=False, batch_size = 64):
        pass

        self.input_x = tf.placeholder(tf.int32, [None, seq_len, seq_width], name='inputx')
        self.input_y = tf.placeholder(tf.float32, [None, num_class], name='inputy')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self._hidden_size = hidden_size

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            x_internal = tf.reshape(self.input_x, (-1, seq_len * seq_width))
            W = tf.Variable(
                tf.random_uniform([vocabsize, embedding_size], -1.0, 1.0), name='W'
            )
            #
            # # The result of the embedding operation is a 3-dimensional tensor of shape [None, sequence_length, embedding_size]
            self.embedding_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedding_chars_expanded = self.embedding_chars
            # self.embedding_chars_expanded = tf.reshape(self.embedding_chars, shape=[-1, seq_len, seq_width*embedding_size, 1])
            # self.embedding_chars_expanded = tf.expand_dims(self.input_x, -1)

        if position_embedding:
            lengths = tf.ones(seq_len, dtype=tf.int32) * seq_width
            position_embed = self._create_position_embedding(embedding_dim=self.input_x.get_shape().as_list()[-1],
                                                             num_positions=seq_len,
                                                             lengths=lengths,
                                                             maxlen=tf.shape(self.input_x)[2])
            self.embedding_chars_expanded = tf.add(position_embed, self.embedding_chars_expanded)

        # Convolutional and max-pooling layers
        # use filters of differents sizes
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope('conv-maxpool-%s' % filter_size):
                # Conv layer
                # filter_shape = [filter_size, , embedding_size, num_filters]
                filter_shape = [filter_size, seq_width, embedding_size, num_filters]


                conv_w = self.create_conv(self.embedding_chars_expanded, filter_shape, "linear")
                conv_v = self.create_conv(self.embedding_chars_expanded, filter_shape, "gated")
                h = conv_w * tf.sigmoid(conv_v)

                h = tf.reshape(h, [-1, seq_len, seq_width*num_filters , 1])

                pooled = tf.nn.max_pool(h, ksize=[1, 1, seq_width*num_filters , 1],
                                        # seq_len - filter_size 让卷积得到的结果变成一个单一的值
                                        strides=[1, 1, 1, 1],
                                        padding="VALID",
                                        name='pool')
                pooled_outputs.append(pooled)
        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(axis=3, values=pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [batch_size, -1])

        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        with tf.name_scope('dense_layer'):
            W_d = tf.Variable(tf.truncated_normal([seq_len * len(filter_sizes), self._hidden_size], stddev=0.1), name="W_d")
            b_d = tf.Variable(tf.constant(0.2, shape=[self._hidden_size]), name='b_d')
            self.hidden_out = tf.sigmoid(tf.nn.xw_plus_b(self.h_drop, W_d, b_d))

        with tf.name_scope("dense_layer2"):
            W_d2 = tf.Variable(tf.truncated_normal([self._hidden_size, self._hidden_size], stddev=0.1), name="W_d2")
            b_d2 = tf.Variable(tf.constant(0.2, shape=[self._hidden_size]), name='b_d2')
            self.hidden_out2 = tf.sigmoid(tf.nn.xw_plus_b(self.hidden_out, W_d2, b_d2))

        with tf.name_scope('output'):
            W = tf.Variable(tf.truncated_normal([self._hidden_size, num_class], stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[num_class]), name="b")
            self.scores = tf.nn.xw_plus_b(self.hidden_out2, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def create_conv(self, input, shape, name):
        W = tf.get_variable("%s_W"%name, shape, tf.float32, tf.random_normal_initializer(0.0, 0.1))
        b = tf.get_variable("%s_b"%name, shape[-1], tf.float32, tf.constant_initializer(1.0))
        return tf.add(tf.nn.conv2d(input, W, strides=[1,1,1,1], padding="SAME"), b)

    def position_encoding(self, sentence_size, embedding_size):
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
        return encoding

    def _create_position_embedding(self, embedding_dim, num_positions, lengths, maxlen):
        """Creates position embeddings.

        Args:
          embedding_dim: Dimensionality of the embeddings. An integer.
          num_positions: The number of positions to be embedded. For example,
            if you have inputs of length up to 100, this should be 100. An integer.
          lengths: The lengths of the inputs to create position embeddings for.
            An int32 tensor of shape `[batch_size]`.
          maxlen: The maximum length of the input sequence to create position
            embeddings for. An int32 tensor.

        Returns:
          A tensor of shape `[batch_size, maxlen, embedding_dim]` that contains
          embeddings for each position. All elements past `lengths` are zero.
        """
        # Create constant position encodings
        position_encodings = tf.constant(
            self.position_encoding(num_positions, embedding_dim),
            name="position_encoding")

        # Slice to size of current sequence
        pe_slice = position_encodings[:maxlen, :]
        # Replicate encodings for each element in the batch
        batch_size = tf.shape(lengths)[0]
        pe_batch = tf.tile([pe_slice], [batch_size, 1, 1])

        # Mask out positions that are padded
        positions_mask = tf.sequence_mask(
            lengths=lengths, maxlen=maxlen, dtype=tf.float32)
        positions_embed = pe_batch * tf.expand_dims(positions_mask, 2)

        return positions_embed

