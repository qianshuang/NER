# -*- coding: utf-8 -*-

# from tensorflow.contrib.rnn import DropoutWrapper
import tensorflow as tf
# import numpy as np


class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 64      # 词向量维度
    seq_length = 50        # 序列长度
    num_classes = 0        # tag数
    num_labels = 0         # 类别数
    num_filters = 128        # 卷积核数目

    hidden_dim = 256        # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 64         # 每批训练大小
    num_epochs = 200         # 总迭代轮次

    print_per_batch = 10    # 每多少轮输出一次结果


class TextCNN(object):
    """文本分类，CNN模型"""
    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_y')
        self.input_label = tf.placeholder(tf.float32, [None, self.config.num_labels])
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())
        self.seq_length = tf.placeholder(tf.int32, [None, ])

        self.cnn()

    def conv_1d(self, x, gram, input_channel, output_channel, pool=False):
        filter_w_1 = tf.Variable(tf.truncated_normal([gram, input_channel, output_channel], stddev=0.1))
        filter_b_1 = tf.Variable(tf.constant(0.1, shape=[output_channel]))
        conv_1 = tf.nn.conv1d(x, filter_w_1, padding='SAME', stride=1) + filter_b_1
        h_conv_1 = tf.nn.relu(conv_1)
        if pool:
            return tf.reduce_max(h_conv_1, reduction_indices=[1])
        else:
            return h_conv_1

    def network_bcnn(self, embedding_inputs):
        flaten_1 = self.conv_1d(embedding_inputs, 1, self.config.embedding_dim, 128) # (-1, 100, 128)
        flaten_2 = self.conv_1d(embedding_inputs, 2, self.config.embedding_dim, 128)
        flaten_3 = self.conv_1d(embedding_inputs, 3, self.config.embedding_dim, 128)
        flaten_4 = self.conv_1d(embedding_inputs, 4, self.config.embedding_dim, 128)
        flaten_5 = self.conv_1d(embedding_inputs, 5, self.config.embedding_dim, 128)
        h_pool1 = tf.concat([flaten_1, flaten_2, flaten_3, flaten_4, flaten_5], -1)  # 列上做concat
        return h_pool1

    def fc(self, embedding_inputs, input_dim, output_dim):
        W_ = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
        b_ = tf.Variable(tf.constant(0.1, shape=[output_dim]))
        h_fc = tf.nn.relu(tf.matmul(embedding_inputs, W_) + b_)
        return h_fc

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
        embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("score"):
            # classification-specific features
            rep_fenlei = self.network_bcnn(embedding_inputs)  # [-1, 100, 128 * 5]

            # shared features
            rep_share = self.network_bcnn(embedding_inputs)  # [-1, 100, 128 * 5]

            # NER-specific features
            rep_ner = self.network_bcnn(embedding_inputs)  # [-1, 100, 128 * 5]

            ## 分类任务
            feature_cf = tf.concat([rep_fenlei, rep_share], axis=-1)  # [-1, 100, 128 * 10]
            feature_cf = tf.reduce_max(feature_cf, reduction_indices=[1])  # [-1, 128 * 10]
            feature_cf = tf.nn.dropout(feature_cf, self.keep_prob)
            feature_cf = self.fc(feature_cf, 128 * 10, 2048)
            logits_cf = tf.layers.dense(feature_cf, self.config.num_labels, name='fc1')
            self.y_pred_cls = tf.argmax(logits_cf, 1)  # 预测类别
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits_cf, labels=self.input_label)
            loss_cf = tf.reduce_mean(cross_entropy)

            ## NER任务
            feature_ner = tf.concat([rep_share, rep_ner], axis=-1)  # [-1, 100, 128 * 10]
            feature_ner = tf.reshape(feature_ner, [-1, 128 * 10])
            feature_ner = self.fc(feature_ner, 128 * 10, 2048)
            feature_ner = tf.nn.dropout(feature_ner, self.keep_prob)
            self.logits_in = tf.layers.dense(feature_ner, self.config.num_classes, name='fc2')
            logits_ner = tf.reshape(self.logits_in, [-1, self.config.seq_length, self.config.num_classes])
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(logits_ner, self.input_y, self.seq_length)
            loss_ner = tf.reduce_mean(-log_likelihood)

            # NER result
            # np_logits = np.squeeze(np_logits)
            # print(self.logits_in)
            # self.y_pred_tag, _ = tf.contrib.crf.viterbi_decode(self.logits_in, self.transition_params) # viterbi_decode方法只能同于测试阶段，这样用会报错

        with tf.name_scope("optimize"):
            # 优化器
            self.loss = 0.75 * loss_cf + 0.25 * loss_ner
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
