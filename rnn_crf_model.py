# -*- coding: utf-8 -*-

from tensorflow.contrib.rnn import DropoutWrapper
import tensorflow as tf


class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 64      # 词向量维度
    seq_length = 50        # 序列长度
    num_classes = 0        # 类别数

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
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())
        self.seq_length = tf.placeholder(tf.int32, [None, ])

        self.cnn()

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
        embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("score"):
            GRU_cell_fw = tf.contrib.rnn.GRUCell(300)
            GRU_cell_bw = tf.contrib.rnn.GRUCell(300)

            GRU_cell_fw = DropoutWrapper(GRU_cell_fw, input_keep_prob=1.0, output_keep_prob=self.keep_prob)
            GRU_cell_bw = DropoutWrapper(GRU_cell_bw, input_keep_prob=1.0, output_keep_prob=self.keep_prob)

            ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_cell_fw,
                                                                                 cell_bw=GRU_cell_bw,
                                                                                 inputs=embedding_inputs,
                                                                                 sequence_length=self.seq_length,
                                                                                 dtype=tf.float32)
            outputs = tf.concat((fw_outputs, bw_outputs), 2)    # ?, 50, 300 * 2
            print(outputs)
            h_pool1 = tf.reshape(outputs, [-1, 2 * 300])
            W_fc1 = tf.Variable(tf.truncated_normal([2 * 300, 1024], stddev=0.1))
            b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
            h_fc1 = tf.nn.relu(tf.matmul(h_pool1, W_fc1) + b_fc1)
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

            # 分类器
            self.logits = tf.layers.dense(h_fc1_drop, self.config.num_classes, name='fc2')
            # 再reshape回去
            logits_in = tf.reshape(self.logits, [-1, self.config.seq_length, self.config.num_classes])


        with tf.name_scope("optimize"):
            self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(logits_in, self.input_y, self.seq_length)
            self.loss = tf.reduce_mean(-self.log_likelihood)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
