# -*- coding: utf-8 -*-

# from cnn_crf_model import *
from rnn_crf_model import *
from data.cnews_loader import *

import time
from datetime import timedelta

base_dir = 'data/cnews'
ner_train_source_dir = os.path.join(base_dir, 'ner.train.source.txt')
ner_train_target_dir = os.path.join(base_dir, 'ner.train.target.txt')
ner_test_source_dir = os.path.join(base_dir, 'ner.test.source.txt')
ner_test_target_dir = os.path.join(base_dir, 'ner.test.target.txt')
vocab_dir = os.path.join(base_dir, 'ner.vocab.txt')

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch, len_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob,
        model.seq_length: len_batch
    }
    return feed_dict


def musk(x, len_actual, len_total):
    bool_musk = [1] * len_actual + [0] * (len_total - len_actual)
    return np.array(x)[np.array(bool_musk) > 0]


def preds(x_batch, y_batch, len_batch, sess):
    res = []
    batch_loss = 0.0
    for ii in range(len(x_batch)):
        feed_dict = {
            model.input_x: [x_batch[ii]],
            model.input_y: [y_batch[ii]],
            model.keep_prob: 1.0,
            model.seq_length: [len_batch[ii]]
        }
        loss, np_logits, np_transition_params = sess.run([model.loss, model.logits, model.transition_params], feed_dict=feed_dict)
        batch_loss += loss
        np_logits = np.squeeze(np_logits)
        np_viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(np_logits, np_transition_params)
        res.extend(musk(np_viterbi_sequence, len_batch[ii], config.seq_length))
    return batch_loss, res


def evaluate(sess, x_, y_, len_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, len_, 64)
    total_batch_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch, len_batch in batch_eval:
        batch_len = len(x_batch)
        y_batch_actual = []
        for ii in range(batch_len):
            y_batch_actual.extend(musk(y_batch[ii], len_batch[ii], config.seq_length))
        total_loss, np_viterbi_sequence = preds(x_batch, y_batch, len_batch, sess)
        correct_pred = np.equal(np.array(y_batch_actual), np_viterbi_sequence)
        acc = np.count_nonzero(correct_pred) / len(np.array(np_viterbi_sequence).flatten())
        total_acc += acc * batch_len
        total_batch_loss += total_loss
    return total_batch_loss / data_len, total_acc / data_len


if __name__ == '__main__':
    print('Configuring CNN model...')
    config = TCNNConfig()
    if not os.path.exists(vocab_dir):
        build_vocab(ner_train_source_dir, vocab_dir)
    categories, cat_to_id = read_category(ner_train_target_dir)
    words, word_to_id = read_vocab(vocab_dir)

    config.vocab_size = len(words)
    config.num_classes = len(categories)

    model = TextCNN(config)

    """train model"""
    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 载入训练集与验证集
    print("Loading training and val data...")
    x_train, len_trains = process_nn_crf_source_file(ner_train_source_dir, word_to_id, config.seq_length)
    x_val, len_vals = process_nn_crf_source_file(ner_test_source_dir, word_to_id, config.seq_length)
    y_train = process_nn_crf_target_file(ner_train_target_dir, cat_to_id, config.seq_length)
    y_val = process_nn_crf_target_file(ner_test_target_dir, cat_to_id, config.seq_length)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 100  # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, len_trains, 128)
        for x_batch, y_batch, len_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, len_batch, config.dropout_keep_prob)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                loss_train, np_logits, np_transition_params = session.run([model.loss, model.logits, model.transition_params], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val, len_vals)

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, 0.0, loss_val, acc_val, time_dif, improved_str))

            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break
