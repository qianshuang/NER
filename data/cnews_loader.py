# -*- coding: utf-8 -*-

import re
import numpy as np
import os

base_dir = 'data'
stopwords_dir = os.path.join(base_dir, 'stop_words.txt')


def open_file(filename, mode='r'):
    return open(filename, mode, encoding='utf-8', errors='ignore')


def read_file(filename):
    contents = []
    with open_file(filename) as f:
        for line in f:
            try:
                conts = re.split('\s+', line.strip())
                contents.append(conts)
            except:
                print(line)
    return contents


def stopwords_list(filename):
    stopwords = []
    with open_file(filename) as f:
        for line in f:
            try:
                content = line.strip()
                stopwords.append(content)
            except:
                pass
    return stopwords


stopwords = stopwords_list(stopwords_dir)


def remove_stopwords(content):
    return list(set(content).difference(set(stopwords)))


def word2features(words, i):
    # TODO 还可以取词的ngram特征
    word = words[i]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),  # 当前词
        # 'word[-3:]': word[-3:],
        # 'word[-2:]': word[-2:],
        # 'word.isupper()': word.isupper(),
        # 'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        # 'postag': postag,
        # 'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = words[i-1]
        features.update({
            '-1:word.lower()': word1.lower(),  # 当前词的前一个词
            # '-1:word.istitle()': word1.istitle(),
            # '-1:word.isupper()': word1.isupper(),
            # '-1:postag': postag1,
            # '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(words)-1:
        word1 = words[i+1]
        features.update({
            '+1:word.lower()': word1.lower(),  # 当前词的后一个词
            # '+1:word.istitle()': word1.istitle(),
            # '+1:word.isupper()': word1.isupper(),
            # '+1:postag': postag1,
            # '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def process_crf_file(crf_train_source_dir, crf_train_target_dir):
    features = []
    labels = []
    with open_file(crf_train_source_dir) as f:
        for line in f:
            feature = []
            words = re.split('\s+', line.strip())
            for i in range(len(words)):
                feature.append(word2features(words, i))
            features.append(feature)

    with open_file(crf_train_target_dir) as f:
        for line in f:
            label = []
            ls = re.split('\s+', line.strip())
            for i in range(len(ls)):
                label.append(ls[i])
            labels.append(label)

    return np.array(features), np.array(labels)


def build_vocab(total_dir, vocab_dir):
    """根据训练集构建词汇表，存储"""
    print("building vacab...")
    final_words = ["Padding", "Unknown"]
    with open_file(total_dir) as f:
        for line in f:
            conts = re.split('\s+', line.strip())
            for con in conts:
                final_words.append(con)
    open_file(vocab_dir, mode='w').write('\n'.join(set(final_words)) + '\n')


def read_vocab(vocab_dir):
    """读取词汇表"""
    words = open_file(vocab_dir).read().strip().split('\n')
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category(target_dir):
    cates = ['Padding']
    with open_file(target_dir) as f:
        for line in f:
            cates.extend(re.split('\s+', line.strip()))
    categories = list(set(cates))
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id


def pad_sequences(sequences,
                  maxlen=None,
                  dtype='int32',
                  padding='post',
                  truncating='post',
                  value=0.):
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:  # pylint: disable=g-explicit-length-test
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):  # pylint: disable=g-explicit-length-test
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]  # pylint: disable=invalid-unary-operand-type
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                'Shape of sample %s of sequence at position %s is different from '
                'expected shape %s'
                % (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def process_nn_crf_source_file(source_total_dir, word_to_id, seq_length):
    """将文件转换为id表示"""
    contents = read_file(source_total_dir)
    len_ = len(contents)
    len_texts = []

    data_id = []
    for i in range(len_):
        data_id_in_text = []
        for x in contents[i]:
            if x in word_to_id:
                data_id_in_text.append(word_to_id[x])
            else:
                data_id_in_text.append(word_to_id['Unknown'])
        data_id.append(data_id_in_text)

        if len(data_id_in_text) >= seq_length:
            len_texts.append(seq_length)
        else:
            len_texts.append(len(data_id_in_text))

    x_pad = pad_sequences(data_id, seq_length, value=word_to_id['Padding'])

    return x_pad, len_texts


def process_nn_crf_target_file(target_total_dir, cat_to_id, seq_length):
    tags = read_file(target_total_dir)
    tag_id = []
    for i in range(len(tags)):
        tag_id_in_text = []
        for y in tags[i]:
            tag_id_in_text.append(cat_to_id[y])
        tag_id.append(tag_id_in_text)
    y_pad = pad_sequences(tag_id, seq_length, value=cat_to_id['Padding'])

    return y_pad


def batch_iter(x, y, len_, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = []
    y_shuffle = []
    len_shuffle = []
    for i in range(len(indices)):
        x_shuffle.append(x[indices[i]])
        y_shuffle.append(y[indices[i]])
        len_shuffle.append(len_[indices[i]])

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id], len_shuffle[start_id:end_id]
