# -*- coding: utf-8 -*-

import os
import re

"""
数据清洗，去除标注错误的行
"""

base_dir = 'data/cnews'
crf_train_source_dir = os.path.join(base_dir, 'crf.train.source.txt')
crf_train_target_dir = os.path.join(base_dir, 'crf.train.target.txt')
crf_test_source_dir = os.path.join(base_dir, 'crf.test.source.txt')
crf_test_target_dir = os.path.join(base_dir, 'crf.test.target.txt')


def open_file(filename, mode='r'):
    return open(filename, mode, encoding='utf-8', errors='ignore')


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


train = []
features = []
with open_file(crf_test_source_dir) as f:
    for line in f:
        feature = []
        words = re.split('\s+', line.strip())
        for i in range(len(words)):
            feature.append(word2features(words, i))
        features.append(feature)
        train.append(words)

target = []
with open_file(crf_test_target_dir) as f:
    for line in f:
        label = []
        ls = re.split('\s+', line.strip())
        for i in range(len(ls)):
            label.append(ls[i])
        target.append(label)


for i in range(len(target)):
    if len(features[i]) != len(target[i]):
        print(" ".join(train[i]))
        print(target[i])
