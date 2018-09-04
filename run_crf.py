# -*- coding: utf-8 -*-

import sklearn_crfsuite
from sklearn_crfsuite import metrics

from data.cnews_loader import *

base_dir = 'data/cnews'
crf_train_source_dir = os.path.join(base_dir, 'crf.train.source.txt')
crf_train_target_dir = os.path.join(base_dir, 'crf.train.target.txt')
crf_test_source_dir = os.path.join(base_dir, 'crf.test.source.txt')
crf_test_target_dir = os.path.join(base_dir, 'crf.test.target.txt')


def train():
    print("start training...")
    # 处理CRF训练数据
    train_feature, train_target = process_crf_file(crf_train_source_dir, crf_train_target_dir)
    # 模型训练
    crf_model.fit(train_feature, train_target)


def test():
    print("start testing...")
    # 处理测试数据
    test_feature, test_target = process_crf_file(crf_test_source_dir, crf_test_target_dir)
    # 去除无意义的标记O
    labels = list(crf_model.classes_)
    labels.remove('O')
    print(labels)
    # 返回预测标记
    test_predict = crf_model.predict(test_feature)
    # test_predict = crf_model.predict_single(test_feature[0])  # 预测单个样本
    accuracy = metrics.flat_f1_score(test_target, test_predict, average='weighted', labels=labels)

    # accuracy
    print()
    print("accuracy is %f" % accuracy)

    # precision    recall  f1-score
    print()
    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )
    print(metrics.flat_classification_report(test_target, test_predict, labels=sorted_labels, digits=3))


# CRF
crf_model = sklearn_crfsuite.CRF(c1=0.1, c2=0.1, max_iterations=200, all_possible_transitions=True)

train()
test()
