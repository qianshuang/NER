# -*- coding: utf-8 -*-

import numpy as np

y_batch_actual = [[1,2],[1,3]]
np_viterbi_sequence = [[1,2],[1,2]]

correct = np.equal(np.array(y_batch_actual), np.array(np_viterbi_sequence))
print(correct)
acc = np.count_nonzero(correct) / len(np.array(np_viterbi_sequence).flatten())
print(acc)
