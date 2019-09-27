# encoding: utf-8
"""
@author: xiongzhaung
@contact: 1013852341@qq.com
"""
import math
import os
import shutil
from collections import Counter

import numpy as np


def reorg_data(train_dir, valid_dir, valid_ratio):
    idx_label = dict()
    for c in os.listdir(train_dir):
        if c == '.DS_Store':
            continue
        for d in os.listdir(os.path.join(train_dir, c)):
            if d == '.DS_Store':
                continue
            idx_label[d] = c
    min_num_per_class = Counter(idx_label.values()).most_common()[-1][1]
    num_valid_per_class = math.floor(min_num_per_class * valid_ratio)

    for cls in os.listdir(train_dir):
        if cls == '.DS_Store':
            continue
        valid_idx = np.random.choice(
            os.listdir(os.path.join(train_dir, cls)),
            size=num_valid_per_class,
            replace=False)
        if not os.path.exists(os.path.join(valid_dir, cls)):
            os.mkdir(os.path.join(valid_dir, cls))

        for every_idx in valid_idx:
            shutil.move(
                os.path.join(train_dir, cls, every_idx),
                os.path.join(valid_dir, cls, every_idx))
