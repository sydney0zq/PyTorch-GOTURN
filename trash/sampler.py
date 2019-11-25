#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qiang.zhou <qiang.zhou@yz-gpu029.hogpu.cc>
# Created on 27 14:52

"""Dataset Sampler"""

import numpy as np

class Sampler(object):
    """Our sampler random selects one element from our data source when shuffle=True,
       and it will only iterate our data source only once.
    """
    def __init__(self, data_source, shuffle=True):
        self.data_source = data_source
        self.shuffle = shuffle

    def __iter__(self):
        data_idxs = np.arange(len(self.data_source))
        if self.shuffle:
            np.random.shuffle(data_idxs)

        for idx in data_idxs:
            yield idx

if __name__ == "__main__":
    x = ['a', 'b', 1]
    sampler = Sampler(x)
    p = 0
    for y in sampler:
        print(x[y])
        p += 1
        if p == 10: break




if __name__ == "__main":
    pass

