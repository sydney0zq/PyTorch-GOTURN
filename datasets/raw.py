#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qiang.zhou <qiang.zhou@yz-gpu029.hogpu.cc>
#
# Distributed under terms of the MIT license.

"""
Load C++ Raw images and targets.
"""
import numpy as np
import torch
from multiprocessing.dummy import Pool as ThreadPool
import cv2
import os.path as osp


class RawPair:
    def __init__(self, imroot, labelfn, bs=50, insize=227):
        self.imroot = imroot
        self.labelfn = labelfn
        self.bs = bs
        self.pool = ThreadPool(bs)
        self.label = self.parse_label()
        self.insize = insize

    def parse_label(self):
        with open(self.labelfn, "r") as f:
            labelstr = f.read().split('\n')[:-1]
        label = np.zeros((len(labelstr), 4))
        for i, l in enumerate(labelstr):
            label[i] = np.array(l.split(','))
        return label

    def load(self, i_iter):
        insize = self.insize
        batch_x1 = torch.Tensor(self.bs, 3, insize, insize)
        batch_x2 = torch.Tensor(self.bs, 3, insize, insize)
        batch_y  = torch.Tensor(self.bs, 4)
        imageslist = ["images_iter_{:06d}_batch_{:02d}.jpg".format(i_iter, x) for x in range(50)]
        targetslist = ["targets_iter_{:06d}_batch_{:02d}.jpg".format(i_iter, x) for x in range(50)]
        #targets = self.pool.map(cv2.imread, [osp.join(self.imroot, targetslist[i]) for i in range(self.bs)])
        #images = self.pool.map(cv2.imread, [osp.join(self.imroot, imageslist[i]) for i in range(self.bs)])
        for i in range(self.bs):
            target = cv2.imread(osp.join(self.imroot, targetslist[i])).transpose((2, 0, 1))
            image = cv2.imread(osp.join(self.imroot, imageslist[i])).transpose((2, 0, 1))
            batch_x1[i, :, :, :] = torch.from_numpy(target)
            batch_x2[i, :, :, :] = torch.from_numpy(image)
            batch_y[i]  = torch.from_numpy(self.label[i_iter*self.bs + i])
            #import pdb
            #pdb.set_trace()
        return batch_x1, batch_x2, batch_y

    def __len__(self):
        return int(self.label.shape[0] / self.bs)
        

if __name__ == "__main__":
    r = RawPair("/data/qiang.zhou/RAWDET2014", "/data/qiang.zhou/RAWDET2014/label.txt")
    print (len(r))









































