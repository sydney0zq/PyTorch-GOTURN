#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qiang.zhou <qiang.zhou@yz-gpu029.hogpu.cc>
# Created on 01 14:54

import torch
from torchvision.transforms import Compose
from multiprocessing.dummy import Pool as ThreadPool
import threading
import numpy as np
import math
import sys
import os
CURR_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURR_DIR, ".."))
from datasets.transforms import Rescale, Normalize, ToTensor
from configuration import MODEL_CONFIG, TRAIN_CONFIG

class StepLR:
    def __init__(self, optimizer, stepsize, gamma, logger):
        self.optimizer = optimizer
        self.stepsize = stepsize
        self.gamma = gamma
        self.lr = []
        self.logger = logger

    def step(self, i_iter):
        if i_iter % self.stepsize == 0 and i_iter > 0:
            self.lr = []
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.gamma
                self.lr.append(param_group['lr'])

            self.logger.info("On iter {} LR is set to {}".format(i_iter, self.get_lr()))

    def get_lr(self):
        return self.lr

class BatchLoader:
    def __init__(self, alov_obj, det_obj, exgenerator, netInSize=227, kBatchSize=50, kGenPerImage=10):
        self.alov_obj = alov_obj
        self.det_obj = det_obj
        self.kGenPerImage = kGenPerImage
        self.exgenerator = exgenerator
        self.prev_im = []
        self.curr_im = []
        self.curr_bb = []
        # number of images in each batch
        self.kBatchSize = kBatchSize
        self.batch_x1 = torch.Tensor(self.kBatchSize, 3, netInSize, netInSize)
        self.batch_x2 = torch.Tensor(self.kBatchSize, 3, netInSize, netInSize)
        self.batch_y = torch.Tensor(self.kBatchSize, 4)

        self.pool = ThreadPool(self.kGenPerImage)
        self.pnum_main = 16
        self.pool_main = ThreadPool(self.pnum_main)

    def enqueue(self):
        det_index = np.random.randint(0, len(self.det_obj)-1)
        prev_im, prev_bb = self.det_obj[det_index]
        self.exgenerator.reset(prev_im, prev_im, prev_bb, prev_bb)
        curr_search_reg, target_pad, curr_bb_recenter  = self.exgenerator.make_examples(self.kGenPerImage)
        self.preprocess(curr_search_reg, target_pad, curr_bb_recenter)

        alov_index = np.random.randint(0, len(self.alov_obj)-1)
        prev_im, curr_im, prev_bb, curr_bb = self.alov_obj[alov_index]
        self.exgenerator.reset(prev_im, curr_im, prev_bb, curr_bb)
        curr_search_reg, target_pad, curr_bb_recenter  = self.exgenerator.make_examples(self.kGenPerImage)
        self.preprocess(curr_search_reg, target_pad, curr_bb_recenter)
        
    def preprocess(self, curr_search_reg, target_pad, curr_bb_recenter):
        assert(len(curr_search_reg) == len(target_pad) == len(curr_bb_recenter)), "Length not legal..."
        minibs = len(curr_search_reg)
        def bboxscale(image, bbox):
            h, w = image.shape[0], image.shape[1]
            bbox = np.array([bbox[0]/w, bbox[1]/h, bbox[2]/w, bbox[3]/h])
            bbox *= MODEL_CONFIG['bbox_scale']
            return torch.from_numpy(bbox)
        trans = Compose([Rescale(MODEL_CONFIG['input_size']), 
                         Normalize(),
                         ToTensor()])
        for i in range(minibs):
            self.prev_im.append(trans(target_pad[i]))
            self.curr_im.append(trans(curr_search_reg[i]))
            self.curr_bb.append(bboxscale(curr_search_reg[i], curr_bb_recenter[i]))


    def __call__(self):
        #if len(self.prev_im) < self.kBatchSize:
        #    self.pool_main.map(lambda x: self.enqueue(), 
        #                       [[] for i in range(self.pnum_main)])
        if len(self.prev_im) < self.kBatchSize:
            self.pool_main.map(lambda x: self.enqueue(), 
                               [[] for i in range(self.pnum_main)])
        #if len(self.prev_im) < self.kBatchSize:
        #    self.enqueue()
        #    self.enqueue()

        for i in range(self.kBatchSize):
            self.batch_x1[i, :, :, :] = self.prev_im[i]
            self.batch_x2[i, :, :, :] = self.curr_im[i]
            self.batch_y[i, :] = self.curr_bb[i]
        self.curr_im = self.curr_im[self.kBatchSize:]
        self.prev_im = self.prev_im[self.kBatchSize:]
        self.curr_bb = self.curr_bb[self.kBatchSize:]
        return self.batch_x1, self.batch_x2, self.batch_y


if __name__ == "__main__":
    #from datasets.alov import ALOV300
    #from datasets.det import DET2014
    #from datasets.dataloader import PairData, StaticPairData
    #import time
    #s = time.time()
    #b = BBoxHandler()
    #a = torch.from_numpy(np.ones([50, 4]))
    #print (time.time() - s)
    #print (b.batch_doto_yolo_format(a))
    s = StepLR(None, 1e-6, 100000, 0.1)
    for i in range(500000):
        s.step(i)





