#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qiang.zhou <qiang.zhou@yz-gpu029.hogpu.cc>
# Created on 27 14:52

""" Dataloader For Unified Dataset """

""" Note by cv2.imread, we get a BGR and uint8 image 
    while by io.imread, we get a RGB and 0~1 float image"""


import os.path as osp
import sys
from torch.utils.data import DataLoader
import numpy as np
from torchvision.transforms import Compose
import pickle
import os
import cv2
import time
from multiprocessing.dummy import Pool as ThreadPool
import random

CURR_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURR_DIR, '..'))
from utils.misc_utils import detanno_parser
from configuration import DATASET_PREFIX

class PairData(object):         # For video
    def __init__(self, videos):
        self.build_pair(videos)
        self.pool_im = ThreadPool(2)

    def build_pair(self, videos):
        def build_pair_(videos):
            x, y = [], []
            for i_video in videos:
                i_frames = i_video.frame_abspath_list
                i_bboxes = i_video.bboxes_list
                for i in range(len(i_frames) - 1):
                    x.append([i_frames[i], i_frames[i+1]])
                    y.append([i_bboxes[i], i_bboxes[i+1]])
            return [x, y]
        x, y = build_pair_(videos)
        self.x, self.y = x, y
    
    def __getitem__(self, index):
        x, y = self.x, self.y
        ims = self.pool_im.map(cv2.imread, x[index])
        prev_im, curr_im = ims[0], ims[1]
        prev_bb, curr_bb = y[index][0], y[index][1]
        return prev_im, curr_im, prev_bb, curr_bb

    def __len__(self):
        return len(self.y)


class StaticPairData:           # For images
    def __init__(self, all_impath, all_adpath, det_db):
        self.all_impath = all_impath
        self.all_adpath = all_adpath
        self.det_db = det_db
        self.wrapper()

    def wrapper(self):
        x, y = [], []
        if os.path.exists(self.det_db):
            with open(self.det_db, "rb") as f:
                dbx, dby = pickle.load(f)
            for i_x in dbx:
                x.append(osp.join(DATASET_PREFIX, i_x))
            self.x, self.y = x, dby
        else:
            all_impath = self.all_impath
            all_adpath = self.all_adpath
            for i in range(len(all_impath)):
                i_impath, i_adpath = all_impath[i], all_adpath[i]
                i_anno = detanno_parser(all_adpath[i])
                if len(i_anno) > 0:
                    x.append(i_impath)
                    y.append(i_anno)
            self.x, self.y = x, y
            with open(self.det_db, "wb") as f:
                dbx, dby = [], y
                for i_x in x:
                    dbx.append(i_x.replace(DATASET_PREFIX+"/", ""))
                pickle.dump([dbx, dby], f)
    
    def __getitem__(self, index):
        prev_im = cv2.imread(self.x[index])
        prev_bb = random.choice(self.y[index])
        return prev_im, prev_bb

    def __len__(self):
        return len(self.y)


"""Classical code for advance yield
class LoaderPacker:
    def __init__(self, loaders=[[1, 2], [3, 4]]):
        self.state = 1
        self.loader_iters = [iter(loader) for loader in loaders]

    def __next__(self):
        self.state = self.state ^ 1
        loader_iter = self.loader_iters[self.state]
        yield next(loader_iter)
"""

if __name__ == "__main__":
    from datasets.alov import ALOV300
    from datasets.vid import VID2015
    from datasets.det import DET2014
    from configuration import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, HDFS_CONFIG, DATASET_PREFIX
    alov_videos = ALOV300(DATA_CONFIG['alov_vdroot'], DATA_CONFIG['alov_adroot'],
                          DATA_CONFIG['alov_dbfn'])
    #dataobj = PairData(alov_videos)
    #loader = DataLoader(dataobj, batch_size=10, shuffle=False, num_workers=2)
    det_images = DET2014(DATA_CONFIG['det2014_imroot'], DATA_CONFIG['det2014_adroot'], DATA_CONFIG['det2014_dbfn'])
    det_pair = StaticPairData(det_images.all_impath, det_images.all_adpath, DATA_CONFIG['det_wrapper_dbfn'])
    #det_loader = DataLoader(det_pair, batch_size=10, shuffle=False, num_workers=2)
    #det_videos = VID2015(DATA_CONFIG['vid2014_vdroot'], DATA_CONFIG['vid2014_adroot'], DATA_CONFIG['vid2014_dbfn'])
    #detobj = PairData(det_videos, DATA_CONFIG['vid_wrapper_dbfn'])
    #detlen = len(det_pair)

    #LP = LoaderPacker(loaders=[loader, det_loader])
    #for i in range(len(dataobj)):
    #    print (i)
    #    dpair = dataobj.sampler(i)
    #    print (dpair['prev_im'])




