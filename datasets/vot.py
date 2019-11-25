#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qiang.zhou <qiang.zhou@yz-gpu029.hogpu.cc>
# Created on 2018-03-08 20:05

""" VOT Dataset """


import sys
import os
import os.path as osp
import glob
import pickle
CURR_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURR_DIR, '..'))
from datasets.video import Video
from utils.misc_utils import listdirs
import logging


class VOT2015:

    def __init__(self, vot_vdroot, dbfn=None, logger=None):
        self.vdroot = vot_vdroot
        self.dbfn = dbfn
        if logger is None:
            self.logger = logging
        else:
            self.logger = logger
        self.videos = self.get_videos()


    def get_videos(self):
        logger = self.logger

        """ Get all categories """
        categories = listdirs(self.vdroot)

        """ Get all video absolute paths """
        all_vdpath = []
        for i_cate in categories:
            i_cate_videos = osp.join(self.vdroot, i_cate)
            all_vdpath.append(i_cate_videos)
        all_vdpath = sorted(all_vdpath)
        num_allvd = len(all_vdpath)

        """ Get all anno absolute paths """
        all_adpath = []
        for i_cate in categories:
            i_cate_abspath = osp.join(self.vdroot, i_cate)
            all_adpath.append(osp.join(i_cate_abspath, "groundtruth.txt"))
        all_adpath = sorted(all_adpath)
        num_allad = len(all_adpath)

        """ Check video amount """
        assert (num_allvd == num_allad), "Number of videos and annos should be the same"
        logger.info("VOT benchmark has {} videos...".format(num_allvd))

        """ Get all videos object """
        videos = []

        for i in range(num_allvd):
            i_vdpath, i_adpath = all_vdpath[i], all_adpath[i]
            self.logger.info("Loading video {} -- {} / {}".format(i_vdpath, i+1, num_allvd))
            i_objvideo = self.i_objvideo_loader(i_vdpath, i_adpath)
            #self.logger.info(i_objvideo)
            videos.append(i_objvideo)
        return videos

    def i_objvideo_loader(self, i_vdpath, i_adpath):
        """ Get all frames absolute paths """
        all_frames = sorted(glob.glob(i_vdpath + "/*.jpg"))

        """ Get all available anno and frames """
        with open(i_adpath, "r") as f:
            anno = f.read().rstrip().split('\n')
        all_annos = []
        for i_frame_anno in anno:
            ax, ay, bx, by, cx, cy, dx, dy = i_frame_anno.split(',')
            ax, ay, bx, by, cx, cy, dx, dy = float(ax), float(ay), float(bx), float(by), float(cx), float(cy), float(dx), float(dy)
            minx = min(ax, min(bx, min(cx, dx))) - 1
            miny = min(ay, min(by, min(cy, dy))) - 1
            maxx = max(ax, max(bx, max(cx, dx))) - 1
            maxy = max(ay, max(by, max(cy, dy))) - 1
            all_annos.append([minx, miny, maxx, maxy])
        return Video(i_vdpath, all_frames, all_annos)

    def __getitem__(self, index):
        return self.videos[index]

    def __len__(self):
        return len(self.videos)

if __name__ == "__main__":
    vot_dataset = VOT2015("/data/qiang.zhou/VOT2015", './data/vot2015.pickle')

