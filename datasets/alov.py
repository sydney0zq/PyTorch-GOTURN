#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qiang.zhou <qiang.zhou@yz-gpu029.hogpu.cc>
# Created on 2018-02-27 14:50

"""ALOV300 Dataset"""


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
from configuration import DATA_CONFIG


class ALOV300:
    """
    ALOV_DATASET
        images/          --> `alov_folder`
            01-Light/    --> `alov_sub_folder`
                01-Light_video00001/
                    frame1
                    ...
                01-Light_video00002/
                    frame1
                    ...
                ...
            ...
        gt/              --> `annotations_folder`
            01-Light/    --> `alov_sub_folder`
                01-Light_video00001.ann     --> `ann` / `annotation_file`
                02-Light_video00002.ann
                ...
            ...
    """

    def __init__(self, alov_vdroot=DATA_CONFIG['alov_vdroot'], alov_adroot=DATA_CONFIG['alov_adroot'], logger=None):
        self.vdroot = alov_vdroot
        self.adroot = alov_adroot
        self.logger = logging if logger is None else logger
        self.videos = self.get_videos()

    def get_videos(self):
        """ Get all categories """
        categories = listdirs(self.vdroot)

        """ Get all video absolute paths """
        all_vdpath = []
        for i_cate in categories:
            i_cate_abspath = osp.join(self.vdroot, i_cate)
            i_cate_videos = listdirs(i_cate_abspath)
            for i_video in i_cate_videos:
                all_vdpath.append(osp.join(i_cate_abspath, i_video))
        all_vdpath = sorted(all_vdpath)
        num_allvd = len(all_vdpath)

        """ Get all anno absolute paths """
        all_adpath = []
        for i_cate in categories:
            i_cate_abspath = osp.join(self.adroot, i_cate)
            i_cate_annos = glob.glob(i_cate_abspath + "/*.ann")
            all_adpath += i_cate_annos
        all_adpath = sorted(all_adpath)
        num_allad = len(all_adpath)

        """ Check video amount """
        assert (num_allvd == num_allad), "Number of videos and annos should be the same"
        self.logger.info("ALOV300 has {} videos...".format(num_allvd))

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
        #print (i_vdpath, i_adpath)
        """ Get all frames absolute paths """
        all_frames = sorted(os.listdir(i_vdpath))

        """ Get all available anno and frames """
        filt_frames = []
        filt_annos = []
        with open(i_adpath, "r") as f:
            anno = f.read().rstrip().split('\n')

        for i_frame_anno in anno:
            frame_num, ax, ay, bx, by, cx, cy, dx, dy = i_frame_anno.split()
            frame_num, ax, ay, bx, by, cx, cy, dx, dy = int(frame_num), float(ax), float(ay), float(bx), float(by), float(cx), float(cy), float(dx), float(dy)
            minx = min(ax, min(bx, min(cx, dx))) - 1
            miny = min(ay, min(by, min(cy, dy))) - 1
            maxx = max(ax, max(bx, max(cx, dx))) - 1
            maxy = max(ay, max(by, max(cy, dy))) - 1

            filt_annos.append([minx, miny, maxx, maxy])
            filt_frames.append(all_frames[frame_num-1])
        return Video(i_vdpath, filt_frames, filt_annos)

    def __getitem__(self, index):
        return self.videos[index]

    def __len__(self):
        return len(self.videos)


if __name__ == "__main__":
    data = ALOV300("data/ALOV300/train/images", "data/ALOV300/train/gt",  None)
    print(data[10])

