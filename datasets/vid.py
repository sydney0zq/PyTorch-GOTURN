#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qiang.zhou <qiang.zhou@yz-gpu029.hogpu.cc>
# Created on 01 16:21

"""VID dataset"""

import logging
import os
import pickle
import sys
import os.path as osp
import glob
import xml.etree.ElementTree as ET

CURR_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURR_DIR, '..'))
from utils.misc_utils import listdirs
from datasets.video import Video
from configuration import DATA_CONFIG

class annoXMLparser(object):
    """ Return bbox from VID XML annotation file """
    def __call__(self, xml_fn):
        annotree = ET.parse(xml_fn)
        self.annoroot = annotree.getroot()
        bbox = self.parser()
        return bbox

    def parser(self):
        bbox_obj = self.get_firstrackid()
        bndbox = bbox_obj.find('bndbox')
        maxx = int(bndbox.find('xmax').text)
        minx = int(bndbox.find('xmin').text)
        maxy = int(bndbox.find('ymax').text)
        miny = int(bndbox.find('ymin').text)
        return [minx, miny, maxx, maxy]

    def get_firstrackid(self):
        annoroot = self.annoroot
        #rand_trackid = random.randint(0, self.get_trackidnum()-1)
        first_bbox = annoroot.find('object')
        return first_bbox
        """
        for i, child in enumerate(annoroot):
            if child.tag == "object":
                #return annoroot[i+rand_trackid]
                print ("rand_trackid: {}".format(i))
                return annoroot[i]
        """

    def get_trackidnum(self):
        i = 0
        for child in self.annoroot:
            if child.tag == "object":
                i += 1
        return i



class VID2015:
    """
    VID_DATASET
        train/          --> `vid_train_folder`
            ILSVRC2015_VID_train_0000/    --> `vid_train_subfolder`
                ILSVRC2015_train_00000000/
                    frame1
                    ...
                ILSVRC2015_train_00000001/
                    frame1
                    ...
                ...
            ...
        train_gt/       --> `annotations_folder`
            ILSVRC2015_VID_train_0000/    --> `vid_sub_annofolder`
                ILSVRC2015_train_00000000/
                    frame1.xml
                    ...
                ILSVRC2015_train_00000001/
                    frame1.xml
                    ...
                ...
            ...
    """
    def __init__(self, vid_vdroot, vid_adroot, dbfn=None, logger=None):
        self.vdroot = vid_vdroot
        self.adroot = vid_adroot
        self.dbfn = dbfn
        if logger is None:
            self.logger = logging
        else:
            self.logger = logger
        self.xmlparser = annoXMLparser()
        self.videos = self.get_videos_proxy()

    def get_videos_proxy(self):
        if os.path.isfile(self.dbfn) is False:
            videos = self.get_videos()
            with open(self.dbfn, "wb") as f:
                pickle.dump(videos, f)
        else:
            self.logger.info("Getting videos from {}".format(self.dbfn))
            with open(self.dbfn, "rb") as f:
                videos = pickle.load(f)
        return videos

    def get_videos(self):
        """ Get all categories """
        categories = listdirs(self.vdroot)
        num_categories = len(categories)

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
            i_cate_annos = os.listdir(i_cate_abspath)
            for i_anno in i_cate_annos:
                all_adpath.append(osp.join(i_cate_abspath, i_anno))
        all_adpath = sorted(all_adpath)
        num_allad = len(all_adpath)

        """ Check video amount """
        assert (num_allvd == num_allad), "Number of videos and annos should be the same"
        self.logger.info("DET2014 has {} videos...".format(num_allvd))

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
        xmlparser = self.xmlparser
        """ Get all frames absolute paths """
        all_frames_path = sorted(os.listdir(i_vdpath))
        all_annos_path = sorted(os.listdir(i_adpath))
        # print (i_vdpath, i_adpath)

        all_annos_abspath = []
        for i_anno in all_annos_path:
            all_annos_abspath.append(osp.join(i_adpath, i_anno))

        """ Get all available anno """
        all_frames_anno = []
        for i_anno_xmlfn in all_annos_abspath:
            # print (i_anno_xmlfn)
            bbox = xmlparser(i_anno_xmlfn)
            # print (bbox)
            all_frames_anno.append(bbox)
        return Video(i_vdpath, all_frames_path, all_frames_anno)

    def __getitem__(self, index):
        return self.videos[index]

    def __len__(self):
        return len(self.videos)


if __name__ == "__main__":
    vidobj = VID2015(DATA_CONFIG['vid2014_vdroot'], DATA_CONFIG['vid2014_adroot'], DATA_CONFIG['vid_dbfn'])
    print (vidobj[0])







