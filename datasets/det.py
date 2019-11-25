#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qiang.zhou <qiang.zhou@yz-gpu029.hogpu.cc>
# Created on 28 18:08

"""DET 2014 dataset"""

import sys
import os
import glob
import pickle
import logging
import os.path as osp

CURR_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURR_DIR, ".."))
from utils.misc_utils import listdirs
from configuration import DATA_CONFIG

class DET2014:
    def __init__(self, det_imroot=DATA_CONFIG['det2014_imroot'], det_adroot=DATA_CONFIG['det2014_adroot'], logger=None):
        self.det_imroot = det_imroot
        self.det_adroot = det_adroot
        self.logger = logging if logger is None else logger
        self.get_images()

    def get_images(self):
        """ Get all categories """
        categories = listdirs(self.det_imroot)
        num_categories = len(categories)

        """ Get all image absolute paths """
        all_impath = []
        for i, i_cate in enumerate(categories):
            if i % 100 == 0:
                self.logger.info('Get image abspath list {}/{}'.format(i+1, num_categories))
            i_cate_abspath = osp.join(self.det_imroot, i_cate)
            i_cate_ims = os.listdir(i_cate_abspath)
            for i_im in i_cate_ims:
                all_impath.append(osp.join(i_cate_abspath, i_im))
        all_impath = sorted(all_impath)
        num_allim = len(all_impath)

        """ Get all anno absolute paths """
        all_adpath = []
        for i, i_cate in enumerate(categories):
            if i % 100 == 0:
                self.logger.info('Get annotation abspath list {}/{}'.format(i+1, num_categories))
            i_cate_abspath = osp.join(self.det_adroot, i_cate)
            i_cate_annos = os.listdir(i_cate_abspath)
            for i_anno in i_cate_annos:
                all_adpath.append(osp.join(i_cate_abspath, i_anno))
        all_adpath = sorted(all_adpath)
        num_allad = len(all_adpath)

        """ Check dataset integrity """
        assert (num_allim == num_allad), "Number of image and anno should be the same"
        self.all_impath = all_impath
        self.all_adpath = all_adpath


if __name__ == "__main__":
    det_obj = DET2014(DATA_CONFIG['det2014_imroot'], DATA_CONFIG['det2014_adroot'])

