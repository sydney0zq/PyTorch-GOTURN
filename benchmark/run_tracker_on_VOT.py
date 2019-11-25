#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qiang.zhou <qiang.zhou@yz-gpu029.hogpu.cc>
# Created on 2018-03-04 12:43

import sys
import os
import os.path as osp
import setproctitle
import importlib
import torch
from skimage import io
CURR_DIR=osp.dirname(__file__)
sys.path.append(osp.join(CURR_DIR, ".."))
from benchmark import vot
from logger.logger import setup_logger
from configuration import MODEL_CONFIG, TRACK_CONFIG
from inference.tracker import Tracker

setproctitle.setproctitle('VOT_BENCHMARK_TEST')
logger = setup_logger(logfile=None)

# LOAD MODEL #
model_fn = TRACK_CONFIG['model']
if os.path.exists(model_fn):
    model = importlib.import_module("models." + MODEL_CONFIG['model_id']).GONET(MODEL_CONFIG['pretrained_model_fn'], TRACK_CONFIG['use_gpu'])
    if TRACK_CONFIG['use_gpu']:
        model.load_state_dict(torch.load(model_fn))
    else:
        model.load_state_dict(torch.load(model_fn, map_location=lambda storage, loc: storage))
    model.eval()
else:
    print (model_fn)
    assert False, "The trained model state file not exists..."

# GET INIT IMAGE #
handle = vot.VOT('rectangle')
selection = handle.region()

# PROCESS INIT FRAME #
imgfile = handle.frame()
if not imgfile:
    sys.exit(0)

logger.info(imgfile)

init_left, init_top, init_w, init_h = selection[:]

# Feed minx, miny, maxx, maxy #
init_bbox = [init_left, init_top, init_left+init_w, init_top+init_h]

init_im = io.imread(imgfile)
tracker = Tracker(model, init_im, init_bbox, TRACK_CONFIG['use_gpu'])

while True:
    imgfile = handle.frame()
    logger.info(imgfile)
    if not imgfile:
        handle.quit()
        break
    im = io.imread(imgfile)
    est_bbox = tracker.track(im)
    #logger.info("est_bbox: ", est_bbox)
    w, h = est_bbox[2]-est_bbox[0], est_bbox[3]-est_bbox[1]
    minx, miny = est_bbox[0], est_bbox[1]
    selection = vot.Rectangle(minx, miny, w, h)
    #logger.info(selection)
    handle.report(selection)


