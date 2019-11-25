#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 zhou <zhou@hobot.cc>
#
# Distributed under terms of the MIT license.


import torch
import numpy as np

from torch.autograd import Variable
import importlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os.path as osp
from skimage import io
from torchvision import transforms
import os
import sys
import cv2

CURR_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURR_DIR, '..'))
from configuration import MODEL_CONFIG, TRACK_CONFIG
from utils.infer_utils import inverse_transform
from datasets.transforms import CropTargetWContext, Rescale, Normalize, ToTensor
from datasets.alov import ALOV300
from datasets.vot import VOT2015


class Tracker:
    def __init__(self, model, init_image, init_bbox, use_gpu):
        self.model = model
        #self.init_image = CropTargetWContext()({'image': init_image, 'bbox': init_bbox})['image']
        self.init_image = init_image
        self.init_bbox = init_bbox
        self.prev_im = self.init_image
        self.prev_bb = self.init_bbox
        self.use_gpu = use_gpu

    def track(self, curr_im):
        prev_im = self.prev_im
        prev_bb = self.prev_bb
        #prev_im = self.init_image
        prev_im_ = CropTargetWContext()({'image': prev_im, 'bbox': prev_bb})['image']
        curr_im_ = CropTargetWContext()({'image': curr_im, 'bbox': prev_bb})['image']
        dpair = {'prev_im': prev_im_, 'curr_im': curr_im_}
        est_bbox = self.regression(dpair)
        est_bbox = inverse_transform(est_bbox, prev_bb)

        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(prev_im_)
        ax2.imshow(curr_im_)
        #rect = patches.Rectangle((prev_bb[0], prev_bb[1]), prev_bb[2]-prev_bb[0], prev_bb[3]-prev_bb[1], linewidth=2, edgecolor='r', facecolor="none")
        #ax1.add_patch(rect)
        #rect = patches.Rectangle((est_bbox[0], est_bbox[1]), est_bbox[2]-est_bbox[0], est_bbox[3]-est_bbox[1], linewidth=2, edgecolor='r', facecolor="none")
        #ax2.add_patch(rect)
        plt.show()
        """
        self.prev_bb = est_bbox
        self.prev_im = curr_im
        return est_bbox

    def regression(self, dpair):
        model = self.model
        rescale = Rescale(MODEL_CONFIG['input_size'])
        try:
            prev_im, curr_im = rescale({'image': dpair['prev_im']})['image'], rescale({'image': dpair['curr_im']})['image']
        except:
            return np.array([0, 0, 0, 0])
        trans_tomodel = transforms.Compose([Normalize(), ToTensor()])
        dpair = trans_tomodel({'prev_im': prev_im, 'curr_im': curr_im})
        prev_im, curr_im = dpair['prev_im'], dpair['curr_im']

        if self.use_gpu is True:
            prev_im, curr_im = Variable(prev_im.cuda()), Variable(curr_im.cuda())
            model = model.cuda()
        else:
            prev_im, curr_im = Variable(prev_im), Variable(curr_im)
        prev_im = prev_im[None, :, :, :]
        curr_im = curr_im[None, :, :, :]

        regression_bbox = model(prev_im, curr_im)
        bbox = regression_bbox.data.cpu().numpy()
        bbox = bbox[0, :] / MODEL_CONFIG['bbox_scale']
        return bbox


class TrackerManager:
    def __init__(self, model, videos):
        self.videos = videos
        self.model = model

    def track_video(self, index):
        video = self.videos[index]
        frames = video.frame_abspath_list
        bboxes = video.bboxes_list

        init_image = io.imread(frames[0])
        tracker = Tracker(self.model, init_image, bboxes[0], use_gpu=TRACK_CONFIG['use_gpu'])

        for i, frame in enumerate(frames):
            curr_image = io.imread(frame)
            est_bbox = tracker.track(curr_image)
            print (est_bbox)
            sMatImageDraw = cv2.imread(frame)
            sMatImageDraw = cv2.rectangle(sMatImageDraw, (int(bboxes[i][0]), int(bboxes[i][1])),
                                                         (int(bboxes[i][2]), int(bboxes[i][3])),
                                                         (255, 255, 255), 2)
            sMatImageDraw = cv2.rectangle(sMatImageDraw, (int(est_bbox[0]), int(est_bbox[1])),
                                                         (int(est_bbox[2]), int(est_bbox[3])),
                                                         (255, 0, 0), 2)
            cv2.imshow('Results', sMatImageDraw)
            cv2.waitKey(10)

            if (est_bbox[2]-est_bbox[0])*(est_bbox[3]-est_bbox[1]) > curr_image.shape[0]*curr_image.shape[1]:
                return
            if (est_bbox[2]-est_bbox[0])*(est_bbox[3]-est_bbox[1]) < 10:
                return

    def track_all(self):
        for i in range(len(self.videos)):
            self.track_video(i)


if __name__ == "__main__":
    #alov_videos = ALOV300("/data/qiang.zhou/ALOV300/train/images", "/data/qiang.zhou/ALOV300/train/gt", "./data/alov300.pickle")
    #alov_videos = ALOV300("/data/ALOV300/train/images", "/data/ALOV300/train/gt", "./data/alov300.pickle")
    vot_videos = VOT2015("/data/qiang.zhou/VOT2015")

    # LOAD MODEL #
    model_fn = TRACK_CONFIG['model']
    if TRACK_CONFIG['use_pretrained_model'] is False:
        if os.path.exists(model_fn):
            model = importlib.import_module("models." + MODEL_CONFIG['model_id']).GONET()
            if TRACK_CONFIG['use_gpu']:
                model.load_state_dict(torch.load(model_fn))
            else:
                model.load_state_dict(torch.load(model_fn, map_location=lambda storage, loc: storage))
            model.eval()
        else:
            assert False, "The trained model state file not exists..."
    else:
        if os.path.exists(MODEL_CONFIG['pretrained_model_dir']):
            model = importlib.import_module("models." + MODEL_CONFIG['model_id']).GONET(MODEL_CONFIG['pretrained_model_dir'])
            model.eval()
        else:
            assert False, "The trained model dir not exists..."

    tracker_manager = TrackerManager(model, vot_videos)
    tracker_manager.track_all()










