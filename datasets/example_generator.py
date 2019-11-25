# Date: Thursday 20 July 2017
# Email: nrupatunga@whodat.com
# Name: Nrupatunga
# Description: generating training examples for training

import cv2
import numpy as np
import math
import sys
import os
import random
CURR_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURR_DIR, ".."))
from utils.misc_utils import ImProcer

def sample_rand_uniform():
    RAND_MAX = 2147483647
    return (random.randint(0, RAND_MAX) + 1) * 1.0 / (RAND_MAX + 2)

def sample_exp_two_sides(lambda_):
    RAND_MAX = 2147483647
    randnum = random.randint(0, RAND_MAX)
    if (randnum % 2) == 0:
        pos_or_neg = 1
    else:
        pos_or_neg = -1
    rand_uniform = sample_rand_uniform()
    return math.log(rand_uniform) / (lambda_ * pos_or_neg)

def shift(image, bbox, bbparam, kContextFactor, LaplaceBool):
    lambda_shift_frac = bbparam['lambda_shift_frac']
    lambda_scale_frac = bbparam['lambda_scale_frac']
    min_scale = bbparam['min_scale']
    max_scale = bbparam['max_scale']
    width, height = bbox[2] - bbox[0], bbox[3]-bbox[1]
    center_x, center_y = bbox[0] + width/2., bbox[1] + height/2.

    kMaxNumTries = 10

    new_width = -1
    num_tries_width = 0

    while ((new_width < 0) or (new_width > image.shape[1] - 1)) and (num_tries_width < kMaxNumTries):
        if LaplaceBool:
            width_scale_factor = max(min_scale, min(max_scale, sample_exp_two_sides(lambda_scale_frac)))
        else:
            rand_num = sample_rand_uniform()
            width_scale_factor = rand_num * (max_scale - min_scale) + min_scale

        new_width = width * (1 + width_scale_factor)
        new_width = max(1.0, min((image.shape[1] - 1), new_width))
        num_tries_width = num_tries_width + 1

    new_height = -1
    num_tries_height = 0
    while ((new_height < 0) or (new_height > image.shape[0] - 1)) and (num_tries_height < kMaxNumTries):
        if LaplaceBool:
            height_scale_factor = max(min_scale, min(max_scale, sample_exp_two_sides(lambda_scale_frac)))
        else:
            rand_num = sample_rand_uniform()
            height_scale_factor = rand_num * (max_scale - min_scale) + min_scale

        new_height = height * (1 + height_scale_factor)
        new_height = max(1.0, min((image.shape[0] - 1), new_height))
        num_tries_height = num_tries_height + 1

    first_time_x = True
    new_center_x = -1
    num_tries_x = 0

    while ((first_time_x or (new_center_x < center_x - width * kContextFactor / 2)
           or (new_center_x > center_x + width * kContextFactor / 2)
           or ((new_center_x - new_width / 2) < 0)
           or ((new_center_x + new_width / 2) > image.shape[1]))
           and (num_tries_x < kMaxNumTries)):

        if LaplaceBool:
            new_x_temp = center_x + width * sample_exp_two_sides(lambda_shift_frac)
        else:
            rand_num = sample_rand_uniform()
            new_x_temp = center_x + rand_num * (2 * new_width) - new_width

        new_center_x = min(image.shape[1] - new_width / 2, max(new_width / 2, new_x_temp))
        first_time_x = False
        num_tries_x = num_tries_x + 1

    first_time_y = True
    new_center_y = -1
    num_tries_y = 0

    while ((first_time_y or (new_center_y < center_y - height * kContextFactor / 2)
           or (new_center_y > center_y + height * kContextFactor / 2)
           or ((new_center_y - new_height / 2) < 0)
           or ((new_center_y + new_height / 2) > image.shape[0]))
           and (num_tries_y < kMaxNumTries)):

        if LaplaceBool:
            new_y_temp = center_y + height * sample_exp_two_sides(lambda_shift_frac)
        else:
            rand_num = sample_rand_uniform()
            new_y_temp = center_y + rand_num * (2 * new_height) - new_height

        new_center_y = min(image.shape[0] - new_height / 2, max(new_height / 2, new_y_temp))
        first_time_y = False
        num_tries_y = num_tries_y + 1

    return [new_center_x - new_width/2., new_center_y - new_height/2.,
            new_center_x + new_width/2., new_center_y + new_height/2.]


class Example_generator(ImProcer):
    def __init__(self, bbparam):
        self.ContextFactor = 2  # HARD CODE
        self.bbparam = bbparam

    def reset(self, prev_im, curr_im, prev_bb, curr_bb):
        target_pad, _, _, _ = self.cropPadImage(prev_bb, prev_im)
        self.curr_im = curr_im
        self.curr_bb = curr_bb
        self.prev_bb = prev_bb
        self.target_pad = target_pad

    def make_true_example(self):
        curr_search_reg, curr_search_loc, edge_spx, edge_spy = self.cropPadImage(self.prev_bb, self.curr_im)
        curr_bb_recenter = self.recenter(self.curr_bb, curr_search_loc, edge_spx, edge_spy)
        return curr_search_reg, self.target_pad, curr_bb_recenter

    def make_motion_example(self):
        curr_im = self.curr_im
        curr_bb = self.curr_bb
        curr_bb_shift = shift(curr_im, curr_bb, self.bbparam, self.ContextFactor, True)
        curr_search_reg, curr_search_loc, edge_spx, edge_spy = self.cropPadImage(curr_bb_shift, curr_im)
        curr_bb_recenter = self.recenter(curr_bb, curr_search_loc, edge_spx, edge_spy)
        return curr_search_reg, self.target_pad, curr_bb_recenter

    def make_examples(self, num):
        curr_im, target_pad, curr_bb = [], [], []
        curr_im_, target_pad_, curr_bb_ = self.make_true_example()
        curr_im.append(curr_im_)
        target_pad.append(target_pad_)
        curr_bb.append(curr_bb_)
        for i in range(num):
            curr_im_, target_pad_, curr_bb_ = self.make_motion_example()
            curr_im.append(curr_im_)
            target_pad.append(target_pad_)
            curr_bb.append(curr_bb_)
        return curr_im, target_pad, curr_bb

if __name__ == "__main__":
    random.seed(800)
    objExampleGen = example_generator({
        'lambda_scale_frac': 15,
        'lambda_shift_frac': 5,
        'min_scale': -0.4,
        'max_scale': 0.4
    })
    #prev_im = cv2.imread('test.jpg')[..., ::-1]
    #prev_bb = np.array([  21. ,   96.5 , 113. ,  186.5])
    prev_im = cv2.imread('sample.jpg')
    prev_bb = [  13.5  , 59.5 , 345.5 , 388.5]
    h, w = prev_im.shape[0], prev_im.shape[1]
    objExampleGen.reset(prev_im, prev_im, prev_bb, prev_bb)
    curr_search_reg, target_pad, curr_search_loc = objExampleGen.make_true_example()
    curr_search_reg, target_pad, curr_search_loc = objExampleGen.make_motion_example()
    print (curr_search_loc)
    x1, y1, x2, y2 = curr_search_loc
    h, w = curr_search_reg.shape[0], curr_search_reg.shape[1]
    print(x1/w*10, y1/h*10, x2/w*10, y2/h*10)

    #import pdb
    #pdb.set_trace()

    #import matplotlib.pyplot as plt
    #import matplotlib.patches as patches

    #fig, (ax1, ax2) = plt.subplots(1, 2)
    #ax1.imshow(target_pad)
    #ax2.imshow(curr_im)
    #curr_bb = np.array([curr_bb[0]/w, curr_bb[1]/h, curr_bb[2]/w, curr_bb[3]/h])
    #curr_bb *= 227

    #bbox_gt_scaled.unscale(rand_search_region)
    #rect1 = patches.Rectangle((prev_bb[0], prev_bb[1]), prev_bb[2] - prev_bb[0], prev_bb[3] - prev_bb[1], linewidth=2,
    #                          edgecolor='r', facecolor="none")
    #ax1.add_patch(rect1)
    #rect = patches.Rectangle((curr_bb[0], curr_bb[1]), curr_bb[2] - curr_bb[0], curr_bb[3] - curr_bb[1], linewidth=2,
    #                         edgecolor='r', facecolor="none")
    #ax2.add_patch(rect)
    #print (curr_bb)
    #plt.savefig("test_crop.jpg")
