#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qiang.zhou <qiang.zhou@yz-gpu029.hogpu.cc>
# Created on 27 15:36

"""Miscellaneous Utilities"""

import json
import os
import numpy as np
import re
import sys
import os.path as osp
import xml.etree.ElementTree as ET
import math

def get_center(x):
    return (x - 1.) / 2

def mkdir_p(path):
    """mimic the behavior of mkdir -p in bash"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python > 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def listdirs(dir_root):
    return [i_dir for i_dir in os.listdir(dir_root) if os.path.isdir(os.path.join(dir_root, i_dir))]

def bb_cross2cwh(bbox):
    minx, miny, maxx, maxy = bbox
    return [(minx+maxx)/2., (miny+maxy)/2., maxx-minx, maxy-miny]

def bb_cross2ltwh(bbox):
    minx, miny, maxx, maxy = bbox
    return [minx, miny, maxx - minx, maxy - miny]

def bb_cwh2cross(bbox):
    cx, cy, w, h = bbox
    return [cx-get_center(w), cy-get_center(h), cx+get_center(w), cy+get_center(y)]


# BBOX ###################################################

def bboxes_aveiou(a, b):
    ave_iou = 0
    assert(len(a) == len(b)), "Two numpy bboxes' length not matches..."
    for i in range(len(a)):
        ave_iou += box_iou(a[i, :], b[i, :])
    ave_iou = ave_iou / len(a)
    return ave_iou


def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)


def box_union(a, b):
    i = box_intersection(a, b)
    u = (a[2]-a[0]) * (a[3]-a[1]) + (b[2]-b[0]) * (b[3]-b[1]) - i
    return u


def box_intersection(a, b):
    # a, b should in (minx, miny, maxx, maxy) format
    w = overlap(a[0], a[2]-a[0], b[0], b[2]-b[0])
    h = overlap(a[1], a[3]-a[1], b[1], b[3]-b[1])
    if w <= 0 or h <= 0:
        return 0
    area = w * h
    return area


def overlap(x1, w1, x2, w2):
    l1 = x1 - w1 / 2
    l2 = x2 - w2 / 2
    left = max(l1, l2)
    r1 = x1 + w1 / 2
    r2 = x2 + w2 / 2
    right = min(r1, r2)
    return right - left

# Image Proc ###############################################

class ImProcer:
    def __init__(self):
        self.ContextFactor = 2  # HARD CODE
    def compute_output_w(self, bbox):
        return max(1.0, self.ContextFactor*(bbox[2]-bbox[0]))
    def compute_output_h(self, bbox):
        return max(1.0, self.ContextFactor*(bbox[3]-bbox[1]))
    def compute_edge_spx(self, bbox):
        output_width = self.compute_output_w(bbox)
        return max(0.0, (output_width/2) - (bbox[0]+bbox[2])/2.)
    def compute_edge_spy(self, bbox):
        output_height = self.compute_output_h(bbox)
        return max(0.0, (output_height/2) - (bbox[1]+bbox[3])/2.)
    def recenter(self, bbox, search_loc, edge_sx, edge_sy):
        minx = bbox[0] - search_loc[0] + edge_sx
        miny = bbox[1] - search_loc[1] + edge_sy
        maxx = bbox[2] - search_loc[0] + edge_sx
        maxy = bbox[3] - search_loc[1] + edge_sy
        return [minx, miny, maxx, maxy]
    def cropPadImage(self, bbox, image):
        pad_image_location = self.computeCropPadImageLocation(bbox, image)
        roi_left = min(pad_image_location[0], (image.shape[1] - 1))
        roi_bottom = min(pad_image_location[1], (image.shape[0] - 1))
        roi_width = min(image.shape[1], max(1.0, math.ceil(pad_image_location[2] - pad_image_location[0])))
        roi_height = min(image.shape[0], max(1.0, math.ceil(pad_image_location[3] - pad_image_location[1])))

        err = 0.000000001  # To take care of floating point arithmetic errors
        cropped_image = image[int(roi_bottom + err):int(roi_bottom + roi_height),
                              int(roi_left + err):int(roi_left + roi_width)]
        output_width = max(math.ceil(self.compute_output_w(bbox)), roi_width)
        output_height = max(math.ceil(self.compute_output_h(bbox)), roi_height)
        if image.ndim > 2:
            output_image = np.zeros((int(output_height), int(output_width), image.shape[2]), dtype=image.dtype)
        else:
            output_image = np.zeros((int(output_height), int(output_width)), dtype=image.dtype)

        edge_spacing_x = min(self.compute_edge_spx(bbox), (image.shape[1] - 1))
        edge_spacing_y = min(self.compute_edge_spy(bbox), (image.shape[0] - 1))

        # rounding should be done to match the width and height
        output_image[int(edge_spacing_y):int(edge_spacing_y) + cropped_image.shape[0],
                     int(edge_spacing_x):int(edge_spacing_x) + cropped_image.shape[1]] = cropped_image
        return output_image, pad_image_location, edge_spacing_x, edge_spacing_y
    def computeCropPadImageLocation(self, bbox, image):
        # Center of the bounding box
        bbox_cx = (bbox[0]+bbox[2]) / 2.
        bbox_cy = (bbox[1]+bbox[3]) / 2.
        image_height = image.shape[0]
        image_width = image.shape[1]

        # Padded output width and height
        output_width = self.compute_output_w(bbox)
        output_height = self.compute_output_h(bbox)
        roi_left = max(0.0, bbox_cx - (output_width / 2.))
        roi_bottom = max(0.0, bbox_cy - (output_height / 2.))

        # Padded roi width
        left_half = min(output_width / 2., bbox_cx)
        right_half = min(output_width / 2., image_width - bbox_cx)
        roi_width = max(1.0, left_half + right_half)

        # Padded roi height
        top_half = min(output_height / 2., bbox_cy)
        bottom_half = min(output_height / 2., image_height - bbox_cy)
        roi_height = max(1.0, top_half + bottom_half)

        # Padded image location in the original image
        return [roi_left, roi_bottom, roi_left + roi_width, roi_bottom + roi_height]


# Filter ###################################################

""" DET2014 annotation XML file parser """
def detanno_parser(i_anno_abspath):
    tree = ET.parse(i_anno_abspath)
    annoroot = tree.getroot()
    imsize = [float(annoroot.find('size').find('width').text),
              float(annoroot.find('size').find('height').text)]
    bboxes = []
    for obj in annoroot.findall('object'):
        xmin = obj.find('bndbox').find('xmin').text
        ymin = obj.find('bndbox').find('ymin').text
        xmax = obj.find('bndbox').find('xmax').text
        ymax = obj.find('bndbox').find('ymax').text
        bbox = [float(xmin), float(ymin), float(xmax), float(ymax)]
        bboxes.append(bbox)
    bboxes = deemster(imsize, bboxes)
    return bboxes


""" Filter those objects which covers at least ratio of the image """
def deemster(imsize, annlist):
    filtered_ann = []
    ratio = 0.66
    for ann in annlist:
        ann_w, ann_h = ann[2]-ann[0], ann[3]-ann[1]
        area_gate = ann_w > 0 and ann_h > 0 and ann_w*ann_h > 0
        if ann_w <= ratio*imsize[0] and ann_h <= ratio*imsize[1] and area_gate:
            filtered_ann.append(ann)
    return filtered_ann


if __name__ == "__main__":
    import numpy as np
    a = np.array([[1,1,2,2], [2, 2, 3, 3]])
    b = np.array([[2,2,3,3], [1,1, 2, 2]])
    print(boxes_aveiou(a, b))


