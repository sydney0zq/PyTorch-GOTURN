#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qiang.zhou <qiang.zhou@yz-gpu029.hogpu.cc>
# Created on 28 15:46

"""Inference utils"""

def inverse_transform(est_bbox, prev_bbox):
    # HARD CODE
    w, h = (prev_bbox[2] - prev_bbox[0])*2, (prev_bbox[3] - prev_bbox[1])*2
    unscaledbb = [est_bbox[0]*w,
                  est_bbox[1]*h,
                  est_bbox[2]*w,
                  est_bbox[3]*h]
    left, top = prev_bbox[0]-(prev_bbox[2]-prev_bbox[0])/2, prev_bbox[1]-(prev_bbox[3]-prev_bbox[1])/2
    return [left+unscaledbb[0], top+unscaledbb[1], left+unscaledbb[2], top+unscaledbb[3]]



if __name__ == "__main__":
    pass

