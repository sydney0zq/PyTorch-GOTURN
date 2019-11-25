#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qiang.zhou <qiang.zhou@yz-gpu029.hogpu.cc>
# Created on 27 14:52

""" Transform Module """

import numpy as np
import cv2
import torch

class Rescale(object):
    """ Rescale bbox to 0~1 and image to (outsz, outsz). And the output image type is uint8. """
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.outsz = output_size

    def __call__(self, image):
        image = cv2.resize(image, (self.outsz, self.outsz), interpolation=cv2.INTER_CUBIC)
        return image

class Normalize(object):
    """ Returns image with zero mean """
    def __call__(self, image):
        image = np.float32(image)
        mean = [104, 117, 123]
        image -= mean
        return image

class ToTensor(object):
    """ Convert ndarrays in sample to Tensors """
    def __call__(self, image):
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # Numpy image: HxWxC
        # Torch image: CxHxW
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        return image


if __name__ == "__main__":
    im = cv2.imread('test.jpg')[..., ::-1]
    bbox = np.array([  21. ,   96.5 , 113. ,  186.5])
    prev_bb = bbox + 90
    s = {'curr_im': im, 'curr_bb': bbox, 'prev_bb': prev_bb}
    t = CropMotionTargetWContext()(s)
    image = t['image']
    bb = t['bbox']
    cv2.rectangle(image, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (255, 0, 0), 2)
    cv2.imwrite('t.jpg', image)




















