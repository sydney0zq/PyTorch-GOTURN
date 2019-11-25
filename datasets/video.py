#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qiang.zhou <qiang.zhou@yz-gpu029.hogpu.cc>
# Created on 27 16:10

"""Video Object"""

import os

class Video:
    """video_path"""
    def __init__(self, video_abspath, frame_list, bboxes_list):
        self.video_abspath = video_abspath
        self.frame_list = frame_list
        self.bboxes_list = bboxes_list
        self.frame_abspath_list = self.get_frame_abspath_list()
        assert(len(frame_list) == len(bboxes_list)), "Frame number should be same with bboxes number"

    def get_frame_abspath_list(self):
        frame_abspath_list = []
        for i in self.frame_list:
            frame_abspath_list.append(os.path.join(self.video_abspath, i))
        return frame_abspath_list

    def __len__(self):
        return len(self.frame_list)

    def __str__(self):
        videoinfo = "Video Path: {}\nVideo Len: {}".format(self.video_abspath, self.__len__())
        framesinfo = "Frames Absolute Path: {}".format(self.frame_abspath_list)
        bboxinfo = "Bbox Info: {}".format(self.bboxes_list)
        return "{}\n\n{}\n\n{}".format(videoinfo, framesinfo, bboxinfo)


if __name__ == "__main__":
    pass

