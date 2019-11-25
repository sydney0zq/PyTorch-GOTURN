#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qiang.zhou <qiang.zhou@yz-gpu029.hogpu.cc>
#
# Distributed under terms of the MIT license.

from __future__ import print_function
import caffe
import sys
import numpy as np

def model_dump(prototxt, modelfn):
    print ("Now checking...")
    net = caffe.Net(prototxt, modelfn, caffe.TEST)
    print (net.params.keys())
    netkeys = net.params.keys()
    for k in netkeys:
        if 'conv' in k:
            print (k)
            for ki in range(len(net.params[k])):
                np.save(k+"_"+str(ki), net.params[k][ki].data[...])
                print ("dumped", k+"_"+str(ki))
        elif 'fc' in k:
            print (k)
            for ki in range(len(net.params[k])):
                np.save(k+"_"+str(ki), net.params[k][ki].data[...])
                print ("dumped", k+"_"+str(ki))
    print("------------------------------------------------")

if __name__ == "__main__":
    prototxt = "tracker.prototxt"
    modelfn = "tracker.caffemodel"
    model_dump(prototxt, modelfn)


