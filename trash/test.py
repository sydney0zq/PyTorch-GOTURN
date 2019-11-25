#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qiang.zhou <qiang.zhou@yz-gpu029.hogpu.cc>
#
# Distributed under terms of the MIT license.

"""

"""
from multiprocessing import Process, Queue
import os, time, random

# 写数据进程执行的代码:
def write(q1, q2):
    while True:
        for value in ['A', 'B', 'C', 'D', 'E', 'F']:
            try:
                q1.put_nowait(value + '1')
                q2.put_nowait(value + '2')
                print ('Put %s to queue1 and queue2...' % value)
            except:
                return
    print ("write end")

# 读数据进程执行的代码:
def read(q1, q2, pw):
    time.sleep(2)
    while True:
        try:
            value1 = q1.get_nowait()
            value2 = q2.get_nowait()
            print ("get ", value1, value2)
        except:
            pw.start()


    

if __name__=='__main__':
    # 父进程创建Queue，并传给各个子进程：
    q1 = Queue(maxsize=4)
    q2 = Queue(maxsize=4)
    pw = Process(target=write, args=(q1,q2))
    #pr = Process(target=read, args=(q1,q2))
    # 启动子进程pw，写入:
    pw.start()
    # 启动子进程pr，读取:
    for i in range(100):
        read(q1, q2, pw)
    pw.join()
    pw.terminate()

    # 等待pw结束:
    pw.join()
    # pr进程里是死循环，无法等待其结束，只能强行终止:
    #pr.terminate()
