# -*- coding:utf-8 -*-
"""
project: traceAD
file: Parameters
author: ksy
email: buaaksy@buaa.edu.cn
create date: 2022/7/10 20:35
description:
"""


class Parameters:

    def __init__(self):
        self._batch_size = 300
        self._G_learning_rate = 0.00005
        self._D_learning_rate = 0.0001
        self._ttration = 3
        # 滑动窗口参数
        self._window_length = 1000
        self._window_step = 5000

        self.weight_rs = 0.25
        self.weight_rec = 0.25
        self.weight_lat = 0.25
        self.weight_ld = 0.25

        # WGAP-GP LAMBDAA
        self.gp_lambda = 10

        self.DUpdateFreq = 5

        self.num_epoch = 50

    def get_batch_size(self):
        return self._batch_size

    def set_batch_size(self, batchsize):
        self._batch_size = batchsize

    def get_G_lr(self):
        return self._G_learning_rate

    def get_D_lr(self):
        return self._D_learning_rate

    def set_G_lr(self, lr):
        self._G_learning_rate = lr

    def set_D_lr(self, lr):
        self._D_learning_rate = lr

    def get_ttration(self):
        return self._ttration

    def set_ttration(self,ttration):
        self._ttration = ttration

    def get_window_length(self):
        return self._window_length

    def set_window_length(self,length):
        self._window_length = length

    def get_window_step(self):
        return self._window_step

    def set_window_step(self,step):
        self._window_step = step