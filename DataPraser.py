#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : DataPraser
# @Author: ksy
# @Date  : 2022/7/7:23:43
# @license : Copyright(C), SRLAB-RSE-BUAA
# @Contact : buaaksy@buaa.edu.cn
# @Software : PyCharm
# @Description

import pandas as pd
import torch
import numpy as np
import os
from torch.utils.data import Dataset


class FullDataset(Dataset):
    """map dataset for dataloader，提供完整全部数据"""
    def __init__(self,filelist):
        self.list = filelist

    def __getitem__(self, index):
        file = self.list[index]
        array_data = np.load(file)
        return array_data

    def __len__(self):
        return len(self.list)


class NDDataset(Dataset):
    """
    提供方法名与方法执行时间的数据集。
    """
    def __init__(self,filelist):
        self.list = filelist

    def __getitem__(self, index):
        file = self.list[index]
        array_data = np.load(file)
        return array_data[:,0:1]

    def __len__(self):
        return len(self.list)
