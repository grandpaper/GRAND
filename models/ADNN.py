#!/usr/bin/env python
# -- coding: utf-8 --
#   @Time : 2022/7/7 13:13 
#   @Author : ksy
#   @File : ADNN.py
#   @Software: PyCharm
#   @Descriptions:
import torch
import torch.nn as nn
import models.NeuralComponents as com


class ADNN(nn.Module):
    """GRAND model"""

    def __init__(self,device):
        super(ADNN, self).__init__()
        self.NS = com.NSEncoder().to(device)
        self.ND = com.NDEncoder().to(device)
        self.Enc = com.Encoder(4).to(device)
        self.Dec = com.Decoder().to(device)

    def forward(self, x):
        name = x[:, :, 0]
        nd = x[:, :, :2]
        workload = x[:, :, 2:]
        ns = self.NS(name)
        nd = self.ND(nd)
        mu, log_var = self.Enc(workload, ns, nd)
        out, z, workload = self.Dec(mu, log_var)
        return out, z, workload, mu, log_var
