#!/usr/bin/env python
# -- coding: utf-8 --
#   @Time : 2022/7/7 12:51 
#   @Author : ksy
#   @File : NeuralComponents.py
#   @Software: PyCharm
#   @Descriptions:

import torch
import torch.nn as nn
import pandas as pd
import numpy
import logging
logging.basicConfig(level=logging.DEBUG)


def weights_init(mod):
    """
    init model parameter
    """
    classname = mod.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(mod.weight.data)

    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(mod.weight)
        mod.bias.data.fill_(0.01)

class NSEncoder(nn.Module):
    """method name encoder"""

    def __init__(self):
        super(NSEncoder, self).__init__()
        # attention layer
        self.q_projection = nn.Linear(1, 1)
        self.k_projection = nn.Linear(1, 1)
        self.v_projection = nn.Linear(1, 1)
        self.sm = nn.Softmax(dim=1)
        # conv layer
        self.NSEncoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, stride=1, bias=True),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=10, stride=1, bias=True),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=20, stride=1, bias=True),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=30, stride=1, bias=True),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input):
        input = input.unsqueeze(-1)
        q = self.q_projection(input)
        k = self.k_projection(input)
        k_t = k.permute(0, 2, 1)
        v = self.v_projection(input)
        att = self.sm(torch.matmul(q, k_t)/torch.norm(k_t))
        out = torch.matmul(att, v).permute(0, 2, 1)
        out = self.NSEncoder(out).permute(0, 2, 1)
        return out


class NDEncoder(nn.Module):
    """
    duration encoder
    """

    def __init__(self):
        super(NDEncoder, self).__init__()
        self.q_projection = nn.Linear(2, 2)
        self.k_projection = nn.Linear(2, 2)
        self.v_projection = nn.Linear(2, 2)
        self.sm = nn.Softmax(dim=1)

        self.conv_layer = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=2, kernel_size=5, stride=1, bias=True),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(in_channels=2, out_channels=2, kernel_size=10, stride=1, bias=True),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(in_channels=2, out_channels=2, kernel_size=20, stride=1, bias=True),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(in_channels=2, out_channels=1, kernel_size=30, stride=1, bias=True),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input):
        q = self.q_projection(input)
        k = self.k_projection(input)
        k_t = k.permute(0, 2, 1)
        v = self.v_projection(input)
        att = self.sm(torch.matmul(q, k_t)/torch.norm(k_t))
        out = torch.matmul(att, v).permute(0, 2, 1)
        out = self.conv_layer(out).permute(0, 2, 1)
        return out


class Encoder(nn.Module):
    """
    encoder & workload encoder
    """
    def __init__(self, wd_len):
        super(Encoder, self).__init__()
        self.q_projection = nn.Linear(1, 1)
        self.k_projection = nn.Linear(1, 1)
        self.v_projection = nn.Linear(1, 1)
        self.sm = nn.Softmax(dim=1)
        self.fc_mu = nn.Linear(20, 2)
        self.fc_logvar = nn.Linear(20, 2)
        self.workloadEncoder = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5, stride=1, bias=True),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(in_channels=4, out_channels=3, kernel_size=10, stride=1, bias=True),
            nn.BatchNorm1d(3),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(in_channels=3, out_channels=2, kernel_size=20, stride=1, bias=True),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(in_channels=2, out_channels=2, kernel_size=30, stride=1, bias=True),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.mainEncoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=1, bias=True),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=1, bias=True),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.fc_21 = nn.Linear(2,1) #
        self.fc_120 = nn.Linear(1,20)
        self.softplus = nn.Softplus()

    def forward(self, workload, ns, nd):
        workload = workload.permute(0,2,1)
        wd_weight = self.workloadEncoder(workload)
        cat = torch.cat((ns, nd), dim=2)
        att = self.sm(torch.matmul(cat, wd_weight)/torch.norm(wd_weight))
        input = torch.matmul(att, cat)
        input = self.fc_21(input)
        q = self.q_projection(input)
        k = self.k_projection(input)
        v = self.v_projection(input)
        k_t = k.permute(0, 2, 1)
        att = self.sm(torch.matmul(q, k_t)/torch.norm(k_t))
        out = torch.matmul(att, v)
        out = out.permute(0, 2, 1)
        out = self.mainEncoder(out)
        out = out.permute(0, 2, 1)
        out = self.fc_120(out)
        mu = self.fc_mu(out)
        log_var = self.softplus(self.fc_logvar(out))+1e-4  # 1,x,2
        return mu, log_var


class Decoder(nn.Module):
    """
    Decoder
    """
    def __init__(self):
        super(Decoder, self).__init__()

        self.mainDecoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=2, out_channels=1, kernel_size=2, stride=1, bias=True),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=2, stride=1, bias=True),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose1d(in_channels=1, out_channels=2, kernel_size=30, stride=1, bias=True),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose1d(in_channels=2, out_channels=2, kernel_size=20, stride=1, bias=True),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose1d(in_channels=2, out_channels=2, kernel_size=10, stride=1, bias=True),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose1d(in_channels=2, out_channels=2, kernel_size=5, stride=1, bias=True),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.workload = nn.Sequential(
            nn.Linear(2, 4),
            nn.Sigmoid()
        )

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, mu, log_var):
        z = self.reparametrize(mu, log_var)
        z = z.permute(0,2,1)
        out = self.mainDecoder(z)
        out = out.permute(0,2,1)
        workload = self.workload(out)
        return out, z, workload


class Discriminator_Rec(nn.Module):

    def __init__(self):
        super(Discriminator_Rec, self).__init__()

        self.featureExtraction = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=2, kernel_size=5, stride=1, bias=True),
            # nn.BatchNorm1d(2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(in_channels=2, out_channels=2, kernel_size=10, stride=1, bias=True),
            # nn.BatchNorm1d(2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(in_channels=2, out_channels=2, kernel_size=10, stride=1, bias=True),
            # nn.BatchNorm1d(2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(in_channels=2, out_channels=2, kernel_size=10, stride=1, bias=True),
            # nn.BatchNorm1d(2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(in_channels=2, out_channels=2, kernel_size=20, stride=1, bias=True),
            # nn.BatchNorm1d(2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(in_channels=2, out_channels=1, kernel_size=30, stride=1, bias=True),
            # nn.BatchNorm1d(1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Conv1d(1, 1, 4, 1, 0),
            # nn.Sigmoid()
        )

    def forward(self, x):
        nd = x[:, :, :2]
        batch = nd.size(0)
        nd = nd.permute([0,2,1])
        features = self.featureExtraction(nd)
        features = features
        classifier = self.classifier(features)
        classifier = classifier.view(batch, -1, 1).squeeze(2)
        return classifier, features


class Discriminator_Latent(nn.Module):

    def __init__(self):
        super(Discriminator_Latent, self).__init__()

        self.featureExtraction = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=2, kernel_size=4, stride=1, bias=True),
            # nn.BatchNorm1d(2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(in_channels=2, out_channels=2, kernel_size=4, stride=1, bias=True),
            # nn.BatchNorm1d(2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(in_channels=2, out_channels=2, kernel_size=4, stride=1, bias=True),
            # nn.BatchNorm1d(2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(in_channels=2, out_channels=4, kernel_size=4, stride=1, bias=True),
            # nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=4, stride=1, bias=True),
            # nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=4, stride=1, bias=True),
            # nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Conv1d(16, 1, 4, 1, 0)
            # nn.Sigmoid()
        )

    def forward(self, x):
        batch = x.size(0)
        x = x.permute([0,2,1])
        features = self.featureExtraction(x)
        features = features
        classifier = self.classifier(features)
        classifier = classifier.view(batch,-1, 1).squeeze(2)
        return classifier, features


class Discriminator_workload(nn.Module):

    def __init__(self):
        super(Discriminator_workload, self).__init__()

        self.featureExtraction = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=2, kernel_size=2, stride=1, bias=True),
            # nn.BatchNorm1d(2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(in_channels=2, out_channels=2, kernel_size=2, stride=1, bias=True),
            # nn.BatchNorm1d(2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(in_channels=2, out_channels=2, kernel_size=2, stride=1, bias=True),
            # nn.BatchNorm1d(2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(in_channels=2, out_channels=2, kernel_size=2, stride=1, bias=True),
            # nn.BatchNorm1d(2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(in_channels=2, out_channels=4, kernel_size=2, stride=1, bias=True),
            # nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Conv1d(4, 1, 2, 1, 0)
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = x[:,:,-4:]
        batch = x.size(0)
        x = x.permute([0,2,1])
        features = self.featureExtraction(x)
        features = features
        classifier = self.classifier(features)
        classifier = classifier.view(batch, -1, 1).squeeze(2)
        return classifier, features
