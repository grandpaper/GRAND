# -*- coding:utf-8 -*-
"""
project: traceAD
file: trainer
author: ksy
email: buaaksy@buaa.edu.cn
create date: 2022/7/9 23:20
description: 机器学习训练 控制
"""
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import time
import torch.optim as optim
from torch.utils.data import  DataLoader
from models import Parameters
import DataPraser
from torch.nn.utils.rnn import pad_sequence
from models.ADNN import *
from models.NeuralComponents import *
import pandas as pd
from numpy import *
from tqdm import *
from torch.autograd import grad

class Trainer():

    def __init__(self, data_path, parameters: Parameters.Parameters,device):
        """
        初始化训练参数
        :param data_path: String, 存放npy数据文件的路径
        :param batch_size:
        :param learning_rate:
        """
        self.loss_g_workload = None
        self.loss_g = None
        self.loss_g_ld = None
        self.loss_g_lat = None
        self.loss_g_rec = None
        self.loss_g_rs = None

        self.loss_d_ld = None
        self.loss_d_ld_fake = None
        self.loss_d_ld_real = None

        self.loss_d_lat = None
        self.loss_d_lat_fake = None
        self.loss_d_lat_real = None

        self.loss_d_rec = None
        self.loss_d_rec_fake = None
        self.loss_d_rec_real = None

        self.abnormal_file = [740, 878, 779, 821, 153, 648, 552, 846, 633, 49, 378, 311, 987, 978, 511, 15, 247, 593, 194, 177, 562, 169, 884, 305, 390, 568, 886, 837, 508, 135, 460, 900, 12, 892, 210, 728, 94, 693, 976, 98, 90, 657, 645, 915, 171, 85, 179, 849, 938, 277, 953, 462, 888, 651, 590, 573, 709, 555, 890, 349, 804, 534, 781, 585, 581, 559, 474, 155, 858, 264, 285, 113, 308, 566, 70, 175, 393, 825, 744, 336, 424, 746, 141, 120, 506, 765, 283, 880, 677, 192, 541, 748, 873, 341, 966, 730, 406, 73, 947, 665, 40, 326, 181, 579, 970, 865, 742, 163, 102, 319, 313, 753, 57, 871, 133, 696, 545, 905, 671, 564, 980, 863, 807, 576, 494, 733, 735, 919, 361, 83, 515, 196, 30, 332, 388, 869, 956, 105, 147, 547, 679, 115, 126, 122, 595, 365, 212, 930, 173, 796, 256, 861, 186, 54, 184, 51, 732, 736, 260, 895, 159, 215, 631, 316, 33, 167, 625, 530, 358, 798, 151, 699, 24, 600, 758, 442, 830, 165, 613, 722, 882, 526, 454, 738, 549, 338, 951]


        self.p = parameters
        self.data_path = data_path
        self.data_file_list = self.findDataFile()
        self.batch_size = self.p.get_batch_size()
        self.G_learning_rate = self.p.get_G_lr()
        self.D_learning_rate = self.p.get_D_lr()
        self.ttration = self.p.get_ttration()
        self.train_dataloader, self.test_dataloader = self.createDataLoader()
        self.device = device

        # 定义各种网络
        self.model = ADNN(device).to(device)  # 初始化了深度学习模型（需要update的也是这个里边的参数） 主模型G
        self.model.apply(weights_init)
        self.D_rec = Discriminator_Rec().to(device)
        self.D_rec.apply(weights_init)
        self.D_latent = Discriminator_Latent().to(device)
        self.D_latent.apply(weights_init)
        self.D_workload = Discriminator_workload().to(device)
        self.D_workload.apply(weights_init)

        # 定义各种损失值字典
        self.loss = {
            "D_rec_loss": [],  # 联合的损失函数；根据以下原理，可使用二分类误差，real的时候target设置为1，fake的时候target设置为0
            "D_rec_real": [],  # 更新D_work_load: 使用real input的loss, D 网络应该给real input 打高分
            "D_rec_fake": [],  # 更新D_work_load: 使用fake input的loss, D 网络应该给fake input 打低分
            "D_latent_loss": [],  # 联合的损失函数；根据以下原理，可使用二分类误差，real的时候target设置为1，fake的时候target设置为0
            "D_latent_real": [],  # 更新D_work_load: 使用real input的loss, D 网络应该给real input 打高分
            "D_latent_fake": [],  # 更新D_work_load: 使用fake input的loss, D 网络应该给fake input 打低分
            "D_workload_loss": [],  # 联合的损失函数；根据以下原理，可使用二分类误差，real的时候target设置为1，fake的时候target设置为0
            "D_workload_real": [],  # 更新D_work_load: 使用real input的loss, D 网络应该给real input 打高分
            "D_workload_fake": [],  # 更新D_work_load: 使用fake input的loss, D 网络应该给fake input 打低分
            "G_loss": [],  # 生成器minimax的结果
            "G_loss_rs": [],  # 真正的重构误差
            "G_loss_rec_workload":[],
            "G_loss_rec": [],  # G试图骗过D_rec的误差
            "G_loss_latent": [],  # G试图骗过D_latent的误差
            "G_loss_workload": []  # G试图骗过D_workload的误差
        }

        self.p_z = None # VAE中间的数据分布
        self.input = None
        self.workload = None
        self.real_label = 1.0
        self.fake_label = 0.0
        #
        # output of discriminator_rec
        self.out_d_rec_real = None
        self.feat_rec_real = None
        self.out_d_rec_fake = None
        self.feat_rec_fake = None

        # output of discriminator_lat
        self.out_d_lat_real = None
        self.feat_lat_real = None
        self.out_d_lat_fake = None
        self.feat_lat_fake = None

        # output of discriminator_workload
        self.out_d_ld_real = None
        self.feat_ld_real = None
        self.out_d_ld_fake = None
        self.feat_ld_fake = None

        # output of generator
        self.mu = None
        self.log_var = None
        self.out_g_fake = None
        self.latent_z = None
        self.workload_fake = None



        # 定义损失函数
        self.l1loss = nn.L1Loss()
        self.mse_criterion = nn.MSELoss()
        self.bce_criterion = nn.BCELoss()

        # 设置参数优化器
        self.optimizer_D_rec = optim.Adam(self.D_rec.parameters(), lr=self.D_learning_rate)
        self.optimizer_D_latent = optim.Adam(self.D_latent.parameters(), lr=self.D_learning_rate)
        self.optimizer_D_workload = optim.Adam(self.D_latent.parameters(), lr=self.D_learning_rate)
        self.optimizer_G = optim.Adam(self.model.parameters(), lr=self.G_learning_rate)


    def findDataFile(self):
        """
        在npy数据路径获取所有的数据文件路径，生成list返回，帮助dataset加载
        :return:
        """
        file_list = []
        for root, dir, files in os.walk(self.data_path):
            for file in files:
                if file.find(".npy") != -1:
                    id = file.split("-")[0]
                    if id not in self.abnormal_file:
                        fullpath = os.path.join(root, file)
                        file_list.append(fullpath)
        return file_list

    def collate_fn(self, data):
        """Dataloader 把数据合并为batch 的过程"""
        data.sort(key=lambda x: len(x), reverse=True)
        data_length = [len(sq) for sq in data]
        x = [torch.from_numpy(i) for i in data]
        data = pad_sequence(x, batch_first=True, padding_value=0)
        return data, data_length

    def createDataLoader(self):
        """生成测试集和验证集datalaoder"""
        train_file_list = []
        test_file_list = []
        for index, file_path in enumerate(self.data_file_list):
            if index % self.ttration == 0:
                test_file_list.append(file_path)
            else:
                train_file_list.append(file_path)
        train_dataset = DataPraser.FullDataset(train_file_list)
        test_dataset = DataPraser.FullDataset(test_file_list)
        train_dataloader = DataLoader(train_dataset, self.batch_size, shuffle=False, collate_fn=self.collate_fn)
        test_dataloader = DataLoader(test_dataset, self.batch_size, shuffle=False, collate_fn=self.collate_fn)
        return train_dataloader, test_dataloader

    def yieldIndex(self, max_index, length=1000, step=1):
        """
        使用yield方法，每次生成指定长度的index
        :param step: 滑动步长
        :param max_index: end的最大长度
        :param length: 时间窗口的长度
        :return: start, end
        """
        start = 0
        end = start + length
        while end <= max_index:
            res = [start, end]
            start = start + step
            end = start + length
            yield res

    def train(self):
        """
        训练过程
        控制训练多少个epoch,利用参数p中的num_epoch
        :return:
        """
        glosslist = []
        rslosslist = []
        vloss = []
        wloss = []
        self.train_hist = {}
        self.train_hist['per_epoch_time'] = []
        for i in trange(self.p.num_epoch,desc ='EPOCH Progress:'):
            loss,rsloss = self.train_epoch()
            self.save(i)
            val_loss, workload_loss = self.validate()
            glosslist.append(loss)
            rslosslist.append(rsloss)
            vloss.append(val_loss)
            wloss.append(workload_loss)
            self.save_loss_dict(i)
            print("current epoch {} of {}, train loss {}, test loss {}, workload rec loss {}".format(i, self.p.num_epoch,loss,val_loss,workload_loss))
        f = open("epochloss.txt","w")
        f.writelines(str(glosslist))
        f.close()
        f = open("trainrsloss.txt","w")
        f.writelines(str(rslosslist))
        f.close()
        f = open("testloss.txt","w")
        f.writelines(str(vloss))
        f.close()
        f = open("workloadtestloss.txt","w")
        f.writelines(str(wloss))
        f.close()
        lossdf = pd.DataFrame.from_dict(self.loss)
        lossdf.to_csv("loss_total.csv")

    def save(self,num):
        save_dir = os.path.join("./model/",str(num))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.model.state_dict(),os.path.join(save_dir,"_G.pkl"))
        torch.save(self.D_rec.state_dict(),os.path.join(save_dir,"_D_rec.pkl"))
        torch.save(self.D_latent.state_dict(), os.path.join(save_dir, "_D_latent.pkl"))
        torch.save(self.D_workload.state_dict(), os.path.join(save_dir, "_D_workload.pkl"))

    def train_epoch(self):
        """
        完成一个epoch的训练
        :return:
        """
        epoch_start_time = time.time()
        self.model.train()
        self.D_rec.train()
        self.D_latent.train()
        self.D_workload.train()

        # epoch_iter = 0

        for step, (x, batch_length) in enumerate(tqdm(self.train_dataloader,desc="train dataloader:")):
            # epoch_loss_list = []
            # 调用yieldIndex获取一个滑窗的数据并利用模型进行学习，算作一个step
            for [start, end] in self.yieldIndex(x.shape[1],self.p.get_window_length(),self.p.get_window_step()):
                self.input = x[:, start:end, :].float().to(self.device)
                self.p_z = torch.randn(self.input.size(0), 1000, 2).to(self.device)

                self.optimizing()

                self.save_loss()

        self.printloss()

        self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)

        return self.loss["G_loss"][-1],self.loss["G_loss_rs"][-1]

    def save_loss_dict(self,num):
        lossdf = pd.DataFrame.from_dict(self.loss)
        lossdf.to_csv("loss_epoch_{}.csv".format(num))

    def validate(self):
        """
        在每一个epoch结束的时候，检查在test集合上的loss结果
        :return:
        """
        l1loss = nn.L1Loss()
        mse = nn.MSELoss()
        self.model.eval()
        self.D_rec.eval()
        self.D_latent.eval()
        self.D_workload.eval()

        loss = []
        workloadloss = []
        with torch.no_grad():
            for step, (x, batch_length) in enumerate(tqdm(self.test_dataloader, desc="test dataloader:")):

                for [start, end] in self.yieldIndex(x.shape[1], self.p.get_window_length(), self.p.get_window_step()):

                    input = x[:, start:end, :].float().to(self.device)
                    fake, _, workload_fake, _, _ = self.model(input)
                    # assert torch.isnan(fake).sum() == 0, print(fake)
                    loss.append(l1loss(input[:, :, :2], fake).cpu().numpy())
                    workloadloss.append(l1loss(input[:, :, 2:], workload_fake).cpu().numpy())
            val_loss = np.mean(loss)
            val_wd_loss = np.mean(workloadloss)
        # 返回值 第一个是VAE部分的重构误差，第二个是workload的重构误差部分
        return val_loss, val_wd_loss

    def optimizing(self):
        """
        对网络的模型参数进行优化
        :return:
        """
        # print("update D")
        for i in range(self.p.DUpdateFreq):
            self.update_D_rec()
            self.update_D_latent()
            self.update_D_workload()
        self.update_G()

        # if self.loss_d_rec.item() < 5e-16:
        #     self.reinitialize_netd_rec()
        # if self.loss_d_lat.item() < 5e-16:
        #     self.reinitialize_netd_lat()
        # if self.loss_d_ld.item() < 5e-16:
        #     self.reinitialize_netd_ld()

    def update_G(self):
        """
        对生成器G模型参数进行优化
        :return:
        """
        self.model.zero_grad()
        self.out_g_fake, self.latent_z, self.workload_fake, self.mu, self.log_var = self.model(self.input)

        _, self.feat_rec_real = self.D_rec(self.input)
        _, self.feat_rec_fake = self.D_rec(self.out_g_fake)

        self.latent_z = self.latent_z.permute([0, 2, 1])

        self.p_z = torch.randn_like(self.latent_z).to(self.device)
        _, self.feat_lat_real = self.D_latent(self.p_z)
        _, self.feat_lat_fake = self.D_latent(self.latent_z)

        _, self.feat_ld_real = self.D_workload(self.input)
        _, self.feat_ld_fake = self.D_workload(self.workload_fake)

        self.loss_g_rs = self.l1loss(self.out_g_fake, self.input[:, :, :2])
        self.loss_g_workload = self.l1loss(self.workload_fake,self.input[:,:,2:])

        self.loss_g_rec = -torch.mean(self.feat_rec_fake)
        self.loss_g_lat = -torch.mean(self.feat_lat_fake)
        self.loss_g_ld = -torch.mean(self.feat_ld_fake)

        self.loss_g = self.loss_g_rec + self.loss_g_lat + self.loss_g_ld + self.loss_g_rs + self.loss_g_workload

        self.loss_g.backward()
        nn.utils.clip_grad_norm(self.model.parameters(), 0.1)

        self.optimizer_G.step()


    def printloss(self):
        for k in self.loss.keys():
            print("loss name: {}, values: {}".format(k,self.loss[k][-1]))

    def update_D_rec(self):
        """
        对鉴别器D_rec进行优化
        :return:
        """

        self.D_rec.zero_grad()
        self.out_d_rec_real,self.feat_rec_real = self.D_rec(self.input)

        self.out_g_fake, self.latent_z,self.workload_fake,_, _ = self.model(self.input)

        self.out_d_rec_fake,self.feat_rec_fake = self.D_rec(self.out_g_fake.detach())


        self.loss_d_rec_real = -torch.mean(self.out_d_rec_real)
        self.loss_d_rec_fake = torch.mean(self.out_d_rec_fake)
        # Calculate Gradient penalty
        alpha = torch.rand(self.input.shape[0],1,1).to(self.device)

        x_hat = alpha * self.input[:,:,:2] + (1-alpha)*self.out_g_fake.detach()
        x_hat.requires_grad = True

        pred_hat = self.D_rec(x_hat)[0]

        gradients = grad(outputs=pred_hat,inputs=x_hat,grad_outputs=torch.ones(pred_hat.size()).to(self.device),create_graph=True,retain_graph=True,only_inputs=True)[0]
        gp = self.p.gp_lambda * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()

        self.loss_d_rec = self.loss_d_rec_real + self.loss_d_rec_fake + gp
        self.loss_d_rec.backward()
        self.optimizer_D_rec.step()

    def update_D_latent(self):
        """

        :return:
        """
        self.D_latent.zero_grad()
        self.out_d_lat_real, self.feat_lat_real = self.D_latent(self.p_z)  # 应该在每个step开始的时候把input存进来

        self.out_g_fake, self.latent_z, self.workload_fake, _, _ = self.model(self.input)

        self.latent_z = self.latent_z.permute([0, 2, 1])
        self.p_z = torch.randn_like(self.latent_z).to(self.device)
        self.out_d_lat_fake, self.feat_lat_fake= self.D_latent(self.latent_z.detach())



        self.loss_d_lat_real = -torch.mean(self.out_d_lat_real)
        self.loss_d_lat_fake = torch.mean(self.out_d_lat_fake)

        alpha = torch.rand(self.p_z.shape[0], 1, 1).to(self.device)
        x_hat = alpha * self.p_z + (1 - alpha) * self.latent_z.detach()
        x_hat.requires_grad = True

        pred_hat = self.D_latent(x_hat)[0]

        gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).to(self.device),
                         create_graph=True, retain_graph=True, only_inputs=True)[0]

        gp = self.p.gp_lambda * ((gradients.reshape(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()

        self.loss_d_lat = self.loss_d_lat_real + self.loss_d_lat_fake + gp
        self.loss_d_lat.backward()
        self.optimizer_D_latent.step()

    def update_D_workload(self):
        """

        :return:
        """
        self.D_workload.zero_grad()
        self.out_d_ld_real, self.feat_ld_real = self.D_workload(self.input)  # 应该在每个step开始的时候把input存进来

        self.out_g_fake, self.latent_z, self.workload_fake, _, _ = self.model(self.input)
        self.out_d_ld_fake, self.feat_ld_fake = self.D_workload(self.workload_fake.detach())

        self.loss_d_ld_real = -torch.mean(self.out_d_ld_real)
        self.loss_d_ld_fake = torch.mean(self.out_d_ld_fake)

        alpha = torch.rand(self.input.shape[0], 1, 1).to(self.device)
        x_hat = alpha * self.input[:,:,-4:] + (1 - alpha) * self.workload_fake.detach()
        x_hat.requires_grad = True

        pred_hat = self.D_workload(x_hat)[0]

        gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).to(self.device),
                         create_graph=True, retain_graph=True, only_inputs=True)[0]

        gp = self.p.gp_lambda * ((gradients.reshape(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()

        self.loss_d_ld = self.loss_d_ld_real + self.loss_d_ld_fake + gp

        self.loss_d_ld.backward()
        self.optimizer_D_workload.step()

    def reinitialize_netd_rec(self):
        self.D_rec.apply(weights_init)
        print("重新初始化模块D_rec参数")

    def reinitialize_netd_lat(self):
        self.D_latent.apply(weights_init)
        print("重新初始化模块D_latent参数")

    def reinitialize_netd_ld(self):
        self.D_workload.apply(weights_init)
        print("重新初始化模块D_workload参数")

    def save_loss(self):
        self.loss["G_loss"].append(self.loss_g.item())
        self.loss["G_loss_rs"].append(self.loss_g_rs.item())
        self.loss["G_loss_rec_workload"].append(self.loss_g_workload.item())
        self.loss["G_loss_rec"].append(self.loss_g_rec.item())
        self.loss["G_loss_latent"].append(self.loss_g_lat.item())
        self.loss["G_loss_workload"].append(self.loss_g_ld.item())

        self.loss["D_rec_loss"].append(self.loss_d_rec.item())
        self.loss["D_rec_real"].append(self.loss_d_rec_real.item())
        self.loss["D_rec_fake"].append(self.loss_d_rec_fake.item())

        self.loss["D_latent_loss"].append(self.loss_d_lat.item())
        self.loss["D_latent_real"].append(self.loss_d_lat_real.item())
        self.loss["D_latent_fake"].append(self.loss_d_lat_fake.item())

        self.loss["D_workload_loss"].append(self.loss_d_ld.item())
        self.loss["D_workload_real"].append(self.loss_d_ld_real.item())
        self.loss["D_workload_fake"].append(self.loss_d_ld_fake.item())


if __name__ == "__main__":
    p = Parameters.Parameters()
    d = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainer = Trainer("/ssd/trace/little_npy", p, device=d)

    trainer.train()
