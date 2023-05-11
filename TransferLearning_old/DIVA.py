import torch.nn as nn
import torch
import torch.nn.functional as F
from TransferLearning.Multitrain_ae import ConvBlock
from einops.layers.torch import Rearrange
import torch.distributions as dist


class px(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(px, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(zd_dim + zx_dim + zy_dim, 1024, bias=False), nn.BatchNorm1d(1024), nn.ReLU())
        self.up1 = nn.Upsample(8)
        self.de1 = nn.Sequential(nn.ConvTranspose2d(64, 128, kernel_size=5, stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(128), nn.ReLU())
        self.up2 = nn.Upsample(24)
        self.de2 = nn.Sequential(nn.ConvTranspose2d(128, 256, kernel_size=5, stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(256), nn.ReLU())
        self.de3 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1))

        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        torch.nn.init.xavier_uniform_(self.de1[0].weight)
        torch.nn.init.xavier_uniform_(self.de2[0].weight)
        torch.nn.init.xavier_uniform_(self.de3[0].weight)
        self.de3[0].bias.data.zero_()

    def forward(self, zd, zx, zy):
        if zx is None:
            zdzxzy = torch.cat((zd, zy), dim=-1)
        else:
            zdzxzy = torch.cat((zd, zx, zy), dim=-1)
        h = self.fc1(zdzxzy)
        h = h.view(-1, 64, 4, 4)
        h = self.up1(h)
        h = self.de1(h)
        h = self.up2(h)
        h = self.de2(h)
        loc_img = self.de3(h)

        return loc_img


class pzd(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(pzd, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(d_dim, zd_dim, bias=False), nn.BatchNorm1d(zd_dim), nn.ReLU())
        self.fc21 = nn.Sequential(nn.Linear(zd_dim, zd_dim))
        self.fc22 = nn.Sequential(nn.Linear(zd_dim, zd_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        torch.nn.init.xavier_uniform_(self.fc21[0].weight)
        self.fc21[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc22[0].weight)
        self.fc22[0].bias.data.zero_()

    def forward(self, d):
        hidden = self.fc1(d)
        zd_loc = self.fc21(hidden)
        zd_scale = self.fc22(hidden) + 1e-7

        return zd_loc, zd_scale


class pzy(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(pzy, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(y_dim, zy_dim, bias=False), nn.BatchNorm1d(zy_dim), nn.ReLU())
        self.fc21 = nn.Sequential(nn.Linear(zy_dim, zy_dim))
        self.fc22 = nn.Sequential(nn.Linear(zy_dim, zy_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        torch.nn.init.xavier_uniform_(self.fc21[0].weight)
        self.fc21[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc22[0].weight)
        self.fc22[0].bias.data.zero_()

    def forward(self, y):
        hidden = self.fc1(y)
        zy_loc = self.fc21(hidden)
        zy_scale = self.fc22(hidden) + 1e-7

        return zy_loc, zy_scale


class qzd(nn.Module):
    def __init__(self, f_d, zd_dim):
        super(qzd, self).__init__()
        conv1 = ConvBlock(f_d, 64, kernel_size=3, pool_size=4)
        conv2 = ConvBlock(64, 128, kernel_size=3, pool_size=2)
        conv3 = ConvBlock(128, 256, kernel_size=3, pool_size=3)
        conv4 = ConvBlock(256, 512, kernel_size=3, pool_size=3)
        conv5 = ConvBlock(512, 1024, kernel_size=3, pool_size=4)
        self.encoder = nn.Sequential(
            conv1, conv2, conv3, conv4, conv5, Rearrange("N C L -> N (C L)"),
        )
        self.fc21 = nn.Sequential(
            nn.Linear(1024, zd_dim)
        )
        self.fc22 = nn.Sequential(
            nn.Linear(1024, zd_dim),
            nn.Softplus()
        )
        torch.nn.init.xavier_uniform_(self.fc21[0].weight)
        self.fc21[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc22[0].weight)
        self.fc22[0].bias.data.zero_()

    def forward(self, x):
        x = self.encoder(x)
        zd_loc = self.fc21(x)
        zd_scale = self.fc22(x) + 1e-7

        return zd_loc, zd_scale


class qzx(nn.Module):  # 编码剩余变量
    def __init__(self, f_d, zx_dim):
        super(qzx, self).__init__()
        conv1 = ConvBlock(f_d, 64, kernel_size=3, pool_size=4)
        conv2 = ConvBlock(64, 128, kernel_size=3, pool_size=2)
        conv3 = ConvBlock(128, 256, kernel_size=3, pool_size=3)
        conv4 = ConvBlock(256, 512, kernel_size=3, pool_size=3)
        conv5 = ConvBlock(512, 1024, kernel_size=3, pool_size=4)
        self.encoder = nn.Sequential(
            conv1, conv2, conv3, conv4, conv5, Rearrange("N C L -> N (C L)"),
        )
        self.fc21 = nn.Sequential(
            nn.Linear(1024, zx_dim)
        )
        self.fc22 = nn.Sequential(
            nn.Linear(1024, zx_dim),
            nn.Softplus()
        )
        torch.nn.init.xavier_uniform_(self.fc21[0].weight)
        self.fc21[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc22[0].weight)
        self.fc22[0].bias.data.zero_()

    def forward(self, x):
        x = self.encoder(x)
        zx_loc = self.fc21(x)
        zx_scale = self.fc22(x) + 1e-7

        return zx_loc, zx_scale


class qzy(nn.Module):  # 编码类别特定
    def __init__(self, f_d, zy_dim):
        super(qzy, self).__init__()

        conv1 = ConvBlock(f_d, 64, kernel_size=3, pool_size=4)
        conv2 = ConvBlock(64, 128, kernel_size=3, pool_size=2)
        conv3 = ConvBlock(128, 256, kernel_size=3, pool_size=3)
        conv4 = ConvBlock(256, 512, kernel_size=3, pool_size=3)
        conv5 = ConvBlock(512, 1024, kernel_size=3, pool_size=4)
        self.encoder = nn.Sequential(
            conv1, conv2, conv3, conv4, conv5, Rearrange("N C L -> N (C L)"),
        )
        self.fc21 = nn.Sequential(
            nn.Linear(1024, zy_dim)
        )
        self.fc22 = nn.Sequential(
            nn.Linear(1024, zy_dim),
            nn.Softplus()
        )
        torch.nn.init.xavier_uniform_(self.fc21[0].weight)
        self.fc21[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc22[0].weight)
        self.fc22[0].bias.data.zero_()

    def forward(self, x):
        x = self.encoder(x)
        zy_loc = self.fc21(x)
        zy_scale = self.fc22(x) + 1e-7
        return zy_loc, zy_scale


class qd(nn.Module):  # 预测域
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qd, self).__init__()

        self.fc1 = nn.Linear(zd_dim, d_dim)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()

    def forward(self, zd):
        h = F.relu(zd)
        loc_d = self.fc1(h)

        return loc_d


class qy(nn.Module):  # 预测类别
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qy, self).__init__()

        self.fc1 = nn.Linear(zy_dim, y_dim)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()

    def forward(self, zy):
        h = F.relu(zy)
        loc_y = self.fc1(h)

        return loc_y


class DIVA(nn.Module):
    def __init__(self, args):
        super(DIVA, self).__init__()
        self.zd_dim = 64
        self.zx_dim = 64
        self.zy_dim = 64
        self.d_dim = 2
        self.x_dim = args.filters
        self.y_dim = 2

        self.start_zx = self.zd_dim
        self.start_zy = self.zd_dim + self.zx_dim

        self.px = px(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        self.pzd = pzd(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        self.pzy = pzy(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)

        self.qzd = qzd(args, self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        if self.zx_dim != 0:
            self.qzx = qzx(args, self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        self.qzy = qzy(args, self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)

        self.qd = qd(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        self.qy = qy(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)

        self.aux_loss_multiplier_y = args.aux_loss_multiplier_y
        self.aux_loss_multiplier_d = args.aux_loss_multiplier_d

        self.beta_d = args.beta_d
        self.beta_x = args.beta_x
        self.beta_y = args.beta_y

        self.cuda()

    def forward(self, d, x, y):
        # Encode
        zd_q_loc, zd_q_scale = self.qzd(x)
        if self.zx_dim != 0:
            zx_q_loc, zx_q_scale = self.qzx(x)
        zy_q_loc, zy_q_scale = self.qzy(x)

        # Reparameterization trick
        qzd = dist.Normal(zd_q_loc, zd_q_scale)
        zd_q = qzd.rsample()  # x的域特定特征表示
        if self.zx_dim != 0:
            qzx = dist.Normal(zx_q_loc, zx_q_scale)
            zx_q = qzx.rsample()  # x的剩余特征表示
        else:
            qzx = None
            zx_q = None

        qzy = dist.Normal(zy_q_loc, zy_q_scale)
        zy_q = qzy.rsample()  # x的类别特定特征表示

        # Decode
        x_recon = self.px(zd_q, zx_q, zy_q)  # 根据三个表示重建x

        zd_p_loc, zd_p_scale = self.pzd(d)  # 提取d（旋转角度）的域特定特征表示

        if self.zx_dim != 0:
            zx_p_loc, zx_p_scale = torch.zeros(zd_p_loc.size()[0], self.zx_dim).cuda(), \
                torch.ones(zd_p_loc.size()[0], self.zx_dim).cuda()
        zy_p_loc, zy_p_scale = self.pzy(y)

        # Reparameterization trick
        pzd = dist.Normal(zd_p_loc, zd_p_scale)
        if self.zx_dim != 0:
            pzx = dist.Normal(zx_p_loc, zx_p_scale)
        else:
            pzx = None
        pzy = dist.Normal(zy_p_loc, zy_p_scale)

        # Auxiliary losses
        d_hat = self.qd(zd_q)  # 根据x的域特定表示来判断域
        y_hat = self.qy(zy_q)  # 根据x的类别特定表示来判断类别

        return x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q

    def loss_function(self, d, x, y=None):

        x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q = self.forward(d, x, y)
        # 重建x，预测域  预测类别 x的域特定表示分布 d的域特定表示分布，x的域特定表示，x的剩余表示分布
        x_recon = x_recon.view(-1, 256)
        x_target = (x.view(-1) * 255).long()
        CE_x = F.cross_entropy(x_recon, x_target, reduction='sum')

        zd_p_minus_zd_q = torch.sum(pzd.log_prob(zd_q) - qzd.log_prob(zd_q))  # 损失函数中KL(q_Φd(z_d|x)||p_θd(z_d|d))
        if self.zx_dim != 0:
            KL_zx = torch.sum(pzx.log_prob(zx_q) - qzx.log_prob(zx_q))  # 损失函数中 KL(q_Φx(z_x|x) || p(z_x))
        else:
            KL_zx = 0

        zy_p_minus_zy_q = torch.sum(pzy.log_prob(zy_q) - qzy.log_prob(zy_q))  # 损失函数中KL(q_Φy(z_y|x)||p_θy(z_y|y))

        _, d_target = d.max(dim=1)
        CE_d = F.cross_entropy(d_hat, d_target, reduction='sum')

        _, y_target = y.max(dim=1)
        CE_y = F.cross_entropy(y_hat, y_target, reduction='sum')
        # 返回损失
        return CE_x - self.beta_d * zd_p_minus_zd_q - self.beta_x * KL_zx - self.beta_y * zy_p_minus_zy_q \
               + self.aux_loss_multiplier_d * CE_d + self.aux_loss_multiplier_y * CE_y, CE_y
