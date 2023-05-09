import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from einops.layers.torch import Rearrange

from config import Args
from model import TAB, TAB_DIFF, TAB_ADD, AT_TAB
from utils import accuracy_cal, cal_seq_len

device = "cuda" if torch.cuda.is_available() else 'cpu'


class Multi_model(nn.Module):
    def __init__(self, arg: Args, num_layers=None, seq_len=None):
        super(Multi_model, self).__init__()
        if num_layers is None:
            num_layers = [3, 3]
        if seq_len is None:
            seq_len = [313, 188, 313]
        arg.dilation = num_layers[0] + num_layers[1]

        self.shareNet = nn.Sequential(
            Prepare(arg),
            TAB_ADD(arg, num_layer=num_layers[0])
        )
        arg.seq_len = seq_len[0]
        self.specialNet1 = nn.Sequential(
            TAB_ADD(arg, num_layer=num_layers[1]),
            Classifier(arg, 0),
        )
        arg.seq_len = seq_len[1]
        self.specialNet2 = nn.Sequential(
            TAB_ADD(arg, num_layer=num_layers[1]),
            Classifier(arg, 1),
        )
        arg.seq_len = seq_len[2]
        self.specialNet3 = nn.Sequential(
            TAB_ADD(arg, num_layer=num_layers[1]),
            Classifier(arg, 2)
        )

    def forward(self, x, y, index):
        x_f = self.shareNet[0](x)
        x_b = self.shareNet[0](torch.flip(x, dims=[-1]))
        x_1, x_f, x_b = self.shareNet[1](x_f, x_b)

        if index == 0:
            x_2, _, _ = self.specialNet1[0](x_f, x_b)
            x = self.specialNet1[1:](torch.cat([x_1, x_2], dim=-1))
            loss = F.cross_entropy(x, y)
            correct_num = accuracy_cal(x, y)
            return loss, correct_num
        elif index == 1:
            x_2, _, _ = self.specialNet2[0](x_f, x_b)
            x = self.specialNet2[1:](torch.cat([x_1, x_2], dim=-1))
            loss = F.cross_entropy(x, y)
            correct_num = accuracy_cal(x, y)
            return loss, correct_num
        elif index == 2:
            x_2, _, _ = self.specialNet3[0](x_f, x_b)
            x = self.specialNet3[1:](torch.cat([x_1, x_2], dim=-1))
            loss = F.cross_entropy(x, y, label_smoothing=0.1)
            correct_num = accuracy_cal(x, y)
            return loss, correct_num


class Prepare(nn.Module):
    def __init__(self, arg: Args):
        super(Prepare, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=arg.feature_dim, out_channels=arg.filters, kernel_size=1, dilation=1, padding=0),
            nn.BatchNorm1d(arg.filters),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Classifier(nn.Module):
    def __init__(self, arg: Args, i: int):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(  # 用于MODMA 2分类
            Rearrange('N C L H-> N C (L H)'),
            nn.BatchNorm1d(arg.filters),
            nn.AdaptiveAvgPool1d(1),
            Rearrange('N C L -> N (C L)'),
            nn.Linear(in_features=arg.filters, out_features=arg.num_class[i])
        )

    def forward(self, x):
        return self.net(x)


class MModel(nn.Module):
    def __init__(self, arg: Args, index, num_layers=None, seq_len=None):
        super(MModel, self).__init__()
        if num_layers is None:
            num_layers = [3, 5]
        if seq_len is not None:
            arg.seq_len = seq_len
        arg.dilation = num_layers[0] + num_layers[1]
        self.prepare = Prepare(arg)
        self.shareNet = AT_TAB(arg, num_layers, index=0)
        merge = nn.Sequential(
            nn.Linear(arg.dilation, 1),
            # nn.Dropout(0.2)
        )
        self.specialNet = nn.Sequential(
            TAB_DIFF(arg, num_layer=num_layers, index=1),
            merge,
            Classifier(arg, index),
        )

    def get_generalFeature(self, x):
        x_f = self.prepare(x)
        x_b = self.prepare(torch.flip(x, dims=[-1]))
        _, x_f, _ = self.shareNet(x_f, x_b)
        return x_f

    def get_stackFeature(self, x):
        x_f = self.prepare(x)
        x_b = self.prepare(torch.flip(x, dims=[-1]))
        stack, _, _ = self.shareNet(x_f, x_b)
        return stack

    def get_mmd_feature(self, x):
        x_f = self.prepare(x)
        x_b = self.prepare(torch.flip(x, dims=[-1]))
        _, x_f, _ = self.shareNet(x_f, x_b)
        _, x_f = self.specialNet[0](x_f)
        return x_f

    def get_softmax(self, x):
        x_f = self.prepare(x)
        x_b = self.prepare(torch.flip(x, dims=[-1]))
        x_1, x_f, _ = self.shareNet(x_f, x_b)
        x_2, x_f = self.specialNet[0](x_f)
        x = self.specialNet[1](torch.cat([x_1, x_2], dim=-1))
        x = self.specialNet[2](x)
        x = F.softmax(x, dim=1)
        return x

    def forward(self, x, y):
        x_f = self.prepare(x)
        x_b = self.prepare(torch.flip(x, dims=[-1]))
        x_1, x_f, _ = self.shareNet(x_f, x_b)
        x_2, x_f = self.specialNet[0](x_f)
        x = self.specialNet[1](torch.cat([x_1, x_2], dim=-1))
        x = self.specialNet[2](x)
        loss = F.cross_entropy(x, y)
        correct_num = accuracy_cal(x, y)
        return loss, correct_num


class Discriminator(nn.Module):
    def __init__(self, arg):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=arg.filters, out_channels=arg.filters, kernel_size=3, padding="same"),
            nn.BatchNorm1d(arg.filters),
            # nn.MaxPool1d(2),
            nn.ReLU(),
            # nn.Conv1d(in_channels=arg.filters, out_channels=8, kernel_size=3, padding="same"),
            # nn.BatchNorm1d(8),
            # nn.MaxPool1d(2),
            # nn.ReLU(),
            # nn.Conv1d(in_channels=arg.filters, out_channels=arg.filters, kernel_size=3, padding="same"),
            # nn.BatchNorm1d(arg.filters),
            # # nn.MaxPool1d(2),
            # nn.ReLU(),
            Rearrange("N C L -> N (C L)"),
            nn.Linear(arg.seq_len * arg.filters, 100),
            nn.Dropout(0.3),
            nn.Linear(100, 2),
        )

    def forward(self, x, y):
        x = self.net(x)
        loss = F.cross_entropy(x, y)
        correct_num = accuracy_cal(x, y)
        return loss, correct_num


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    Params:
        source: 源域数据（n * len(x))
        target: 目标域数据（m * len(y))
        kernel_mul:
        kernel_num: 取不同高斯核的数量
        fix_sigma: 不同高斯核的sigma值
    Return:
        sum(kernel_val): 多个核矩阵之和
    """
    n_samples = int(source.size()[0]) + int(target.size()[0])  # 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0)  # 将source,target按列方向合并
    # 将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    L2_distance = ((total0 - total1) ** 2).sum(2)
    # 调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    # 高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    # 得到最终的核矩阵
    return sum(kernel_val)  # /len(kernel_val)


def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    计算源域数据和目标域数据的MMD距离
    Params:
        source: 源域数据（n * len(x))
        target: 目标域数据（m * len(y))
        kernel_mul:
        kernel_num: 取不同高斯核的数量
        fix_sigma: 不同高斯核的sigma值
    Return:
        loss: MMD loss
    """
    batch_size = int(source.size()[0])  # 一般默认为源域和目标域的batch size相同
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss  # 因为一般都是n==m，所以L矩阵一般不加入计算


# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.

    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'

    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze().to(device)
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze().to(device)

        u = torch.zeros_like(mu).to(device)
        v = torch.zeros_like(nu).to(device)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        """Modified cost for logarithmic updates"""
        r"""$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"""
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        """Returns the matrix of $|x_i-y_j|^p$."""
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        """Barycenter subroutine, used by kinetic acceleration through extrapolation."""
        return tau * u + (1 - tau) * u1
