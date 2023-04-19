import datetime
import math
import os

import torchaudio
from transformers import Wav2Vec2Processor

from config import Args
from preprocess.process_utils import MODMA_code
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import dataset
from torchvision import transforms
import random

plt.rcParams['font.sans-serif'] = ['Simhei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

CLASS_LABELS = ["angry", "boredom", "disgust", "fear", "happy", "neutral", "sad"]
IEMOCAP_LABELS = ['angry', 'excited', 'frustrated', 'happy', 'neutral', 'sad']
MODMA_LABELS = ['HC', 'MDD']
RAVDESS_LABELS = ["angry", "calm", "disgust", "fear", "happy", "normal", "sad", "surprised"]
dpi = 300
plt.rcParams['font.family'] = ['SimHei']


class EarlyStopping:
    """Early stops the training if validation accuracy doesn't change after a given patience."""

    def __init__(self, patience=5, use_acc: float = 80.0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved. Default: 5
            use_acc (float): 只有当验证集准确率大于use_acc，才会触发早停止
        """
        self.patience = patience
        self.patience_ = patience
        self.use_acc = use_acc
        self.last_val_acc = 0

    def __call__(self, val_acc) -> bool:
        if (abs(self.last_val_acc - val_acc) < 1e-9) and (val_acc > self.use_acc):
            self.patience -= 1
        else:
            self.patience = self.patience_
        self.last_val_acc = val_acc
        if self.patience == 1:
            print(f"The validation accuracy has not changed in {self.patience_} iterations, stop train")
            print(f"The final validation accuracy is {val_acc}")
            return True
        else:
            return False


def load_dataset(dataset_name, spilt_rate, random_seed, version='V2', order=3, is_cluster=False):
    """
    加载数据集
    """
    if is_cluster:
        data = np.load(f'preprocess/data/{dataset_name}_{version}_order{order}_c_6.npy', allow_pickle=True).item()
    else:
        data = np.load(f'D:/xmj/SER_depression/preprocess/data/{dataset_name}_{version}_order{order}.npy',
                       allow_pickle=True).item()
    x = data['x'][:, :order * 13, :]
    y = data['y']
    Num = x.shape[0]  # 样本数
    dataset = myDataset(x, y)  # input shape of x: [样本数，特征维度，时间步]  input shape of y: [样本数，类别数]
    spilt_num = len(spilt_rate)
    lengths = []
    for i in range(spilt_num):
        lengths.append(int(Num * spilt_rate[i]))
    if spilt_num > 1:
        lengths[-1] = Num - sum(lengths[:-1])

    datasets = torch.utils.data.random_split(dataset, lengths, generator=torch.Generator().manual_seed(random_seed))
    return datasets


def load_loader(dataset_name, spilt_rate, batch_size, random_seed, version='V2', order=3):
    """
    加载数据加载器
    """
    datasets = load_dataset(dataset_name, spilt_rate, random_seed, version, order)
    loaders = []
    for dt in datasets:
        loaders.append(
            torch.utils.data.dataloader.DataLoader(
                dataset=dt,
                batch_size=batch_size,
            ))
    return loaders


def sample_dataset(data_loader, sample_num, batch_size):
    """
    随机采样数据
    """
    sample_index = random.sample(range(len(data_loader.dataset)), sample_num)
    samples = data_loader.dataset[sample_index]
    sample_loader = torch.utils.data.dataloader.DataLoader(
        dataset=myDataset(samples[0], samples[1]),
        batch_size=batch_size
    )
    return sample_loader


def dataset_num_class(dataset_name):
    """
    用在域适应中，返回数据集对应的类别数
    """
    num_class = []
    for name in dataset_name:
        if name == "IEMOCAP":
            num_class.append(6)
        elif name == "MODMA":
            num_class.append(2)
        elif name == "CASIA":
            num_class.append(6)
        elif name == "RAVDESS":
            num_class.append(8)
        elif name == "ABC":
            num_class.append(6)
        elif name == "TESS":
            num_class.append(7)
        elif name == "eNTERFACE":
            num_class.append(6)
        else:
            raise NotImplementedError
    return num_class


def seed_everything(random_seed):
    """确定随机数，以便于复现 但是会很慢
    
    Args:
        random_seed (int): 
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def load_multi_dataset(name, spilt_rate, random_seed, batch_size, out_type=True, length=-1, version="v2", order=3):
    """为多个数据集生成对应的Dataloader

    Args:
        name (str): 数据集名称
        spilt_rate (list): 训练集，验证集，测试集分割比例
        random_seed (int): 随机数种子
        batch_size (int): 批次大小
        length(int): 当length=-1时，选择所有的数据，否则选择length个数据
        out_type(bool): 控制返回哪些loader，True返回训练和验证，false返回测试
        version(str): 决定数据版本
        order(int): 39维，26维，13维MFCC

    Returns:
        Dataloader: 数据加载器
    """

    data = np.load(f"preprocess/data/{name}_{version}_order{order}.npy", allow_pickle=True).item()
    x = data['x']
    y = data['y']
    actual_length = y.shape[0]
    if actual_length > length != -1:
        delete_index = random.sample(range(actual_length), actual_length - length)
        x = np.delete(x, delete_index, axis=0)
        y = np.delete(y, delete_index, axis=0)
    dataset = myDataset(x=x, y=y)
    train_num = int(x.shape[0] * spilt_rate[0])
    val_num = int(x.shape[0] * spilt_rate[1])
    test_num = x.shape[0] - train_num - val_num
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_num, val_num, test_num],
                                                                             generator=torch.Generator().manual_seed(
                                                                                 random_seed))
    if out_type:
        train_loader = torch.utils.data.dataloader.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
        )
        val_loader = torch.utils.data.dataloader.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
        )
        return train_loader, val_loader
    else:
        test_loader = torch.utils.data.dataloader.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
        )
        return test_loader


def load_meta_dataset(name, length=256, version="V2", order=3, return_residual=False, appendix=None):
    """
    加载元学习数据集

    Args:
        name: 数据集的名字
        length: 截取长度
        version: 数据集版本
        order: 数据集阶数
        return_residual: 是否返回剩余的数据集（用作测试数据集）
        appendix: 是否输出数据集标号
    """
    data = np.load(f"../preprocess/data/{name}_{version}_order{order}.npy", allow_pickle=True).item()
    x = data['x']
    y = data['y']
    if y.shape[0] < length:
        length = y.shape[0]
    random_index = np.random.permutation(y.shape[0])
    cut_index = random_index[:length]
    x_cut = x[cut_index]
    y_cut = y[cut_index]
    meta_dataset = myDataset(x=x_cut, y=y_cut, appendix=appendix)
    if return_residual:
        residual_index = random_index[length:]
        x_res = x[residual_index]
        y_res = y[residual_index]
        residual_dataset = myDataset(x=x_res, y=y_res)
        return meta_dataset, residual_dataset
    else:
        return meta_dataset


def accuracy_cal_numpy(y_pred, y_true):
    """
    计算准确率(numpy)
    """
    predict = np.argmax(y_pred.numpy(), 1)
    label = np.argmax(y_true.numpy(), 1)
    true_num = (predict == label).sum()
    return true_num


def accuracy_cal(y_pred, y_true):
    """
    计算准确率(torch)
    """
    predict = torch.max(y_pred.data, 1)[1]  # torch.max()返回[values, indices]，torch.max()[1]返回indices
    if len(y_true.data.shape) == 1:
        label = y_true.data
    else:
        label = torch.max(y_true.data, 1)[1]
    true_num = (predict == label).sum()
    return true_num


def compare_key(src_key: str, tgt_key: str):
    """
    比较两个权重的键，并返回不同的部分（用于加载预训练模型）
    """
    src_key = src_key.split('.')
    tgt_key = tgt_key.split('.')
    mini_len = min(len(src_key), len(tgt_key))
    tmp_src_key = None
    tmp_tgt_key = None
    for i in range(1, mini_len + 1):
        if src_key[-i] != tgt_key[-i]:
            tmp_src_key, tmp_tgt_key = src_key[:len(src_key) - i + 1], tgt_key[:len(tgt_key) - i + 1]
            break
    if isinstance(tmp_src_key, list):
        tmp_src_key = '.'.join(tmp_src_key)
    if isinstance(tmp_tgt_key, list):
        tmp_tgt_key = '.'.join(tmp_tgt_key)
    return tmp_src_key, tmp_tgt_key


def confusion_matrix(pred, labels, conf_matrix):
    """
    更新混淆矩阵
    """
    pred = torch.max(pred, 1)[1]
    labels = torch.max(labels, 1)[1]
    for p, t in zip(pred, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def plot_matrix(cm, labels_name, model_name: str, title='Confusion matrix', normalize=False,
                result_path: str = "results/", best=False):
    """绘制混淆矩阵，保存并返回

    Args:
        cm: 计算出的混淆矩阵的值
        labels_name: 标签名
        model_name: 保存的图片名
        title: 生成的混淆矩阵的标题
        normalize: True:显示百分比, False:显示个数
        result_path: 保存路径
        best: 是否是最优模型的结果

    Returns: 图窗

    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    fig = plt.figure(figsize=(4, 4), dpi=120)
    # 画图，如果希望改变颜色风格，可以改变此部分的cmap=plt.get_cmap('Blues')处
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.colorbar()  # 绘制图例
    # 图像标题
    plt.title(title)
    # 绘制坐标
    num_local = np.array(range(len(labels_name)))
    axis_labels = labels_name
    plt.xticks(num_local, axis_labels, rotation=45)  # 将标签印在x轴坐标上， 并倾斜45度
    plt.yticks(num_local, axis_labels)  # 将标签印在y轴坐标上
    # x,y轴长度一致(问题1解决办法）
    plt.axis("equal")
    # x轴处理一下，如果x轴或者y轴两边有空白的话(问题2解决办法）
    ax = plt.gca()  # 获得当前axis
    left, right = plt.xlim()  # 获得x轴最大最小值
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")
    thresh = cm.max() / 2.
    # 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if normalize:
                plt.text(j, i, format(int(cm[i][j] * 100 + 0.5), 'd') + '%',
                         ha="center", va="center",
                         color="white" if cm[i][j] > thresh else "black")  # 如果要更改颜色风格，需要同时更改此行
            else:
                plt.text(j, i, int(cm[i][j]),
                         ha="center", va="center",
                         color="white" if cm[i][j] > thresh else "black")
    plt.tight_layout()
    # plt.subplots_adjust(left=0.01,bottom=0.1)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    img_path = result_path + "images/" + model_name + "_confusion_matrix_best.jpg" if best else result_path + "images/" + model_name + "_confusion_matrix_final.jpg"
    plt.savefig(img_path, dpi=dpi)
    # 显示
    # plt.show()
    return fig


class myDataset(dataset.Dataset):
    """
    自定义数据集
    """

    def __init__(self, x, y, train=True, appendix=None):
        super(myDataset, self).__init__()
        self.train = train  # 训练和测试时对数据的处理可能不同
        self.appendix = appendix
        # self.transforms = transforms.Compose([
        #     transforms.ToTensor(),
        #     # transforms.Normalize(mean=[0.5], std=[0.5])
        # ])
        self.x = x
        self.y = y

    def __getitem__(self, index):
        data = self.x[index]
        label = self.y[index]
        if self.appendix is not None:
            return data, label, self.appendix
        else:
            return data, label

    def __len__(self):
        return len(self.y)


class myWavLoader(dataset.Dataset):
    """
    直接加载音频原始数据作为输入（wav2vec2, hubert）
    """

    def __init__(self, files, duration=10) -> None:
        super(myWavLoader, self).__init__()
        self.files = files
        self.duration = duration
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

    def __getitem__(self, index):
        data, sr = torchaudio.load(self.files[index])
        label = MODMA_code(self.files[index].split('/')[-2])
        expect_length = sr * self.duration
        if data.shape[1] >= expect_length:
            data = data[:, :expect_length]
        else:
            padding = torch.zeros([1, expect_length - data.shape[1]])
            data = torch.cat([data, padding], dim=1)
        data = data.squeeze(0)
        data = self.processor(data, return_tensors="pt", padding="longest", sampling_rate=sr).input_values
        return data.squeeze(0).float(), label.astype(np.float32)

    def get_seq_len(self):
        data = self.__getitem__(0)
        return data[0].shape[-1]

    def __len__(self):
        return len(self.files)


class Metric:
    """
    存储模型训练和测试时的指标
    """

    def __init__(self, mode="train"):
        if mode == "train":
            self.mode = "train"
            self.train_acc = []
            self.train_loss = []
            self.val_acc = []
            self.val_loss = []
            self.best_val_acc = [0, 0]  # [val_acc, train_acc]
        elif mode == "test":
            self.mode = "test"
            self.test_acc = 0
            self.test_loss = 0
            self.confusion_matrix = None
            self.report = None
        else:
            print("wrong mode !!! use default mode train")
            self.mode = "train"
            self.train_acc = []
            self.train_loss = []
            self.val_acc = []
            self.val_loss = []
            self.best_val_acc = [0, 0]

    def item(self) -> dict:
        """
        返回各种指标的字典格式数据
        Returns: dict

        """
        if self.mode == "train":
            data = {"train_acc": self.train_acc, "train_loss": self.train_loss,
                    "val_acc": self.val_acc, "val_loss": self.val_loss, 'best_val_acc': self.best_val_acc}
        else:
            data = {"test_acc": self.test_acc, "test_loss": self.test_loss}
        return data


def get_newest_file(path, suffix='pt'):
    """
    获取路径下最新的文件
    """
    all_files = os.listdir(path)
    files = []
    for file in all_files:
        if os.path.splitext(file)[1] == f'.{suffix}':  # 目录下包含.json的文件
            files.append(file)
    files.sort(key=lambda x: int(os.path.getmtime((path + "\\" + x))))
    file_new = os.path.join(path, files[-1])
    return file_new


def plot(metric: dict, model_name: str, result_path: str = "results/"):
    """
    绘制训练集，验证集准确率和损失变化曲线图
    """
    train_acc = metric['train_acc']
    train_loss = metric['train_loss']
    val_loss = metric["val_loss"]
    val_acc = metric['val_acc']
    epoch = np.arange(len(train_acc)) + 1

    plt.figure()
    plt.plot(epoch, train_acc)
    plt.plot(epoch, val_acc)
    plt.legend(["train accuracy", "validation accuracy"])
    plt.xlabel("epoch")
    plt.ylabel("accuracy(%)")
    plt.title("train accuracy and validation accuracy")
    plt.savefig(result_path + "images/" + model_name + "_accuracy.png", dpi=dpi)
    # plt.show()

    plt.figure()
    plt.plot(epoch, train_loss)
    plt.plot(epoch, val_loss)
    plt.legend(["train loss", "validation loss"])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("train loss and validation loss")
    plt.savefig(result_path + "images/" + model_name + "_loss.png", dpi=dpi)
    # plt.show()


def plot_meta(metric: dict, epoch: int, dataset_name: list, model_name: str,
              result_path: str = "results/"):
    """
    绘制训练集，验证集准确率和损失变化曲线图(元学习)
    Args:
        metric:
        epoch: 迭代次数
        model_name: 模型名
        dataset_name: 数据集名
        result_path: 保存路径
    """
    train_acc = metric['train_acc']
    train_loss = metric['train_loss']
    val_loss = metric["val_loss"]
    val_acc = metric['val_acc']
    num_dataset = len(dataset_name)
    acc = np.zeros([epoch, num_dataset])
    epoch_range = np.arange(epoch) + 1
    for i in range(num_dataset):
        acc[:, i] = train_acc[i::num_dataset]
    plt.figure()
    plt.plot(epoch_range, acc)
    plt.legend(dataset_name)
    plt.xlabel("epoch")
    plt.ylabel("accuracy(%)")
    plt.title("train accuracy")
    plt.savefig(result_path + "images/" + model_name + "dataset_accuracy.png", dpi=dpi)

    plt.figure()
    plt.plot(epoch_range, acc[:, -1])
    plt.plot(epoch_range, val_acc)
    plt.legend(["train accuracy", "validation accuracy"])
    plt.xlabel("epoch")
    plt.ylabel("accuracy(%)")
    plt.title("train accuracy and validation accuracy")
    plt.savefig(result_path + "images/" + model_name + f"{dataset_name[-1]}_accuracy.png", dpi=dpi)
    # plt.show()

    # plt.figure()
    # plt.plot(epoch, train_loss)
    # plt.plot(epoch, val_loss)
    # plt.legend(["train loss", "validation loss"])
    # plt.xlabel("epoch")
    # plt.ylabel("loss")
    # plt.title("train loss and validation loss")
    # plt.savefig(result_path + "images/" + model_name + "_loss.png", dpi=dpi)


def plot_2(path, name: str, result_path: str = "results/", ):
    """同时画出准确率和损失变化曲线图

    Args:
        path: metric的路径
        name: 保存图片的名称
        result_path: 保存图片的路径

    Returns:

    """
    metric = np.load(path, allow_pickle=True).item()
    train_acc = metric['train_acc']
    train_loss = metric['train_loss']
    val_loss = metric["val_loss"]
    val_acc = metric['val_acc']
    epoch = np.arange(len(train_acc)) + 1
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epoch, train_acc)
    plt.plot(epoch, val_acc)
    plt.legend(["train accuracy", "validation accuracy"])
    plt.xlabel("epoch")
    plt.ylabel("accuracy(%)")
    plt.title("train accuracy and validation accuracy")
    plt.subplot(1, 2, 2)
    plt.plot(epoch, train_loss)
    plt.plot(epoch, val_loss)
    plt.legend(["train loss", "validation loss"])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("train loss and validation loss")
    plt.savefig(result_path + "images/" + name + "acc_and_loss.svg", dpi=dpi)


class logger:
    """
    日志记录（仅作为参考，更多的时候看Tensorboard中的记录）
    """

    def __init__(self, model_name: str, addition: str, filename: str = "log.txt"):
        self.model_name = model_name
        self.addition = addition
        self.time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.filename = filename
        self.is_start = False

    def start(self):
        if not self.is_start:
            with open(self.filename, 'a') as f:
                f.write("\n========================\t" + self.time + "\t========================\n")
                f.write(f"model name: \t{self.model_name}\n")
                f.write(f"addition: \t{self.addition}\n")
            self.is_start = True

    def train(self, train_metric: Metric):
        with open(self.filename, 'a') as f:
            f.write("\n========================\t" + "train begin" + "\t========================\n")
            f.write("train(final): \t\t" + "train loss: {:.4f}\t train accuracy: {:.3f}\t validation loss: {:.4f}\t "
                                           "validation accuracy: {:.3f} \n".format(train_metric.train_loss[-1],
                                                                                   train_metric.train_acc[-1],
                                                                                   train_metric.val_loss[-1],
                                                                                   train_metric.val_acc[-1]))
            f.write("train(max_min): \t" + "train loss: {:.4f}\t train accuracy: {:.3f}\t validation loss: {:.4f}\t "
                                           "validation accuracy: {:.3f} \n".format(min(train_metric.train_loss),
                                                                                   max(train_metric.train_acc),
                                                                                   min(train_metric.val_loss),
                                                                                   max(train_metric.val_acc)))
            f.write("best val accuracy: {:3f} \t corresponding train accuracy: {:3f}\n"
                    .format(train_metric.best_val_acc[0], train_metric.best_val_acc[1]))
            f.write("========================\t" + "train end" + "\t========================\n")

    def test(self, test_metric: Metric):
        with open(self.filename, 'a') as f:
            try:
                f.write("\n========================\t" + "test begin" + "\t========================\n")
                f.write(
                    "test: \t\t\t" + "test loss: \t{:.4f} \t test accuracy:\t {:.3f} \n".format(test_metric.test_loss,
                                                                                                test_metric.test_acc))
                f.write("confusion matrix: \n")
                for i in range(len(test_metric.confusion_matrix)):
                    f.write(str(test_metric.confusion_matrix[i]) + '\n')
                f.write("\n")
                f.write("classification report: \n")
                f.write(test_metric.report)
                f.write("\n")
                f.write("========================\t" + "test end" + "\t========================\n")
            except TypeError:
                f.write("\n========================\t" + "test begin" + "\t========================\n")
                f.write(
                    "test: \t\t\t" + "test loss: \t{:.4f} \t test accuracy:\t {:.3f} \n".format(
                        test_metric.test_loss[0],
                        test_metric.test_acc[0]))
                f.write("confusion matrix: \n")
                for i in range(len(test_metric.confusion_matrix[0])):
                    f.write(str(test_metric.confusion_matrix[0][i]) + '\n')
                f.write("\n")
                f.write("classification report: \n")
                f.write(test_metric.report[0])
                f.write("\n")
                f.write(
                    "test: \t\t\t" + "test loss: \t{:.4f} \t test accuracy:\t {:.3f} \n".format(
                        test_metric.test_loss[1],
                        test_metric.test_acc[1]))
                f.write("confusion matrix: \n")
                for i in range(len(test_metric.confusion_matrix[1])):
                    f.write(str(test_metric.confusion_matrix[1][i]) + '\n')
                f.write("\n")
                f.write("classification report: \n")
                f.write(test_metric.report[1])
                f.write("\n")
                f.write("========================\t" + "test end" + "\t========================\n")


def l2_regularization(model, alpha: float):
    """
    L2正则化（加损失）

    Args:
        model: 模型
        alpha: L2正则化参数

    Returns: L2损失

    """
    l2_loss = []
    for module in model.modules():
        if type(module) is torch.nn.Conv2d or type(module) is torch.nn.Linear:
            l2_loss.append((module.weight ** 2).sum() / 2.0)
    return alpha * sum(l2_loss)


def onehot2int(onehot):
    """
    onehot -> int(应该没用)
    """
    label = torch.max(onehot.data, 1)[1]
    return label


def smooth_labels(labels, factor=0.1):
    """
    标签平滑（在loss函数设置label_smoothing=0.1能起到一样的效果）

    Args:
        labels: 原始标签
        factor: 平滑超参数

    Returns: 平滑后的标签

    """
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    return labels


class NoamScheduler:
    def __init__(self, optimizer, d_model, initial_lr, warmup):
        self.optimizer = optimizer
        self.d_model = d_model
        self.initial_lr = initial_lr
        self.warmup = warmup
        self.lr = 0

    def step(self, steps):
        self.lr = self.initial_lr * noam(d_model=self.d_model, step=steps + 1, warmup=self.warmup)
        for param_groups in self.optimizer.param_groups:
            param_groups['lr'] = self.lr

    def get_lr(self):
        return self.lr


def noam(d_model, step, warmup):
    """
    noam scheduler
    """
    fact = min(step ** (-0.5), step * warmup ** (-1.5))
    return fact * (d_model ** (-0.5))


def plot_noam(args: Args):
    """
    绘制noam scheduler的学习率变化曲线
    """
    lr = []
    for i in range(1000):
        lr.append(args.initial_lr * noam(d_model=args.d_model, step=i + 1, warmup=args.warmup))
    plt.plot(lr)
    plt.show()


def cleanup():
    """
    获取清除文件的命令行方法
    """
    print("如果想要删除一个目录下所有的png格式图片，可以切换到该目录下在控制台输入:  del /a /f /s /q  \"*.png\"")


class WarmupScheduler:
    """
    用在Transformer的训练中，主要用在CNN_ML_Transformer中
    """

    def __init__(self, optimizer, n_step, warmup, d_model, initial_lr):
        self.optimizer = optimizer
        self.n_step = n_step
        self.warmup = warmup
        self.d_model = d_model
        self.initial_lr = initial_lr

    def step(self):
        fact = min(self.n_step ** (-0.5), self.n_step * self.warmup ** (-1.5)) * (self.d_model ** (-0.5))
        self.optimizer.param_groups[0]['lr'] = self.initial_lr * fact


def cal_seq_len(seq_len: int, scale: int):
    """
    计算经过池化后的序列长度（向下取整）

    Args:
        seq_len: 原始序列长
        scale: 池化大小

    Returns: 经过池化操作的序列长度

    """
    return math.floor(seq_len / scale)


def check_dir():
    """
    创建models, results/images/, results/data 文件夹
    """
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists("results/images"):
        os.makedirs("results/images")
    if not os.path.exists("results/data"):
        os.makedirs("results/data")


def print_model(model):
    """
    打印模型参数，梯度
    """
    for name, params in model.named_parameters():
        print('-->name:', name)
        print('-->para:', params)
        print('-->grad_requires:', params.requires_grad)
        print('-->grad_value:', params.grad)
        print("===")


def log_model(model, val_acc):
    """
    记录模型参数，梯度
    """
    with open("model.txt", 'a') as f:
        f.write(str(val_acc) + "\n")
        for name, params in model.named_parameters():
            f.write(f'-->name: {name} \n')
            # print('-->para:', params)
            # f.write(f'-->grad_requires:{params.requires_grad}')
            f.write(f'-->grad_value:{params.grad} \n')
            f.write("===\n")


def model_structure(model_path, save_path=None):
    """
    查看模型结构
    """
    if not os.path.exists(model_path):
        print(f"cannot find path:{model_path}")
        return
    model = torch.load(model_path)
    if save_path is None:
        save_path = model_path.split('.')[-2].split('/')[-1] + ".txt"
    blank = ' '
    with open(save_path, 'w') as f:
        f.write('-' * 90 + "\n")
        print('-' * 90)
        f.write('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
                + ' ' * 3 + 'number' + ' ' * 3 + '|\n')
        print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
                + ' ' * 3 + 'number' + ' ' * 3 + '|')
        f.write('-' * 90 + "\n")
        print('-' * 90)
        num_para = 0
        type_size = 1  # 如果是浮点数就是4
        for index, (key, w_variable) in enumerate(model.named_parameters()):
            if len(key) <= 40:
                key = key + (40 - len(key)) * blank
            shape = str(w_variable.shape)
            if len(shape) <= 30:
                shape = shape + (30 - len(shape)) * blank
            each_para = 1
            for k in w_variable.shape:
                each_para *= k
            num_para += each_para
            str_num = str(each_para)
            if len(str_num) <= 10:
                str_num = str_num + (10 - len(str_num)) * blank

            f.write('| {} | {} | {} |\n'.format(key, shape, str_num))
            print('| {} | {} | {} |'.format(key, shape, str_num))
        f.write('-' * 90 + "\n")
        print('-' * 90)
        f.write('The total number of parameters: ' + str(num_para) + "\n")
        print('The total number of parameters: ' + str(num_para))
        f.write('The parameters of Model {}: {:4f}M \n'.format(model._get_name(), num_para * type_size / 1000 / 1000))
        print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
        f.write('-' * 90 + "\n")
        print('-' * 90)
        for name in list(model.named_modules()):
            f.write("{}\n".format(name))


def mask_input(x, p):
    """
    对输入进行遮盖

    Args:
        x: 输入特征
        p: 遮盖概率
    """
    batch_size, _, seq_len = x.shape
    mask = torch.rand(batch_size, 1, seq_len).to(x.device)
    mask = mask.ge(p)
    mask_x = x * mask
    return mask_x


if __name__ == "__main__":
    # plot_2("results/data/train_drop1_mfcc_smoothTrue_epoch80_l2re1_lr0002_pretrainTrue_train_metric.npy",
    #        "pretrain_True1")
    # args = Args()
    # plot_noam(args=args)
    model_structure(model_path="models/MultiTIM_AT_DIFF_MODMA_order3_drop1_mfcc_epoch100_l2re1_lr0004_pretrainFalse_clusterFalse.pt")
    pass
