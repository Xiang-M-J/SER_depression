import os

import numpy as np
from torch.utils.data import DataLoader

from utils import myWavLoader


def get_files(base_path: str = "MODMA"):
    """
    获取路径下的所有的文件
    Args:
        base_path: 路径名
    Returns: 文件名集合
    """
    files = []
    for dir_ in os.listdir(base_path):
        for file in os.listdir(base_path + "/" + dir_):
            files.append(base_path + "/" + dir_ + "/" + file)
    return files


class MODMADataModule:
    def __init__(self, spilt_rate=None, data_dir: str = "../preprocess/MODMA", duration: int = 10):
        super().__init__()
        if spilt_rate is None:
            spilt_rate = [0.8, 0.1, 0.1]
        self.data_dir = data_dir
        self.duration = duration
        self.num_classes = 2
        self.spilt_rate = spilt_rate
        self.files = np.array(get_files(self.data_dir))
        self.train_num = int(len(self.files) * spilt_rate[0])
        self.val_num = int(len(self.files) * spilt_rate[1])
        self.test_num = len(self.files) - self.train_num - self.val_num

    def setup(self):
        random_index = np.random.permutation(len(self.files))
        train_wavs = self.files[random_index[:self.train_num]]
        val_wavs = self.files[random_index[self.train_num:self.train_num + self.val_num]]
        test_wavs = self.files[random_index[self.train_num + self.val_num:]]
        self.train_dataset = myWavLoader(train_wavs)
        self.val_dataset = myWavLoader(val_wavs)
        self.test_dataset = myWavLoader(test_wavs)

    def get_seq_len(self):
        return self.train_dataset.get_seq_len()

    def train_dataloader(self, batch_size):
        return DataLoader(self.train_dataset, batch_size=batch_size)

    def val_dataloader(self, batch_size):
        return DataLoader(self.val_dataset, batch_size=batch_size)

    def test_dataloader(self, batch_size):
        return DataLoader(self.test_dataset, batch_size=batch_size)
