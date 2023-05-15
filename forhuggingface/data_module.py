import os

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, dataset
from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor


class myWavLoader(dataset.Dataset):
    """
    直接加载音频原始数据作为输入（wav2vec2, hubert）
    """

    def __init__(self, files, labels, processor, duration=10) -> None:
        super(myWavLoader, self).__init__()
        self.files = files
        self.labels = labels
        self.duration = duration
        self.processor = processor

    def __getitem__(self, index):
        data, sr = torchaudio.load(self.files[index])
        label = self.labels[index]
        data = data.squeeze(0)
        data = self.processor(data, padding="max_length", truncation=True, max_length=self.duration * sr, return_tensors="pt", sampling_rate=sr).input_values
        return data.squeeze(0).float(), label.astype(np.float32)

    def get_seq_len(self):
        data = self.__getitem__(0)
        return data[0].shape[-1]

    def __len__(self):
        return len(self.files)


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


class PretrainDataModule:
    def __init__(self, num_class, label_path, data_dir, processor, spilt_rate=None, duration: int = 10):
        super().__init__()
        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        self.processor = processor
        if spilt_rate is None:
            spilt_rate = [0.6, 0.2, 0.2]
        self.data_dir = data_dir
        self.duration = duration
        self.label_path = label_path
        self.num_classes = num_class
        self.spilt_rate = spilt_rate
        self.files = np.array(get_files(self.data_dir))
        self.train_num = int(len(self.files) * spilt_rate[0])
        self.val_num = int(len(self.files) * spilt_rate[1])
        self.test_num = len(self.files) - self.train_num - self.val_num

    def setup(self):
        labels = np.load(self.label_path)
        np.random.seed(34)
        random_index = np.random.permutation(len(self.files))
        train_wavs = self.files[random_index[:self.train_num]]
        train_labels = labels[random_index[:self.train_num]]
        val_wavs = self.files[random_index[self.train_num:self.train_num + self.val_num]]
        val_labels = labels[random_index[self.train_num:self.train_num + self.val_num]]
        test_wavs = self.files[random_index[self.train_num + self.val_num:]]
        test_labels = labels[random_index[self.train_num + self.val_num:]]
        self.train_dataset = myWavLoader(train_wavs, train_labels, self.processor, self.duration)
        self.val_dataset = myWavLoader(val_wavs, val_labels, self.processor, self.duration)
        self.test_dataset = myWavLoader(test_wavs, test_labels, self.processor, self.duration)

    def get_seq_len(self):
        return self.train_dataset.get_seq_len()

    def train_dataloader(self, batch_size):
        return DataLoader(self.train_dataset, batch_size=batch_size)

    def val_dataloader(self, batch_size):
        return DataLoader(self.val_dataset, batch_size=batch_size)

    def test_dataloader(self, batch_size):
        return DataLoader(self.test_dataset, batch_size=batch_size)
