import math
import os

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from sklearn.mixture import GaussianMixture as GMM
from collections import Counter


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


def get_MODMA_conf(path="LANZHOU/info.xlsx"):
    """
    返回关于MODMA数据集的具体信息
    Args:
        path: 信息文件

    Returns: 包含关于数据集文件信息的字典

    """
    excel = pd.read_excel(path)
    conf = {}
    value = np.array(excel.values[:, :-2])
    value[:, 0] = np.array(value[:, 0], dtype=str)
    value[:, 1] = np.array(value[:, 1] == 'MDD', dtype=int)
    value[:, 3] = np.array(value[:, 3] == 'M', dtype=int)
    for i in range(value.shape[0]):
        conf['0' + value[i, 0]] = value[i, 1:]
    return conf


def get_wav_dict(path="MODMA"):
    """
    返回每个受试者所对应的音频文件
    Args:
        path: 存放所有音频文件的文件夹

    Returns: 字典: 键为受试者id，值为对应文件名列表

    """
    wav_files = get_files(path)
    conf = np.load('data/conf.npy', allow_pickle=True)
    subjection_id = conf.item().keys()
    wav_dict = {}
    for key in subjection_id:
        wav_dict[key] = []
    for wav in wav_files:
        wav_dict[wav[6:14]].append(wav)
    return wav_dict


def data_distribution(path, step=5):
    """
    计算音频文件时长分布
    Args:
        step: 分割步长
        path: 音频文件路径

    Returns:

    Examples:
        >>> distribute = data_distribution("MODMA")
        >>> index = np.array(list(distribute.index))
        >>> value = distribute.values
        >>> no_zero_index = value != 0              # 去除分布为0
        >>> index = index[no_zero_index]
        >>> value = value[no_zero_index]
        >>> index_ = list(index[:9].astype(str))    # 只单独展示前9个，后面的归为一类，共10个
        >>> index_.append("(46,]")
        >>> value_ = list(value[:9])
        >>> value_.append(np.sum(value[9:]))
    """
    wav_files = get_files(path)
    lens = []
    for it in wav_files:
        try:
            lens.append(librosa.get_duration(filename=it))
        except:
            print(it)
    print("mean: ")
    print(np.mean(lens))
    print("std: ")
    print(np.std(lens))
    print("max: ")
    print(np.max(lens))
    print("min: ")
    print(np.min(lens))
    minN = math.floor(min(lens))
    maxN = math.ceil(max(lens))
    s = pd.cut(lens, bins=[x for x in range(minN, maxN + step, step)])
    return s.value_counts()


def wav_split(data, sr, threshold, base_duration=10, overlap=0):
    """
    分割大于base_duration的音频
    Args:
        data: 音频数据
        sr: 采样率
        threshold: 阈值（小于截断，大于补零）
        base_duration: 分段时长
        overlap: 重叠部分时长

    Returns:

    """
    real_duration = librosa.get_duration(y=data, sr=sr)
    residual = real_duration % base_duration
    if residual < threshold:
        split_num = math.floor(real_duration / base_duration)
        data = data[:, :split_num * base_duration * sr]
    else:
        split_num = math.floor(real_duration / base_duration) + 1
        padding_len = int(sr * base_duration * split_num - data.shape[1])
        data = np.hstack([data, torch.zeros([1, padding_len])])
    data = data.reshape([split_num, sr * base_duration])
    return torch.Tensor(data), split_num


def wav_split_overlap(data, sr, threshold, base_duration=10, overlap=0.0):
    """
    分割大于base_duration的音频
    Args:
        data: 音频数据
        sr: 采样率
        threshold: 阈值（小于截断，大于补零）
        base_duration: 分段时长
        overlap: 重叠部分时长

    Returns:

    """

    stride = base_duration - overlap
    real_duration = librosa.get_duration(y=data, sr=sr)
    cut_duration = real_duration - base_duration
    residual = cut_duration % stride
    stride_length = int(stride * sr)
    base_length = int(base_duration * sr)

    if residual <= threshold:
        split_num = math.floor(cut_duration / stride)
        data = data[:, :split_num * stride_length + base_length]
    else:
        split_num = math.floor(cut_duration / stride) + 1
        padding_len = int(stride_length * split_num + base_length - data.shape[1])
        data = torch.cat([data, torch.zeros([1, padding_len])], dim=1)
    index = 0
    split_data = torch.zeros([split_num + 1, base_length])
    for i in range(split_num + 1):
        split_data[i] = data[:, index:index + base_length]
        index += stride_length
    return torch.Tensor(split_data), split_num + 1


def get_file_dict(path):
    """
    返回文件分类和其所对应的文件的字典
    Args:
        path: 文件存放路径

    Returns: 字典： 键为文件分类，值为文件名集合

    """
    file_dict = {}
    for dir_ in os.listdir(path=path):
        file_dict[dir_] = []
        for file in os.listdir(path + "/" + dir_):
            file_dict[dir_].append(path + "/" + dir_ + "/" + file)
    return file_dict


def get_mfcc(concat_data, resample_rate, order, frame_size):
    if order == 1:
        mfcc = librosa.feature.mfcc(y=concat_data, sr=resample_rate, n_mfcc=13, n_fft=frame_size)
        mfcc = mfcc.astype(np.float32)
    elif order == 2:
        mfcc = librosa.feature.mfcc(y=concat_data, sr=resample_rate, n_mfcc=13, n_fft=frame_size)
        mfcc_delta = librosa.feature.delta(mfcc, width=3)
        mfcc = np.concatenate([mfcc, mfcc_delta], axis=1).astype(np.float32)
    else:
        mfcc = librosa.feature.mfcc(y=concat_data, sr=resample_rate, n_mfcc=13, n_fft=frame_size)
        mfcc_delta = librosa.feature.delta(mfcc, width=3)
        mfcc_acc = librosa.feature.delta(mfcc_delta, width=3)
        mfcc = np.concatenate([mfcc, mfcc_delta, mfcc_acc], axis=1).astype(np.float32)
    return mfcc


def resample_wavs(path: str, resample_wavs_path: str, resample_rate: int = 16000):
    """
    重采样音频
    Args:
        path: 待采样的音频文件夹
        resample_wavs_path: 重采样后的文件夹
        resample_rate: 重采样率

    Returns:

    """
    wav_files = get_files(path)
    _, sr = librosa.load(wav_files[0], sr=None)
    if sr == resample_rate:
        print(f"音频采样率已经为{resample_rate} Hz")
        return
    for wav in tqdm(wav_files):
        data, sr = librosa.load(wav, sr=None)
        resample_data = librosa.resample(y=data, orig_sr=sr, target_sr=resample_rate).astype(np.float32)
        new_path = resample_wavs_path + wav[wav.find('/'):]
        if not os.path.exists(new_path[:new_path.rfind('/')]):
            os.makedirs(new_path[:new_path.rfind('/')])
        # sf.write(new_path, resample_data, samplerate=resample_rate)   # 存在一定损失
        torchaudio.save(new_path, torch.Tensor(resample_data).unsqueeze(0), sample_rate=16000)  # 无损


def load_MultiDataset(path: str, code, frame_length=0.05, padding_sec=0.5, duration=10, resample_rate=16000,
                      extra=False, order=3):
    """
    使用torch读取音频数据，加速读取 推荐使用
    Args:
        order: [1,2,3] 1:13阶MFCC 2:13阶MFCC+一阶差分 3:13阶MFCC+一阶差分+二阶差分
        extra: 是否保存分段信息(MODMA)
        path: 音频文件存放路径
        code: one-hot编码
        frame_length: 帧长
        padding_sec: 每两段音频之间的补零时长
        duration: 划分时长
        resample_rate: 重采样

    Returns:

    """
    dataset = path.split('/')[-1]
    if dataset == "MODMA":
        extra = True
    print(f"fast: {dataset}")
    wav_dict = get_file_dict(path=path)
    mfccs = None
    labels = None
    extra_info = {}
    sample_wav = wav_dict[list(wav_dict.keys())[0]][0]
    _, sr = librosa.load(sample_wav, sr=None)
    padding_len = int(padding_sec * sr)
    padding = torch.zeros([1, padding_len])
    frame_size = int(frame_length * resample_rate)
    # resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=resample_rate)
    for key in (wav_dict.keys()):
        print(f"process: {key}")
        concat_data = None
        for wav in tqdm(wav_dict[key]):
            data, sr = torchaudio.load(wav)
            if concat_data is None:
                concat_data = data
            else:
                concat_data = torch.cat([concat_data, padding, data], dim=1)
        concat_data = concat_data.squeeze(0).numpy()
        total_duration = len(concat_data) / sr  # 连接后的音频时长
        slice_length = duration * sr  # 分段长度
        res = total_duration % duration
        if res < 1:
            slice_num = math.floor(total_duration / duration)
            concat_data = concat_data[:slice_num * slice_length]
        else:
            slice_num = math.ceil(total_duration / duration)  # 分段数
            padding_len = int(slice_length * slice_num - len(concat_data))
            concat_data = torch.cat([concat_data, np.zeros(padding_len)], dim=0)
        extra_info[key] = slice_num
        concat_data = concat_data.reshape([slice_num, -1])
        mfcc = get_mfcc(concat_data, resample_rate, order, frame_size)
        if mfccs is None:
            mfccs = mfcc
        else:
            mfccs = np.vstack([mfccs, mfcc])
        label_one_hot = code(key)
        if labels is None:
            labels = np.tile(label_one_hot, (slice_num, 1))
        else:
            labels = np.vstack([labels, np.tile(label_one_hot, (slice_num, 1))])
    labels = labels.astype(np.float32)
    if extra:
        data = {'x': mfccs, 'y': labels, 'extra': extra_info}
    else:
        data = {'x': mfccs, 'y': labels}
    try:
        np.save(f"data/{dataset}_order{order}.npy", data)
        print(f"保存至 data/{dataset}_order{order}.npy")
    except:
        print("保存错误")
        np.save("temp.data", data)


def load_MultiDataset_V1(path: str, code, frame_length=0.05, duration=10, resample_rate=16000, threshold=1,
                         extra=False, order=3, overlap=2.5):
    """
    使用torch读取音频数据，加速读取 推荐使用，（将小于10s的短音频补零至10s，将大于10s的音频分成若干段(包含重叠)）
    Args:
        threshold: 对于多出来的部分，大于threshold补零，小于threshold截断
        order: [1,2,3] 1:13阶MFCC 2:13阶MFCC+一阶差分 3:13阶MFCC+一阶差分+二阶差分
        extra: 是否保存分段信息(MODMA)
        path: 音频文件存放路径
        code: one-hot编码
        frame_length: 帧长
        duration: 划分时长
        resample_rate: 重采样
        overlap: 重叠时长

    Returns:

    """
    dataset = path.split('/')[-1]
    if dataset == "MODMA" or dataset == "MODMA_16kHz":
        extra = True
    print(f"fastV1: {dataset}")
    wav_dict = get_file_dict(path=path)
    mfccs = None
    labels = None
    extra_info = {}
    sample_wav = wav_dict[list(wav_dict.keys())[0]][0]
    _, sr = librosa.load(sample_wav, sr=None)
    frame_size = int(frame_length * resample_rate)
    # resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=resample_rate)
    expect_length = sr * duration
    for key in (wav_dict.keys()):
        print(f"process: {key}")
        slice_num = 0
        stack_data = None
        for wav in tqdm(wav_dict[key]):
            data, sr = torchaudio.load(wav)
            actual_length = data.shape[1]
            if actual_length < expect_length:
                data = torch.cat([data, torch.zeros([1, expect_length - actual_length])], dim=1)
                slice_num += 1
            elif actual_length > expect_length:
                data, split_num = wav_split_overlap(data, sr, threshold, duration, overlap)
                slice_num += split_num
            else:
                slice_num += 1
            if stack_data is None:
                stack_data = data
            else:
                stack_data = torch.cat([stack_data, data], dim=0)

        extra_info[key] = slice_num
        stack_data = stack_data.numpy().astype(np.float32)
        if sr != resample_rate:
            stack_data = librosa.resample(stack_data, orig_sr=sr, target_sr=resample_rate)
        mfcc = get_mfcc(stack_data, resample_rate, order, frame_size)
        if mfccs is None:
            mfccs = mfcc
        else:
            mfccs = np.vstack([mfccs, mfcc])
        label_one_hot = code(key)
        if labels is None:
            labels = np.tile(label_one_hot, (slice_num, 1))
        else:
            labels = np.vstack([labels, np.tile(label_one_hot, (slice_num, 1))])
    labels = labels.astype(np.float32)
    if extra:
        data = {'x': mfccs, 'y': labels, 'extra': extra_info}
    else:
        data = {'x': mfccs, 'y': labels}
    try:
        np.save(f"data/{dataset}_V1_order{order}.npy", data)
        print(f"保存至 data/{dataset}_V1_order{order}.npy")
    except (PermissionError, FileNotFoundError):
        print("保存错误")
        np.save("temp.data", data)


def load_MultiDataset_V2(path: str, code, frame_length=0.05, duration=10, resample_rate=16000,
                         extra=False, order=3):
    """
   使用torch读取音频数据，加速读取 推荐使用，（将小于10s的短音频补零至10s，将大于10s的音频直接截断）
   Args:
       order: [1,2,3] 1:13阶MFCC 2:13阶MFCC+一阶差分 3:13阶MFCC+一阶差分+二阶差分
       extra: 是否保存分段信息(MODMA)
       path: 音频文件存放路径
       code: one-hot编码
       frame_length: 帧长
       duration: 划分时长
       resample_rate: 重采样

   Returns:

    """

    dataset = path.split('/')[-1]
    if dataset == "MODMA" or dataset == "MODMA_16kHz":
        extra = True
    print(f"fastV2: {dataset}")
    wav_dict = get_file_dict(path=path)
    mfccs = None
    labels = None
    extra_info = {}
    sample_wav = wav_dict[list(wav_dict.keys())[0]][0]
    _, sr = librosa.load(sample_wav, sr=None)
    frame_size = int(frame_length * resample_rate)
    expect_length = sr * duration
    for key in (wav_dict.keys()):
        print(f"process: {key}")
        stack_data = None
        for wav in tqdm(wav_dict[key]):
            data, sr = torchaudio.load(wav)
            actual_length = data.shape[1]
            if actual_length < expect_length:
                data = torch.cat([data, torch.zeros([1, expect_length - actual_length])], dim=1)
            elif actual_length > expect_length:
                data = data[:, :expect_length]
            if stack_data is None:
                stack_data = data
            else:
                stack_data = torch.cat([stack_data, data], dim=0)
        slice_num = len(wav_dict[key])
        extra_info[key] = slice_num
        stack_data = stack_data.numpy().astype(np.float32)
        if sr != resample_rate:
            stack_data = librosa.resample(stack_data, orig_sr=sr, target_sr=resample_rate)

        mfcc = get_mfcc(stack_data, resample_rate, order, frame_size)
        if mfccs is None:
            mfccs = mfcc
        else:
            mfccs = np.vstack([mfccs, mfcc])
        label_one_hot = code(key)
        if labels is None:
            labels = np.tile(label_one_hot, (slice_num, 1))
        else:
            labels = np.vstack([labels, np.tile(label_one_hot, (slice_num, 1))])
    labels = labels.astype(np.float32)
    if extra:
        data = {'x': mfccs, 'y': labels, 'extra': extra_info}
    else:
        data = {'x': mfccs, 'y': labels}
    try:
        np.save(f"data/{dataset}_V2_order{order}.npy", data)
        print(f"保存至 data/{dataset}_V2_order{order}.npy")
    except:
        print("保存错误")
        np.save("temp.data", data)


def get_MODMA_extra(path):
    """
    得到MODMA的附加信息(id2mfcc: id所对应的MFCC行索引，scores:id对应的PHQ-9 CTQ-SF LES SSRS GAD-7 PSQI 分数)
    Args:
        path: npy文件存放路径
    Returns:

    """
    conf = np.load("data/conf.npy", allow_pickle=True).item()
    tmp = np.load(path, allow_pickle=True).item()
    extra = tmp['extra']
    id2mfcc = {}
    path_info = path.split('/')[-1].split('.')[0][5:]
    dirs = extra.keys()
    splice = list(extra.values())
    scores = np.zeros([sum(splice), 6])
    for i, dir_ in enumerate(dirs):
        id2mfcc[dir_] = np.arange(sum(splice[:i]), sum(splice[:i + 1]))
        score = conf[dir_][4:]
        scores[id2mfcc[dir_]] = score
    scores = scores - np.min(scores)
    np.save(f"data/scores{path_info}.npy", scores)
    np.save(f"data/id2mfcc{path_info}.npy", id2mfcc)


# one-hot编码
def EMODB_code(class_name):
    if class_name == 'anger':
        return np.array([1, 0, 0, 0, 0, 0, 0])
    elif class_name == 'boredom':
        return np.array([0, 1, 0, 0, 0, 0, 0])
    elif class_name == 'disgust':
        return np.array([0, 0, 1, 0, 0, 0, 0])
    elif class_name == 'fear':
        return np.array([0, 0, 0, 1, 0, 0, 0])
    elif class_name == 'happy':
        return np.array([0, 0, 0, 0, 1, 0, 0])
    elif class_name == 'neutral':
        return np.array([0, 0, 0, 0, 0, 1, 0])
    elif class_name == 'sad':
        return np.array([0, 0, 0, 0, 0, 0, 1])
    else:
        return np.zeros([1, 7])


def MODMA_code(class_name):
    conf = np.load(r"D:\graduate_code\SER_depression\preprocess\data\conf.npy", allow_pickle=True).item()
    MDD_keys = [k for k, v in conf.items() if v[0] == 1]
    HC_keys = [k for k, v in conf.items() if v[0] == 0]
    if class_name in HC_keys:
        return np.array([1, 0])
    elif class_name in MDD_keys:
        return np.array([0, 1])
    else:
        raise NameError


def CASIA_code(class_name):
    if class_name == 'anger':
        return np.array([1, 0, 0, 0, 0, 0])
    elif class_name == 'fear':
        return np.array([0, 1, 0, 0, 0, 0])
    elif class_name == 'happy':
        return np.array([0, 0, 1, 0, 0, 0])
    elif class_name == 'neutral':
        return np.array([0, 0, 0, 1, 0, 0])
    elif class_name == 'sad':
        return np.array([0, 0, 0, 0, 1, 0])
    elif class_name == 'surprise':
        return np.array([0, 0, 0, 0, 0, 1])
    else:
        raise NameError


def CASIA_code_(class_name):
    if class_name == 'anger':
        return np.array([1, 0, 0, 0])
    elif class_name == 'happy':
        return np.array([0, 1, 0, 0])
    elif class_name == 'sad':
        return np.array([0, 0, 1, 0])
    elif class_name == 'surprise':
        return np.array([0, 0, 0, 1])
    else:
        raise NameError


def IEMOCAP_code(class_name):
    if class_name == 'anger':
        return np.array([1, 0, 0, 0, 0, 0])
    elif class_name == 'exc':
        return np.array([0, 1, 0, 0, 0, 0])
    elif class_name == 'frustrated':
        return np.array([0, 0, 1, 0, 0, 0])
    elif class_name == 'happy':
        return np.array([0, 0, 0, 1, 0, 0])
    elif class_name == 'neutral':
        return np.array([0, 0, 0, 0, 1, 0])
    elif class_name == 'sad':
        return np.array([0, 0, 0, 0, 0, 1])
    else:
        raise NameError


def RAVDESS_code(class_name):
    if class_name == 'anger':
        return np.array([1, 0, 0, 0, 0, 0, 0, 0])
    elif class_name == 'calm':
        return np.array([0, 1, 0, 0, 0, 0, 0, 0])
    elif class_name == 'disgust':
        return np.array([0, 0, 1, 0, 0, 0, 0, 0])
    elif class_name == 'fear':
        return np.array([0, 0, 0, 1, 0, 0, 0, 0])
    elif class_name == 'happy':
        return np.array([0, 0, 0, 0, 1, 0, 0, 0])
    elif class_name == 'normal':
        return np.array([0, 0, 0, 0, 0, 1, 0, 0])
    elif class_name == 'sad':
        return np.array([0, 0, 0, 0, 0, 0, 1, 0])
    elif class_name == 'surprised':
        return np.array([0, 0, 0, 0, 0, 0, 0, 1])
    else:
        raise NameError


def RAVDESS_code_(class_name):
    if class_name == 'anger':
        return np.array([1, 0, 0, 0])
    elif class_name == 'happy':
        return np.array([0, 1, 0, 0])
    elif class_name == 'sad':
        return np.array([0, 0, 1, 0])
    elif class_name == 'surprised':
        return np.array([0, 0, 0, 1])
    else:
        raise NameError


def eNTERFACE_code(class_name):
    if class_name == 'anger':
        return np.array([1, 0, 0, 0, 0, 0])
    elif class_name == 'disgust':
        return np.array([0, 1, 0, 0, 0, 0])
    elif class_name == 'fear':
        return np.array([0, 0, 1, 0, 0, 0])
    elif class_name == 'happy':
        return np.array([0, 0, 0, 1, 0, 0])
    elif class_name == 'sad':
        return np.array([0, 0, 0, 0, 1, 0])
    elif class_name == 'surprise':
        return np.array([0, 0, 0, 0, 0, 1])
    else:
        raise NameError


def DAIC_code(class_name):
    if class_name == 'f_depr':
        return np.array([1, 0, 0, 0])
    elif class_name == 'f_nor':
        return np.array([0, 1, 0, 0])
    elif class_name == 'm_depr':
        return np.array([0, 0, 1, 0])
    elif class_name == 'm_nor':
        return np.array([0, 0, 0, 1])
    else:
        raise NameError


def ABC_code(class_name):
    if class_name == 'anger':
        return np.array([1, 0, 0, 0, 0, 0])
    elif class_name == 'boredom':
        return np.array([0, 1, 0, 0, 0, 0])
    elif class_name == 'fear':
        return np.array([0, 0, 1, 0, 0, 0])
    elif class_name == 'happy':
        return np.array([0, 0, 0, 1, 0, 0])
    elif class_name == 'neutral':
        return np.array([0, 0, 0, 0, 1, 0])
    elif class_name == 'surprise':
        return np.array([0, 0, 0, 0, 0, 1])
    else:
        raise NameError


def TESS_code(class_name):
    if class_name == 'anger':
        return np.array([1, 0, 0, 0, 0, 0, 0])
    elif class_name == 'disgust':
        return np.array([0, 1, 0, 0, 0, 0, 0])
    elif class_name == 'fear':
        return np.array([0, 0, 1, 0, 0, 0, 0])
    elif class_name == 'happy':
        return np.array([0, 0, 0, 1, 0, 0, 0])
    elif class_name == 'neutral':
        return np.array([0, 0, 0, 0, 1, 0, 0])
    elif class_name == 'sad':
        return np.array([0, 0, 0, 0, 0, 1, 0])
    elif class_name == 'surprise':
        return np.array([0, 0, 0, 0, 0, 0, 1])
    else:
        raise NameError


def stereo2mono(path):
    """
    立体声变单通道
    Args:
        path: 可能存放立体声的音频路径

    Returns:

    """
    wav_files = get_files(path)
    for wav in wav_files:
        data, sr = librosa.load(wav, mono=False, sr=None)
        if len(data.shape) == 2:
            # print(wav)
            data = librosa.to_mono(y=data)
            torchaudio.save(wav, torch.Tensor(data).unsqueeze(0), sample_rate=sr)


def order3_order2(data):
    """
    39维MFCC -> 26维MFCC
    """
    if data.shape[1] != 39:
        print(f"error data shape, expect shape[1]=39, but receive shape[1]={data.shape[1]}")
        return
    return data[:, :26, :]


def order3_order1(data):
    """
    39维MFCC -> 13维MFCC
    """
    if data.shape[1] != 39:
        print(f"error data shape, expect shape[1]=39, but receive shape[1]={data.shape[1]}")
        return
    return data[:, :13, :]


def mGMM(X, K, n_components):
    label2index = []
    for k in range(K):
        l2i = []
        tmp_labels = []
        gmm = GMM(n_components=n_components).fit(X)
        labels = gmm.predict(X)
        for i in range(len(labels)):
            if labels[i] not in tmp_labels:
                l2i.append(i)
                tmp_labels.append(labels[i])
        label2index.append(l2i)
    return label2index


def no_repeat_column(X):
    diff_x = np.diff(X, axis=0)
    counts = np.count_nonzero(diff_x, axis=0)
    return max(counts)


def multiGMM(path, K, n_components, save_path: str):
    data = np.load(path, allow_pickle=True).item()
    x = data['x']
    y = data['y']
    sample_num = x.shape[0]
    feature_dim = x.shape[1]
    new_x = np.zeros([sample_num, feature_dim, n_components])
    for s in tqdm(range(sample_num)):
        X = x[s].T
        if no_repeat_column(X) + 1 > n_components:
            label2index = mGMM(X=X, K=K, n_components=n_components)
            label2index = np.array(label2index).flatten()
            label2index_dict = Counter(label2index)
            label2index_sorted = sorted(label2index_dict.items(), key=lambda x: x[1], reverse=True)
            label2index = np.array(label2index_sorted[:n_components])
            index = np.sort(label2index[:, 0])
            try:
                new_x[s] = x[s, :, index].T
            except IndexError:
                print(s)
                np.save("temp.npy", new_x)
        else:
            new_x[s, :, :] = x[s, :, :n_components]
    new_data = {'x': new_x.astype(np.float32), 'y': y}
    np.save(save_path, new_data)
    return new_x


def process_MODMA(path: str, duration=10, threshold=1, overlap=2.5, new_wavs_path="MODMA_Plus"):
    """
    处理MODMA
    """
    wav_dict = get_file_dict(path=path)
    sample_wav = wav_dict[list(wav_dict.keys())[0]][0]
    _, sr = librosa.load(sample_wav, sr=None)
    expect_length = sr * duration
    for key in (wav_dict.keys()):
        print(f"process: {key}")
        for wav in tqdm(wav_dict[key]):
            data, sr = torchaudio.load(wav)
            actual_length = data.shape[1]
            if actual_length <= expect_length:
                data = torch.cat([data, torch.zeros([1, expect_length - actual_length])], dim=1)
                new_path = new_wavs_path + wav[wav.find('/'):]
                if not os.path.exists(new_path[:new_path.rfind('/')]):
                    os.makedirs(new_path[:new_path.rfind('/')])
                torchaudio.save(new_path, torch.Tensor(data), sample_rate=16000)  # 无损

            elif actual_length > expect_length:
                data, split_num = wav_split_overlap(data, sr, threshold, duration, overlap)
                for i in range(data.shape[0]):
                    new_path = new_wavs_path + wav[wav.find('/'):][:-4] + f"_{i}.wav"
                    if not os.path.exists(new_path[:new_path.rfind('/')]):
                        os.makedirs(new_path[:new_path.rfind('/')])
                    torchaudio.save(new_path, torch.Tensor(data[i]).unsqueeze(0), sample_rate=16000)


if __name__ == "__main__":
    # load_MultiDataset_V2("datasets/RAVDESS", code=RAVDESS_code, order=3, duration=6)
    # resample_wavs("datasets/TESS", resample_wavs_path="datasets/TESS_16kHz", resample_rate=16000)
    # load_MultiDataset_V2("MODMA", code=MODMA_code)
    # get_MODMA_extra(path="data/MODMA_V2_order3.npy")
    # load_MultiDataset_V1("MODMA", frame_length=0.05, code=MODMA_code, duration=10, resample_rate=16000, threshold=1,
    #                      extra=True, order=3, overlap=2.5)
    # multiGMM("data/MODMA_V1_order3.npy", 6, n_components=100, save_path="data/MODMA_V1_order3_c_6.npy")
    load_MultiDataset_V1("datasets/IEMOCAP", code=IEMOCAP_code, duration=10, overlap=2.5)
    pass
