import math
import os
import random

import librosa
import numpy as np
import torch
import torchaudio
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from preprocess.process_utils import get_files, MODMA_code
from utils import myDataset, accuracy_cal


def sample_wavs(wav_list: list, code, frame_length=0.05, duration=10, resample_rate=16000,
                order=3):
    """
   将小于10s的短音频补零至10s，将大于10s的音频直接截断
   Args:
       wav_list: 音频文件路径
       order: [1,2,3] 1:13阶MFCC 2:13阶MFCC+一阶差分 3:13阶MFCC+一阶差分+二阶差分
       path: 音频文件存放路径
       code: one-hot编码
       frame_length: 帧长
       duration: 划分时长
       resample_rate: 重采样

   Returns:

    """
    wav_dict = {}
    for wav in wav_list:
        dir_ = wav.split('/')[-2]
        if dir_ in wav_dict.keys():
            wav_dict[dir_].append(wav)
        else:
            wav_dict[dir_] = [wav]
    mfccs = None
    labels = None
    sample_wav = wav_dict[list(wav_dict.keys())[0]][0]
    _, sr = librosa.load(sample_wav, sr=None)
    frame_size = int(frame_length * resample_rate)
    expect_length = sr * duration
    for key in (wav_dict.keys()):
        stack_data = None
        for wav in wav_dict[key]:
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
        stack_data = stack_data.numpy().astype(np.float32)
        if sr != resample_rate:
            stack_data = librosa.resample(stack_data, orig_sr=sr, target_sr=resample_rate)
        if order == 1:
            mfcc = librosa.feature.mfcc(y=stack_data, sr=resample_rate, n_mfcc=13, n_fft=frame_size)
            mfcc = mfcc.astype(np.float32)
        elif order == 2:
            mfcc = librosa.feature.mfcc(y=stack_data, sr=resample_rate, n_mfcc=13, n_fft=frame_size)
            mfcc_delta = librosa.feature.delta(mfcc, width=3)
            mfcc = np.concatenate([mfcc, mfcc_delta], axis=1).astype(np.float32)
        else:
            mfcc = librosa.feature.mfcc(y=stack_data, sr=resample_rate, n_mfcc=13, n_fft=frame_size)
            mfcc_delta = librosa.feature.delta(mfcc, width=3)
            mfcc_acc = librosa.feature.delta(mfcc_delta, width=3)
            mfcc = np.concatenate([mfcc, mfcc_delta, mfcc_acc], axis=1).astype(np.float32)
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

    return mfccs, labels


def test_(model_path, wav_path, num, num_class, batch_size=32, sr=16000, duration=10, order=3, frame_length=0.05):
    if not os.path.exists(model_path):
        print(f"error! cannot find the model in {model_path}")
        return
    model = torch.load(model_path)
    print(f"load model: {model_path}")
    wav_list = get_files(wav_path)
    if num > len(wav_list):
        print("抽样次数太多")
        return
    sample_wav = random.sample(wav_list, num)
    mfccs, labels = sample_wavs(sample_wav, code=MODMA_code, order=order, duration=duration,
                                resample_rate=sr, frame_length=frame_length)

    dataset = myDataset(x=mfccs, y=labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    test_num = len(dataset)
    model.eval()
    test_correct = 0
    test_loss = 0
    loss_fn = torch.nn.CrossEntropyLoss()
    y_pred = torch.zeros(test_num)
    y_true = torch.zeros(test_num)
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()
    for step, (vx, vy) in enumerate(dataloader):
        if torch.cuda.is_available():
            vx = vx.cuda()
            vy = vy.cuda()
        with torch.no_grad():
            output = model(vx)
            y_pred[step * batch_size: step * batch_size + vy.shape[0]] = torch.max(output.data, 1)[1]
            y_true[step * batch_size: step * batch_size + vy.shape[0]] = torch.max(vy.data, 1)[1]
            loss = loss_fn(output, vy)
            test_correct += accuracy_cal(output, vy)
            test_loss += loss
    conf_matrix = confusion_matrix(y_true, y_pred, labels=np.arange(num_class))
    test_loss = test_loss / math.ceil(test_num / batch_size)
    test_acc = float((test_correct * 100) / test_num)
    print("test Loss:{:.4f} test Accuracy:{:.3f}".format(test_loss, test_acc))
    print(conf_matrix)


# def create_sample_dir():
#     wav_files = np.load("sample_wav.data")
#     sample_wavs_path = "sample"
#     for wav in wav_files:
#         data, sr = torchaudio.load(wav)
#         wav = wav[wav.find('/') + 1:]
#         new_path = sample_wavs_path + wav[wav.find('/'):]
#         if not os.path.exists(new_path[:new_path.rfind('/')]):
#             os.makedirs(new_path[:new_path.rfind('/')])
#         torchaudio.save(new_path, data, sample_rate=16000)  # 无损


if __name__ == "__main__":
    test_(
        model_path="models/train_order3_drop1_mfcc_smoothTrue_epoch60_l2re1_lr0002_pretrainTrue_best.pt",
        wav_path="preprocess/MODMA_16kHz", num=300, num_class=2, order=3)
    pass
