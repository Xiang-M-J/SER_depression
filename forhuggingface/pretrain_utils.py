import math
from config import Args
import torch
from torch.cuda.amp import autocast

from utils import accuracy_cal


def cal_hidden(seq_len):
    h1 = math.floor((seq_len - 10) / 5 + 1)
    h2 = math.floor((h1 - 3) / 2 + 1)
    h3 = math.floor((h2 - 3) / 2 + 1)
    h4 = math.floor((h3 - 3) / 2 + 1)
    h5 = math.floor((h4 - 3) / 2 + 1)
    h6 = math.floor((h5 - 2) / 2 + 1)
    h7 = math.floor((h6 - 2) / 2 + 1)
    return h7


def train_step(model, optimizer, bx, by, use_amp, scaler):
    if torch.cuda.is_available():
        bx = bx.cuda()
        by = by.cuda()
    optimizer.zero_grad()
    if use_amp:
        with autocast():
            output = model(bx)
            loss = torch.nn.CrossEntropyLoss()(output, by)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        output = model(bx)
        loss = torch.nn.CrossEntropyLoss()(output, by)
        loss.backward()
        optimizer.step()
    return accuracy_cal(output, by), loss.data.item()


def val_step(model, vx, vy, use_amp):
    if torch.cuda.is_available():
        vx = vx.cuda()
        vy = vy.cuda()
    if use_amp:
        with autocast():
            output = model(vx)
            loss = torch.nn.CrossEntropyLoss()(output, vy)
    else:
        output = model(vx)
        loss = torch.nn.CrossEntropyLoss()(output, vy)
    return accuracy_cal(output, vy), loss.data.item()


def test_step(model, vx, vy, use_amp):
    return val_step(model, vx, vy, use_amp)
