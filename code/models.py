# -*- coding: utf-8 -*-
"""
models.py — W4 模型库（v1.0）
====================================================================
监督保底基线：LSTM / GRU / 1D-CNN
主线网络：轻量注意力编码器（<0.5M 参数，CPU 单步毫秒级目标）
输出统一为各轮滑移系数 s_hat [B, N, 4]，经 s_max*tanh 有界映射。

输入特征（7 通道，与数据集口径一致）：
  ch0-3: 四轮编码器线速度 u (m/s)
  ch4  : 陀螺偏航角速度 g (rad/s)
  ch5-6: IMU 水平速度增量 dv (m/s)，末步补零（先归一化后补零）
"""

import torch
import torch.nn as nn

S_MAX = 0.45
IN_CH = 7


def count_params(m):
    return sum(p.numel() for p in m.parameters())


class ScaledHead(nn.Module):
    """线性头 + s_max*tanh 有界输出（保证 (1-s)>0）。"""
    def __init__(self, d_in, n_out=4, s_max=S_MAX):
        super().__init__()
        self.fc = nn.Linear(d_in, n_out)
        self.s_max = s_max

    def forward(self, x):
        return self.s_max * torch.tanh(self.fc(x))


class LSTMNet(nn.Module):
    def __init__(self, hidden=64, layers=2, bidir=True):
        super().__init__()
        self.rnn = nn.LSTM(IN_CH, hidden, layers, batch_first=True,
                           bidirectional=bidir)
        self.head = ScaledHead(hidden * (2 if bidir else 1))

    def forward(self, x):
        y, _ = self.rnn(x)
        return self.head(y)


class GRUNet(nn.Module):
    def __init__(self, hidden=64, layers=2, bidir=True):
        super().__init__()
        self.rnn = nn.GRU(IN_CH, hidden, layers, batch_first=True,
                          bidirectional=bidir)
        self.head = ScaledHead(hidden * (2 if bidir else 1))

    def forward(self, x):
        y, _ = self.rnn(x)
        return self.head(y)


class CNNNet(nn.Module):
    """1D-CNN：感受野 1+6+12+24 = 43 步（< 窗长 50）。"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(IN_CH, 32, 7, padding=3), nn.ReLU(),
            nn.Conv1d(32, 64, 7, padding=6, dilation=2), nn.ReLU(),
            nn.Conv1d(64, 64, 7, padding=12, dilation=4), nn.ReLU(),
            nn.Conv1d(64, 4, 1),
        )
        self.s_max = S_MAX

    def forward(self, x):
        y = self.net(x.transpose(1, 2)).transpose(1, 2)   # [B,N,4]
        return self.s_max * torch.tanh(y)


class AttentionEncoder(nn.Module):
    """
    轻量注意力编码器（主线网络）：
      Linear 投影(d=64) + 学习式位置编码 + 2 层 TransformerEncoder(heads=4, FFN=128)
      + 线性头
    参数量约 72k（<0.5M 目标），支持后续导出注意力权重做逐轮置信度热图。
    """
    def __init__(self, d_model=64, nhead=4, layers=2, ff=128, n_max=64):
        super().__init__()
        self.inp = nn.Linear(IN_CH, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_max, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=ff, dropout=0.0,
            batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = ScaledHead(d_model)

    def forward(self, x):
        h = self.inp(x) + self.pos[:, :x.shape[1]]
        h = self.enc(h)
        return self.head(h)


MODEL_ZOO = {
    "lstm": lambda: LSTMNet(),
    "gru": lambda: GRUNet(),
    "cnn": lambda: CNNNet(),
    "attn": lambda: AttentionEncoder(),
}


def build_model(name):
    if name not in MODEL_ZOO:
        raise ValueError(f"unknown model: {name}")
    return MODEL_ZOO[name]()
