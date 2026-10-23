"""A causal temporal residual learner and validation-selected convex gating."""

import itertools
import numpy as np
import torch
from torch import nn
from .metrics import metrics


class CausalBlock(nn.Module):
    def __init__(self, inputs, channels, dilation):
        super().__init__()
        pad = 2 * dilation
        self.net = nn.Sequential(nn.ConstantPad1d((pad, 0), 0), nn.Conv1d(inputs, channels, 3, dilation=dilation), nn.ReLU(),
                                 nn.ConstantPad1d((pad, 0), 0), nn.Conv1d(channels, channels, 3, dilation=dilation), nn.ReLU())
        self.skip = nn.Conv1d(inputs, channels, 1) if inputs != channels else nn.Identity()

    def forward(self, x):
        return torch.relu(self.net(x) + self.skip(x))


class ResidualTCN(nn.Module):
    def __init__(self, inputs, channels, horizon, dilations):
        super().__init__()
        blocks = []
        for dilation in dilations:
            blocks.append(CausalBlock(inputs, channels, dilation))
            inputs = channels
        self.encoder = nn.Sequential(*blocks)
        self.head = nn.Linear(channels, horizon)

    def forward(self, x):
        return self.head(self.encoder(x.transpose(1, 2))[:, :, -1])


def weighted_mae(y, p, quantile, tail_weight):
    weights = np.where(y >= np.quantile(y, quantile), tail_weight, 1.0)
    return float(np.average(np.abs(y - p), weights=weights))


class ConservativeGate:
    def fit(self, y, arx, rpc, quantile):
        self.arx_threshold = float(np.quantile(arx, quantile))
        self.rpc_threshold = float(np.quantile(rpc, quantile))
        risk = (arx >= self.arx_threshold) | (rpc >= self.rpc_threshold)
        best = None
        for risk_lambda, normal_lambda in itertools.product([0.0, 0.1, 0.2, 0.5, 1.0], repeat=2):
            prediction = arx + np.where(risk, risk_lambda, normal_lambda) * (rpc - arx)
            score = metrics(y, prediction, quantile)
            key = (score["AbsPeakBias"], score["ExtremeMAE"], score["MAE"], abs(risk_lambda - normal_lambda))
            if best is None or key < best:
                best = key
                self.risk_lambda, self.normal_lambda = risk_lambda, normal_lambda
        return self

    def predict(self, arx, rpc):
        risk = (arx >= self.arx_threshold) | (rpc >= self.rpc_threshold)
        strength = np.where(risk, self.risk_lambda, self.normal_lambda)
        return arx + strength * (rpc - arx), strength
