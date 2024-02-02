from __future__ import print_function
import torch
from torch import optim, nn, utils, Tensor

import torch.nn as nn
import torch.nn.functional as F

import lightning as L  # Note it is no longer pytorch_lightning!


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class MNISTModel(L.LightningModule):

    def __init__(self):
        super().__init__()
        self.net = Net()

    def configure_optimisers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

    def compute_loss(self, batch):
        input, target = batch
        output = self.net(input)
        loss = F.mse_loss(output, target)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("val/loss", loss)
        return

    def test_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("test/loss", loss)