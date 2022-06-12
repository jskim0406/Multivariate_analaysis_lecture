import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import copy
import lightly
import numpy as np

import lightly.models as models
import lightly.loss as loss
import lightly.data as data

from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum
from lightly.models.utils import batch_shuffle
from lightly.models.utils import batch_unshuffle


class Classifier(pl.LightningModule):
    def __init__(self, c, backbone, save_path='clf_weight/temp.pth'):
        super().__init__()
        self.c = c
        self.backbone = backbone
        self.fc = nn.Linear(512, 10)

        self.criterion = nn.CrossEntropyLoss()
        self.best_valacc = 0.0
        self.save_path = save_path

    def forward(self, x):
        y_hat = self.backbone(x).flatten(start_dim=1)
        y_hat = self.fc(y_hat)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss_fc", loss)
        return loss

    def training_epoch_end(self, outputs):
        self.custom_histogram_weights()

    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(
                name, params, self.current_epoch
            )

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        y_hat = torch.nn.functional.softmax(y_hat, dim=1)

        _, predicted = torch.max(y_hat, 1)
        num = predicted.shape[0]
        correct = (predicted == y).float().sum()
        return num, correct

    def validation_epoch_end(self, outputs):
        if outputs:
            total_num = 0
            total_correct = 0
            for num, correct in outputs:
                total_num += num
                total_correct += correct
            acc = total_correct / total_num
            if self.best_valacc < acc.cpu().numpy() :
                state_dict = {
                    'backbone': self.backbone.state_dict(), 
                    'fc': self.fc.state_dict()
                }
                torch.save(state_dict, self.save_path)
                self.best_valacc = acc.cpu().numpy()
            self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optim = torch.optim.Adam(list(self.backbone.parameters()) + list(self.fc.parameters()), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.c.max_epochs)
        return [optim], [scheduler]