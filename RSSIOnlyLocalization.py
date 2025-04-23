import pytorch_lightning as pl
import torch
from torch import nn
import numpy as np
import os
import matplotlib.pyplot as plt

# from MobileNetV2 import mobilenet_v2
from MobileVit import mobilevitorigin1d_xs

class RSSIOnlyLocalization(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.deivce = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # self.rssi_net = mobilenet_v2(
        #     input_channels=1,num_classes= self.hparams.num_classes
        # ).to(self.deivce)
        # self.rssi_net = mobilenet_v2(
        #     in_channels=1,
        #     num_classes= self.hparams.num_classes,   # 假设要分5个类
        #         input_length=self.hparams.heq_len,
        #     width_mult=1.0
        # ).to(self.deivce)
        self.rssi_net = mobilevitorigin1d_xs(self.hparams.heq_len, self.hparams.num_classes).to(self.deivce)
        
        self.loss_fn = nn.CrossEntropyLoss()
 

    def forward(self, data):
        rssi_pred = self.rssi_net(data)
        return rssi_pred

    def training_step(self, batch, batch_idx):
        data, targets = batch
        rssi_pred = self(data)
        loss = self.loss_fn(rssi_pred, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,batch_size=self.hparams.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        data, targets = batch
        rssi_pred = self(data)
        loss = self.loss_fn(rssi_pred, targets)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,batch_size=self.hparams.batch_size)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        data, targets = batch
        rssi_pred = self(data)
        loss = self.loss_fn(rssi_pred, targets)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True,batch_size=self.hparams.batch_size)
        
        # Optional: Additional operations such as trajectory reconstruction could be added here if needed
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.hparams.lr_factor, mode="min",patience=self.hparams.lr_patience, verbose=True, eps=self.hparams.lr_eps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }