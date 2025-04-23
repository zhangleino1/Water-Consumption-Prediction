
import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger
from tensorboardX import SummaryWriter
from dataset import BeaconDataModule
from RSSIOnlyLocalization import RSSIOnlyLocalization
import torch

def load_callbacks():
    callbacks = []
    callbacks.append(plc.EarlyStopping(
       monitor= 'val_loss',
        patience=10,
        min_delta=0.0001,
        verbose=True
    ))
    

    callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))
    
    #add ModelCheckpoint
    callbacks.append(plc.ModelCheckpoint(
        monitor= 'val_loss',
        filename=f"rssi-best-{{epoch:02d}}-{{val_loss:.3f}}",
        save_top_k=1,
        mode='min',
        verbose=True,
        save_last=True
    ))

    return callbacks


def get_model(args):
    return RSSIOnlyLocalization(hparams=args)


def get_data_module(args):

    data_module = BeaconDataModule(batch_size=args.batch_size, num_workers=args.num_workers)
    return data_module

def train(args):
    # pl.seed_everything(args.seed)

    model = get_model(args)
    # # If you want to change the logger's saving folder
    logger = TensorBoardLogger(
        save_dir="./logs",
        name=f"rssi"
    )

    trainer = pl.Trainer(accelerator=args.accelerator,devices=args.devices, logger=logger, callbacks=load_callbacks(), min_epochs=args.min_epochs,max_epochs=args.max_epochs
                         ,fast_dev_run=args.fast_dev_run )

    trainer.fit(model=model, datamodule=get_data_module(args))

def test(args):
    # pl.seed_everything(args.seed)
    logger = TensorBoardLogger(
        save_dir="./logs",
        name=f"{args.model_type}test"
    )

     # 将 Namespace 对象转换为字典
    hparams_dict = vars(args)

    model = RSSIOnlyLocalization.load_from_checkpoint(args.cpt_path,**hparams_dict)

    
    # # If you want to change the logger's saving folder

    trainer = pl.Trainer(accelerator=args.accelerator,devices=args.devices, logger=logger, callbacks=load_callbacks(), min_epochs=args.min_epochs,max_epochs=args.max_epochs
                         ,fast_dev_run=args.fast_dev_run)

    trainer.test(model=model, datamodule=get_data_module(args))

import torch
import os

def to_onnx(args):
    # torch model to onnx
    model = RSSIOnlyLocalization.load_from_checkpoint(args.cpt_path)
    model.eval()

    # 创建一个示例输入
    example_input = torch.randn(1, 1, args.heq_len).to(model.device)

    # Export the model to ONNX
    onnx_path = os.path.join(os.getcwd(), "model.onnx")

    torch.onnx.export(
        model.rssi_net,
        example_input,
        onnx_path,
        # opset_version=11,  # Important: Specify the opset version (11 or higher is recommended)
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None
    )
    print(f"Model has been converted to ONNX and saved at {onnx_path}")



if __name__ == '__main__':
    parser = ArgumentParser()
    # Basic Training Control
    # Basic Training Control
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--accelerator', default="gpu", type=str)
    parser.add_argument('--devices', default=1, type=int)
    parser.add_argument('--min_epochs', default=50, type=int)
    parser.add_argument('--max_epochs', default=200, type=int)
    parser.add_argument('--fast_dev_run', default=False, type=bool)

    # LR Scheduler
    parser.add_argument('--lr_factor', default=0.1, type=float)
    parser.add_argument('--lr_patience', default=5, type=int)
    parser.add_argument('--lr_eps', default=1e-12, type=float)

    parser.add_argument('--heq_len', default=8, type=int)
    parser.add_argument('--num_classes', default=7, type=int)

    # Restart Control
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--load_v_num', default=None, type=int)

    # Extra
    parser.add_argument('--mode', choices=['test', 'train','onnx'], type=str, default='onnx')
    parser.add_argument('--cpt_path', default=os.getcwd()+"/logs/rssi/version_5/checkpoints/last.ckpt", type=str)

    args = parser.parse_args()

    if args.mode == 'test':
        test(args)
    if args.mode == 'onnx':
        to_onnx(args)
    else:
        train(args)