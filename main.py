
import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger
from tensorboardX import SummaryWriter
from dataset import TimeSeriesDataModule # Updated import
from cnn_lstm_net_model import CNN_LSTM_Net # Model class name is the same, but implementation changed
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
    # Pass necessary hyperparameters to the updated model
    return CNN_LSTM_Net(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_size=args.output_size,
        lr=args.lr,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        lr_eps=args.lr_eps
    )


def get_data_module(args):
    # Instantiate the new TimeSeriesDataModule
    data_module = TimeSeriesDataModule(
        data_path=args.data_path,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    return data_module

def train(args):
    # pl.seed_everything(args.seed)

    model = get_model(args)
    # # If you want to change the logger's saving folder
    logger = TensorBoardLogger(
        save_dir="./logs",
        name=f"water_consumption_lstm" # Updated logger name
    )

    trainer = pl.Trainer(accelerator=args.accelerator,devices=args.devices, logger=logger, callbacks=load_callbacks(), min_epochs=args.min_epochs,max_epochs=args.max_epochs
                         ,fast_dev_run=args.fast_dev_run )

    trainer.fit(model=model, datamodule=get_data_module(args))

def test(args):
    # pl.seed_everything(args.seed)
    logger = TensorBoardLogger(
        save_dir="./logs",
        name=f"water_consumption_lstm_test" # Updated logger name for test
    )

     # 将 Namespace 对象转换为字典
    hparams_dict = vars(args)

    # Use the new model class for loading checkpoint
    # Ensure checkpoint is compatible with the new model structure
    # You might need to retrain or ensure hparams match
    # For now, assuming checkpoint loading works with saved hyperparameters
    model = CNN_LSTM_Net.load_from_checkpoint(args.cpt_path, hparams_file=os.path.join(os.path.dirname(args.cpt_path), 'hparams.yaml'))
    # Update model hparams if needed, though load_from_checkpoint usually handles this
    model.hparams.update(hparams_dict)

    
    # # If you want to change the logger's saving folder

    trainer = pl.Trainer(accelerator=args.accelerator,devices=args.devices, logger=logger, callbacks=load_callbacks(), min_epochs=args.min_epochs,max_epochs=args.max_epochs
                         ,fast_dev_run=args.fast_dev_run)

    trainer.test(model=model, datamodule=get_data_module(args))

# import torch # torch already imported
# import os # os already imported

# def to_onnx(args):
#     # This function needs significant updates for the new model structure
#     # Input shape and model architecture have changed.
#     # Commenting out for now.
#     print("ONNX export function is currently disabled due to model changes.")
#     pass
    # # torch model to onnx
    # model = CNN_LSTM_Net.load_from_checkpoint(args.cpt_path)
    # model.eval()
    #
    # # Create a sample input matching the new model's expected input
    # # Shape: (batch_size, sequence_length, input_size)
    # example_input = torch.randn(1, args.sequence_length, args.input_size).to(model.device)
    #
    # # Export the model to ONNX
    # onnx_path = os.path.join(os.getcwd(), "water_consumption_model.onnx")
    #
    # try:
    #     torch.onnx.export(
    #         model, # Export the whole LightningModule or the specific nn.Module part
    #         example_input,
    #         onnx_path,
    #         input_names=['input_sequence'],
    #         output_names=['predicted_consumption'],
    #         dynamic_axes={'input_sequence': {0: 'batch_size'}, # Optional: if batch size can vary
    #                       'predicted_consumption': {0: 'batch_size'}}
    #     )
    #     print(f"Model has been converted to ONNX and saved at {onnx_path}")
    # except Exception as e:
    #     print(f"Error during ONNX export: {e}")



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

    # Arguments for TimeSeriesDataModule
    parser.add_argument('--data_path', default='data.csv', type=str, help='Path to the data file')
    parser.add_argument('--sequence_length', default=12, type=int, help='Length of the input sequence for LSTM')

    # Arguments for the LSTM Model (CNN_LSTM_Net)
    parser.add_argument('--input_size', default=3, type=int, help='Number of input features (precipitation, temp, supply)')
    parser.add_argument('--hidden_size', default=50, type=int, help='Number of hidden units in LSTM')
    parser.add_argument('--num_layers', default=1, type=int, help='Number of LSTM layers')
    parser.add_argument('--output_size', default=1, type=int, help='Number of output values (predicted consumption)')

    # parser.add_argument('--heq_len', default=8, type=int) # Obsolete
    # parser.add_argument('--num_classes', default=7, type=int) # Obsolete

    # Restart Control
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--load_v_num', default=None, type=int)

    # Extra
    parser.add_argument('--mode', choices=['test', 'train'], type=str, default='train', help='Operation mode: train or test') # Removed 'onnx'
    # Update default checkpoint path or make it required for test mode
    parser.add_argument('--cpt_path', default=None, type=str, help='Path to checkpoint for testing or resuming training')

    args = parser.parse_args()

    # Ensure checkpoint path is provided for test mode
    if args.mode == 'test' and not args.cpt_path:
        raise ValueError("Checkpoint path (--cpt_path) must be provided for test mode.")
    # Optionally load from checkpoint if provided in train mode (for resuming)
    ckpt_path_arg = {'ckpt_path': args.cpt_path} if args.cpt_path and args.mode == 'train' else {}

    if args.mode == 'test':
        test(args)
    # if args.mode == 'onnx': # ONNX mode removed
    #     to_onnx(args)
    elif args.mode == 'train':
        # Pass ckpt_path to trainer.fit if resuming
        trainer.fit(model=model, datamodule=data_module, **ckpt_path_arg)
    else:
        print(f"Unknown mode: {args.mode}")