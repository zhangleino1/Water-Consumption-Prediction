import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
import numpy as np
import os

class CNN_LSTM_Net(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size, lr, lr_factor, lr_patience, lr_eps):
        super().__init__()
        # Save hyperparameters
        self.save_hyperparameters()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lr = lr
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.lr_eps = lr_eps

        # LSTM layer
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True) # batch_first=True expects input shape (batch, seq_len, features)

        # Output layer
        self.fc = nn.Linear(self.hidden_size, self.output_size)

        self.test_preds = []
        self.test_targets = []

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # We only need the output from the last time step
        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        last_time_step_out = lstm_out[:, -1, :]
        # last_time_step_out shape: (batch_size, hidden_size)
        output = self.fc(last_time_step_out)
        # output shape: (batch_size, output_size)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.lr_factor,
            patience=self.lr_patience,
            eps=self.lr_eps,
            verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss', # Monitor validation loss for scheduler
            },
        }

    def _common_step(self, batch, batch_idx):
        features, targets = batch
        # features shape: (batch_size, sequence_length, input_size)
        # targets shape: (batch_size, 1)
        outputs = self(features)
        # outputs shape: (batch_size, output_size) which should be (batch_size, 1)
        loss = F.mse_loss(outputs, targets)
        return loss, outputs, targets

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, outputs, targets = self._common_step(batch, batch_idx)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # Store predictions and targets for potential analysis later
        self.test_preds.append(outputs.detach().cpu().numpy())
        self.test_targets.append(targets.detach().cpu().numpy())
        return loss

    def on_test_epoch_end(self):
        # Example: Calculate overall MSE on test set
        if not self.test_preds or not self.test_targets:
            print("No test predictions or targets collected.")
            return

        all_preds = np.concatenate(self.test_preds)
        all_targets = np.concatenate(self.test_targets)

        test_mse = np.mean((all_preds - all_targets)**2)
        print(f"\nOverall Test MSE: {test_mse:.4f}")

        # You could add plotting or other analysis here if needed
        # For example, plot predictions vs actuals
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 6))
        # plt.plot(all_targets, label='Actual Consumption')
        # plt.plot(all_preds, label='Predicted Consumption', alpha=0.7)
        # plt.title('Test Set: Actual vs Predicted Water Consumption')
        # plt.xlabel('Sample Index')
        # plt.ylabel('Normalized Consumption')
        # plt.legend()
        # plt.grid(True)
        # plt.savefig(os.path.join(self.logger.log_dir, 'test_predictions_vs_actuals.png'))
        # plt.close()

        # Clear lists for next potential test run
        self.test_preds = []
        self.test_targets = []
        print('Finished test epoch.')