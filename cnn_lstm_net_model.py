import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
import numpy as np
import os
import matplotlib.pyplot as plt

class CNN_LSTM_Net(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size, lr, lr_factor, lr_patience, lr_eps, 
                 use_attention=True, dropout_rate=0.2):
        super().__init__()
        # 保存超参数
        self.save_hyperparameters()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lr = lr
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.lr_eps = lr_eps
        self.use_attention = use_attention
        self.dropout_rate = dropout_rate
        
        # 深度增加的CNN组件
        # 第一层1D CNN
        self.conv1 = nn.Conv1d(
            in_channels=input_size,  # 输入特征数量
            out_channels=32,         # 输出通道数量
            kernel_size=3,           # 卷积核大小
            padding=1                # 相同填充
        )
        
        # 第二层1D CNN
        self.conv2 = nn.Conv1d(
            in_channels=32,          # 匹配conv1的输出通道
            out_channels=64,         # 增加通道数
            kernel_size=3,           # 卷积核大小
            padding=1                # 相同填充
        )
        
        # 第三层1D CNN用于更深层特征提取
        self.conv3 = nn.Conv1d(
            in_channels=64,          # Match output channels of conv2
            out_channels=128,        # 增加通道数
            kernel_size=3,           # 卷积核大小
            padding=1                # 相同填充
        )
        
        # CNN层后的批量归一化
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(128)
        
        # 用于正则化的Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # 双向LSTM用于更好的序列建模
        self.lstm = nn.LSTM(
            input_size=128,           # CNN输出通道
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,      # 使用双向LSTM
            dropout=dropout_rate if num_layers > 1 else 0  # LSTM层间的Dropout
        )
        
        # 注意力机制
        if use_attention:
            # LSTM后的自注意力层
            self.attention = nn.MultiheadAttention(
                embed_dim=self.hidden_size * 2,  # *2表示双向
                num_heads=4,
                dropout=dropout_rate
            )

        # 带跳跃连接的最终全连接层
        self.fc1 = nn.Linear(self.hidden_size * 2, self.hidden_size)  # *2表示双向
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

        self.test_preds = []
        self.test_targets = []
        self.test_metadata = []
        
        # 层归一化用于稳定训练
        self.layer_norm = nn.LayerNorm(self.hidden_size)

    def forward(self, x):
        batch_size, seq_length, features = x.size()
        
        # 重塑用于CNN: (batch_size, input_size, sequence_length)
        x_cnn = x.permute(0, 2, 1)
        
        # 应用CNN层，使用ReLU激活和批量归一化
        x_cnn = F.relu(self.batch_norm1(self.conv1(x_cnn)))
        x_cnn = F.relu(self.batch_norm2(self.conv2(x_cnn)))
        x_cnn = F.relu(self.batch_norm3(self.conv3(x_cnn)))
        
        # 应用dropout进行正则化
        x_cnn = self.dropout(x_cnn)
        
        # 重塑回LSTM格式: (batch_size, sequence_length, cnn_output_channels)
        x_lstm = x_cnn.permute(0, 2, 1)
        
        # 应用LSTM
        lstm_out, (h_n, c_n) = self.lstm(x_lstm)
        
        # 如果启用则应用注意力机制
        if self.use_attention:
            # 准备多头注意力 (seq_len, batch_size, hidden_size*2)
            lstm_out_for_attn = lstm_out.permute(1, 0, 2)
            
            # 自注意力机制
            attn_output, _ = self.attention(
                lstm_out_for_attn, lstm_out_for_attn, lstm_out_for_attn
            )
            
            # 返回原始形状 (batch_size, seq_len, hidden_size*2)
            attn_output = attn_output.permute(1, 0, 2)
            
            # 使用带有注意力的最后时间步输出
            output_features = attn_output[:, -1, :]
        else:
            # 如果没有注意力，只使用最后时间步
            output_features = lstm_out[:, -1, :]
        
        # 应用dropout进行正则化
        output_features = self.dropout(output_features)
        
        # 应用全连接层，带有跳跃连接和层归一化
        fc1_out = self.fc1(output_features)
        fc1_out = self.layer_norm(fc1_out)
        fc1_out = F.relu(fc1_out)
        fc1_out = self.dropout(fc1_out)
        
        # 最终输出预测
        output = self.fc2(fc1_out)
        
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
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
                'monitor': 'val_loss', # 监控验证损失来调整学习率
            },
        }

    def _common_step(self, batch, batch_idx):
        features, targets = batch
        # 特征形状: (batch_size, sequence_length, input_size)
        # 目标形状: (batch_size, 1)
        outputs = self(features)
        # 输出形状: (batch_size, output_size) 应该是 (batch_size, 1)
        loss = F.mse_loss(outputs, targets)
        return loss, outputs, targets

    def training_step(self, batch, batch_idx):
        loss, outputs, targets = self._common_step(batch, batch_idx)
        # 计算额外的评估指标
        mae = F.l1_loss(outputs, targets)
        # 记录指标
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_mae', mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, outputs, targets = self._common_step(batch, batch_idx)
        # 计算额外的评估指标
        mae = F.l1_loss(outputs, targets)
        # 记录指标
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mae', mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, outputs, targets = self._common_step(batch, batch_idx)
        # 计算额外的评估指标
        mae = F.l1_loss(outputs, targets)
        # 记录指标
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_mae', mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # 存储预测值和目标值，用于后续分析
        self.test_preds.append(outputs.detach().cpu().numpy())
        self.test_targets.append(targets.detach().cpu().numpy())
        
        # 尝试获取元数据（如果可用）
        dataset = self.trainer.datamodule.test_dataset.dataset
        indices = self.trainer.datamodule.test_dataset.indices
        if hasattr(dataset, 'get_metadata') and batch_idx < len(indices):
            try:
                metadata = [dataset.get_metadata(indices[i + batch_idx * self.trainer.datamodule.batch_size]) 
                           for i in range(min(len(outputs), len(indices) - batch_idx * self.trainer.datamodule.batch_size))]
                self.test_metadata.extend(metadata)
            except (AttributeError, IndexError) as e:
                print(f"无法获取元数据: {e}")
        
        return loss

    def on_test_epoch_end(self):
        # 示例：计算测试集上的整体评估指标
        if not self.test_preds or not self.test_targets:
            print("没有收集到测试预测值或目标值。")
            return

        all_preds = np.concatenate(self.test_preds)
        all_targets = np.concatenate(self.test_targets)

        # 计算评估指标
        test_mse = np.mean((all_preds - all_targets)**2)
        test_mae = np.mean(np.abs(all_preds - all_targets))
        test_rmse = np.sqrt(test_mse)
        
        # 打印评估指标
        print(f"\n测试指标:")
        print(f"MSE: {test_mse:.4f}")
        print(f"MAE: {test_mae:.4f}")
        print(f"RMSE: {test_rmse:.4f}")

        # 如果可能，创建可视化
        if self.logger and hasattr(self.logger, 'experiment') and len(all_preds) > 0:
            try:
                # 绘制预测值与实际值对比图
                plt.figure(figsize=(12, 6))
                plt.plot(all_targets, label='实际用水量', marker='o')
                plt.plot(all_preds, label='预测用水量', marker='x')
                plt.title('测试集：实际与预测水资源消耗对比')
                plt.xlabel('样本索引')
                plt.ylabel('水资源消耗量')
                plt.legend()
                plt.grid(True)
                
                # 保存图表
                log_dir = self.logger.log_dir if hasattr(self.logger, 'log_dir') else '.'
                plt.savefig(os.path.join(os.getcwd(), 'results/test_predictions_vs_actuals.png'))
                plt.close()
                
                # 如果元数据可用，创建时间序列图
                if self.test_metadata and len(self.test_metadata) == len(all_preds):
                    # 获取年份（如果可用）并确保它们是整数
                    years = []
                    for i, item in enumerate(self.test_metadata):
                        year = item.get('year', i)
                        # 确保年份是整数
                        if isinstance(year, (int, float)):
                            years.append(int(year))
                        else:
                            years.append(i)
                    
                    province = self.test_metadata[0].get('province', 'Unknown')
                    
                    # 绘制时间序列
                    plt.figure(figsize=(12, 6))
                    plt.plot(years, all_targets, label='实际用水量', marker='o')
                    plt.plot(years, all_preds, label='预测用水量', marker='x')
                    plt.title(f'省份 {province}：实际与预测水资源消耗随时间变化')
                    plt.xlabel('年份')
                    plt.ylabel('水资源消耗量')
                    plt.legend()
                    plt.grid(True)
                    
                    # 使用整数刻度
                    plt.xticks(years)
                    
                    # 保存时间序列图
                    plt.savefig(os.path.join(os.getcwd(), f'results/province_{province}_time_series.png'))
                    plt.close()
                    
            except Exception as e:
                print(f"创建可视化时出错: {e}")

        # 清空列表，为下一次可能的测试运行做准备
        self.test_preds = []
        self.test_targets = []
        self.test_metadata = []
        print('测试轮次完成。')
        
    def predict_future(self, initial_sequence, future_steps=5, future_years=None, device='cpu'):
        """
        基于初始序列预测未来值
        
        参数:
            initial_sequence: 初始输入序列张量，形状为 [1, seq_len, features]
            future_steps: 预测未来的步数
            future_years: 可选的未来年份列表，用于特征生成
            device: 运行预测的设备 ('cpu' 或 'cuda')
            
        返回:
            形状为 [future_steps] 的预测张量
        """
        self.eval()  # 设置模型为评估模式
        self.to(device)  # 将模型移动到指定设备
        
        # 将输入移动到设备
        if isinstance(initial_sequence, np.ndarray):
            initial_sequence = torch.tensor(initial_sequence, dtype=torch.float32)
        
        initial_sequence = initial_sequence.to(device)
        
        # 使用提供的序列初始化
        current_sequence = initial_sequence.clone()
        predictions = []
        
        with torch.no_grad():  # 预测不需要梯度计算
            for i in range(future_steps):
                # 获取下一步的预测
                output = self(current_sequence)
                predictions.append(output.item() if output.numel() == 1 else output[0].item())
                
                # 更新序列用于下一次预测（滚动窗口方法）
                # 移除第一个时间步并附加预测作为新的时间步
                next_input = current_sequence.clone()
                next_input = next_input[:, 1:, :]  # 移除第一个时间步
                
                # 为下一个时间步创建特征向量
                # 如果有年份信息则使用它；否则复制最后的特征
                if future_years and i < len(future_years):
                    # 需要创建适当的特征向量，包含归一化年份等
                    # 这取决于你的特征工程逻辑
                    next_timestep = next_input[:, -1, :].clone()
                    # 如果包含年份特征，则更新它
                    year_idx = 2  # 假设年份是第3个特征（索引2）
                    if next_timestep.size(1) > year_idx:
                        # 类似于训练数据的年份归一化
                        next_timestep[:, year_idx] = (future_years[i] - 2000) / 10
                else:
                    next_timestep = next_input[:, -1, :].clone()
                
                # 更新预测值（假设上一步的目标是最后一个特征）
                if next_timestep.size(1) > 3:  # 如果供水量作为特征包含（索引3）
                    next_timestep[:, 3] = output  # 将供水量设置为当前预测的消费量
                
                # 将新的时间步添加到序列中
                next_input = torch.cat([next_input, next_timestep.unsqueeze(1)], dim=1)
                current_sequence = next_input
                
        return predictions