import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger
from tensorboardX import SummaryWriter
from dataset import TimeSeriesDataModule # 更新的导入
from cnn_lstm_net_model import CNN_LSTM_Net # 模型类名称相同，但实现已更改
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib


# 指定默认字体
matplotlib.rcParams['font.sans-serif'] = ['SimSun'] 
matplotlib.rcParams['font.family']='sans-serif'
# 解决负号'-'显示为方块的问题
matplotlib.rcParams['axes.unicode_minus'] = False 



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
    
    #添加模型检查点
    callbacks.append(plc.ModelCheckpoint(
        monitor= 'val_loss',
        filename=f"water-consumption-best-{{epoch:02d}}-{{val_loss:.3f}}",
        save_top_k=1,
        mode='min',
        verbose=True,
        save_last=True
    ))

    return callbacks


def get_model(args):
    # 传递必要的超参数到更新后的模型
    return CNN_LSTM_Net(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_size=args.output_size,
        lr=args.lr,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        lr_eps=args.lr_eps,
        use_attention=args.use_attention,
        dropout_rate=args.dropout_rate
    )


def get_data_module(args):
    # 实例化TimeSeriesDataModule，如果指定了province_id则包含它
    data_module = TimeSeriesDataModule(
        data_dir=args.data_path,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        province_id=args.province_id
    )
    return data_module

def train(args):
    # 创建模型和数据模块
    model = get_model(args)
    data_module = get_data_module(args)

    # 如果提供了省份信息，则设置日志记录
    province_info = f"_province_{args.province_id}" if args.province_id else ""
    
    logger = TensorBoardLogger(
        save_dir="./logs",
        name=f"water_consumption_lstm{province_info}"
    )

    # 配置训练器
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices, 
        logger=logger, 
        callbacks=load_callbacks(), 
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        fast_dev_run=args.fast_dev_run
    )

    # 训练模型
    trainer.fit(model=model, datamodule=data_module)
    
    # 训练后测试模型
    trainer.test(model=model, datamodule=data_module)
    
    # 保存最佳检查点的路径
    if trainer.checkpoint_callback.best_model_path:
        print(f"最佳模型保存于: {trainer.checkpoint_callback.best_model_path}")
        
        # 如果指定了预测年份，进行未来预测
        if args.predict_years and args.province_id:
            predict_future_consumption(
                model=model,
                data_module=data_module,
                province_id=args.province_id,
                years=args.predict_years,
                checkpoint_path=trainer.checkpoint_callback.best_model_path
            )

def test(args):
    logger = TensorBoardLogger(
        save_dir="./logs",
        name=f"water_consumption_lstm_test"
    )

    # 从检查点加载模型
    if not os.path.exists(args.cpt_path):
        raise FileNotFoundError(f"未找到检查点文件: {args.cpt_path}")

    model = CNN_LSTM_Net.load_from_checkpoint(
        args.cpt_path,
        map_location=torch.device('cpu' if args.accelerator=="cpu" else 'cuda')
    )
    
    # 创建数据模块
    data_module = get_data_module(args)

    # 为测试配置训练器
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices, 
        logger=logger
    )

    # 测试模型
    trainer.test(model=model, datamodule=data_module)
    
    # 如果指定了预测年份，进行未来预测
    if args.predict_years and args.province_id:
        predict_future_consumption(
            model=model,
            data_module=data_module,
            province_id=args.province_id,
            years=args.predict_years,
            checkpoint_path=args.cpt_path
        )

def list_provinces(args):
    """列出数据集中所有可用的省份及其ID"""
    # 创建临时数据模块以访问省份信息
    data_module = TimeSeriesDataModule(data_dir=args.data_path)
    
    # 强制设置以加载数据
    data_module.setup()
    
    # 获取并打印省份信息
    if data_module.full_dataset and hasattr(data_module.full_dataset, 'data'):
        # 加载合并的数据文件以获取省份ID和名称
        try:
            csv_path = os.path.join(args.data_path, 'merged_water_data.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                province_info = df[['province_id', 'province']].drop_duplicates()
                print("\n可用省份:")
                print("-------------------")
                for _, row in province_info.sort_values('province_id').iterrows():
                    print(f"ID: {row['province_id']}, 名称: {row['province']}")
            else:
                print("找不到merged_water_data.csv文件。")
        except Exception as e:
            print(f"加载省份信息时出错: {e}")
            # 回退到仅显示可用省份而不显示ID
            print("\n可用省份 (仅名称):")
            for province in data_module.get_province_names():
                print(f"- {province}")
    else:
        print("没有可用的省份数据。")

def predict_future_consumption(model, data_module, province_id, years, checkpoint_path=None):
    """
    为特定省份进行未来预测
    
    参数:
        model: 训练好的模型或检查点路径
        data_module: 包含数据集的数据模块
        province_id: 要预测的省份ID
        years: 要预测的未来年份列表
        checkpoint_path: 可选的模型检查点路径
    """
    # 确保如果提供了检查点路径，则加载模型
    if checkpoint_path and isinstance(checkpoint_path, str):
        try:
            model = CNN_LSTM_Net.load_from_checkpoint(checkpoint_path)
            print(f"从检查点加载模型: {checkpoint_path}")
        except Exception as e:
            print(f"从检查点加载模型时出错: {e}")
            return
    
    # 确保数据模块已设置
    if not data_module.full_dataset:
        data_module.setup()
    
    # 如果年份是字符串，则转换为整数列表
    if isinstance(years, str):
        try:
            # 解析逗号分隔的年份
            years = [int(y.strip()) for y in years.split(",")]
            print(f"预测年份: {years}")
        except ValueError:
            print("无效的年份格式。请提供逗号分隔的年份（例如，'2024,2025,2026'）")
            return
    
    # 获取省份的最新可用数据
    if not data_module.full_dataset or not hasattr(data_module.full_dataset, 'data'):
        print("没有可用于预测的数据集")
        return
    
    # 在数据集中查找省份数据
    province_data = None
    province_name = "未知"
    for name, df in data_module.full_dataset.data.items():
        # 检查这个数据框是否属于我们的目标省份
        if 'province_id' in df.columns and province_id in df['province_id'].values:
            province_data = df
            province_name = name
            break
    
    if province_data is None:
        print(f"未找到省份ID {province_id} 的数据")
        return
    
    # 获取训练中使用的序列长度
    seq_length = data_module.sequence_length
    
    # 按年份排序数据以获取最新年份
    province_data = province_data.sort_values('year')
    
    # 获取训练中使用的特征
    features = data_module.features
    
    # 从最新可用数据准备输入序列
    if len(province_data) < seq_length:
        print(f"省份 {province_name} (ID: {province_id}) 的历史数据不足")
        return
    
    # 获取最新的数据序列
    latest_data = province_data.iloc[-seq_length:][features].values
    
    # 为未来预测归一化年份（假设包含年份特征）
    future_years_norm = [(year - 2000) / 10 for year in years]
    
    # 准备模型的输入张量
    input_sequence = torch.tensor(latest_data).unsqueeze(0).float()  # 添加批次维度
    
    # 进行预测
    model.eval()  # 设置模型为评估模式
    device = next(model.parameters()).device
    input_sequence = input_sequence.to(device)
    
    # 使用我们增强模型中的predict_future方法
    predictions = model.predict_future(
        initial_sequence=input_sequence,
        future_steps=len(years),
        future_years=years,
        device=device
    )
    
    # 创建结果数据框
    results = pd.DataFrame({
        'Year': years,
        'Predicted_Consumption': predictions
    })
    
    print(f"\n{province_name} (ID: {province_id}) 的未来水资源消耗预测:")
    print(results.to_string(index=False))
    
    # 可视化结果
    try:
        # 获取历史数据作为上下文
        historical_years = province_data['year'].tolist()
        historical_consumption = province_data['consumption'].tolist()
        
        plt.figure(figsize=(12, 6))
        
        # 绘制历史数据
        plt.plot(historical_years, historical_consumption, 'o-', label='历史数据')
        
        # 绘制预测
        plt.plot(years, predictions, 'x--', color='red', label='预测')
        
        # 添加标签和标题
        plt.xlabel('年份')
        plt.ylabel('水资源消耗量')
        plt.title(f'{province_name} (ID: {province_id}) 的水资源消耗预测')
        plt.grid(True)
        plt.legend()
        
        # 保存图表
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = 'predictions'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{province_name}_id{province_id}_{timestamp}.png'))
        print(f"预测图表已保存到 {output_dir}")
        
        plt.close()
    except Exception as e:
        print(f"可视化预测时出错: {e}")
    
    # 将预测结果保存到CSV
    try:
        # 创建包含历史和预测数据的数据框
        all_data = pd.DataFrame({
            'Year': historical_years + years,
            'Consumption': historical_consumption + [None] * len(years),
            'Predicted': [None] * len(historical_years) + predictions,
            'Province': province_name,
            'Province_ID': province_id
        })
        
        # 保存到CSV
        csv_file = os.path.join(output_dir, f'{province_name}_id{province_id}_{timestamp}.csv')
        all_data.to_csv(csv_file, index=False)
        print(f"预测数据已保存到 {csv_file}")
    except Exception as e:
        print(f"保存预测数据时出错: {e}")


if __name__ == '__main__':
    parser = ArgumentParser()
    # 基本训练控制参数
    parser.add_argument('--batch_size', default=32, type=int, help='批次大小')
    parser.add_argument('--num_workers', default=0, type=int, help='数据加载器使用的工作线程数')
    parser.add_argument('--seed', default=1234, type=int, help='随机种子，确保结果可复现')
    parser.add_argument('--lr', default=1e-3, type=float, help='学习率')
    parser.add_argument('--accelerator', default="gpu", type=str, help='训练加速器类型（gpu 或 cpu）')
    parser.add_argument('--devices', default=1, type=int, help='使用的设备数量')
    parser.add_argument('--min_epochs', default=50, type=int, help='最小训练轮数')
    parser.add_argument('--max_epochs', default=200, type=int, help='最大训练轮数')
    parser.add_argument('--fast_dev_run', default=False, type=bool, help='快速开发运行模式，用于调试')

    # 学习率调度器参数
    parser.add_argument('--lr_factor', default=0.1, type=float, help='学习率衰减因子')
    parser.add_argument('--lr_patience', default=5, type=int, help='学习率调整前的等待轮数')
    parser.add_argument('--lr_eps', default=1e-12, type=float, help='学习率衰减的最小阈值')

    # 时间序列数据模块参数
    parser.add_argument('--data_path', default='dataset', type=str, help='数据文件路径')
    parser.add_argument('--sequence_length', default=5, type=int, help='LSTM输入序列长度')

    # LSTM模型参数 (CNN_LSTM_Net)
    parser.add_argument('--input_size', default=4, type=int, help='输入特征数量（降水量、温度、标准化年份、供水量）')
    parser.add_argument('--hidden_size', default=64, type=int, help='LSTM隐藏单元数量')
    parser.add_argument('--num_layers', default=2, type=int, help='LSTM层数')
    parser.add_argument('--output_size', default=1, type=int, help='输出值数量（预测的消耗量）')
    parser.add_argument('--dropout_rate', default=0.2, type=float, help='用于正则化的dropout比率')
    parser.add_argument('--use_attention', default=True, type=bool, help='是否使用注意力机制')

    # 省份特定参数
    parser.add_argument('--province_id', type=int, help='要训练/预测的省份ID')
    parser.add_argument('--predict_years', type=str, help='预测水资源消耗的年份，以逗号分隔（例如："2024,2025,2026"）',default="2024,2025,2026")

    # 重启控制参数
    parser.add_argument('--load_best', action='store_true', help='加载最佳模型')
    parser.add_argument('--load_dir', default=None, type=str, help='加载模型的目录')
    parser.add_argument('--load_ver', default=None, type=str, help='加载模型的版本')
    parser.add_argument('--load_v_num', default=None, type=int, help='加载模型的版本号')

    # 额外选项
    parser.add_argument('--mode', choices=['train', 'test', 'list_provinces'], type=str, default='test', help='操作模式')
    parser.add_argument('--cpt_path', default=os.getcwd()+'/logs/water_consumption_lstm/version_0/checkpoints/water-consumption-best-epoch=04-val_loss=0.015.ckpt', type=str, help='用于测试或恢复训练的检查点路径')

    args = parser.parse_args()

    # 确保测试模式下提供了检查点路径
    if args.mode == 'test' and not args.cpt_path:
        raise ValueError("测试模式下必须提供检查点路径 (--cpt_path)。")
    
    # 处理省份ID（如果以字符串形式提供，用于命令行使用）
    if hasattr(args, 'province_id') and args.province_id is not None:
        if isinstance(args.province_id, str):
            try:
                args.province_id = int(args.province_id)
            except ValueError:
                print(f"警告：无效的province_id格式：{args.province_id}。使用None代替。")
                args.province_id = None

    # 以指定模式运行
    if args.mode == 'test':
        test(args)
    elif args.mode == 'train':
        train(args)
    elif args.mode == 'list_provinces':
        list_provinces(args)
    else:
        print(f"未知模式：{args.mode}")