import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 时间序列数据的数据集类
class TimeSeriesDataset(Dataset):
    def __init__(self, data_dir, sequence_length=12, features=['precipitation', 'temperature', 'supply'], target='consumption', province_id=None):
        """
        参数:
            data_dir (str): 包含数据集文件的目录路径
            sequence_length (int): 输入序列的长度
            features (list): 特征列名的列表
            target (str): 目标列名
            province_id (str or int): 可选的省份ID用于过滤数据
        """
        super().__init__()
        self.sequence_length = sequence_length
        self.features = features
        self.target = target
        self.province_id = province_id
        # 标准化器需要可访问以便后续可能的逆变换
        self.scalers = {} # 为每个省份存储标准化器
        self.data = self._load_and_preprocess_data(data_dir)
        print(self.data)
        self.sequences = self._create_sequences()
        
        # 存储省份名称以了解数据集中包含哪些省份
        self.province_names = list(self.data.keys()) if self.data else []

    def _load_and_preprocess_data(self, data_dir):
        # 如果存在合并后的CSV文件则直接加载
        csv_path = os.path.join(data_dir, 'merged_water_data.csv')
        
        
        try:
            # 加载合并的CSV文件
            df_merged = pd.read_csv(csv_path)
            print(f"从 {csv_path} 加载了合并数据")
        except Exception as e:
            raise RuntimeError(f"加载合并CSV文件时出错: {e}")
      

        # 如果province_id为None，则按年份聚合所有省份的数据
        if self.province_id is None:
            # 创建全国的数据
            print("未指定省份ID，将创建全国汇总数据")
            
            # 确保数值类型列已正确转换
            for col in self.features + [self.target]:
                if col in df_merged.columns:
                    df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')
            
            # 按年份分组并聚合数据
            national_data = df_merged.groupby('year').agg({
                'supply': 'sum',
                'consumption': 'sum',
                # 对温度和降水量取平均值
                'precipitation': 'mean',
                'temperature': 'mean'
            }).reset_index()
            
            # 添加"全国"标识
            national_data['province'] = '全国'
            national_data['province_id'] = 0  # 使用0作为全国的ID
            
            df_merged = national_data
        # 如果提供了province_id则进行过滤
        elif 'province_id' in df_merged.columns:
            df_merged = df_merged[df_merged['province_id'] == self.province_id]
            print(f"已为省份ID过滤数据: {self.province_id}")
            if len(df_merged) == 0:
                raise ValueError(f"未找到省份ID {self.province_id} 的数据")
        else:
            print(f"警告: 在数据中未找到'province_id'列，无法按省份ID {self.province_id} 进行过滤")

        # 将特征和目标转换为数值类型，强制转换错误
        for col in self.features + [self.target]:
            if col in df_merged.columns:
                df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')
            else:
                print(f"警告: 在合并数据中未找到列 '{col}'。")

        # 处理可能由合并或强制转换产生的NaN值
        df_merged.dropna(inplace=True)

        # 为创建序列排序数据
        df_merged.sort_values(by=['province', 'year'], inplace=True)

        # 添加年份作为特征(自2000年起按十年标准化)
        df_merged['year_normalized'] = (df_merged['year'] - 2000) / 10
        if 'year_normalized' not in self.features and 'year' in self.features:
            self.features = [f if f != 'year' else 'year_normalized' for f in self.features]
            print("已将归一化年份添加到特征中")
        
        # --- 标准化(按省份) ---
        processed_data = {}
        feature_cols_to_scale = self.features  # 只标准化特征列，不标准化目标列
        
        # 处理特定省份ID或全国数据的情况
        if self.province_id is not None or 'province' in df_merged.columns:
            provinces_to_process = df_merged['province'].unique()
        else:
            provinces_to_process = ['全国']  # 默认使用全国作为省份名称
            
        for province in provinces_to_process:
            if province == '全国':
                group = df_merged  # 全国数据已经聚合好了
            else:
                group = df_merged[df_merged['province'] == province]
                
            # 确保组足够大以进行标准化和序列创建
            if len(group) > self.sequence_length:
                scaler = MinMaxScaler()
                # 仅对特征列进行标准化，不标准化目标列
                cols_to_scale_in_group = [col for col in feature_cols_to_scale if col in group.columns]
                if cols_to_scale_in_group:
                    group_scaled = group.copy()
                    group_scaled[cols_to_scale_in_group] = scaler.fit_transform(group[cols_to_scale_in_group])
                    self.scalers[province] = scaler # 存储此省份的标准化器
                    processed_data[province] = group_scaled
                else:
                    print(f"警告: 省份 {province} 没有要标准化的列。")
            else:
                print(f"警告: 由于数据不足，跳过省份 {province}（发现 {len(group)} 条数据，需要 > {self.sequence_length} 条）。")

        return processed_data # 返回以省份为键的数据框字典

    def _create_sequences(self):
        sequences = []
        # self.data 现在是一个数据框字典 {province: df}
        for province, df_province in self.data.items():
            # 确保我们使用存在的列名
            feature_cols = [col for col in self.features if col in df_province.columns]
            if not feature_cols:
                print(f"警告: 未找到省份 {province} 的有效特征")
                continue
                
            if self.target not in df_province.columns:
                print(f"警告: 未找到省份 {province} 的目标列 '{self.target}'")
                continue
            
            # 获取特征和目标数据
            feature_values = df_province[feature_cols].values
            target_values = df_province[self.target].values
            num_samples = len(feature_values)
            province_id = df_province['province_id'].iloc[0] if 'province_id' in df_province.columns else None
            
            if num_samples > self.sequence_length:
                for i in range(num_samples - self.sequence_length):
                    # 只使用特征列创建序列，目标值保持原样
                    seq = feature_values[i:i + self.sequence_length]  # 特征序列
                    label = target_values[i + self.sequence_length]   # 未缩放的目标值
                    
                    # 包含元数据以便追踪
                    metadata = {
                        'province': province,
                        'province_id': province_id,
                        'year': df_province['year'].iloc[i + self.sequence_length] if 'year' in df_province.columns else None
                    }
                    
                    sequences.append((seq, label, metadata))
            # else: 省份在预处理中已被跳过
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence, label, metadata = self.sequences[idx]
        # 转换为张量
        features = torch.tensor(sequence, dtype=torch.float32)
        target = torch.tensor(label, dtype=torch.float32).unsqueeze(0) # 确保目标是 [1]
        return features, target
    
    def get_metadata(self, idx):
        """返回特定序列的元数据"""
        return self.sequences[idx][2]

    def get_province_names(self):
        """返回数据集中的省份名称列表"""
        return self.province_names


class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='dataset', sequence_length=12, batch_size=32, num_workers=0, 
                 train_val_test_split=(0.7, 0.15, 0.15), province_id=None):
        super().__init__()
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        self.province_id = province_id
        self.features = ['precipitation', 'temperature', 'year'] # 输入特征更新为包含年份
        self.target = 'consumption' # 目标变量
        # 为数据集添加占位符
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.full_dataset = None # 保留引用以备需要

    def setup(self, stage=None):
        # 设置在每个进程中调用一次
        if not self.train_dataset and not self.val_dataset and not self.test_dataset:
            self.full_dataset = TimeSeriesDataset(
                data_dir=self.data_dir,
                sequence_length=self.sequence_length,
                features=self.features,
                target=self.target,
                province_id=self.province_id
            )
            total_size = len(self.full_dataset)

            if total_size == 0:
                 raise ValueError("处理后数据集为空。检查数据文件和预处理步骤。")
            
            # 提取所有序列的元数据
            sequences_with_metadata = [(idx, self.full_dataset.get_metadata(idx)['year']) 
                                      for idx in range(total_size)]
            
            # 按年份排序索引
            sorted_indices = [idx for idx, _ in sorted(sequences_with_metadata, key=lambda x: x[1])]
            
            # 计算用于拆分的年份
            years = [self.full_dataset.get_metadata(idx)['year'] for idx in sorted_indices]
            min_year = min(years)
            max_year = max(years)
            
            # 使用所有可用年份的数据
            # 计算年份拆分点: 70% 训练, 15% 验证, 15% 测试
            train_end_year = min_year + int((max_year - min_year) * 0.7)
            val_end_year = train_end_year + int((max_year - min_year) * 0.15)
            
            # 根据年份范围拆分数据
            train_indices = [idx for idx in sorted_indices 
                            if self.full_dataset.get_metadata(idx)['year'] <= train_end_year]
            val_indices = [idx for idx in sorted_indices 
                          if self.full_dataset.get_metadata(idx)['year'] > train_end_year 
                          and self.full_dataset.get_metadata(idx)['year'] <= val_end_year]
            test_indices = [idx for idx in sorted_indices 
                           if self.full_dataset.get_metadata(idx)['year'] > val_end_year]
            
            # 创建拆分的数据集子集
            from torch.utils.data import Subset
            self.train_dataset = Subset(self.full_dataset, train_indices)
            self.val_dataset = Subset(self.full_dataset, val_indices)
            self.test_dataset = Subset(self.full_dataset, test_indices)
            
            print(f"已按年份顺序加载数据集，共 {total_size} 个样本")
            print(f"训练: {len(self.train_dataset)} ({int(len(self.train_dataset)/total_size*100)}%), " 
                  f"验证: {len(self.val_dataset)} ({int(len(self.val_dataset)/total_size*100)}%), "
                  f"测试: {len(self.test_dataset)} ({int(len(self.test_dataset)/total_size*100)}%)")
            
            # 显示每个子集的年份范围
            if len(train_indices) > 0:
                train_years = [self.full_dataset.get_metadata(idx)['year'] for idx in train_indices]
                print(f"训练集年份范围: {min(train_years)} - {max(train_years)}")
            
            if len(val_indices) > 0:
                val_years = [self.full_dataset.get_metadata(idx)['year'] for idx in val_indices]
                print(f"验证集年份范围: {min(val_years)} - {max(val_years)}")
            
            if len(test_indices) > 0:
                test_years = [self.full_dataset.get_metadata(idx)['year'] for idx in test_indices]
                print(f"测试集年份范围: {min(test_years)} - {max(test_years)}")
                
            if self.province_id:
                print(f"正在使用省份ID: {self.province_id}")

    def train_dataloader(self):
        if not self.train_dataset:
            self.setup('fit') # 如果直接访问，确保调用setup
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )

    def val_dataloader(self):
        if not self.val_dataset:
             self.setup('validate')
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )

    def test_dataloader(self):
        if not self.test_dataset:
             self.setup('test')
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )
        
    def get_province_names(self):
        """从数据集返回省份名称列表"""
        if self.full_dataset:
            return self.full_dataset.get_province_names()

