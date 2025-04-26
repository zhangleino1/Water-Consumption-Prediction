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
        self.sequences = self._create_sequences()
        
        # 存储省份名称以了解数据集中包含哪些省份
        self.province_names = list(self.data.keys()) if self.data else []

    def _load_and_preprocess_data(self, data_dir):
        # 如果存在合并后的CSV文件则直接加载
        csv_path = os.path.join(data_dir, 'merged_water_data.csv')
        
        if os.path.exists(csv_path):
            try:
                # 加载合并的CSV文件
                df_merged = pd.read_csv(csv_path)
                print(f"从 {csv_path} 加载了合并数据")
            except Exception as e:
                raise RuntimeError(f"加载合并CSV文件时出错: {e}")
        else:
            # 回退到处理单独的文件
            water_data_path = os.path.join(data_dir, '15-分省水资源供水和用水数据.xls')
            prec_data_path = os.path.join(data_dir, '省逐年降水.xlsx')
            temp_data_path = os.path.join(data_dir, '省逐年平均气温.xlsx')

            try:
                # 加载水资源供水和消耗数据 (.xls)
                df_water = pd.read_excel(water_data_path, engine='xlrd', sheet_name=0, header=2)
                df_water = df_water.rename(columns={'地区': 'province', '年份': 'year', '供水总量': 'supply', '用水总量': 'consumption'})
                df_water = df_water[['province', 'year', 'supply', 'consumption']]

                # 加载降水数据 (.xlsx)
                df_prec = pd.read_excel(prec_data_path, sheet_name=0, header=0)
                df_prec = df_prec.rename(columns={'省份': 'province', '年份': 'year', '降水量': 'precipitation'})
                df_prec = df_prec[['province', 'year', 'precipitation']]

                # 加载温度数据 (.xlsx)
                df_temp = pd.read_excel(temp_data_path, sheet_name=0, header=0)
                df_temp = df_temp.rename(columns={'省份': 'province', '年份': 'year', '平均气温': 'temperature'})
                df_temp = df_temp[['province', 'year', 'temperature']]

                # --- 数据清洗 ---
                # 标准化省份名称(移除后缀)
                def clean_province(name):
                    if isinstance(name, str):
                        return name.replace('省', '').replace('市', '').replace('自治区', '').replace('壮族', '').replace('回族', '').replace('维吾尔', '')
                    return name

                df_water['province'] = df_water['province'].apply(clean_province)
                df_prec['province'] = df_prec['province'].apply(clean_province)
                df_temp['province'] = df_temp['province'].apply(clean_province)

                # 将年份转换为数值类型，强制转换错误
                df_water['year'] = pd.to_numeric(df_water['year'], errors='coerce')
                df_prec['year'] = pd.to_numeric(df_prec['year'], errors='coerce')
                df_temp['year'] = pd.to_numeric(df_temp['year'], errors='coerce')

                # 删除无效年份的行
                df_water.dropna(subset=['year'], inplace=True)
                df_prec.dropna(subset=['year'], inplace=True)
                df_temp.dropna(subset=['year'], inplace=True)

                # 将年份转换为整数
                df_water['year'] = df_water['year'].astype(int)
                df_prec['year'] = df_prec['year'].astype(int)
                df_temp['year'] = df_temp['year'].astype(int)

                # --- 合并数据框 ---
                df_merged = pd.merge(df_water, df_prec, on=['province', 'year'], how='inner')
                df_merged = pd.merge(df_merged, df_temp, on=['province', 'year'], how='inner')

            except FileNotFoundError as e:
                raise FileNotFoundError(f"加载数据集文件时出错: {e}。确保文件存在于 {data_dir}")
            except Exception as e:
                raise RuntimeError(f"处理Excel文件时出错: {e}。检查文件结构、工作表名称和列名。")

        # 如果提供了province_id则进行过滤
        if self.province_id is not None:
            if 'province_id' in df_merged.columns:
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
        all_cols_to_scale = self.features + [self.target]
        
        # 处理特定省份ID的情况
        if self.province_id is not None:
            provinces_to_process = df_merged['province'].unique()
        else:
            provinces_to_process = df_merged['province'].unique()
            
        for province in provinces_to_process:
            group = df_merged[df_merged['province'] == province]
            # 确保组足够大以进行标准化和序列创建
            if len(group) > self.sequence_length:
                scaler = MinMaxScaler()
                # 仅对数值特征/目标列进行标准化
                cols_to_scale_in_group = [col for col in all_cols_to_scale if col in group.columns]
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
            
            data_values = df_province[feature_cols + [self.target]].values
            num_samples = len(data_values)
            province_id = df_province['province_id'].iloc[0] if 'province_id' in df_province.columns else None
            
            if num_samples > self.sequence_length:
                for i in range(num_samples - self.sequence_length):
                    seq = data_values[i:i + self.sequence_length, :-1]  # 特征
                    label = data_values[i + self.sequence_length, -1]   # 目标
                    
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
        self.features = ['precipitation', 'temperature', 'year', 'supply'] # 输入特征更新为包含年份
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

            train_size = int(self.train_val_test_split[0] * total_size)
            val_size = int(self.train_val_test_split[1] * total_size)
            # 确保测试集大小至少为0
            test_size = max(0, total_size - train_size - val_size)

            # 如果由于四舍五入或小数据集导致总和超过total_size，则调整train_size
            if train_size + val_size + test_size > total_size:
                 train_size = total_size - val_size - test_size

            # 确保大小非负
            train_size = max(0, train_size)
            val_size = max(0, val_size)

            # 处理数据集太小无法拆分的情况
            if train_size == 0 or val_size == 0 or test_size == 0:
                print(f"警告: 数据集大小 ({total_size}) 对于请求的拆分 ({self.train_val_test_split}) 太小。调整拆分大小。")
                # 简单调整示例：主要分配给训练集，如果可能的话，验证/测试集各1个
                if total_size > 2:
                    val_size = 1
                    test_size = 1
                    train_size = total_size - 2
                elif total_size == 2:
                    val_size = 1
                    test_size = 0
                    train_size = 1
                else: # total_size == 1
                    val_size = 0
                    test_size = 0
                    train_size = 1
                print(f"调整后的大小: 训练={train_size}, 验证={val_size}, 测试={test_size}")


            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                self.full_dataset, [train_size, val_size, test_size]
            )
            
            print(f"已加载数据集，共 {total_size} 个样本")
            print(f"训练: {train_size}, 验证: {val_size}, 测试: {test_size}")
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
        else:
            self.setup()
            return self.full_dataset.get_province_names()

