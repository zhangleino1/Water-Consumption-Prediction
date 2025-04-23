import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import redis
import json
# Dataset class
class BeaconSequenceDataset(Dataset):
    """
    示例：从 Redis 读取数据，生成 [batch_size, 8] 的指纹特征向量。
    """

    def __init__(self):
        super().__init__()
        self.redis_client = redis.Redis(
            host='43.138.39.243',
            port=6379,
            db=3,
            decode_responses=True,
            password='qmkj$2022$Uping'
        )

        self.location_labels = {}
        self.label_to_location = {}
        self.all_fingerprints = self.getAndTrainFingerprints()
        # location_labels 和 label_to_location 用于将位置编码和位置名称相互转换，输出到json文件中
        with open("label_to_location.json", "w") as f:
            json.dump(self.label_to_location, f)
        

        print("数据总量: ", len(self.all_fingerprints))

    def getAndTrainFingerprints(self):
        """
        从 Redis 中读取指纹数据，并对位置进行编码。
        """
        all_fingerprints = []
        label_counter = 0
        keys = self.redis_client.keys("*")
        for key in keys:
            # 简单判断，如果 key 中含有逗号，则认为是有效位置
            if "," in key:
                fingerprint_data = self.redis_client.lrange(key, 0, -1)
                for beacon_data in fingerprint_data:
                    beacons = json.loads(beacon_data)
                    rssi_values = []
                    for beacon in beacons:
                        rssi_values.append(beacon['rssi'])

                  

                    # 做一个简单的标准化(可根据需要替换成别的方法)
                    # rssi_tensor = torch.tensor(rssi_values, dtype=torch.float32)
                    # rssi_norm = (rssi_tensor - rssi_tensor.mean()) / (rssi_tensor.std() + 1e-6)

                    all_fingerprints.append({
                        "feature": rssi_values,
                        "target": label_counter
                    })

                self.location_labels[key] = label_counter
                self.label_to_location[label_counter] = key
                label_counter += 1

        # 打乱数据顺序
        np.random.shuffle(all_fingerprints)
        print("位置数量: ", len(self.location_labels))
        print("all_fingerprints: ", all_fingerprints)

        return all_fingerprints

    def __len__(self):
        return len(self.all_fingerprints)

    def __getitem__(self, idx):
        features = torch.tensor(self.all_fingerprints[idx]["feature"], dtype=torch.float32)
        features = features.unsqueeze(0)
        target = torch.tensor(self.all_fingerprints[idx]["target"], dtype=torch.long)
        return features, target


class BeaconDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4, train_val_test_split=(0.7, 0.15, 0.15)):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split

    def setup(self, stage=None):
        full_dataset = BeaconSequenceDataset()
        total_size = len(full_dataset)
        train_size = int(self.train_val_test_split[0] * total_size)
        val_size = int(self.train_val_test_split[1] * total_size)
        test_size = total_size - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

   