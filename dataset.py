import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Dataset class for Time Series Data
class TimeSeriesDataset(Dataset):
    def __init__(self, data_dir, sequence_length=12, features=['precipitation', 'temperature', 'supply'], target='consumption'):
        """
        Args:
            data_dir (str): Path to the directory containing the dataset files.
            sequence_length (int): Length of the input sequence.
            features (list): List of feature column names.
            target (str): Target column name.
        """
        super().__init__()
        self.sequence_length = sequence_length
        self.features = features
        self.target = target
        # Scaler needs to be accessible for potential inverse transform later
        self.scalers = {} # Store scalers per province
        self.data = self._load_and_preprocess_data(data_dir)
        self.sequences = self._create_sequences()

    def _load_and_preprocess_data(self, data_dir):
        water_data_path = os.path.join(data_dir, '15-分省水资源供水和用水数据.xls')
        prec_data_path = os.path.join(data_dir, '省逐年降水.xlsx')
        temp_data_path = os.path.join(data_dir, '省逐年平均气温.xlsx')

        try:
            # Load water supply and consumption data (.xls)
            # Assuming the relevant data starts from the 4th row (index 3) and province/year are in specific columns
            # Adjust header row and column names/indices based on actual file structure
            df_water = pd.read_excel(water_data_path, engine='xlrd', sheet_name=0, header=2) # Adjust header index if needed
            # Select and rename relevant columns (assuming column names or indices)
            # Example: df_water = df_water[['地区', '年份', '供水总量', '用水总量']]
            # Need to inspect the actual file for correct column names/indices
            # For now, let's assume standard names for demonstration
            df_water = df_water.rename(columns={'地区': 'province', '年份': 'year', '供水总量': 'supply', '用水总量': 'consumption'})
            # Keep only necessary columns
            df_water = df_water[['province', 'year', 'supply', 'consumption']]

            # Load precipitation data (.xlsx)
            df_prec = pd.read_excel(prec_data_path, sheet_name=0, header=0) # Adjust header index if needed
            df_prec = df_prec.rename(columns={'省份': 'province', '年份': 'year', '降水量': 'precipitation'}) # Adjust column names
            df_prec = df_prec[['province', 'year', 'precipitation']]

            # Load temperature data (.xlsx)
            df_temp = pd.read_excel(temp_data_path, sheet_name=0, header=0) # Adjust header index if needed
            df_temp = df_temp.rename(columns={'省份': 'province', '年份': 'year', '平均气温': 'temperature'}) # Adjust column names
            df_temp = df_temp[['province', 'year', 'temperature']]

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error loading dataset files: {e}. Ensure files exist in {data_dir}")
        except Exception as e:
            # Catch other potential errors during loading (e.g., wrong sheet name, column names)
            # Provide more specific error handling based on pandas exceptions if possible
            raise RuntimeError(f"Error processing Excel files: {e}. Check file structure, sheet names, and column names.")

        # --- Data Cleaning ---
        # Standardize province names (remove suffixes)
        def clean_province(name):
            if isinstance(name, str):
                return name.replace('省', '').replace('市', '').replace('自治区', '').replace('壮族', '').replace('回族', '').replace('维吾尔', '')
            return name

        df_water['province'] = df_water['province'].apply(clean_province)
        df_prec['province'] = df_prec['province'].apply(clean_province)
        df_temp['province'] = df_temp['province'].apply(clean_province)

        # Convert year to numeric, coercing errors
        df_water['year'] = pd.to_numeric(df_water['year'], errors='coerce')
        df_prec['year'] = pd.to_numeric(df_prec['year'], errors='coerce')
        df_temp['year'] = pd.to_numeric(df_temp['year'], errors='coerce')

        # Drop rows with invalid year
        df_water.dropna(subset=['year'], inplace=True)
        df_prec.dropna(subset=['year'], inplace=True)
        df_temp.dropna(subset=['year'], inplace=True)

        # Convert year to integer
        df_water['year'] = df_water['year'].astype(int)
        df_prec['year'] = df_prec['year'].astype(int)
        df_temp['year'] = df_temp['year'].astype(int)

        # --- Merging DataFrames ---
        df_merged = pd.merge(df_water, df_prec, on=['province', 'year'], how='inner')
        df_merged = pd.merge(df_merged, df_temp, on=['province', 'year'], how='inner')

        # Convert features and target to numeric, coercing errors
        for col in self.features + [self.target]:
            if col in df_merged.columns:
                df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')
            else:
                 print(f"Warning: Column '{col}' not found in merged data.") # Add warning

        # Handle NaNs that might result from merge or coercion
        df_merged.dropna(inplace=True)

        # Sort data for sequence creation
        df_merged.sort_values(by=['province', 'year'], inplace=True)

        # --- Normalization (per province) ---
        processed_data = {}
        all_cols_to_scale = self.features + [self.target]
        for province, group in df_merged.groupby('province'):
            # Ensure group is large enough for scaling and sequence creation
            if len(group) > self.sequence_length:
                scaler = MinMaxScaler()
                # Scale only the numeric feature/target columns
                cols_to_scale_in_group = [col for col in all_cols_to_scale if col in group.columns]
                if cols_to_scale_in_group:
                    group_scaled = group.copy()
                    group_scaled[cols_to_scale_in_group] = scaler.fit_transform(group[cols_to_scale_in_group])
                    self.scalers[province] = scaler # Store scaler for this province
                    processed_data[province] = group_scaled
                else:
                    print(f"Warning: No columns to scale for province {province}.")
            else:
                print(f"Warning: Skipping province {province} due to insufficient data (found {len(group)}, need > {self.sequence_length}).")

        return processed_data # Return dict of dataframes keyed by province

    def _create_sequences(self):
        sequences = []
        # self.data is now a dictionary of DataFrames {province: df}
        for province, df_province in self.data.items():
            data_values = df_province[self.features + [self.target]].values
            num_samples = len(data_values)

            if num_samples > self.sequence_length:
                for i in range(num_samples - self.sequence_length):
                    seq = data_values[i:i + self.sequence_length, :-1]  # Features
                    label = data_values[i + self.sequence_length, -1]   # Target
                    sequences.append((seq, label))
            # else: province was already skipped in preprocessing
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence, label = self.sequences[idx]
        # Convert to tensors
        features = torch.tensor(sequence, dtype=torch.float32)
        target = torch.tensor(label, dtype=torch.float32).unsqueeze(0) # Ensure target is [1]
        return features, target


class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='dataset', sequence_length=12, batch_size=32, num_workers=0, train_val_test_split=(0.7, 0.15, 0.15)):
        super().__init__()
        self.data_dir = data_dir # Changed from data_path to data_dir
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        self.features = ['precipitation', 'temperature', 'supply'] # Input features
        self.target = 'consumption' # Target variable
        # Add placeholders for datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.full_dataset = None # Keep reference if needed

    def setup(self, stage=None):
        # Setup is called once per process
        if not self.train_dataset and not self.val_dataset and not self.test_dataset:
            self.full_dataset = TimeSeriesDataset(
                data_dir=self.data_dir, # Pass the directory
                sequence_length=self.sequence_length,
                features=self.features,
                target=self.target
            )
            total_size = len(self.full_dataset)

            if total_size == 0:
                 raise ValueError("Dataset is empty after processing. Check data files and preprocessing steps.")

            train_size = int(self.train_val_test_split[0] * total_size)
            val_size = int(self.train_val_test_split[1] * total_size)
            # Ensure test_size is at least 0
            test_size = max(0, total_size - train_size - val_size)

            # Adjust train_size if the sum exceeds total_size due to rounding or small dataset
            if train_size + val_size + test_size > total_size:
                 train_size = total_size - val_size - test_size

            # Ensure sizes are non-negative
            train_size = max(0, train_size)
            val_size = max(0, val_size)

            # Handle cases where dataset is too small for split
            if train_size == 0 or val_size == 0 or test_size == 0:
                print(f"Warning: Dataset size ({total_size}) is small for the requested split ({self.train_val_test_split}). Adjusting split sizes.")
                # Example simple adjustment: allocate primarily to train, maybe 1 to val/test if possible
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
                print(f"Adjusted sizes: Train={train_size}, Val={val_size}, Test={test_size}")


            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                self.full_dataset, [train_size, val_size, test_size]
            )

    def train_dataloader(self):
        if not self.train_dataset:
            self.setup('fit') # Ensure setup is called if accessed directly
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

   