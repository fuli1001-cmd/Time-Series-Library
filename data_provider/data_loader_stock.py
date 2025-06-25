import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.timefeatures import time_features
from joblib import Parallel, delayed
from utils.device import try_gpu
import warnings

warnings.filterwarnings('ignore')

def process_group(group, seq_len):
    indices = []
    group_index = group.index.values
    for i in range(len(group) - seq_len + 1):
        start_idx = group_index[i]
        end_idx = group_index[i + seq_len - 1]
        indices.append((start_idx, end_idx))
    return indices

class Dataset_Stock(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path=None,
                 target='OT', scale=False, timeenc=0, freq='d', seasonal_patterns=None):
        self.args = args
        self.seq_len = args.seq_len
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.timeenc = timeenc
        self.freq = freq
        self.group_column = args.group_column
        self.datetime_column = args.datetime_column

        self.root_path = root_path
        self.train_file = args.train_file
        self.val_file = args.val_file
        self.test_file = args.test_file
        self.label_column = args.label_column
        self.__read_data__()

    def __read_data__(self):
        device = try_gpu()
        print(f"using {device}")

        if self.set_type == 0:
            file_path = os.path.join(self.root_path, self.train_file)
        elif self.set_type == 1:
            file_path = os.path.join(self.root_path, self.val_file)
        else:
            file_path = os.path.join(self.root_path, self.test_file)
        
        df = pd.read_parquet(file_path)
        
        # 1. Initial sort (once at the beginning)
        df.sort_values(by=[self.group_column, self.datetime_column], inplace=True)
        df.reset_index(drop=True, inplace=True) # Reset index after sorting to get contiguous indices

        # 2. Create full tensors for features, labels, and timestamps
        feature_cols = [col for col in df.columns if col not in [self.group_column, self.datetime_column, self.label_column]]
        feature_tensor = torch.tensor(df[feature_cols].values, dtype=torch.float32, device=device)
        label_tensor = torch.tensor(df[self.label_column].values, dtype=torch.long, device=device)

        # # Create timestamp features
        # df_stamp = df[[self.datetime_column]]
        # if self.timeenc == 1:
        #     data_stamp = time_features(pd.to_datetime(df_stamp[self.datetime_column].values), freq=self.freq)
        #     data_stamp = data_stamp.transpose(1, 0)
        # else:
        #     time_stamps = pd.to_datetime(df_stamp[self.datetime_column].values)
        #     df_stamp['month'] = time_stamps.month
        #     df_stamp['day'] = time_stamps.day
        #     df_stamp['weekday'] = time_stamps.weekday
        #     df_stamp['hour'] = time_stamps.hour
        #     data_stamp = df_stamp.drop(columns=[self.datetime_column]).values
        
        # stamp_tensor = torch.tensor(data_stamp, dtype=torch.float32, device=device)

        self.samples = []
        for start_idx, end_idx in self._build_indices_parallel(df):
            x = feature_tensor[start_idx : end_idx + 1]
            y = label_tensor[end_idx]
            # x_mark = stamp_tensor[start_idx : end_idx + 1]
            # # For classification, pred_len and label_len are 0, so seq_y_mark is a dummy tensor.
            # y_mark = torch.zeros((0, stamp_tensor.shape[-1]), dtype=torch.float32, device=device)
            # self.samples.append((x, y, x_mark, y_mark))
            self.samples.append((x, y))

        print(f"feature shape: {feature_tensor.shape}, size: {feature_tensor.element_size() * feature_tensor.nelement() / 1024**2} MB")
        print(f"Pre-cached {len(self.samples)} samples.")

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

    def inverse_transform(self, data):
        # Data is not scaled, so just return it
        return data
    
    def _build_indices_parallel(self, df):
        print("parallel building indices for sequences...")
        seq_len = self.seq_len
        group_column = self.group_column

        # 使用 Parallel 并行处理
        results = Parallel(n_jobs=-1)(
            delayed(process_group)(group, seq_len)
            for _, group in df.groupby(group_column)
            if len(group) >= seq_len
        )

        # 将所有结果合并
        indices = [item for sublist in results for item in sublist]
        return indices
