import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


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
        if self.set_type == 0:
            file_path = os.path.join(self.root_path, self.train_file)
        elif self.set_type == 1:
            file_path = os.path.join(self.root_path, self.val_file)
        else:
            file_path = os.path.join(self.root_path, self.test_file)
        
        df_raw = pd.read_parquet(file_path)
        
        # 3. Sort by group_column and datetime_column once for efficiency
        df_raw.sort_values(by=[self.group_column, self.datetime_column], inplace=True)

        data_by_group = df_raw.groupby(self.group_column)

        data_x_list = []
        data_y_list = []
        data_stamp_list = []

        feature_cols = [col for col in df_raw.columns if col not in [self.group_column, self.datetime_column, self.label_column]]
        
        for _, group_df in data_by_group:
            group_df.reset_index(drop=True, inplace=True)
            
            # 2. No need for redundant conversion if dtype is already datetime
            df_stamp = group_df[[self.datetime_column]]
            if self.timeenc == 1:
                data_stamp = time_features(df_stamp[self.datetime_column].values, freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)
            else:
                # Use .dt accessor which requires datetime-like values
                time_stamps = pd.to_datetime(df_stamp[self.datetime_column].values)
                df_stamp['month'] = time_stamps.month
                df_stamp['day'] = time_stamps.day
                df_stamp['weekday'] = time_stamps.weekday
                df_stamp['hour'] = time_stamps.hour
                data_stamp = df_stamp.drop(columns=[self.datetime_column]).values

            data = group_df[feature_cols].values
            labels = group_df[[self.label_column]].values

            for i in range(len(group_df) - self.seq_len + 1):
                data_x_list.append(data[i:i + self.seq_len])
                # 4. Correctly get the label from the last day of the window
                data_y_list.append(labels[i + self.seq_len - 1])
                data_stamp_list.append(data_stamp[i:i + self.seq_len])

        # 6. Pre-construct tensors for efficiency and move to target device
        device = self.args.device
        self.data_x = torch.tensor(np.array(data_x_list), dtype=torch.float32).to(device)
        # 5. Use torch.long for classification labels and squeeze the last dimension
        self.data_y = torch.tensor(np.array(data_y_list), dtype=torch.long).squeeze(-1).to(device)
        self.data_stamp = torch.tensor(np.array(data_stamp_list), dtype=torch.float32).to(device)


    def __getitem__(self, index):
        # For classification, pred_len and label_len are 0, so seq_y_mark is a dummy tensor.
        seq_y_mark = torch.zeros((0, self.data_stamp.shape[-1]), dtype=torch.float32)
        return self.data_x[index], self.data_y[index], self.data_stamp[index], seq_y_mark

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        # Data is not scaled, so just return it
        return data
