from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from model_configs import Config, TransformersConfig
from PatchTST_supervised.utils.timefeatures import time_features
from torch.utils.data import Dataset
import pandas as pd

class NSWDataSet(Dataset):
    """For PatchTSTConfig and LSTMConfig that require lookback windows"""
    
    def __init__(self, config:TransformersConfig, train_df, val_df, test_df, flag='train'):
        self.config = config
        self.timeenc = 0 if config.time_encoding != 'timeF' else 1
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.__read_data__()


    def __read_data__(self):
        """Shared border/slicing logic for all subclasses."""
        df_raw = pd.concat([self.train_df, self.val_df, self.test_df], axis=0).reset_index() # reset index to bring out date column for time encoding for TimeXer
        df_raw = df_raw[[self.config.date_col] + self.config.feature_cols + [self.config.target_col]]
        
        num_train = len(self.train_df)
        num_test = len(self.test_df)
        num_vali = len(self.val_df)
        
        border1s = [0, num_train - self.config.lookback_window, len(df_raw) - num_test - self.config.lookback_window]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.config.variate == 'M' or self.config.variate == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.config.variate == 'S':
            df_data = df_raw[[self.config.target_col]]

        data = df_data.values

        df_stamp = df_raw[[self.config.date_col]][border1:border2]
        df_stamp[self.config.date_col] = pd.to_datetime(df_stamp[self.config.date_col])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp[self.config.date_col].apply(lambda row: row.month)
            df_stamp['day'] = df_stamp[self.config.date_col].apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp[self.config.date_col].apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp[self.config.date_col].apply(lambda row: row.hour)
            data_stamp = df_stamp.drop([self.config.date_col], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp[self.config.date_col].values), freq='30min')
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.time_stamp = df_stamp[self.config.date_col].values.astype(np.int64)
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.config.lookback_window
        r_begin = s_end - 0 # self.label_len only for decoder transformers, which PatchTST is not.
        r_end = r_begin + 0 + self.config.forecast_horizon #self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_time = self.time_stamp[r_begin:r_end]

        return seq_x.copy(), seq_y.copy(), seq_x_mark.copy(), seq_y_mark.copy(), seq_time.copy()

    def __len__(self):
        return len(self.data_x) - self.config.lookback_window - self.config.forecast_horizon + 1   