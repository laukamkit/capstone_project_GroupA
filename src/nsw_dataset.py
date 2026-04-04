
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from model_configs import Config
from PatchTST.PatchTST_supervised.utils.timefeatures import time_features
from torch.utils.data import Dataset
import pandas as pd

__all__ = ["WindowedNSWDataSet", "SequentialNSWDataSet"]


class BaseNSWDataSet(Dataset):
    def __init__(self, config, train_df, val_df, test_df, flag='train'):
        self.config = config
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.train_df.index.freq = self.config.time_freq
        self.val_df.index.freq = self.config.time_freq
        self.test_df.index.freq = self.config.time_freq
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        # Call the subclass-specific read logic
        self.__read_data__()

    def __read_data__(self):
        """Shared border/slicing logic for all subclasses."""
        df_raw = pd.concat([self.train_df, self.val_df, self.test_df], axis=0).reset_index()
        df_raw = df_raw[[self.config.date_col] + self.config.feature_cols + [self.config.target_col]]
        
        num_train = len(self.train_df)
        num_test = len(self.test_df)
        num_vali = len(self.val_df)
        
        border1s = [0, num_train - self.config.lookback_window, len(df_raw) - num_test - self.config.lookback_window]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        self.df_slice = df_raw.iloc[border1:border2]  # expose the slice to subclasses
        self.border1 = border1
        self.border2 = border2
        self.df_raw = df_raw  # expose full df so subclasses can index future windows

    def __getitem__(self, index):
        raise NotImplementedError("Subclasses must implement __getitem__")

    def __len__(self):
        raise NotImplementedError("Subclasses must implement __len__")

class WindowedNSWDataSet(BaseNSWDataSet):
    """For PatchTSTConfig and LSTMConfig that require lookback windows"""
    
    def __init__(self, config, train_df, val_df, test_df, flag='train', features='S', timeenc=0,):
        self.features = features
        self.timeenc = timeenc
        self.freq = config.time_freq
        super().__init__(config, train_df, val_df, test_df, flag)


    def __read_data__(self):
        super().__read_data__()

        if self.features == 'M' or self.features == 'MS':
            cols_data = self.df_slice.columns[1:]
            df_data = self.df_slice[cols_data]
        elif self.features == 'S':
            df_data = self.df_slice[[self.config.target_col]]

        data = df_data.values

        df_stamp = self.df_slice[[self.config.date_col]]
        df_stamp[self.config.date_col] = pd.to_datetime(df_stamp[self.config.date_col])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp[self.config.date_col].apply(lambda row: row.month)
            df_stamp['day'] = df_stamp[self.config.date_col].apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp[self.config.date_col].apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp[self.config.date_col].apply(lambda row: row.hour)
            data_stamp = df_stamp.drop([self.config.date_col], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp[self.config.date_col].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[self.border1:self.border2]
        self.data_y = data[self.border1:self.border2]
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


class SequentialNSWDataSet(BaseNSWDataSet):
    def __init__(self, config, train_df, val_df, test_df, flag='val'):
        assert flag in ['val','test'] # SARIMAX doesn't train via dataloader, so we only only allow val and test sets here
        super().__init__(config, train_df, val_df, test_df, flag)

    def __read_data__(self):
        super().__read_data__()
        self.data_x = self.df_slice[self.config.feature_cols].values
        self.data_y = self.df_slice[self.config.target_col].values
        self.time_stamp = self.df_slice[self.config.date_col].values.astype(np.int64)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.config.forecast_horizon
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[s_begin:s_end]
        seq_time = self.time_stamp[s_begin:s_end]
        return seq_x.copy(), seq_y.copy(), seq_time.copy()

    def __len__(self):
        return len(self.data_x)