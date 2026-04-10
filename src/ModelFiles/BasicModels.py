import torch
import random
import numpy as np
from NSWData.NSWDataLoader import NSWDataLoader
from ModelFiles.ModelConfigs import Config, DeepLearningConfig
from typing import Callable
import pandas as pd

class BaseModel:
    def __init__(self, config: Config, func: Callable[[pd.DataFrame, str, list[int], list[int],list[tuple[str,int]]], pd.DataFrame] | None = None):
        self.config: Config = config
        self.nsw_data_loader = NSWDataLoader()
        self.train, self.validation, self.test, self.train_scaled, self.val_scaled, self.test_scaled, self.scaler = self.nsw_data_loader.load_data()
        training_data = self.train_scaled if self.config.scale else self.train
        validation_data = self.val_scaled if self.config.scale else self.validation
        test_data = self.test_scaled if self.config.scale else self.test
        self.training_data = func(training_data, config.target_col, config.target_lags, config.target_mas, config.feature_lag_cols) if func else training_data
        self.validation_data = func(validation_data, config.target_col, config.target_lags, config.target_mas, config.feature_lag_cols) if func else validation_data
        self.test_data = func(test_data, config.target_col, config.target_lags, config.target_mas, config.feature_lag_cols) if func else test_data
        self.val_step_size = min(config.eval_step_size, config.forecast_horizon)
        if self.config.seed is not None:
            self.set_seed(self.config.seed)
            print(f"Set random seed to {self.config.seed}")

    def set_seed(self, seed: int = 42, deterministic: bool = True):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _get_data(self, flag):
            data_set, data_loader = self.nsw_data_loader.data_provider(self.training_data, self.validation_data, self.test_data, self.config, flag)
            return data_set, data_loader
            
    def train_model(self):
        raise NotImplementedError
    
    def evaluate_model(self):
        raise NotImplementedError
  
class DeepLearningModel(BaseModel):
    def __init__(self, config: DeepLearningConfig, func: Callable[[pd.DataFrame, str, list[int], list[int], list[tuple[str, int]]], pd.DataFrame] | None = None):
        super().__init__(config, func)
        self.config: DeepLearningConfig = config
        self.device = self._acquire_device()
        self.training_data = self.training_data.dropna().reset_index()
        self.validation_data = self.validation_data.dropna().reset_index()
        self.test_data = self.test_data.dropna().reset_index()
        
    def _acquire_device(self):
        # taken from official repository:
        if self.config.use_gpu:
            device = torch.device('cuda:{}'.format(self.config.gpu))
            print('Use GPU: cuda:{}'.format(self.config.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    def _debug_batch_timing(self, loop_name, y_batch_time):
        print("="*50)
        print(f"{loop_name} Loop Debug Info:")
        print("="*50)
        times = np.array(y_batch_time, dtype='datetime64[us]')
        step = np.timedelta64(30, 'm')
        val_step_size = min(self.val_step_size, y_batch_time.shape[1]) # in case eval_step_size is greater than number of time steps in batch
        within_batch_diff = int((times[0, -1] - times[0, 0]) / step) + 1
        across_batch_diff = ((float((times[-1, 0] - times[0, 0]) / step)) / val_step_size) + 1
        print(f"\tFirst datetime in first batch: {np.array(y_batch_time[0][0], '<M8[us]')}\n\tLast datetime in first batch: {np.array(y_batch_time[0][-1], '<M8[us]')}\n"
              f"\tFirst datetime in last batch: {np.array(y_batch_time[-1][0], '<M8[us]')}\n\tLast datetime in last batch: {np.array(y_batch_time[-1][-1], '<M8[us]')}\n"
            f"\t\tTotal steps within a batch: {within_batch_diff}\tConfig Horizon: {self.config.forecast_horizon}\n"
            f"\t\tTotal steps across batches: {across_batch_diff}\tConfig Batch Size: {self.config.batch_size}\n"
            f"\tNOTE: If 'TOTAL STEPS ACROSS BATCHES' do not match the expected numbers, this is ok for training loop as the dataloader is set to 'shuffle=True'.")
        user_input = input("Press Enter to continue to next iteration or enter 'q' to skip this: ")
        if user_input.lower() == 'q':
            return False
        return True