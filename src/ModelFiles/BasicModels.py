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
    
    # def _compute_inverse_scaling(self, shape, pred, true):
    #     pos = self.nsw_data_loader.scaler.feature_names_in_.tolist().index(self.config.target_col)
    #     mean = self.nsw_data_loader.scaler.mean_[pos]
    #     var = self.nsw_data_loader.scaler.var_[pos]
    #     pred_tile = np.tile(pred, [1, 1, 1])
    #     pred_inverse = pred_tile.reshape(shape[0] * shape[1], 1)
    #     pred_inverse = pred_inverse* var**0.5 + mean
    #     pred_inverse = pred_inverse[:, -1:].reshape(shape)
    #     true_tiled = np.tile(true, [1, 1, 1])
    #     true_inverse = true_tiled.reshape(shape[0] * shape[1], 1)
    #     true_inverse = true_inverse*var**0.5 + mean
    #     true_inverse = true_inverse[:, -1:].reshape(shape)
    #     return pred_inverse, true_inverse
            
    def train_model(self):
        raise NotImplementedError
    
    def evaluate_model(self):
        raise NotImplementedError
    
    # @classmethod
    # def plot_predictions(cls, y_true_real, y_pred_real, n_points=500, title="Forecast vs Actual"):
    #     plt.figure(figsize=(12, 5))
    #     plt.plot(y_true_real[:n_points], label="Actual")
    #     plt.plot(y_pred_real[:n_points], label="Predicted")
    #     plt.xlabel("Test Sample")
    #     plt.ylabel("Demand")
    #     plt.title(title)
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)
    #     plt.tight_layout()
    #     plt.show()


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