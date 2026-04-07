import torch
import random
import numpy as np
from nsw_data_loader.nsw_data_loader import NSWDataLoader
from model_configs import Config

class Base_Model:
    def __init__(self, config: Config):
        self.config: Config = config
        self.nsw_data_loader = NSWDataLoader()
        self.train, self.validation, self.test, self.train_scaled, self.val_scaled, self.test_scaled, self.scaler = self.nsw_data_loader.load_data()
        self.training_data = self.train_scaled if self.config.scale else self.train
        self.validation_data = self.val_scaled if self.config.scale else self.validation
        self.test_data = self.test_scaled if self.config.scale else self.test
        if self.config.seed is not None:
            random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            np.random.seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)

    def train_model(self):
        raise NotImplementedError
    
    def test_model(self):
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