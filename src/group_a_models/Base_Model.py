from nsw_data_loader.nsw_data_loader import NSWDataLoader
from model_configs import Config

class Base_Model:
    def __init__(self, config: Config):
        self.config: Config = config
        self.nsw_data_loader = NSWDataLoader()
        self.train, self.validation, self.test, self.train_scaled, self.val_scaled, self.test_scaled = self.nsw_data_loader.load_data()

    def train_model(self):
        raise NotImplementedError
    
    def test_model(self):
        raise NotImplementedError
    
    def data_loader(self):
        raise NotImplementedError