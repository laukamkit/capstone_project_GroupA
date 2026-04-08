from dataclasses import dataclass, field

#__all__ = ["Config", "PatchTSTConfig", "SARIMAXConfig", "LSTMConfig"]

@dataclass
class Config:
    task_id: str
    forecast_horizon: int
    lookback_window: int | None = None
    date_col: str = 'DATETIME'
    target_col: str = 'TOTALDEMAND'
    used_log_target: bool = False
    feature_cols: list[str] = field(default_factory=list)
    feature_lag_cols: list[str] = field(default_factory=list)
    demand_lags: list[int] = field(default_factory=list)
    feature_lags: list[int] = field(default_factory=list)
    scale: bool = True # if scale, then make sure to set inverse to True as well
    seed: int | None = None
    @property
    def all_feature_cols(self) -> list[str]:
        lag_feature_cols = []
        for lag in self.demand_lags:
            lag_feature_cols.append(f"{self.target_col}_lag_{lag}")
        for feature in self.feature_lag_cols:
            for lag in self.feature_lags:
                lag_feature_cols.append(f"{feature}_lag_{lag}")
        return self.feature_cols + lag_feature_cols

@dataclass(kw_only=True)
class DeepLearningConfig(Config):
    shuffle_flag: bool = True
    training_epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.0001
    patience: int = 5 # for early stopping
    num_attention_heads: int = 4
    dropout: float = 0.1 # dropout for residual connections. Used before the encoder layer but after the input embedding + positional encoding, used inside the encoder layer after the multihead attention and after the fully connected layer of each encoder block.
    # also used for LSTM's. default is 0.4 for LSTMs.

    use_gpu: bool = True
    gpu: int = 0
    num_workers: int = 0 # for data loading. Set to 0 for debugging, can increase for faster data loading in production.
    time_encoding: str = 'timeF' # not used for PatchTST, LSTMs and SARIMAX. Only used for TimeXer.

@dataclass(kw_only=True)
class TransformersConfig(DeepLearningConfig):
    model:str
    variate: str = 'MS' # 'S' for single variate, 'MS' for multiple predictors but single output, 'M' for multiple predictors and multiple outputs
    patch_len: int = 16
    stride: int = 1
    d_model: int = 128
    num_encoder_layers: int = 3
    dim_ff: int = 256
    dropout: float = 0.1 # dropout for residual connections. Used before the encoder layer but after the input embedding + positional encoding, used inside the encoder layer after the multihead attention and after the fully connected layer of each encoder block.
    dropout_head_fc: float = 0.1 # dropout for the fully connected layers in the head (after the transformer backbone)
    output_attention: bool = False
    lradj: str = 'TST' # options are 'type1', 'type2', 'type3', 'TST', 'constant', '1', '2', '3', '4', '5', '6'. See tools.py for details on each type of learning rate adjustment strategy.
    pct_start: float = 0.3 # for OneCycleLR scheduler, the percentage of the cycle spent increasing the learning rate.
    padding_patch: str = 'end'
    revin: int = 1
    affine: int = 0
    subtract_last: int = 0
    decomposition: int = 0
    kernel_size: int = 25
    dropout_ff: float = 0.1 # dropout for a fully connected layer head only if pretrain_head is True, but pretrain_head is never used in official repo.
    individual: int = 0 # whether to use individual head for each channel input.
    @property
    def enc_in(self) -> int:
        return len(self.feature_cols) if self.feature_cols is not None else 1
    @property
    def c_out(self) -> int:
        return 1

@dataclass(kw_only=True)
class SARIMAXConfig(Config):
    # If any of the number of elements in these lists is greater than 1, then grid search will be performed
    p: list[int] = field(default_factory=list)
    d: list[int] = field(default_factory=list)
    q: list[int] = field(default_factory=list)
    P: list[int] = field(default_factory=list)
    D: list[int] = field(default_factory=list)
    Q: list[int] = field(default_factory=list)
    seasonality_period: int
    enforce_stationarity: bool
    enforce_invertibility: bool
    val_step_size: int = 48 # every 1 day

@dataclass(kw_only=True)
class LSTMBaseConfig(DeepLearningConfig):
    model_type: str = "multihead_attention_bilstm"
    hidden_size: int = 64
    num_layers: int = 2
    # dropout: float = 0.4
    # learning_rate: float = 5e-5
    # batch_size: int = 64
    # epochs: int = 100
    # patience: int = 8
    use_mlp_head: bool = True
    mlp_hidden_size: int = 64
    # target_col: str = "TOTALDEMAND"
    # temp_col: str = "TEMPERATURE"
    # lookback: int = 168*5
    # horizon: int = 168
    weight_decay: float = 1e-4
    scheduler: str | None = None
    optimizer: str = "adam"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.5
