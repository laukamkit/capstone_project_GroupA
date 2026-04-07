from dataclasses import dataclass, field

#__all__ = ["Config", "PatchTSTConfig", "SARIMAXConfig", "LSTMConfig"]

@dataclass
class Config:
    task_id: str
    forecast_horizon: int
    lookback_window: int | None = None
    target_col: str = 'TOTALDEMAND'
    used_log_target: bool = False
    feature_cols: list[str] = field(default_factory=list)
    scale: bool = True # if scale, then make sure to set inverse to True as well
    seed: int | None = None

@dataclass(kw_only=True)
class TransformersConfig(Config):
    model:str
    date_col: str = 'DATETIME'
    variate: str = 'MS' # 'S' for single variate, 'MS' for multiple predictors but single output, 'M' for multiple predictors and multiple outputs
    patch_len: int = 16
    stride: int = 1
    d_model: int = 128
    num_attention_heads: int = 4
    num_encoder_layers: int = 3
    dim_ff: int = 256
    dropout: float = 0.1 # dropout for residual connections. Used before the encoder layer but after the input embedding + positional encoding, used inside the encoder layer after the multihead attention and after the fully connected layer of each encoder block.
    dropout_head_fc: float = 0.1 # dropout for the fully connected layers in the head (after the transformer backbone)
    use_gpu: bool = True
    time_encoding: str = 'timeF'
    shuffle_flag: bool = True
    training_epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.0001
    output_attention: bool = False
    lradj: str = 'TST' # options are 'type1', 'type2', 'type3', 'TST', 'constant', '1', '2', '3', '4', '5', '6'. See tools.py for details on each type of learning rate adjustment strategy.
    patience: int = 5 # for early stopping
    pct_start: float = 0.3 # for OneCycleLR scheduler, the percentage of the cycle spent increasing the learning rate.
    padding_patch: str = 'end'
    revin: int = 1
    affine: int = 0
    subtract_last: int = 0
    decomposition: int = 0
    kernel_size: int = 25
    gpu: int = 0
    dropout_ff: float = 0.1 # dropout for a fully connected layer head only if pretrain_head is True, but pretrain_head is never used in official repo.
    individual: int = 0 # whether to use individual head for each channel input.
    num_workers: int = 0 # for data loading. Set to 0 for debugging, can increase for faster data loading in production.
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
class LSTMConfig(Config):
    pass