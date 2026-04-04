from dataclasses import dataclass, field

#__all__ = ["Config", "PatchTSTConfig", "SARIMAXConfig", "LSTMConfig"]

@dataclass
class Config:
    task_id: str
    forecast_horizon: int
    lookback_window: int | None = None
    date_col: str = 'DATETIME'
    target_col: str = 'TOTALDEMAND'
    feature_cols: list[str] = field(default_factory=list)
    is_training: bool = True
    time_freq: str = '30min'
    scale: bool = True # if scale, then make sure to set inverse to True as well
    seed: int | None = None

@dataclass
class PatchTSTConfig(Config):
    variate: str = 'MS' # 'S' for single variate, 'MS' for multiple predictors but single output, 'M' for multiple predictors and multiple outputs
    patch_size: int = 16
    stride: int = 1
    d_model: int = 128
    num_attention_heads: int = 4
    num_encoder_layers: int = 3
    dim_ff: int = 256
    dropout_ff: float = 0.1
    dropout_head_fc: float = 0.1
    @property
    def enc_in(self) -> int:
        return len(self.feature_cols) if self.feature_cols is not None else 1
    @property
    def c_out(self) -> int:
        return 1

@dataclass(kw_only=True)
class SARIMAXConfig(Config):
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    enforce_stationarity: bool
    enforce_invertibility: bool
    val_iter: int = 20

@dataclass(kw_only=True)
class LSTMConfig(Config):
    pass