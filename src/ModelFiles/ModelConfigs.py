from dataclasses import dataclass, field
from ModelFiles.ModelEnums import *

__all__ = ["TransformersConfig", "SARIMAXConfig", "LSTMConfig", "HORIZONS", "CONTEXT_LENGTHS", "SEEDS"]

HORIZONS = [48, 336, 720]
CONTEXT_LENGTHS = [144, 336, 720]
SEEDS = [31415, 27182, 14142, 17320, 22360, 57721, 66987, 11235, 98765, 43210]

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
    @property
    def config_params_to_results(self) -> dict:
        return {
            "model_type": "SARIMAX",
            "p": self.p,
            "d": self.d,
            "q": self.q,
            "P": self.P,
            "D": self.D,
            "Q": self.Q,
            "seasonality_period": self.seasonality_period,
            "enforce_stationarity": self.enforce_stationarity,
            "enforce_invertibility": self.enforce_invertibility,
            "lookback": self.lookback_window,
            "horizon": self.forecast_horizon,
            "demand_lags": self.demand_lags,
            "feature_lags": self.feature_lags,
            "val_step_size": self.val_step_size,
        }
    
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
    model: TransformerModelType = TransformerModelType.PATCHTST
    variate: str = 'MS' # 'S' for single variate, 'MS' for multiple predictors but single output, 'M' for multiple predictors and multiple outputs
    patch_len: int = 16
    stride: int = 1
    d_model: int = 128
    num_encoder_layers: int = 3
    dim_ff: int = 256
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

    @property
    def config_params_to_results(self) -> dict:
        return {
            "model_type": self.model.value,
            "variate": self.variate,
            "patch_len": self.patch_len,
            "stride": self.stride,
            "d_model": self.d_model,
            "num_encoder_layers": self.num_encoder_layers,
            "dim_ff": self.dim_ff,
            "dropout": self.dropout,
            "dropout_head_fc": self.dropout_head_fc,
            "lookback": self.lookback_window,
            "horizon": self.forecast_horizon,
            "demand_lags": self.demand_lags,
            "feature_lags": self.feature_lags,
        }
    


@dataclass(kw_only=True)
class LSTMConfig(DeepLearningConfig):
    model_type: LSTMModelType = LSTMModelType.MULTIHEAD_ATTENTION_BILSTM
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

    @property
    def config_params_to_results(self) -> dict:
        return {
            "model_type": self.model_type.value,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "use_mlp_head": self.use_mlp_head,
            "mlp_hidden_size": self.mlp_hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "lookback": self.lookback_window,
            "horizon": self.forecast_horizon,
            "demand_lags": self.demand_lags,
            "feature_lags": self.feature_lags,
        }

