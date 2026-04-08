import math

import torch
from PatchTST_supervised.models import PatchTST, TimeXer
from PatchTST_supervised.models import iTransformer
from PatchTST_supervised.utils.tools import EarlyStopping, adjust_learning_rate, test_params_flop
from PatchTST_supervised.utils.metrics import metric
import torch.nn as nn
from GroupAModels import LSTMModel, PatchTSTModel, SarimaxModel
from nsw_data_loader.nsw_data_loader import NSWDataLoader
from model_configs import LSTMBaseConfig, SARIMAXConfig, TransformersConfig
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from time import time
from copy import deepcopy
from itertools import product
import json
from model_configs import LSTMBaseConfig

def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "TOTALDEMAND",
    feature_cols: list[str] = ["TEMPERATURE"],
    demand_lags: list[int] | None = None,
    feature_lags: list[int] | None = None,
):
    df = df.copy()
    demand_lags = demand_lags or []
    feature_lags = feature_lags or []

    for lag in demand_lags:
        df[f"demand_lag_{lag}"] = df[target_col].shift(lag)

    for feature_col in feature_cols:
        if feature_col in df.columns:
            for lag in feature_lags:
                df[f"{feature_col}_lag_{lag}"] = df[feature_col].shift(lag)

    df = df.dropna().reset_index()
    return df

def _make_config_name(config: LSTMBaseConfig):
    demand_lags_str = "-".join(map(str, config.demand_lags if config.demand_lags else [])) if config.demand_lags else "none"
    temp_lags_str = "-".join(map(str, config.feature_lags if config.feature_lags else [])) if config.feature_lags else "none"

    name = (
        f"{config.model_type}"
        f"_lb{config.lookback_window}"
        f"_hs{config.hidden_size}"
        f"_nl{config.num_layers}"
        f"_do{config.dropout}"
        f"_lr{config.learning_rate}"
        f"_mlp{int(config.use_mlp_head if hasattr(config, 'use_mlp_head') else False)}"
        f"_dlags{demand_lags_str}"
        f"_tlags{temp_lags_str}"
    )

    if hasattr(config, 'num_attention_heads'):
        name += f"_heads{config.num_attention_heads}"

    return name

def run_experiment(config: LSTMBaseConfig, experiment_name=None, save_best_model=True, show_live_plots=False):
    """
    Atomic run function.
    Assumes the following already exist earlier in the notebook:
    - SEED
    - DEVICE
    - OUTPUT_DIR
    - set_seed
    - load_data
    - build_dataloaders
    - build_model
    - train_model
    - evaluate_model
    - plot_training_history
    - plot_predictions
    - nn (torch.nn)
    """
    lstm_model = LSTMModel(config, add_lag_features)
    experiment_name = experiment_name or _make_config_name(config)

    print("\n" + "=" * 90)
    print(f"Running experiment: {experiment_name}")
    print("=" * 90)
    for k, v in config.__dict__.items():
        print(f"{k}: {v}")

    #set_seed(config.seed)
    #print(f"Using run seed: {run_seed}")

    # data_dict = load_data(config)
    train_data, train_loader = lstm_model._get_data("train") 
    val_data, val_loader = lstm_model._get_data("val")
    test_data, test_loader = lstm_model._get_data("test")
    
    # DEBUG: Inspect input features
    print("\n" + "="*60)
    print("DEBUG: FEATURE INSPECTION")
    print("="*60)

    # 1. Feature names
    if config.all_feature_cols is not None:
        print("\nFeature names:")
        for i, f in enumerate(config.all_feature_cols):
            print(f"{i}: {f}")
    else:
        print("\nNo feature list found in config.feature_cols.")

    # 2. Shape check
    train_x = np.array([batch_x for i, (batch_x, _, _, _, _) in enumerate(train_loader)])
    train_x = train_x.reshape(-1,train_x.shape[2],train_x.shape[-1])
    train_y = np.array([batch_y for i, (_, batch_y, _, _, _) in enumerate(train_loader)])
    train_y = train_y.reshape(-1,train_y.shape[-1])
    val_x = np.array([batch_x for i, (batch_x, _, _, _, _) in enumerate(val_loader)])
    val_x = val_x.reshape(-1,val_x.shape[2],val_x.shape[-1])
    val_y = np.array([batch_y for i, (_, batch_y, _, _, _) in enumerate(val_loader)])
    val_y = val_y.reshape(-1,val_y.shape[-1])
    test_x = np.array([batch_x for i, (batch_x, _, _, _, _) in enumerate(test_loader)])
    test_x = test_x.reshape(-1,test_x.shape[2],test_x.shape[-1])
    test_y = np.array([batch_y for i, (_, batch_y, _, _, _) in enumerate(test_loader)])
    test_y = test_y.reshape(-1,test_y.shape[-1])

    train_batch_x, train_batch_y, train_batch_x_mark, train_batch_y_mark, train_batch_time = next(iter(train_loader))
    val_batch_x, val_batch_y, val_batch_x_mark, val_batch_y_mark, val_batch_time = next(iter(val_loader))
    print("\nSequence shape (lookback, num_features):", train_x[0].shape)

    # 3. Convert to DataFrame for readability
    # if config.feature_cols is not None:
    #     df_seq = pd.DataFrame(train_x[0], columns=config.feature_cols)
    #     print("\nFirst 5 timesteps:")
    #     print(df_seq.head())

    #     print("\nLast 5 timesteps:")
    #     print(df_seq.tail())
    #else:
    print("\nRaw sequence values (first 5 timesteps):")
    print(train_x[0][:5])

    # 4. Target check
    print("\nTarget (scaled):", train_y[0][0])

    # Optional: inverse transform
    if config.scale:
        pos = lstm_model.scaler.feature_names_in_.tolist().index(config.target_col)
        mean = lstm_model.scaler.mean_[pos]
        var = lstm_model.scaler.var_[pos]
        y_real = train_y[0][0]* var**0.5 + mean
        print("Target (real scale):", y_real)

    print("="*60 + "\n")

    print("\nSequence shapes:")
    print("X_train:", train_x.shape, "| y_train:", train_y.shape)
    print("X_val:  ", val_x.shape,   "| y_val:  ", val_y.shape)
    print("X_test: ", test_x.shape,  "| y_test: ", test_y.shape)

    #model = build_model(config, input_size=seq_dict["X_train"].shape[2])

    # Quick debug checks
    for name, param in lstm_model.model.named_parameters():
        print(f"[DEBUG] First parameter tensor: {name}")
        print("[DEBUG] First 5 values:", param.detach().view(-1)[:5].cpu().numpy())
        break

    #xb0, yb0 = next(iter(seq_dict["train_loader"]))
    print("[DEBUG] First 5 y values from first train batch:", train_y[0][:5])

    criterion = nn.MSELoss()

    # Optimizer
    optimizer_name = config.optimizer.lower()
    lr = config.learning_rate
    weight_decay = config.weight_decay

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(lstm_model.model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(lstm_model.model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(lstm_model.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    scheduler = None
    scheduler_name = config.scheduler
    if scheduler_name:
        scheduler_name = scheduler_name.lower()
        if scheduler_name == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=config.scheduler_factor,
                patience=config.scheduler_patience,
            )
        elif scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.scheduler_step_size,
                gamma=config.scheduler_gamma,
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    train_output = lstm_model.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=config.training_epochs,
        patience=config.patience,
        scheduler=scheduler,
        show_live_plots=show_live_plots,
    )

    train_losses = train_output["train_losses"]
    val_losses = train_output["val_losses"]
    best_state = train_output["best_state"]
    best_epoch = train_output["best_epoch"]
    best_val_loss = train_output["best_val_loss"]

    lstm_model.model.load_state_dict(best_state)

    eval_dict = lstm_model.evaluate_model(
        test_loader=test_loader,
        tolerance_pct=10.0,
    )

    result = {
        "experiment_name": experiment_name,
        "seed": config.seed,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss) if best_val_loss is not None else np.nan,
        "test_rmse": eval_dict.get("rmse"),
        "test_mae": eval_dict.get("mae"),
        "test_mape": eval_dict.get("mape"),
        "test_r2": eval_dict.get("r2"),
        "test_acc_within_10pct": eval_dict.get("within_tol_acc"),
    }

    # Add selected config parameters into result row for easier sorting/filtering
    for k in [
        "model_type",
        "hidden_size",
        "num_layers",
        "dropout",
        "learning_rate",
        "use_mlp_head",
        "mlp_hidden_size",
        "lookback",
        "horizon",
        "demand_lags",
        "temp_lags",
        "num_attention_heads",
    ]:
        if hasattr(config, k):
            result[k] = getattr(config, k)

    # Optional checkpoint save
    if save_best_model:
        checkpoint_path = os.path.join(lstm_model.nsw_data_loader.output_dir, "LSTM_checkpoints")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        model_path = os.path.join(checkpoint_path, f"{experiment_name}_best.pt")
        torch.save(best_state, model_path)       
        print(f"Saved best model state to: {model_path}")

    artifact = {
        "config": deepcopy(config),
        "result": result,
        "model_class": lstm_model.model.__class__.__name__,
        "input_size": train_x.shape[2],
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_state": deepcopy(best_state),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "eval_dict": eval_dict,
    }

    return artifact

def build_config_list(base_config:LSTMBaseConfig, param_grid=None):
    """
    Builds a list of config dictionaries from a base config and optional param grid.
    If param_grid is None or empty, returns [base_config].
    """
    if not param_grid:
        return [deepcopy(base_config)]

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    configs = []
    for combo in product(*values):
        cfg = deepcopy(base_config)
        for k, v in zip(keys, combo):
            setattr(cfg, k, v)
        configs.append(cfg)
    return configs

def run_experiment_suite(
    base_config,
    param_grid=None,
    n_repeats=1,
    base_seed=42,
    save_dir=None,
    save_best_model=False,
    show_live_plots=False,
):
    """
    Unified runner for:
    - single experiment (param_grid=None, n_repeats=1)
    - parameter sweep (param_grid=..., n_repeats=1)
    - repeated experiments (param_grid=..., n_repeats>1)
    """

    config_list = build_config_list(base_config, param_grid)
    total_runs = len(config_list) * n_repeats

    all_rows = []
    all_artifacts = []

    run_counter = 0
    print(f"Total configurations: {len(config_list)}")
    print(f"Repeats per configuration: {n_repeats}")
    print(f"Total runs: {total_runs}")

    for cfg_idx, cfg in enumerate(config_list, start=1):
        config_name = _make_config_name(cfg)
        print(f"\n{'='*100}")
        print(f"[CONFIG {cfg_idx}/{len(config_list)}] {config_name}")
        print(f"{'='*100}")

        for repeat_idx in range(n_repeats):
            run_counter += 1
            run_seed = base_seed + repeat_idx

            cfg_run = deepcopy(cfg)
            setattr(cfg_run, "seed", run_seed)

            experiment_name = config_name if n_repeats == 1 else f"{config_name}__seed{run_seed}"

            print(f"\nRun {run_counter}/{total_runs} | repeat {repeat_idx + 1}/{n_repeats} | seed={run_seed}")

            artifact = run_experiment(
                config=cfg_run,
                experiment_name=experiment_name,
                save_best_model=save_best_model,
                show_live_plots=show_live_plots,
            )

            result = artifact.get("result", {})
            eval_dict = artifact.get("eval_dict", {})

            row = {
                "config_name": config_name,
                "experiment_name": experiment_name,
                "repeat_idx": repeat_idx + 1,
                "seed": run_seed,
                "model_type": getattr(cfg_run, "model_type", None),
                "hidden_size": getattr(cfg_run, "hidden_size", None),
                "num_layers": getattr(cfg_run, "num_layers", None),
                "dropout": getattr(cfg_run, "dropout", None),
                "learning_rate": getattr(cfg_run, "learning_rate", None),
                "use_mlp_head": getattr(cfg_run, "use_mlp_head", None),
                "mlp_hidden_size": getattr(cfg_run, "mlp_hidden_size", None),
                "lookback": getattr(cfg_run, "lookback", None),
                "horizon": getattr(cfg_run, "horizon", None),
                "num_attention_heads": getattr(cfg_run, "num_attention_heads", np.nan),
                "demand_lags": str(getattr(cfg_run, "demand_lags", [])),
                "temp_lags": str(getattr(cfg_run, "temp_lags", [])),
                "best_epoch": result.get("best_epoch"),
                "best_val_loss": result.get("best_val_loss"),
                "test_rmse": eval_dict.get("rmse"),
                "test_mae": eval_dict.get("mae"),
                "test_mape": eval_dict.get("mape"),
                "test_r2": eval_dict.get("r2"),
                "test_acc_within_10pct": eval_dict.get("within_tol_acc"),
            }

            all_rows.append(row)
            all_artifacts.append(artifact)

    runs_df = pd.DataFrame(all_rows)

    metric_cols = [
        "best_val_loss",
        "test_rmse",
        "test_mae",
        "test_mape",
        "test_r2",
        "test_acc_within_10pct",
    ]
    for col in metric_cols:
        if col in runs_df.columns:
            runs_df[col] = pd.to_numeric(runs_df[col], errors="coerce")

    if not runs_df.empty and "best_val_loss" in runs_df.columns:
        runs_df = runs_df.sort_values(
            by=["best_val_loss", "test_rmse"],
            ascending=[True, True]
        ).reset_index(drop=True)

    if save_dir:
        results_path = os.path.join(NSWDataLoader.output_dir, "LSTM_results")
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        runs_df.to_csv(os.path.join(results_path, f"{config_list[0].task_id}_all_runs_raw.csv"), index=False)
        print(f"Saved detailed rolling forecast results to {results_path}/{config_list[0].task_id}_all_runs_raw.csv")

        # Save a lightweight JSON copy of row-wise results
        with open(os.path.join(results_path, f"{config_list[0].task_id}_all_runs_raw.json"), "w") as f:
            json.dump(runs_df.to_dict(orient="records"), f, indent=2)

    return runs_df, all_artifacts

def summarise_runs(runs_df):
    metric_cols = [
        "best_val_loss",
        "test_rmse",
        "test_mae",
        "test_mape",
        "test_r2",
        "test_acc_within_10pct",
    ]

    group_cols = [
        "config_name",
        "model_type",
        "hidden_size",
        "num_layers",
        "dropout",
        "learning_rate",
        "use_mlp_head",
        "mlp_hidden_size",
        "lookback",
        "horizon",
        "demand_lags",
        "temp_lags",
        "num_attention_heads",
    ]

    # only keep columns that actually exist
    group_cols = [c for c in group_cols if c in runs_df.columns]

    summary_rows = []

    grouped = runs_df.groupby(group_cols, dropna=False)
    for group_key, group_df in grouped:
        row = dict(zip(group_cols, group_key))
        n = len(group_df)
        row["n_runs"] = n

        for metric in metric_cols:
            if metric not in group_df.columns:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
                row[f"{metric}_sem"] = np.nan
                row[f"{metric}_ci95_low"] = np.nan
                row[f"{metric}_ci95_high"] = np.nan
                continue

            vals = group_df[metric].dropna().astype(float).values

            if len(vals) == 0:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
                row[f"{metric}_sem"] = np.nan
                row[f"{metric}_ci95_low"] = np.nan
                row[f"{metric}_ci95_high"] = np.nan
                continue

            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            sem_val = std_val / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
            ci95 = 1.96 * sem_val

            row[f"{metric}_mean"] = mean_val
            row[f"{metric}_std"] = std_val
            row[f"{metric}_sem"] = sem_val
            row[f"{metric}_ci95_low"] = mean_val - ci95
            row[f"{metric}_ci95_high"] = mean_val + ci95

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    if not summary_df.empty and "test_rmse_mean" in summary_df.columns:
        summary_df = summary_df.sort_values(
            by=["test_rmse_mean", "test_mae_mean"],
            ascending=[True, True]
        ).reset_index(drop=True)

    return summary_df
    

if __name__ == "__main__":
    sarimax_config = SARIMAXConfig(
        task_id="sarimax_test",
        forecast_horizon=48,
        lookback_window=1440, # For SARIMAX, this is the number of most recent time steps to use for training.
        target_col='LOG_TOTALDEMAND',
        used_log_target=True,
        feature_cols=['demand_1_week_ago', 'demand_1_year_ago', 'TEMPERATURE','TEMP_SQUARED', 'IS_WEEKEND'],
        scale=True,
        p=[3],#2, 3, 4],
        d=[0],
        q=[0],
        P=[1],
        D=[1],
        Q=[1],
        seasonality_period=48,
        enforce_stationarity=True,
        enforce_invertibility=True
    )
    sarimax_model = SarimaxModel(sarimax_config)
    sarimax_model.train_model()
    all_origins, all_timestamps, all_actuals, all_predictions, mae, rmse, mse = sarimax_model.evaluate_model(None, test_mode=True)

    patchtst_config = TransformersConfig(
        task_id="patchtst_test",
        model="PatchTST",
        forecast_horizon=48,
        lookback_window=336,
        used_log_target=True,
        target_col='LOG_TOTALDEMAND',
        feature_cols=['TEMPERATURE', 'TEMP_SQUARED', 'IS_WEEKEND', 'demand_1_year_ago'],
        scale=True,
        date_col='DATETIME',
        variate='MS',
        patch_len=16,
        stride=8,
        d_model=128,
        num_attention_heads=4,
        num_encoder_layers=3,
        dim_ff=256,
        dropout=0.1,
        dropout_head_fc=0.1,
        use_gpu=True,
        time_encoding='timeF',
        shuffle_flag=True,
        training_epochs=1,
        batch_size=64,
        learning_rate=0.0001,
        output_attention=False,
        lradj='TST',
        patience=10,
    )
    patch_tst_model = PatchTSTModel(patchtst_config)
    patch_tst_model.train_model()
    patch_tst_model.evaluate_model(test_mode=1)

    RUN_SINGLE = False
    RUN_SWEEP = True
    RUN_REPEATED = False
    SEED = 42
    BASE_CONFIG = LSTMBaseConfig(
        task_id= "sweep_run_test",
        model_type= "multihead_attention_bilstm",
        hidden_size= 64,
        num_layers= 2,
        dropout= 0.4,
        learning_rate= 5e-5,
        batch_size= 64,
        training_epochs= 100,
        patience= 8,
        use_mlp_head= True,
        mlp_hidden_size= 64,
        target_col= "TOTALDEMAND",
        feature_cols= ["TEMPERATURE"],
        feature_lag_cols= ["TEMPERATURE"],
        demand_lags= [],
        feature_lags= [0,2,50],
        lookback_window= 168,
        forecast_horizon= 48,
        weight_decay= 1e-4,
        num_attention_heads= 4,
    )

    summary_path = os.path.join(NSWDataLoader.output_dir,"LSTM_summary")
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)

    # SINGLE EXPERIMENT
    if RUN_SINGLE:
        runs_df, all_artifacts = run_experiment_suite(
            base_config=BASE_CONFIG,
            param_grid=None,
            n_repeats=1,
            base_seed=SEED,
            save_dir=1,
            save_best_model=True,
            show_live_plots=False,
        )

        summary_df = summarise_runs(runs_df)
        summary_df.to_csv(os.path.join(summary_path, "single_run_summary.csv"), index=False)
        

        #best_artifact = plot_best_run(runs_df, all_artifacts, n_points=500)

        # if best_artifact["config"]["model_type"] in ["attention_bilstm", "multihead_attention_bilstm"]:
        #     plot_attention_for_sample(best_artifact, sample_index=0)

    if RUN_SWEEP:

        # PARAMETER SWEEP

        PARAM_GRID = {
            "target_col": ["LOG_TOTALDEMAND"], # ["TOTALDEMAND"]
            "used_log_target": [True], # [False]
            "lookback": [168], #[168, 336, 672],
            "horizon": [336],
            "model_type": ["lstm", "bilstm", "multihead_attention_bilstm"],
            "hidden_size": [64],
            "num_layers": [2],
            "dropout": [0.1],
            "learning_rate": [5e-5],
        }

        runs_df, all_artifacts = run_experiment_suite(
            base_config=BASE_CONFIG,
            param_grid=PARAM_GRID,
            n_repeats=1,
            base_seed=SEED,
            save_dir=1,
            save_best_model=True,
            show_live_plots=False
        )

        summary_df = summarise_runs(runs_df)
        summary_df.to_csv(os.path.join(summary_path, "sweep_run_summary.csv"), index=False)
        # print("\nRaw run results:")
        # display(runs_df.head())

        # print("\nSummary:")
        # display(summary_df)

        # best_artifact = plot_best_run(runs_df, all_artifacts, n_points=500)

        # if best_artifact["config"]["model_type"] in ["attention_bilstm", "multihead_attention_bilstm"]:
        #     plot_attention_for_sample(best_artifact, sample_index=0)

    # if RUN_REPEATED:
    #     # REPEATED EXPERIMENTS

    #     REPEAT_N = 5
    #     SAVE_DIR = "repeated_experiment_outputs"

    #     PARAM_GRID = {
    #         "lookback": [168, 336, 672],
    #         "horizon": [336],
    #         "model_type": ["lstm", "bilstm", "multihead_attention_bilstm"],
    #         "hidden_size": [64],
    #         "num_layers": [2],
    #         "dropout": [0.1],
    #         "learning_rate": [5e-5],
    #     }

    #     runs_df, all_artifacts = run_experiment_suite(
    #             base_config=BASE_CONFIG,
    #             param_grid=PARAM_GRID,
    #             n_repeats=REPEAT_N,
    #             base_seed=1000,
    #             save_dir=SAVE_DIR,
    #             save_best_model=False,
    #             show_live_plots=False,
    #     )

    #     summary_df = summarise_runs(runs_df)

    #     runs_df.to_csv(os.path.join(SAVE_DIR, "all_runs_raw.csv"), index=False)
    #     summary_df.to_csv(os.path.join(SAVE_DIR, "summary_stats.csv"), index=False)

    #     print("\nRaw run results:")
    #     display(runs_df.head())

    #     print("\nSummary results:")
    #     display(summary_df)

    #     plot_metric_distribution_with_summary(
    #         runs_df,
    #         metric="test_rmse",
    #         title=f"Test RMSE Distribution ({len(runs_df)} Runs)"
    #     )



    pass