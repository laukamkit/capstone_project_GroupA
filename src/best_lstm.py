from ModelFiles.LSTM.LSTMUtils import *
from ModelFiles.ModelConfigs import LSTMConfig, SEEDS
from ModelFiles.ModelEnums import LSTMModelType
from ModelFiles.ModelPlots import *

use_log_target = False
EVAL_STEP_SIZE = 12
for seed in SEEDS:
    biLSTM_h48_c168_config = LSTMConfig(
        task_id= f"biLSTM_h48_c168_s{seed}",
        model_type= LSTMModelType.BILSTM,
        hidden_size= 128,
        num_layers= 2,
        dropout= 0.2,
        learning_rate= 0.00001,
        batch_size= 64,
        training_epochs= 100,
        patience= 8,
        use_mlp_head= True,
        mlp_hidden_size= 64,
        target_col= "TOTALDEMAND" if not use_log_target else "LOG_TOTALDEMAND",
        used_log_target= use_log_target,
        target_lags= [48, 336],
        target_mas= [],
        feature_cols= ["TEMPERATURE"],
        feature_lag_cols= [],
        lookback_window= 168,
        forecast_horizon= 48,
        num_attention_heads= 4,
        weight_decay= 1e-4,
        seed=seed,
        save_training_log= True,
        eval_step_size=EVAL_STEP_SIZE,
    )
    experiment_name_h48_c168 = make_config_name(biLSTM_h48_c168_config)
    runs_df_h48_c168_artifact = run_experiment(config=biLSTM_h48_c168_config)
    print("=" * 200)
    print("\n")

    biLSTM_h336_c336_config = LSTMConfig(
        task_id= f"biLSTM_h336_c336_s{seed}",
        model_type= LSTMModelType.BILSTM,
        hidden_size= 128,
        num_layers= 2,
        dropout= 0.2,
        learning_rate= 0.00001,
        batch_size= 64,
        training_epochs= 100,
        patience= 8,
        use_mlp_head= True,
        mlp_hidden_size= 64,
        target_col= "TOTALDEMAND" if not use_log_target else "LOG_TOTALDEMAND",
        used_log_target= use_log_target,
        feature_cols= ["TEMPERATURE"],
        feature_lag_cols= [],
        target_lags= [48, 336],
        target_mas= [],
        lookback_window= 336,
        forecast_horizon= 336,
        num_attention_heads= 4,
        weight_decay= 1e-4,
        seed=seed,
        save_training_log= True,
        eval_step_size=EVAL_STEP_SIZE,
    )
    experiment_name_h336_c336 = make_config_name(biLSTM_h336_c336_config)
    runs_df_h336_c336_artifacts = run_experiment(config=biLSTM_h336_c336_config)
    print("=" * 200)
    print("\n")

    biLSTM_h720_c720_config = LSTMConfig(
        task_id= f"biLSTM_h720_c720_s{seed}",
        model_type= LSTMModelType.BILSTM,
        hidden_size= 128,
        num_layers= 2,
        dropout= 0.2,
        learning_rate= 0.00001,
        batch_size= 64,
        training_epochs= 100,
        patience= 8,
        use_mlp_head= True,
        mlp_hidden_size= 64,
        target_col= "LOG_TOTALDEMAND" if use_log_target else "TOTALDEMAND",
        used_log_target= use_log_target,
        feature_cols= ["TEMPERATURE"],
        feature_lag_cols= [],
        target_lags= [48, 336],
        target_mas= [],
        lookback_window= 720,
        forecast_horizon= 720,
        num_attention_heads= 4,
        weight_decay= 1e-4,
        seed=seed,
        save_training_log= True,
        eval_step_size=EVAL_STEP_SIZE,
    )
    experiment_name_h720_c720 = make_config_name(biLSTM_h720_c720_config)
    runs_df_h720_c720_artifacts = run_experiment(config=biLSTM_h720_c720_config)
    print("=" * 200)
    print("\n")