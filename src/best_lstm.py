from ModelFiles.LSTM.LSTMUtils import *
from ModelFiles.ModelConfigs import LSTMConfig, SEEDS
from ModelFiles.ModelEnums import LSTMModelType
from ModelFiles.ModelPlots import *

use_log_target = True
EVAL_STEP_SIZE = 96
FORECAST_LAST_STEP_ONLY = [False, True]
NUM_EPOCHS = 100
PATIENCE = 10
DEBUG = False

def make_task_id(config: LSTMConfig, other_suffix: str = "") -> LSTMConfig:
    name = f"{config.model_type.value}_h{config.forecast_horizon}_c{config.lookback_window}_s{config.seed}"
    if use_log_target:
        name += "_logtarget"
    if config.forecast_last_step_only:
        name += "_laststep"
    else:
        name += "_multistep"
    if other_suffix:
        name += f"_{other_suffix}"
    config.task_id = name
    return config

for forecast_last_step_only in FORECAST_LAST_STEP_ONLY:
    for seed in SEEDS:
        if seed == SEEDS[-1]:
            save_prediction_results = True
        else:
            save_prediction_results = False
            
        biLSTM_h48_c168_config = LSTMConfig(
            task_id= "ph",
            model_type= LSTMModelType.BILSTM,
            hidden_size= 128,
            num_layers= 2,
            dropout= 0.2,
            learning_rate= 0.00001,
            batch_size= 64,
            training_epochs= NUM_EPOCHS,
            patience= PATIENCE,
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
            forecast_last_step_only=forecast_last_step_only,
            num_attention_heads= 4,
            weight_decay= 1e-4,
            scale=True,
            seed=seed,
            save_training_log= True,
            save_test_results= save_prediction_results,
            eval_step_size=EVAL_STEP_SIZE,
            debug= DEBUG,
        )
        biLSTM_h48_c168_config = make_task_id(biLSTM_h48_c168_config)
        experiment_name_h48_c168 = make_config_name(biLSTM_h48_c168_config)
        runs_df_h48_c168_artifact = run_experiment(config=biLSTM_h48_c168_config)
        print("=" * 200)
        print("\n")

        biLSTM_h336_c336_config = LSTMConfig(
            task_id= "ph",
            model_type= LSTMModelType.BILSTM,
            hidden_size= 128,
            num_layers= 2,
            dropout= 0.2,
            learning_rate= 0.00001,
            batch_size= 64,
            training_epochs= NUM_EPOCHS,
            patience= PATIENCE,
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
            scale=True,
            seed=seed,
            save_training_log= True,
            save_test_results= save_prediction_results,
            eval_step_size=EVAL_STEP_SIZE,
            forecast_last_step_only=forecast_last_step_only,
            debug= DEBUG,
        )
        biLSTM_h336_c336_config = make_task_id(biLSTM_h336_c336_config)
        experiment_name_h336_c336 = make_config_name(biLSTM_h336_c336_config)
        runs_df_h336_c336_artifacts = run_experiment(config=biLSTM_h336_c336_config)
        print("=" * 200)
        print("\n")

        biLSTM_h720_c720_config = LSTMConfig(
            task_id= "ph",
            model_type= LSTMModelType.BILSTM,
            hidden_size= 128,
            num_layers= 2,
            dropout= 0.2,
            learning_rate= 0.00001,
            batch_size= 64,
            training_epochs= NUM_EPOCHS,
            patience= PATIENCE,
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
            scale=True,
            seed=seed,
            save_training_log= True,
            save_test_results= save_prediction_results,
            eval_step_size=EVAL_STEP_SIZE,
            forecast_last_step_only=forecast_last_step_only,
            debug= DEBUG,
        )
        biLSTM_h720_c720_config = make_task_id(biLSTM_h720_c720_config)
        experiment_name_h720_c720 = make_config_name(biLSTM_h720_c720_config)
        runs_df_h720_c720_artifacts = run_experiment(config=biLSTM_h720_c720_config)
        print("=" * 200)
        print("\n")