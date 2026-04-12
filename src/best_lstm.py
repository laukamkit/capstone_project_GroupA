from datetime import datetime

from ModelFiles.LSTM.LSTMUtils import *
from ModelFiles.ModelConfigs import LSTMConfig, SEEDS
from ModelFiles.ModelEnums import LSTMModelType
from ModelFiles.ModelPlots import *

use_log_target = True
EVAL_STEP_SIZE = 48
FORECAST_LAST_STEP_ONLY = [False, True]
NUM_EPOCHS = 100
PATIENCE = 10
DEBUG = False
HORIZON_LOOKBACK = [(48, 168), (336, 336), (720, 720)]
# hidden_size, num_layers, dropout, learning_rate, num_attention_heads, batch_size
HYPERPARAMETER_COMBINATIONS = [(512, 8, 0.2, 0.00001, 12, 64), (512, 12, 0.2, 0.00001, 16, 64)]
SEEDS = [31415] # run once first to see preliminary results.
for horizon, lookback in HORIZON_LOOKBACK:
    for hidden_size, num_layers, dropout, learning_rate, num_attention_heads, batch_size in HYPERPARAMETER_COMBINATIONS:
        for forecast_last_step_only in FORECAST_LAST_STEP_ONLY:
            for seed in SEEDS:
                if seed == SEEDS[-1]:
                    save_prediction_results = True
                else:
                    save_prediction_results = False
                biLSTM_config = LSTMConfig(
                    task_id= f"biLSTM_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    model_type= LSTMModelType.BILSTM,
                    hidden_size= hidden_size,
                    num_layers= num_layers,
                    dropout= dropout,
                    learning_rate= learning_rate,
                    batch_size= batch_size,
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
                    lookback_window= lookback,
                    forecast_horizon= horizon,
                    forecast_last_step_only=forecast_last_step_only,
                    num_attention_heads= num_attention_heads,
                    weight_decay= 1e-4,
                    scale=True,
                    seed=seed,
                    save_training_log= True,
                    save_test_results= save_prediction_results,
                    eval_step_size=EVAL_STEP_SIZE,
                    debug= DEBUG,
                )
                runs_df_artifact = run_experiment(config=biLSTM_config, experiment_name=biLSTM_config.task_id)
                print("=" * 200)
                print("\n")