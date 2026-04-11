from ModelFiles.GroupAModels import SarimaxModel
from ModelFiles.ModelConfigs import SARIMAXConfig, HORIZONS, SEEDS
from ModelFiles.ModelPlots import *

# This is the number of most recent time steps to use for training.
CONTEXT_LENGTHS = [720]
# For SARIMAX, we are not doing multiple seeds since it is a deterministic model.
for horizon in HORIZONS:
    for context_length in CONTEXT_LENGTHS:
            sarimax_config = SARIMAXConfig(
                task_id=f"sarimax_run_h{horizon}_c{context_length}",
                forecast_horizon=horizon,
                lookback_window=context_length, # For SARIMAX, this is the number of most recent time steps to use for training.
                target_col='LOG_TOTALDEMAND',
                used_log_target=True,
                feature_cols=['demand_1_week_ago', 'demand_1_year_ago', 'TEMPERATURE','TEMP_SQUARED', 'IS_WEEKEND'],
                scale=True,
                # will perform grid search if any of the following parameters have more than 1 element
                p=[3], 
                d=[0],
                q=[0],
                P=[1],
                D=[1],
                Q=[1],
                seasonality_period=48,
                enforce_stationarity=True,
                enforce_invertibility=True,
                seed=None,
                save_training_log= True,
                eval_step_size=48,
                debug=True
            )
            sarimax_model = SarimaxModel(sarimax_config)
            sarimax_model.train_model()
            all_origins, all_timestamps, all_actuals, all_predictions, mae, rmse, mse = sarimax_model.evaluate_model(None, test_mode=True)
            print("=" * 200)
            print("\n")