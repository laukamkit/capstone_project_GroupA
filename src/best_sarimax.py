from ModelFiles.GroupAModels import SarimaxModel
from ModelFiles.ModelConfigs import SARIMAXConfig, HORIZONS, CONTEXT_LENGTHS, SEEDS
from ModelFiles.ModelPlots import *

for horizon in HORIZONS:
    for context_length in CONTEXT_LENGTHS:
        for seed in SEEDS:
            sarimax_config = SARIMAXConfig(
                task_id=f"sarimax_run_h{horizon}_c{context_length}_s{seed}",
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
                seed=seed,
                val_step_size=48, # For SARIMAX, this is the step size to use when evaluating against validation or test set.
            )
            sarimax_model = SarimaxModel(sarimax_config)
            sarimax_model.train_model()
            all_origins, all_timestamps, all_actuals, all_predictions, mae, rmse, mse = sarimax_model.evaluate_model(None, test_mode=True)
            print("=" * 50)
            print("\n")