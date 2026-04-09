from ModelFiles.GroupAModels import GradientBoostingModel
from ModelFiles.ModelConfigs import GradientBoostingConfig, HORIZONS, CONTEXT_LENGTHS, SEEDS
from ModelFiles.ModelPlots import *
from ModelFiles.LSTM.LSTMUtils import add_lag_features # shared utility function with LSTM

for horizon in HORIZONS:
    for context_length in CONTEXT_LENGTHS:
        for seed in SEEDS:
            gb_config = GradientBoostingConfig(
                task_id=f"gb_run_h{horizon}_c{context_length}_s{seed}",
                forecast_horizon=horizon,
                lookback_window=context_length, # For Gradient Boosting, this is the number of most recent time steps to use for training.
                target_col='LOG_TOTALDEMAND',
                target_lags=[1,2,3,48,96],
                target_mas=[6, 48],
                used_log_target=True,
                feature_cols=['TEMPERATURE','TEMP_SQUARED', 'HOUR', 'DAYOFWEEK', 'IS_WEEKEND'],
                n_estimators=300,
                learning_rate=0.1,
                max_depth=3,
                verbose=1,
                scale=True,
                seed=seed,
                eval_step_size=96, # This is the step size to use when evaluating against validation or test set.
            )
            print(f"\nRunning Gradient Boosting with config: {gb_config}\n")
            gb_model = GradientBoostingModel(gb_config, add_lag_features)
            gb_model.train_model()
            all_actuals, all_predictions, rmse, mae = gb_model.evaluate_model(None, test_mode=1)
            print("=" * 50)
            print("\n")