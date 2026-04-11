from ModelFiles.GroupAModels import GradientBoostingModel
from ModelFiles.ModelConfigs import GradientBoostingConfig, HORIZONS, SEEDS
from ModelFiles.ModelPlots import *
from ModelFiles.LSTM.LSTMUtils import add_lag_features # shared utility function with LSTM

CONTEXT_LENGTH = None # GB we are fitting entire results set. Need to find literature to support this though.
HORIZONS = [336,720]
skip = [57721, 66987, 11235, 98765, 43210]
SEEDS = [31415, 27182, 14142, 17320, 22360, 57721, 66987, 11235, 98765, 43210]
N_ESTIMATORS = 10
DEBUG = True
for horizon in HORIZONS:
    for seed in SEEDS:
        if seed in skip and horizon == 336:
            continue
        gb_config = GradientBoostingConfig(
            task_id=f"gb_run_h{horizon}_c{'None' if CONTEXT_LENGTH is None else CONTEXT_LENGTH}_s{seed}",
            forecast_horizon=horizon,
            lookback_window=CONTEXT_LENGTH, # For Gradient Boosting, this is the number of most recent time steps to use for training.
            target_col='LOG_TOTALDEMAND',
            target_lags=[1,2,3,48,96],
            target_mas=[6, 48],
            used_log_target=True,
            feature_cols=['TEMPERATURE','TEMP_SQUARED', 'HOUR', 'DAYOFWEEK', 'IS_WEEKEND'],
            n_estimators=N_ESTIMATORS,
            learning_rate=0.1,
            max_depth=3,
            verbose=1,
            scale=True,
            seed=seed,
            eval_step_size=96, # This is the step size to use when evaluating against validation or test set.
            debug=DEBUG,
            save_training_log=True,
        )
        print(f"\nRunning Gradient Boosting with config: {gb_config}\n")
        gb_model = GradientBoostingModel(gb_config, add_lag_features)
        gb_model.train_model()
        all_actuals, all_predictions, rmse, mae = gb_model.evaluate_model(None, test_mode=1)
        print("=" * 200)
        print("\n")