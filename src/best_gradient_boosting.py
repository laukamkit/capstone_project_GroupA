from ModelFiles.GroupAModels import GradientBoostingModel
from ModelFiles.ModelConfigs import GradientBoostingConfig, HORIZONS, SEEDS
from ModelFiles.ModelPlots import *
from ModelFiles.LSTM.LSTMUtils import add_lag_features # shared utility function with LSTM

CONTEXT_LENGTH = None # GB we are fitting entire results set. Need to find literature to support this though.
USE_LOG_TARGET = True
EVAL_STEP_SIZE = 96
N_ESTIMATORS = [100, 200, 300]
DEBUG = False

def make_task_id(config: GradientBoostingConfig, other_suffix: str = "") -> GradientBoostingConfig:
    name = f"gb_run_h{config.forecast_horizon}_c{'None' if CONTEXT_LENGTH is None else CONTEXT_LENGTH}_s{config.seed}"
    if USE_LOG_TARGET:
        name += "_logtarget"
    if other_suffix:
        name += f"_{other_suffix}"
    config.task_id = name
    return config

for n_estimator in N_ESTIMATORS:
    for horizon in HORIZONS:
        for seed in SEEDS:
            if seed == SEEDS[-1]:
                save_prediction_results = True
            else:
                save_prediction_results = False

            gb_config = GradientBoostingConfig(
                task_id=f"gb_run_h{horizon}_c{'None' if CONTEXT_LENGTH is None else CONTEXT_LENGTH}_s{seed}",
                forecast_horizon=horizon,
                lookback_window=CONTEXT_LENGTH, # For Gradient Boosting, this is the number of most recent time steps to use for training.
                target_col='LOG_TOTALDEMAND' if USE_LOG_TARGET else 'TOTALDEMAND',
                target_lags=[1,2,3,48,96],
                target_mas=[6, 48],
                used_log_target=USE_LOG_TARGET,
                feature_cols=['TEMPERATURE','TEMP_SQUARED', 'HOUR', 'DAYOFWEEK', 'IS_WEEKEND'],
                n_estimators=n_estimator,
                learning_rate=0.1,
                max_depth=3,
                verbose=1,
                scale=True,
                seed=seed,
                eval_step_size=EVAL_STEP_SIZE, # This is the step size to use when evaluating against validation or test set.
                debug=DEBUG,
                save_training_log=True,
                save_test_results=save_prediction_results,
            )
            gb_config = make_task_id(gb_config)
            print(f"\nRunning Gradient Boosting with config: {gb_config}\n")
            gb_model = GradientBoostingModel(gb_config, add_lag_features)
            gb_model.train_model()
            all_actuals, all_predictions, rmse, mae = gb_model.evaluate_model(None, test_mode=1)
            print("=" * 200)
            print("\n")