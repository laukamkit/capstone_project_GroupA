from datetime import datetime

from ModelFiles.GroupAModels import GradientBoostingModel
from ModelFiles.ModelConfigs import GradientBoostingConfig, HORIZONS, SEEDS
from ModelFiles.ModelPlots import *
from ModelFiles.LSTM.LSTMUtils import add_lag_features # shared utility function with LSTM

CONTEXT_LENGTH = None # GB we are fitting entire results set. Need to find literature to support this though.
USE_LOG_TARGET = True
EVAL_STEP_SIZE = 48
N_ESTIMATORS = [100,200,300,600,1000,1500]
DEBUG = False

for n_estimator in N_ESTIMATORS:
    for horizon in HORIZONS:
        for seed in SEEDS:
            if seed == SEEDS[-1]:
                save_prediction_results = True
            else:
                save_prediction_results = False

            gb_config = GradientBoostingConfig(
                task_id=f"gb_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
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
            print(f"\nRunning Gradient Boosting with config: {gb_config}\n")
            gb_model = GradientBoostingModel(gb_config, add_lag_features)
            gb_model.train_model()
            all_actuals, all_predictions, rmse, mae = gb_model.evaluate_model(None, test_mode=1)
            print("=" * 200)
            print("\n")