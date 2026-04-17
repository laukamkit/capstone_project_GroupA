from datetime import datetime

from ModelFiles.GroupAModels import SarimaxModel
from ModelFiles.ModelConfigs import SARIMAXConfig, HORIZONS
from ModelFiles.ModelPlots import *

CONTEXT_LENGTHS = [720, 1440, 2880, 5760, 48*365] # 15 days, 30 days, 60 days, 120 days, 1 year.
USE_LOG_TARGET = True
DEBUG = False
EVAL_STEP_SIZE = 48

# For SARIMAX, we are not doing multiple seeds since fitting is deterministic.
for horizon in HORIZONS:
    for context_length in CONTEXT_LENGTHS:
            sarimax_config = SARIMAXConfig(
                task_id=f"sarimax_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                forecast_horizon=horizon,
                lookback_window=context_length, # For SARIMAX, this is the number of most recent time steps to use for training.
                target_col='LOG_TOTALDEMAND' if USE_LOG_TARGET else 'TOTALDEMAND',
                used_log_target=USE_LOG_TARGET,
                feature_cols=['demand_1_week_ago', 'demand_1_year_ago', 'TEMPERATURE','TEMP_SQUARED', 'IS_WEEKEND'],
                scale=True,
                # will perform grid search if any of the following parameters have more than 1 element
                p=[5,4,3], 
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
                save_test_results= False,
                eval_step_size=EVAL_STEP_SIZE,
                debug=DEBUG
            )
            sarimax_model = SarimaxModel(sarimax_config)
            sarimax_model.train_model()
            print("=" * 200)
            print("\n")