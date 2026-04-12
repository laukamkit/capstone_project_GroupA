from datetime import datetime
from ModelFiles.GroupAModels import TransformersModel
from ModelFiles.ModelConfigs import TransformersConfig, HORIZONS, SEEDS
from ModelFiles.ModelEnums import TransformerModelType
from ModelFiles.ModelPlots import *

USE_LOG_TARGET = True
CONTEXT_LENGTHS = [336, 720]
EVAL_STEP_SIZE = 48
NUM_EPOCHS = 1
PATIENCE = 10
DEBUG = False

for horizon in HORIZONS:
    for context_length in CONTEXT_LENGTHS:
        if context_length >= horizon:
            for seed in SEEDS:
                if seed == SEEDS[-1]:
                    save_prediction_results = True
                else:
                    save_prediction_results = False

                timexer_config = TransformersConfig(
                    task_id=f"timexer_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    model=TransformerModelType.TIMEXER,
                    forecast_horizon=horizon,
                    lookback_window=context_length,
                    used_log_target=USE_LOG_TARGET,
                    target_col="LOG_TOTALDEMAND" if USE_LOG_TARGET else "TOTALDEMAND",
                    feature_cols=['TEMPERATURE', 'TEMP_SQUARED', 'IS_WEEKEND', 'demand_1_year_ago'],
                    scale=True,
                    date_col='DATETIME',
                    variate='M',
                    patch_len=16,
                    stride=16,  # TimeXer uses patch_len as stride internally
                    d_model=256,
                    num_attention_heads=16,
                    num_encoder_layers=4,
                    dim_ff=256,
                    dropout=0.2,
                    dropout_head_fc=0.2,
                    use_gpu=True,
                    time_encoding='timeF',
                    shuffle_flag=True,
                    training_epochs=NUM_EPOCHS,
                    batch_size=32,
                    learning_rate=0.0001,
                    output_attention=False,
                    lradj='type1',
                    patience=PATIENCE,
                    seed=seed,
                    eval_step_size=EVAL_STEP_SIZE,
                    save_test_results=False,#save_prediction_results,
                    debug=DEBUG,
                    save_training_log=False,#True,
                    use_norm=True,
                    activation='gelu',
                )
                timexer_model = TransformersModel(timexer_config)
                timexer_model.train_model()
                timexer_model.evaluate_model(test_mode=1)
                print("=" * 200)
                print("\n")
