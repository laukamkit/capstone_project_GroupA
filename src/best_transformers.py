from ModelFiles.GroupAModels import PatchTSTModel
from ModelFiles.ModelConfigs import TransformersConfig, HORIZONS, SEEDS
from ModelFiles.ModelEnums import TransformerModelType
from ModelFiles.ModelPlots import *

USE_LOG_TARGET = True
CONTEXT_LENGTHS = [336, 512] # two variants tested in PatchTST paper
EVAL_STEP_SIZE = 96
NUM_EPOCHS = 100
PATIENCE = 10
DEBUG = False

def make_task_id(config: TransformersConfig, other_suffix: str = "") -> TransformersConfig:
    name = f"patchtst_run_h{config.forecast_horizon}_c{config.lookback_window}_s{config.seed}"
    if USE_LOG_TARGET:
        name += "_logtarget"
    if other_suffix:
        name += f"_{other_suffix}"
    config.task_id = name
    return config

for horizon in HORIZONS:
    for context_length in CONTEXT_LENGTHS:
        for seed in SEEDS:
            if seed == SEEDS[-1]:
                save_prediction_results = True
            else:
                save_prediction_results = False

            patchtst_config = TransformersConfig(
                task_id=f"patchtst_run_h{horizon}_c{context_length}_s{seed}",
                model=TransformerModelType.PATCHTST,
                forecast_horizon=horizon,
                lookback_window=context_length,
                used_log_target=USE_LOG_TARGET,
                target_col= "LOG_TOTALDEMAND" if USE_LOG_TARGET else "TOTALDEMAND",
                feature_cols=['TEMPERATURE', 'TEMP_SQUARED', 'IS_WEEKEND', 'demand_1_year_ago'],
                scale=True,
                date_col='DATETIME',
                variate='MS',
                patch_len=16,
                stride=8,
                d_model=128,
                num_attention_heads=16,
                num_encoder_layers=3,
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
                lradj='TST',
                patience=PATIENCE,
                seed=seed,
                eval_step_size=EVAL_STEP_SIZE,
                save_test_results=save_prediction_results,
                debug=DEBUG,
                save_training_log=True,
            )
        patch_tst_config = make_task_id(patchtst_config)
        patch_tst_model = PatchTSTModel(patch_tst_config)
        patch_tst_model.train_model()
        patch_tst_model.evaluate_model(test_mode=1)
        print("=" * 200)
        print("\n")