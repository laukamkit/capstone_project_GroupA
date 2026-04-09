from ModelFiles.GroupAModels import PatchTSTModel
from ModelFiles.ModelConfigs import TransformersConfig, HORIZONS, CONTEXT_LENGTHS, SEEDS
from ModelFiles.ModelEnums import TransformerModelType
from ModelFiles.ModelPlots import *

for horizon in HORIZONS:
    for context_length in CONTEXT_LENGTHS:
        for seed in SEEDS:
            patchtst_config = TransformersConfig(
                task_id=f"patchtst_run_h{horizon}_c{context_length}_s{seed}",
                model=TransformerModelType.PATCHTST,
                forecast_horizon=horizon,
                lookback_window=context_length,
                used_log_target=True,
                target_col='LOG_TOTALDEMAND',
                feature_cols=['TEMPERATURE', 'TEMP_SQUARED', 'IS_WEEKEND', 'demand_1_year_ago'],
                scale=True,
                date_col='DATETIME',
                variate='MS',
                patch_len=16,
                stride=8,
                d_model=128,
                num_attention_heads=4,
                num_encoder_layers=3,
                dim_ff=256,
                dropout=0.1,
                dropout_head_fc=0.1,
                use_gpu=True,
                time_encoding='timeF',
                shuffle_flag=True,
                training_epochs=100,
                batch_size=64,
                learning_rate=0.0001,
                output_attention=False,
                lradj='TST',
                patience=10,
                seed=seed
            )
            patch_tst_model = PatchTSTModel(patchtst_config)
            patch_tst_model.train_model()
            patch_tst_model.evaluate_model(test_mode=1)
            print("=" * 50)
            print("\n")