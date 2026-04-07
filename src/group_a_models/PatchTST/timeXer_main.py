import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TimeXer Runs for Single-Variable and Multi-Variable Time Series Forecasting')
    parser.add_argument('--pred_type', type=str, default='single', help='type of experiment: single or multi')
    parser.add_argument('--is_training', type=int, default=1, help='whether to train the model: 1 for training, 0 for testing')
    parser.add_argument('--model_id', type=str, required=False, default='UNK', help='model id')
    args = parser.parse_args()

    HORIZONS = [336]#[48, 96, 168, 336, 720]
    CONTEXT_LENGTHS = [336, 720]#[168, 336, 720]   # 7d, 14d, 30d history
    SEEDS = [31415]  # example seeds for reproducibility
    BATCH_SIZE = 64
    EPOCHS = 50
    PATIENCE = 10

    for horizon in HORIZONS:
        for context_length in CONTEXT_LENGTHS:
            if context_length >= horizon:
                for seed in SEEDS:
                    print(f'Running experiment with horizon {horizon}, context length {context_length}, and seed {seed}...')
                    # You can set the hyperparameters for each experiment here. For simplicity, I'm just setting seq_len and pred_len.
                    # You may want to set other hyperparameters as well (e.g. learning rate, batch size, etc.) based on the specific experiment.
                    model_id = args.model_id if args.model_id != 'UNK' else f'TimeXer_h{horizon}_c{context_length}_seed{seed}'
                    no_features_var_run_string = (f'python src/PatchTST/patchtst_supervised.py --model_id '
                                f'"{model_id}_Single_Variate" '
                                f'--model TimeXer '
                                f'--is_training {args.is_training} '
                                f'--features S '
                                f'--enc_in 1 '
                                f'--dec_in 1 '
                                f'--c_out 1 '
                                f'--e_layers 4 '
                                f'--factor 3 '
                                f'--d_model 256 '
                                f'--d_ff 256 '
                                f'--embed timeF '
                                f'--train_epochs {EPOCHS} '
                                f'--batch_size {BATCH_SIZE} '
                                f'--patience {PATIENCE} '
                                f'--seq_len {context_length} '
                                f'--pred_len {horizon} '
                                f'--scale '
                                f'--inverse '
                                f'--lradj type1 '
                                f'--random_seed {seed}')
                    
                    temp_only_var_run_string = (f'python src/PatchTST/patchtst_supervised.py --model_id '
                                f'"{model_id}_temp_only" '
                                f'--model TimeXer '
                                f'--is_training {args.is_training} '
                                f'--features MS '
                                f'--enc_in 2 '
                                f'--dec_in 2 '
                                f'--c_out 1 '
                                f'--feature_cols TEMPERATURE '
                                f'--e_layers 4 '
                                f'--factor 3 '
                                f'--d_model 256 '
                                f'--d_ff 256 '
                                f'--embed timeF '
                                f'--train_epochs {EPOCHS} '
                                f'--batch_size {BATCH_SIZE} '
                                f'--patience {PATIENCE} '
                                f'--seq_len {context_length} '
                                f'--pred_len {horizon} '
                                f'--scale '
                                f'--inverse '
                                f'--lradj type1 '
                                f'--random_seed {seed}')
                    
                    lag_only_var_run_string = (f'python src/PatchTST/patchtst_supervised.py --model_id '
                                f'"{model_id}_w_Temp_Lag" '
                                f'--model TimeXer '
                                f'--is_training {args.is_training} '
                                f'--features MS '
                                f'--enc_in 4 '
                                f'--dec_in 4 '
                                f'--c_out 1 '
                                f'--feature_cols demand_1_day_ago demand_1_week_ago demand_1_year_ago '
                                f'--e_layers 4 '
                                f'--factor 3 '
                                f'--d_model 256 '
                                f'--d_ff 256 '
                                f'--embed timeF '
                                f'--train_epochs {EPOCHS} '
                                f'--batch_size {BATCH_SIZE} '
                                f'--patience {PATIENCE} '
                                f'--seq_len {context_length} '
                                f'--pred_len {horizon} '
                                f'--scale '
                                f'--inverse '
                                f'--lradj type1 '
                                f'--random_seed {seed}')
                    
                    rolling_only_var_run_string = (f'python src/PatchTST/patchtst_supervised.py --model_id '
                                f'"{model_id}_w_Temp_Lag" '
                                f'--model TimeXer '
                                f'--is_training {args.is_training} '
                                f'--features MS '
                                f'--enc_in 3 '
                                f'--dec_in 3 '
                                f'--c_out 1 '
                                f'--feature_cols rolling_mean_1_day rolling_mean_1_week '
                                f'--e_layers 4 '
                                f'--factor 3 '
                                f'--d_model 256 '
                                f'--d_ff 256 '
                                f'--embed timeF '
                                f'--train_epochs {EPOCHS} '
                                f'--batch_size {BATCH_SIZE} '
                                f'--patience {PATIENCE} '
                                f'--seq_len {context_length} '
                                f'--pred_len {horizon} '
                                f'--scale '
                                f'--inverse '
                                f'--lradj type1 '
                                f'--random_seed {seed}')

                    stly_only_var_run_string = (f'python src/PatchTST/patchtst_supervised.py --model_id '
                                f'"{model_id}_w_Temp_Lag" '
                                f'--model TimeXer '
                                f'--is_training {args.is_training} '
                                f'--features MS '
                                f'--enc_in 2 '
                                f'--dec_in 2 '
                                f'--c_out 1 '
                                f'--feature_cols target_demand_1_year_ago_h_{horizon} '
                                f'--e_layers 4 '
                                f'--factor 3 '
                                f'--d_model 256 '
                                f'--d_ff 256 '
                                f'--embed timeF '
                                f'--train_epochs {EPOCHS} '
                                f'--batch_size {BATCH_SIZE} '
                                f'--patience {PATIENCE} '
                                f'--seq_len {context_length} '
                                f'--pred_len {horizon} '
                                f'--scale '
                                f'--inverse '
                                f'--lradj type1 '
                                f'--random_seed {seed}')

                    if args.pred_type == '1':
                        os.system(no_features_var_run_string)
                    elif args.pred_type == '2':
                        os.system(temp_only_var_run_string)
                    elif args.pred_type == '3':
                        os.system(lag_only_var_run_string)
                    elif args.pred_type == '4':
                        os.system(rolling_only_var_run_string)
                    elif args.pred_type == '5':
                        os.system(stly_only_var_run_string)
