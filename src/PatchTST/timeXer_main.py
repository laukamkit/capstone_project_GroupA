import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TimeXer Runs for Single-Variable and Multi-Variable Time Series Forecasting')
    parser.add_argument('--pred_type', type=str, default='single', help='type of experiment: single or multi')
    parser.add_argument('--is_training', type=int, default=1, help='whether to train the model: 1 for training, 0 for testing')
    args = parser.parse_args()

    HORIZONS = [48, 336, 720]#[48, 96, 168, 336, 720]
    CONTEXT_LENGTHS = [144, 336, 720]#[168, 336, 720]   # 7d, 14d, 30d history
    SEEDS = [31415, 27182, 14142]  # example seeds for reproducibility
    BATCH_SIZE = 64
    EPOCHS = 20
    PATIENCE = 5
    model_ids = [f'h{h}_c{c}' for h in HORIZONS for c in CONTEXT_LENGTHS if c >= h]
    for horizon in HORIZONS:
        for context_length in CONTEXT_LENGTHS:
            if context_length >= horizon:
                for seed in SEEDS:
                    print(f'Running experiment with horizon {horizon}, context length {context_length}, and seed {seed}...')
                    # You can set the hyperparameters for each experiment here. For simplicity, I'm just setting seq_len and pred_len.
                    # You may want to set other hyperparameters as well (e.g. learning rate, batch size, etc.) based on the specific experiment.
                    single_var_run_string = (f'python src/PatchTST/patchtst_supervised.py --model_id '
                                f'"TimeXer_h{horizon}_c{context_length}_seed{seed}_Single_Variate" '
                                f'--model TimeXer '
                                f'--is_training {args.is_training} '
                                f'--features MS '
                                f'--enc_in 2 '
                                f'--dec_in 2 '
                                f'--c_out 1 '
                                f'--feature_cols TEMPERATURE '
                                f'--e_layers 4 '
                                f'--factor 3 '
                                f'--d_model 512 '
                                f'--d_ff 512 '
                                f'--embed timeF '
                                f'--train_epochs {EPOCHS} '
                                f'--batch_size {BATCH_SIZE} '
                                f'--patience {PATIENCE} '
                                f'--seq_len {context_length} '
                                f'--pred_len {horizon} '
                                f'--lradj type1 '
                                f'--random_seed {seed}')

                    multi_var_run_string = (f'python src/PatchTST/patchtst_supervised.py --model_id '
                                f'"TimeXer_h{horizon}_c{context_length}_seed{seed}_Multi_Variate" '
                                f'--model TimeXer '
                                f'--is_training {args.is_training} '
                                f'--features M '
                                f'--enc_in 2 '
                                f'--dec_in 2 '
                                f'--c_out 2 '
                                f'--feature_cols TEMPERATURE '
                                f'--e_layers 4 '
                                f'--factor 3 '
                                f'--d_model 512 '
                                f'--d_ff 512 '
                                f'--embed timeF '
                                f'--train_epochs {EPOCHS} '
                                f'--batch_size {BATCH_SIZE} '
                                f'--patience {PATIENCE} '
                                f'--seq_len {context_length} '
                                f'--pred_len {horizon} '
                                f'--lradj type1 '
                                f'--random_seed {seed}')

                    if args.pred_type == 'single':
                        os.system(single_var_run_string)
                    elif args.pred_type == 'multi':
                        os.system(multi_var_run_string)
