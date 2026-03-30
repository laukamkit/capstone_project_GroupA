import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PatchTST Runs for Single-Variable and Multi-Variable Time Series Forecasting')
    parser.add_argument('--type', type=str, default='single', help='type of experiment: single or multi')
    args = parser.parse_args()

    HORIZONS = [48, 96, 168]#[48, 96, 168, 336, 720]
    CONTEXT_LENGTHS = [336]#[168, 336, 720]   # 7d, 14d, 30d history
    SEEDS = [2021, 2022, 2023]
    model_ids = [f'h{h}_c{c}' for h in HORIZONS for c in CONTEXT_LENGTHS if c >= h]
    for horizon in HORIZONS:
        for context_length in CONTEXT_LENGTHS:
            if context_length >= horizon:
                for seed in SEEDS:
                    print(f'Running experiment with horizon {horizon}, context length {context_length}, and seed {seed}...')
                    # You can set the hyperparameters for each experiment here. For simplicity, I'm just setting seq_len and pred_len.
                    # You may want to set other hyperparameters as well (e.g. learning rate, batch size, etc.) based on the specific experiment.
                    single_var_run_string = (f'python src/patchtst_supervised.py --model_id '
                                  f'"PatchTST_h{horizon}_c{context_length}_Single_Variate" '
                                  f'--features S '
                                  f'--enc_in 1 '
                                  f'--scale '
                                  f'--seq_len {context_length} '
                                  f'--pred_len {horizon} '
                                  f'--random_seed {seed}')

                    multi_var_run_string = (f'python src/patchtst_supervised.py --model_id '
                                  f'"PatchTST_h{horizon}_c{context_length}_Multi_Variate" '
                                  f'--features MS '
                                  f'--enc_in 2 '
                                  f'--feature_cols TEMPERATURE '
                                  f'--scale '
                                  f'--seq_len {context_length} '
                                  f'--pred_len {horizon} '
                                  f'--random_seed {seed}')

                    if args.type == 'single':
                        os.system(single_var_run_string)
                    elif args.type == 'multi':
                        os.system(multi_var_run_string)
