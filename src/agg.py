import numpy as np
import pandas as pd
from pathlib import Path

results_dir = Path(__file__).parent.parent / 'results'

pred_arrays = {}
true_arrays = {}
timestamps_arrays = {}
results = [results_dir.rglob('pred.npy'), results_dir.rglob('true.npy'), results_dir.rglob('ts.npy')]

for pred_npy_file, true_npy_file, ts_npy_file in zip(*results):
    key = f'{pred_npy_file.parent.name}/{pred_npy_file.name}'
    pred_arrays[key] = np.load(pred_npy_file)
    true_arrays[key] = np.load(true_npy_file)
    timestamps_arrays[key] = np.load(ts_npy_file)
    print(f'Loaded {key}: shape={pred_arrays[key].shape}, {true_arrays[key].shape}, {timestamps_arrays[key].shape}')

pred_df_list = []
for key, _ in pred_arrays.items():
    print(f'{key}: pred shape={pred_arrays[key].shape}, true shape={true_arrays[key].shape}')
    temp_df = pd.DataFrame(list(pred_arrays[key][:,:,0]))
    temp_df['MODEL'] = key.split('/')[0]
    pred_df_list.append(temp_df)

true_df_list = []
for key, _ in true_arrays.items():
    print(f'{key}: true shape={true_arrays[key].shape}, true shape={true_arrays[key].shape}')
    temp_df = pd.DataFrame(list(true_arrays[key][:,:,0]))
    temp_df['MODEL'] = key.split('/')[0]
    true_df_list.append(temp_df)

ts_df_list = []
for key, _ in timestamps_arrays.items():
    print(f'{key}: ts shape={timestamps_arrays[key].shape}')
    temp_df = pd.DataFrame(pd.to_datetime(list(timestamps_arrays[key]), unit='us'))
    temp_df['MODEL'] = key.split('/')[0]
    ts_df_list.append(temp_df)

pass