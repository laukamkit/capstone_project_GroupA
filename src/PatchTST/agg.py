import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

results_dir = Path(__file__).parent.parent / 'results'

pred_arrays = {}
true_arrays = {}
timestamps_arrays = {}
metrics_arrays = {}
results = [results_dir.rglob('pred.npy'), results_dir.rglob('true.npy'), results_dir.rglob('metrics.npy'), results_dir.rglob('ts.npy')]

for pred_npy_file, true_npy_file, metrics_npy_file, ts_npy_file in zip(*results):
    key = f'{pred_npy_file.parent.name}/{pred_npy_file.name}'
    pred_arrays[key] = np.load(pred_npy_file)
    true_arrays[key] = np.load(true_npy_file)
    metrics_arrays[key] = np.load(metrics_npy_file)
    timestamps_arrays[key] = np.load(ts_npy_file)

horizon = np.inf
pred_df_list = []
master_merged_df = pd.DataFrame()

def save_metrics():
    metrics_df = pd.concat(
        [pd.DataFrame(list(metrics_arrays.keys()),columns=['model']), 
         pd.DataFrame(list(metrics_arrays.values()), columns=['mae', 'mse', 'rmse', 'mape', 'mspe', 'rse', 'corr_mean'])], axis=1)
    metrics_df.to_csv(results_dir / 'metrics_summary.csv', index=False)

def load_and_merge_data():
    print(f'Loading...')
    for key in pred_arrays.keys():
        if pred_arrays[key].shape[1] <= horizon:
            horizon = pred_arrays[key].shape[1]-1
        temp_pred_df = pd.DataFrame(list(pred_arrays[key][:, :, 0]))
        temp_true_df = pd.DataFrame(list(true_arrays[key][:, :, 0]))
        temp_ts_df = pd.DataFrame(pd.to_datetime(list(timestamps_arrays[key]), unit='us'))
        merged_df = pd.merge(temp_pred_df.iloc[:,horizon].rename('pred'), temp_true_df.iloc[:,horizon].rename('true'), left_index=True, right_index=True, suffixes=('_pred', '_true'))
        merged_df = pd.merge(merged_df, temp_ts_df.iloc[:,horizon].rename('timestamp'), left_index=True, right_index=True)
        model = key.split('/')[0]
        merged_df['model'] = model
        merged_df['horizon'] = horizon+1
        merged_df = merged_df[['model','horizon'] + [col for col in merged_df.columns if col not in ['model','horizon']]]
        master_merged_df = pd.concat([master_merged_df, merged_df], axis=0)


def plot_models():
    load_and_merge_data()
    days_start_offset = 400
    days_end_offset = 7
    for model in master_merged_df['model'].unique():
        print(f'\nPlotting {model}...')
        start_offset_timedelta = pd.Timedelta(days=days_start_offset)  # adjust as needed
        end_offset_timedelta = pd.Timedelta(days=days_end_offset)  # adjust as needed
        sns.set_style('whitegrid')
        plt.figure(figsize=(12, 6))
        alt_model = None
        variate = 'Single_Variate' if 'Single_Variate' in model else 'Multi_Variate'
        if 'Single_Variate' in model:
            alt_model = model.replace('Single_Variate','Multi_Variate').replace('ftS','ftMS')
        elif 'Multi_Variate' in model:
            alt_model = model.replace('Multi_Variate','Single_Variate').replace('ftMS','ftS')
        temp_master_df = master_merged_df.copy()
        temp_master_df = temp_master_df[(temp_master_df['model'] == model) | (temp_master_df['model'] == alt_model)]
        date_start = pd.to_datetime(temp_master_df['timestamp'].min()) + start_offset_timedelta
        if date_start > temp_master_df['timestamp'].max():
            date_start = temp_master_df['timestamp'].max() - end_offset_timedelta
        date_end = date_start + end_offset_timedelta
        horizon = temp_master_df['horizon'].iloc[0]
        sns.lineplot(
            data=temp_master_df[(temp_master_df['timestamp'] >= date_start) & (temp_master_df['timestamp'] <= date_end)], 
            x='timestamp', y='pred', markers=True, dashes=False, color='blue', alpha=0.7, label='Predicted')
        sns.lineplot(
            data=temp_master_df[(temp_master_df['timestamp'] >= date_start) & (temp_master_df['timestamp'] <= date_end)], 
            x='timestamp', y='true', markers=True, dashes=True, color='red', alpha=0.7, label='True')
        sns.lineplot(
            data=temp_master_df[(temp_master_df['model'] == alt_model) & (temp_master_df['timestamp'] >= date_start) & (temp_master_df['timestamp'] <= date_end)], 
            x='timestamp', y='pred', markers=True, dashes=False, color='green', alpha=0.7, label=f'Alt. Predicted')
        plt.title(f'{variate} Horizon {horizon} from {date_start.date()} to {date_end.date()}')
        plt.xlabel('Timestamp')
        plt.ylabel('Value')
        # x axis units to show year, month, day, hour and 30 minute intervals
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=45)
        plt.savefig(results_dir / f'{model}_horizon{horizon}.png', bbox_inches='tight', dpi=1000)
        plt.close()

    print('Done.')

save_metrics()