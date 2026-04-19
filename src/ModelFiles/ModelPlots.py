import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

def plot_training_history(train_losses, val_losses, best_epoch=None, title="Training History"):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    if best_epoch is not None:
        plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.3, label='Best Epoch')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (scaled)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_predictions(results_dict: dict[str,dict[int,pd.DataFrame]], horizon:int, start_date: str):
    plt.figure(figsize=(12, 5))
    model_names = results_dict.keys()
    for i, model_name in enumerate(model_names):
        min_date = pd.to_datetime(max(results_dict[model_name][horizon]['timestamp'].min(), results_dict['SARIMAX'][horizon]['timestamp'].min()))
        max_date = pd.to_datetime(min(results_dict[model_name][horizon]['timestamp'].max(), results_dict['SARIMAX'][horizon]['timestamp'].max()))
        _start_date = pd.to_datetime(start_date)
        assert _start_date >= min_date and _start_date <= max_date, f"start_date is out of range: {_start_date} not in [{min_date}, {max_date}]"
        _end_date = _start_date + pd.Timedelta(hours=horizon // 2)
        assert _end_date >= min_date and _end_date <= max_date, f"end_date is out of range: {_end_date} not in [{min_date}, {max_date}]"
        df = results_dict[model_name][horizon]
        start_index = df[(df['timestamp'] >= start_date) & (df['horizon'] == 0)].index[0]
        end_index = start_index + horizon - 1
        df = df.loc[start_index:end_index]
        df = df[['timestamp', 'y_pred', 'y_actual']]
        if i == 0:
            actuals = df['y_actual']  # use the first model's actuals for plotting
        plt.plot(df['timestamp'], df['y_pred'], label=model_name)
    plt.plot(df['timestamp'], actuals, label='Actual', linestyle='--')
    plt.legend()
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    # xticks every 6 hours (12 data points) for better readability
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(12))
    plt.xlabel('Timestamp')
    plt.ylabel('Electricity Demand (MW)')
    plt.title(f'Model Predictions vs Actuals (Horizon={horizon} steps)')
    plt.grid(True, alpha=0.3)
    plt.show()