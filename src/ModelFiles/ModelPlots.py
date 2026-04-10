import matplotlib.pyplot as plt

def plot_training_history(train_losses, val_losses, title="Training History"):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (scaled)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_predictions(y_true_real, y_pred_real, n_points=500, title="Forecast vs Actual"):
    plt.figure(figsize=(12, 5))
    plt.plot(y_true_real[:n_points], label="Actual")
    plt.plot(y_pred_real[:n_points], label="Predicted")
    plt.xlabel("Test Sample")
    plt.ylabel("Demand")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def live_plot_losses(train_losses, val_losses, title="Training History"):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()   