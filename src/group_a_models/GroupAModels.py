import pandas as pd, numpy as np, os, torch, torch.nn as nn
from PatchTST_supervised.models import PatchTST, TimeXer
from PatchTST_supervised.models import iTransformer
from PatchTST_supervised.utils.tools import EarlyStopping, adjust_learning_rate, test_params_flop
from PatchTST_supervised.utils.metrics import metric
from LSTM.models import AttentionBiLSTMForecaster, MultiHeadAttentionBiLSTMForecaster, SequenceForecaster
from nsw_data_loader.nsw_data_loader import NSWDataLoader
from model_configs import TransformersConfig, SARIMAXConfig, LSTMBaseConfig
from BasicModels import BaseModel, DeepLearningModel
from torch.utils.data import DataLoader
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from time import time
from copy import deepcopy
from torch import optim
from torch.optim import lr_scheduler
from typing import Callable

class LSTMModel(DeepLearningModel):
    def __init__(self, config: LSTMBaseConfig, func: Callable[[pd.DataFrame,str ,list[str],list[int] | None, list[int] | None], pd.DataFrame] | None = None):
        super().__init__(config, func)
        self.config: LSTMBaseConfig = config
        self.model = self._build_model(config, input_size=len(self.config.all_feature_cols)+1)
        
    def _build_model(self, config, input_size):
        model_type = config.model_type.lower()

        if model_type == "lstm":
            model = SequenceForecaster(
                input_size=input_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                dropout=config.dropout,
                bidirectional=False,
                use_mlp_head=config.use_mlp_head,
                mlp_hidden_size=config.mlp_hidden_size,
            )

        elif model_type == "bilstm":
            model = SequenceForecaster(
                input_size=input_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                dropout=config.dropout,
                bidirectional=True,
                use_mlp_head=config.use_mlp_head,
                mlp_hidden_size=config.mlp_hidden_size,
            )

        elif model_type == "attention_bilstm":
            model = AttentionBiLSTMForecaster(
                input_size=input_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                dropout=config.dropout,
                bidirectional=True,
                use_mlp_head=config.use_mlp_head,
                mlp_hidden_size=config.mlp_hidden_size,
            )

        elif model_type == "multihead_attention_bilstm":
            model = MultiHeadAttentionBiLSTMForecaster(
                input_size=input_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                dropout=config.dropout,
                bidirectional=True,
                use_mlp_head=config.use_mlp_head,
                mlp_hidden_size=config.mlp_hidden_size,
                num_attention_heads=config.num_attention_heads,
            )

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        return model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    
    # def live_plot_losses(self, train_losses, val_losses, title="Training History"):
    #     plt.figure(figsize=(8, 4))
    #     plt.plot(train_losses, label="Train Loss")
    #     plt.plot(val_losses, label="Val Loss")
    #     plt.xlabel("Epoch")
    #     plt.ylabel("MSE Loss")
    #     plt.title(title)
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)
    #     plt.tight_layout()
    #     plt.show()


    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion,
        optimizer,
        epochs=50,
        patience=10,
        scheduler=None,
        show_live_plots=False,
        title="Training History",
    ):
        train_losses = []
        val_losses = []

        best_val_loss = float("inf")
        best_epoch = -1
        best_state = None
        epochs_no_improve = 0

        for epoch in range(epochs):
            self.model.train()
            running_train_loss = 0.0

            for X_batch, y_batch, _, _, _ in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch[:, -1].to(self.device) # predicting only the last timestep in the forecast horizon

                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item() * X_batch.size(0)

            epoch_train_loss = running_train_loss / len(train_loader.dataset)
            train_losses.append(epoch_train_loss)

            self.model.eval()
            running_val_loss = 0.0

            with torch.no_grad():
                for X_batch, y_batch, _, _, _ in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch[:, -1].to(self.device) # predicting only the last timestep in the forecast horizon

                    y_pred = self.model(X_batch)
                    loss = criterion(y_pred, y_batch)

                    running_val_loss += loss.item() * X_batch.size(0)

            epoch_val_loss = running_val_loss / len(val_loader.dataset)
            val_losses.append(epoch_val_loss)

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(epoch_val_loss)
                else:
                    scheduler.step()

            # Stops the training when the val loss meets a certain condition
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_epoch = epoch
                best_state = deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            print(
                f"Epoch [{epoch+1}/{epochs}] | "
                f"Train Loss: {epoch_train_loss:.6f} | "
                f"Val Loss: {epoch_val_loss:.6f}"
            )

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        if best_state is None:
            best_state = deepcopy(self.model.state_dict())

        # if show_live_plots:
        #     plot_training_history(train_losses, val_losses, title=title)

        return {
            "model": self.model,
            "best_state": best_state,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }
    
    def _compute_inverse_scaling(self, shape, pred, true):
            pos = self.nsw_data_loader.scaler.feature_names_in_.tolist().index(self.config.target_col)
            mean = self.nsw_data_loader.scaler.mean_[pos]
            var = self.nsw_data_loader.scaler.var_[pos]
            pred_tile = np.tile(pred, [1, 1, 1])
            pred_inverse = pred_tile.reshape(shape[0] * shape[1], 1)
            pred_inverse = pred_inverse* var**0.5 + mean
            pred_inverse = pred_inverse[:, -1:].reshape(shape)
            true_tiled = np.tile(true, [1, 1, 1])
            true_inverse = true_tiled.reshape(shape[0] * shape[1], 1)
            true_inverse = true_inverse*var**0.5 + mean
            true_inverse = true_inverse[:, -1:].reshape(shape)
            return pred_inverse, true_inverse
    
    def evaluate_model(self, test_loader, tolerance_pct=10.0):
        self.model.eval()
        y_pred_scaled = []
        y_true_scaled = []

        with torch.no_grad():
            for X_batch, y_batch, _, _, _ in test_loader:
                X_batch = X_batch.to(self.device)
                preds = self.model(X_batch).cpu().numpy()

                y_pred_scaled.extend(preds)
                y_true_scaled.extend(y_batch[:, -1].numpy())

        y_pred_scaled = np.array(y_pred_scaled).reshape(-1, 1)
        y_true_scaled = np.array(y_true_scaled).reshape(-1, 1)

        y_pred_real, y_true_real = self._compute_inverse_scaling(y_pred_scaled.shape, y_pred_scaled, y_true_scaled)
        if self.config.used_log_target:
            y_pred_real = np.exp(y_pred_real)
            y_true_real = np.exp(y_true_real)
        y_pred_real = y_pred_real.ravel()
        y_true_real = y_true_real.ravel()

        rmse = np.sqrt(mean_squared_error(y_true_real, y_pred_real))
        mae = mean_absolute_error(y_true_real, y_pred_real)
        r2 = r2_score(y_true_real, y_pred_real)

        mask = np.abs(y_true_real) > 1e-8
        mape = (
            np.mean(np.abs((y_true_real[mask] - y_pred_real[mask]) / y_true_real[mask])) * 100
            if np.any(mask) else np.nan
        )

        tolerance = tolerance_pct / 100.0
        within_tol_acc = np.mean(
            np.abs(y_pred_real - y_true_real) <= tolerance * np.abs(y_true_real)
        ) * 100

        return {
            "y_pred_scaled": y_pred_scaled,
            "y_true_scaled": y_true_scaled,
            "y_pred_real": y_pred_real,
            "y_true_real": y_true_real,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2": r2,
            "within_tol_acc": within_tol_acc,
        }







    # def plot_training_history(self, train_losses, val_losses, title="Training History"):
    #     plt.figure(figsize=(8, 4))
    #     plt.plot(train_losses, label="Train Loss")
    #     plt.plot(val_losses, label="Val Loss")
    #     plt.xlabel("Epoch")
    #     plt.ylabel("MSE Loss (scaled)")
    #     plt.title(title)
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)
    #     plt.tight_layout()
    #     plt.show()


    # def plot_predictions(self,y_true_real, y_pred_real, n_points=500, title="Forecast vs Actual"):
    #     plt.figure(figsize=(12, 5))
    #     plt.plot(y_true_real[:n_points], label="Actual")
    #     plt.plot(y_pred_real[:n_points], label="Predicted")
    #     plt.xlabel("Test Sample")
    #     plt.ylabel("Demand")
    #     plt.title(title)
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)
    #     plt.tight_layout()
    #     plt.show()


class PatchTSTModel(DeepLearningModel):
    # This class is a wrapper around the official PatchTST implementation by the official authors but their data
    def __init__(self, config: TransformersConfig):
        self.variate = config.variate
        self.patch_len = config.patch_len
        self.stride = config.stride
        self.d_model = config.d_model
        self.num_attention_heads = config.num_attention_heads
        self.num_encoder_layers = config.num_encoder_layers
        self.dim_ff = config.dim_ff
        self.dropout_ff = config.dropout_ff
        self.dropout_head_fc = config.dropout_head_fc
        super().__init__(config)
        self.config: TransformersConfig = config
        self.model = self._build_model().to(self.device)
   
    def _build_model(self):
        model_dict = {
            'PatchTST': PatchTST,
            'iTransformer': iTransformer,
            'TimeXer': TimeXer
        }
        model = model_dict[self.config.model].Model(self.config).float()
        return model
    
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def _compute_inverse_scaling(self, shape, pred, true):
        pos = self.nsw_data_loader.scaler.feature_names_in_.tolist().index(self.config.target_col)
        mean = self.nsw_data_loader.scaler.mean_[pos]
        var = self.nsw_data_loader.scaler.var_[pos]
        pred_tile = np.tile(pred, [1, 1, 1])
        pred_inverse = pred_tile.reshape(shape[0] * shape[1], 1)
        pred_inverse = pred_inverse* var**0.5 + mean
        pred_inverse = pred_inverse[:, -1:].reshape(shape)
        true_tiled = np.tile(true, [1, 1, 1])
        true_inverse = true_tiled.reshape(shape[0] * shape[1], 1)
        true_inverse = true_inverse*var**0.5 + mean
        true_inverse = true_inverse[:, -1:].reshape(shape)
        return pred_inverse, true_inverse
    
    def evaluate_model(self, test_mode=0):
        criterion = self._select_criterion()
        test_data, test_loader = self._get_data(flag='test' if test_mode else 'val')

        if test_mode:
            results_path = os.path.join(NSWDataLoader.output_dir, f"{self.config.model}_results")
            if not os.path.exists(results_path):
                os.makedirs(results_path)
            checkpoint_path = os.path.join(NSWDataLoader.output_dir, f"{self.config.model}_checkpoints")
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(checkpoint_path, f'{self.config.task_id}_checkpoint.pth')))

        preds = []
        trues = []
        timestamps = []
        indices = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_time) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if 'Linear' in self.config.model or 'TST' in self.config.model:
                        outputs = self.model(batch_x)
                else:
                    if self.config.output_attention:
                        outputs = self.model(batch_x, batch_x_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark)

                f_dim = -1 if self.config.variate == 'MS' else 0
                outputs = outputs[:, -self.config.forecast_horizon:, f_dim:]
                batch_y = batch_y[:, -self.config.forecast_horizon:, f_dim:].to(self.device)
                batch_time = batch_time[:, -self.config.forecast_horizon:]
                outputs = outputs.detach().cpu()
                batch_y = batch_y.detach().cpu()
                
                batch_time = batch_time.detach().cpu().numpy()
                ts = batch_time
                idx = np.zeros(ts.shape)+i
                indices.append(idx)
                
                if self.config.scale:
                    pred, true = self._compute_inverse_scaling(batch_y.shape, outputs, batch_y)
                else:
                    pred = outputs.numpy()
                    true = batch_y.numpy()
                if self.config.used_log_target:
                    pred = np.exp(pred)
                    true = np.exp(true)
                preds.append(pred)
                trues.append(true)
                timestamps.append(ts)

        preds = np.array(preds)
        trues = np.array(trues)
        timestamps = np.array(timestamps, dtype='<M8[us]')
        indices = np.array(indices)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        timestamps = timestamps.reshape(-1, timestamps.shape[-1])
        indices = indices.reshape(-1, indices.shape[-1])
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        
        # result save
        results_df = pd.DataFrame({
            'index': indices.flatten().astype(int),
            'timestamp': timestamps.flatten(),
            'y_actual': trues.flatten(),
            'y_pred': preds.flatten(),
            "rmse": rmse,
            "mse": mse,
            "mae": mae
        })
        if test_mode:
            print(f"Test Results - MAE: {mae:.2f} MW | RMSE: {rmse:.2f} MW | MSE: {mse:.2f} MW^2")
            results_df.to_csv(os.path.join(results_path, f"{self.config.task_id}_test_results.csv"), index=False)
            print(f"Saved detailed rolling forecast results to {results_path}/{self.config.task_id}_test_results.csv")
        else:
            self.model.train()
        return indices.flatten().astype(int), timestamps.flatten(), trues.flatten(), preds.flatten(), mae, rmse, mse

    
    def train_model(self):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        checkpoint_path = os.path.join(NSWDataLoader.output_dir, f"{self.config.model}_checkpoints")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        results_path = os.path.join(NSWDataLoader.output_dir, f"{self.config.model}_results")
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        time_now = time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.config.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.config.pct_start,
                                            epochs = self.config.training_epochs,
                                            max_lr = self.config.learning_rate)
        progress_log = {
            'model_name': [],
            'lookback_window': [],
            'epoch': [],
            'validation_horizon': [],
            'validation_rmse': [],
            'validation_mse': [],
            'time_taken_seconds': []
        }

        for epoch in range(self.config.training_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, _) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # encoder - decoder
                if 'Linear' in self.config.model or 'TST' in self.config.model:
                        outputs = self.model(batch_x)
                else:
                    if self.config.output_attention:
                        outputs = self.model(batch_x, batch_x_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark)
                # print(outputs.shape,batch_y.shape)
                f_dim = -1 if self.config.variate == 'MS' else 0
                outputs = outputs[:, -self.config.forecast_horizon:, f_dim:]
                batch_y = batch_y[:, -self.config.forecast_horizon:, f_dim:].to(self.device)

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time() - time_now) / iter_count
                    left_time = speed * ((self.config.training_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time()

                loss.backward()
                model_optim.step()
                    
                if self.config.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.config, printout=False)
                    scheduler.step()
            cost_time = time() - epoch_time
            print("Epoch: {} cost time: {}".format(epoch + 1, cost_time))
            train_loss = np.average(train_loss)
            _, _, _, _, mae_val, rmse_val, mse_val = self.evaluate_model()
            progress_log['model_name'].append(self.config.task_id)
            progress_log['lookback_window'].append(self.config.lookback_window)
            progress_log['epoch'].append(epoch + 1)
            progress_log['validation_horizon'].append(self.config.forecast_horizon)
            progress_log['validation_rmse'].append(rmse_val)
            progress_log['validation_mse'].append(mse_val)
            progress_log['time_taken_seconds'].append(cost_time)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss (RMSE): {3:.7f}".format(
                epoch + 1, train_steps, train_loss, rmse_val))
            early_stopping(rmse_val, self.model, checkpoint_path, self.config.task_id)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.config.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.config)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = checkpoint_path + '/' + f'{self.config.task_id}_checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        fitting_progress_log_df = pd.DataFrame(progress_log)
        fitting_progress_log_df.to_csv(os.path.join(results_path, f"{self.config.task_id}_fitting_log.csv"), index=False)
        return self.model


class SarimaxModel(BaseModel):
    def __init__(self, config: SARIMAXConfig):
        super().__init__(config)
        self.p = config.p
        self.d = config.d
        self.q = config.q
        self.P = config.P
        self.D = config.D
        self.Q = config.Q
        self.seasonality_period = config.seasonality_period
        self.enforce_stationarity = config.enforce_stationarity
        self.enforce_invertibility = config.enforce_invertibility
        self.val_step_size = config.val_step_size
        self.best_model: SARIMAXResultsWrapper | None = None
        self.best_order = None
        self.best_seasonal_order = None
        self.config: SARIMAXConfig = config

    def evaluate_model(self, model_fit:SARIMAXResultsWrapper | str | None, test_mode: bool = False):
        if model_fit is None:
            if self.best_model is None:
                raise ValueError("No model provided for testing. Please provide a fitted SARIMAXResultsWrapper object or a model file name, or ensure that train_model() has been called to train and set the best_model.")
            else:
                _model_fit = self.best_model
        elif isinstance(model_fit, str):
            _model_fit = SARIMAXResultsWrapper.load(os.path.join(self.nsw_data_loader.output_dir, "sarimax_models", model_fit))
            if _model_fit is None:
                raise ValueError("Model not found in sarimax_models directory. Please check the file name and try again or use train_model() to train a new model.")
        elif isinstance(model_fit, SARIMAXResultsWrapper):
            _model_fit = model_fit
        else:
            raise ValueError("model_fit must be either None, a file name string, or a SARIMAXResultsWrapper object.")

        rolling_state = deepcopy(_model_fit)

        if test_mode:
            test_df = self.test_data.copy()
        else:
            test_df = self.validation_data.copy()

        exog_vars_df = test_df[self.config.feature_cols] if self.config.feature_cols else None
        test_df = test_df[self.config.target_col]

        assert len(test_df) >= self.config.forecast_horizon, "Test target data points must be at least as many as the forecast horizon"
        assert exog_vars_df is None or len(exog_vars_df) >= self.config.forecast_horizon, "Exogenous variables data points must be at least as many as the forecast horizon"

        origins = range(0, len(test_df) - self.config.forecast_horizon, self.config.val_step_size)
        print(f"\tNumber of rolling forecast origins: {len(list(origins))}")

        all_actuals = []
        all_predictions = []
        all_timestamps = []   
        all_origins = []  

        for i in origins:
            # 1. First, make the forecast from the current state
            future_exog = exog_vars_df.iloc[i : i + self.config.forecast_horizon] if exog_vars_df is not None else None
            forecast = rolling_state.forecast(steps=self.config.forecast_horizon, exog=future_exog)
            actuals = test_df.iloc[i : i + self.config.forecast_horizon]
            if self.config.scale:
                pos = self.nsw_data_loader.scaler.feature_names_in_.tolist().index(self.config.target_col)
                mean = self.nsw_data_loader.scaler.mean_[pos]
                var = self.nsw_data_loader.scaler.var_[pos]
                forecast = forecast*var**0.5 + mean
                actuals = actuals*var**0.5 + mean
            if self.config.used_log_target:
                forecast = np.exp(forecast)
                actuals = np.exp(actuals)
            # Store predictions and actuals
            all_predictions.extend(forecast.values)
            all_actuals.extend(actuals.values)
            all_timestamps.extend(test_df.index[i : i + self.config.forecast_horizon].tolist())
            all_origins.extend([i // self.config.val_step_size] * self.config.forecast_horizon)
            
            if (i // self.config.val_step_size) % 100 == 0 and i > 0:
                mae_so_far  = mean_absolute_error(all_actuals, all_predictions)
                mse_so_far  = mean_squared_error(all_actuals, all_predictions)
                rmse_so_far = np.sqrt(mse_so_far)
                print(f"\t\tOrigin {i//self.config.val_step_size} out of {len(origins)}: Validation MAE: {mae_so_far:.2f} MW | Validation RMSE so far: {rmse_so_far:.2f} MW")

            # 2. Then, advance the Origin forward by 'val_step_size'
            # We do this by APPENDING the model state with the actuals that occurred during that step
            new_endog = test_df.iloc[i : i + self.config.val_step_size]
            new_exog = exog_vars_df.iloc[i : i + self.config.val_step_size] if exog_vars_df is not None else None
            
            # .extend() updates the auto-regressive state of the model with the new observed data, so that the next forecast will be based on this updated state.
            # This simulates the real-world scenario where after making a forecast, we observe the actual outcome and then use that information to make the next forecast.
            rolling_state = rolling_state.extend(new_endog, exog=new_exog)

        mae  = mean_absolute_error(all_actuals, all_predictions)
        mse  = mean_squared_error(all_actuals, all_predictions)
        rmse = np.sqrt(mse)
        print(f"Rolling Forecast (h={self.config.forecast_horizon} steps) — Validation MAE: {mae:.2f} MW | Validation RMSE: {rmse:.2f} MW")
        results_df = pd.DataFrame({
            'index': all_origins,
            'timestamp': all_timestamps,
            'y_actual': all_actuals,
            'y_pred': all_predictions,
            'rmse': [rmse] * len(all_timestamps),
            'mse': [mse] * len(all_timestamps),
            'mae': [mae] * len(all_timestamps)
        })
        if test_mode:
            os.makedirs(os.path.join(NSWDataLoader.output_dir, "sarimax_results"), exist_ok=True)
            results_df.to_csv(os.path.join(NSWDataLoader.output_dir, "sarimax_results", f"{self.config.task_id}_test_results.csv"), index=False)
            print(f"Saved detailed rolling forecast results to sarimax_results/{self.config.task_id}_test_results.csv")
        return all_origins, all_timestamps, all_actuals, all_predictions, mae, rmse, mse
        
    def train_model(self):
        # only use the most recent 'lookback_window' data points for training, if lookback_window is specified in the config
        self.training_data = self.training_data.iloc[-self.config.lookback_window:] if self.config.lookback_window else self.training_data
        best_training_aic = np.inf
        best_val_mse = np.inf
        best_order = None
        best_seasonal_order = None
        progress_log = {
            'model_name': [],
            'order': [],
            'seasonal_order': [],
            'training_aic': [],
            'validation_horizon': [],
            'validation_iters': [],
            'validation_mae': [],
            'validation_mse': [],
            'validation_rmse': [],
            'time_taken_seconds': []
        }
        for p in self.p:
            for d in self.d:
                for q in self.q:
                    for p_s in self.P:
                        for d_s in self.D:
                            for q_s in self.Q:
                                    print(f"Fitting SARIMAX({p},{d},{q})({p_s},{d_s},{q_s},{self.seasonality_period})...")
                                    start = time()
                                    model = SARIMAX(
                                        endog=self.training_data[self.config.target_col], 
                                        exog=self.training_data[self.config.feature_cols] if self.config.feature_cols else None,
                                        order=(p,d,q), 
                                        seasonal_order=(p_s,d_s,q_s,self.seasonality_period),
                                        enforce_stationarity=self.enforce_stationarity,
                                        enforce_invertibility=self.enforce_invertibility
                                    )
                                    model_fit = model.fit(disp=False, warn_convergence=False)
                                    end = time()
                                    print(f"\tFitted SARIMAX({p},{d},{q})({p_s},{d_s},{q_s},{self.seasonality_period}) - Training AIC: {model_fit.aic:.2f} | Time taken: {end - start:.2f} seconds\n")
                                    try:
                                        print(f"\tValidating SARIMAX({p},{d},{q})({p_s},{d_s},{q_s},{self.seasonality_period}) on horizon {self.config.forecast_horizon} with step size {self.config.val_step_size}...")
                                        _, _, _, _, mae, rmse, mse = self.evaluate_model(model_fit)
                                        if mse < best_val_mse:
                                            print(f"New best model found: order ({p},{d},{q}) | seasonal_order ({p_s},{d_s},{q_s},{self.seasonality_period}) - RMSE: {rmse:.2f}")
                                            best_val_mse = mse
                                            best_order = (p,d,q)
                                            best_seasonal_order = (p_s,d_s,q_s,self.seasonality_period)
                                            best_model = model_fit
                                            best_training_aic = model_fit.aic
                                            progress_log['model_name'].append(self.config.task_id)
                                            progress_log['order'].append(best_order)
                                            progress_log['seasonal_order'].append(best_seasonal_order)
                                            progress_log['validation_horizon'].append(self.config.forecast_horizon)
                                            progress_log['validation_iters'].append(self.config.val_step_size)
                                            progress_log['training_aic'].append(model_fit.aic)
                                            progress_log['validation_mae'].append(mae)
                                            progress_log['validation_rmse'].append(rmse)
                                            progress_log['validation_mse'].append(mse)
                                            progress_log['time_taken_seconds'].append(end - start)
                                    except Exception as e:
                                        print(e)
                                        print(f"\tError validating SARIMAX({p},{d},{q})({p_s},{d_s},{q_s},{self.seasonality_period}) on horizon {self.config.forecast_horizon} with step size {self.config.val_step_size}: {e}")
        print(f"Best SARIMAX{best_order}{best_seasonal_order} - AIC: {best_training_aic:.2f} | Validation MSE: {best_val_mse:.2f} | Validation RMSE: {np.sqrt(best_val_mse):.2f} | Time taken: {progress_log['time_taken_seconds'][-1]:.2f} seconds")
        print("\nNow fitting the best model with both training and validation data with lookback_window applied...")
        train_val_data = pd.concat([self.training_data, self.validation_data])[-self.config.lookback_window:] if self.config.lookback_window else pd.concat([self.training_data, self.validation_data])
        best_model = SARIMAX(
            endog=train_val_data[self.config.target_col],
            exog=train_val_data[self.config.feature_cols] if self.config.feature_cols else None,
            order=best_order,
            seasonal_order=best_seasonal_order,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility
        ).fit(disp=False, warn_convergence=False)
        self.best_model = best_model
        self.best_order = best_order
        self.best_seasonal_order = best_seasonal_order
        fitting_progress_log_df = pd.DataFrame(progress_log)
        os.makedirs(os.path.join(NSWDataLoader.output_dir, "sarimax_results"), exist_ok=True)
        fitting_progress_log_df.to_csv(os.path.join(NSWDataLoader.output_dir, "sarimax_results", f"{self.config.task_id}_fitting_log.csv"), index=False)
        os.makedirs(os.path.join(NSWDataLoader.output_dir, "sarimax_models"), exist_ok=True)
        self.best_model.save(os.path.join(NSWDataLoader.output_dir, "sarimax_models", f"{self.config.task_id}_model.pkl"))
        print("Training complete. Saved best model and fitting progress log.")