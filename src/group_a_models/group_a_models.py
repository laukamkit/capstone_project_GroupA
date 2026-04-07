import torch
import random

from PatchTST_supervised.models import PatchTST, TimeXer
from PatchTST_supervised.models import iTransformer
from PatchTST_supervised.utils.tools import EarlyStopping, adjust_learning_rate, test_params_flop
from PatchTST_supervised.utils.metrics import metric
import torch.nn as nn
from model_configs import TransformersConfig, SARIMAXConfig
from Base_Model import Base_Model
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from time import time
from copy import deepcopy

from torch import optim
from torch.optim import lr_scheduler 

class PatchTSTModel(Base_Model):
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
        if self.config.seed is not None:
            random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            np.random.seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        # taken from official repository:
        if self.config.use_gpu:
            device = torch.device('cuda:{}'.format(self.config.gpu))
            print('Use GPU: cuda:{}'.format(self.config.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    def _build_model(self):
        model_dict = {
            'PatchTST': PatchTST,
            'iTransformer': iTransformer,
            'TimeXer': TimeXer
        }
        model = model_dict[self.config.model].Model(self.config).float()
        return model
    
    def _get_data(self, flag):
            data_set, data_loader = self.nsw_data_loader.data_provider(self.training_data, self.validation_data, self.test_data, self.config, flag)
            return data_set, data_loader
    
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        total_loss_inverse = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

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
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                if self.config.scale:
                    loss_inverse = self._calculate_inverse_loss(criterion, batch_y, pred, true)
                total_loss.append(loss)
                if self.config.scale:
                    total_loss_inverse.append(loss_inverse)
        total_loss = np.average(total_loss)
        if self.config.scale:
            total_loss_inverse = np.average(total_loss_inverse)
        self.model.train()
        return total_loss, total_loss_inverse

    def _calculate_inverse_loss(self, criterion, batch_y, pred, true):
        shape = batch_y.shape
        pred_tile = np.tile(pred, [1, 1, 1])
        pred_inverse = pred_tile.reshape(shape[0] * shape[1], 1)
        pred_inverse = pred_inverse*np.sqrt(self.nsw_data_loader.scaler.var_[0]) + self.nsw_data_loader.scaler.mean_[0]
        pred_inverse = pred_inverse[:, -1:].reshape(shape)
        true_tiled = np.tile(true, [1, 1, 1])
        true_inverse = true_tiled.reshape(shape[0] * shape[1], 1)
        true_inverse = true_inverse*np.sqrt(self.nsw_data_loader.scaler.var_[0]) + self.nsw_data_loader.scaler.mean_[0]
        true_inverse = true_inverse[:, -1:].reshape(shape)
        loss_inverse = criterion(torch.from_numpy(pred_inverse), torch.from_numpy(true_inverse))
        return loss_inverse

    def train_model(self, task_id):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.nsw_data_loader.output_dir, f"{self.config.model}_checkpoints", task_id)
        if not os.path.exists(path):
            os.makedirs(path)

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
            vali_loss, vali_loss_inverse = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_loss_inverse = self.vali(test_data, test_loader, criterion)
            progress_log['model_name'].append(self.config.task_id)
            progress_log['lookback_window'].append(self.config.lookback_window)
            progress_log['epoch'].append(epoch + 1)
            progress_log['validation_horizon'].append(self.config.forecast_horizon)
            progress_log['validation_rmse'].append(np.sqrt(vali_loss) if not self.config.scale else np.sqrt(vali_loss_inverse))
            progress_log['validation_mse'].append(vali_loss if not self.config.scale else vali_loss_inverse)
            progress_log['time_taken_seconds'].append(cost_time)
            print("Epoch: {0}, Steps: {1} | Train Loss (MSE): {2:.7f} Train Loss (RMSE): {3:.7f} Vali Loss (MSE): {4:.7f} Vali Loss (RMSE): {5:.7f} Test Loss (MSE): {6:.7f} Test Loss (RMSE): {7:.7f}".format(
                epoch + 1, train_steps, train_loss, np.sqrt(train_loss), vali_loss, np.sqrt(vali_loss), test_loss, np.sqrt(test_loss)))
            print(f"\tVali Loss (MSE) Inverse Transformed: {vali_loss_inverse:.7f} Vali Loss (RMSE) Inverse Transformed: {np.sqrt(vali_loss_inverse):.7f} Test Loss (MSE) Inverse Transformed: {test_loss_inverse:.7f} Test Loss (RMSE) Inverse Transformed: {np.sqrt(test_loss_inverse):.7f}")
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.config.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.config)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        fitting_progress_log_df = pd.DataFrame(progress_log)
        os.makedirs(os.path.join(self.nsw_data_loader.output_dir, f"{self.config.model}_results"), exist_ok=True)
        fitting_progress_log_df.to_csv(os.path.join(self.nsw_data_loader.output_dir, f"{self.config.model}_results", f"fitting_log_{self.config.task_id}.csv"), index=False)
        return self.model

    def test_model(self, task_id, test=0):
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.nsw_data_loader.output_dir, f"{self.config.model}_checkpoints", task_id)
        if not os.path.exists(path):
            os.makedirs(path)

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))

        preds = []
        trues = []
        preds_inverse = []
        trues_inverse = []
        timestamps = []
        inputx = []
        folder_path = os.path.join(self.nsw_data_loader.output_dir, f"{self.config.model}_results", task_id)
        os.makedirs(folder_path, exist_ok=True)

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
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.config.forecast_horizon:, f_dim:]
                batch_y = batch_y[:, -self.config.forecast_horizon:, f_dim:].to(self.device)
                batch_time = batch_time[:, -self.config.forecast_horizon:]
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                # deal with inverse transform. self.nsw_data_loader.scaler
                if self.config.scale:
                    shape = batch_y.shape
                    if self.config.variate == 'MS':
                        n_features = self.nsw_data_loader.scaler.n_features_in_
                        outputs_inverse = np.tile(outputs, [1, 1, n_features])
                        outputs_inverse = self.nsw_data_loader.scaler.inverse_transform(outputs_inverse.reshape(shape[0] * shape[1], n_features))[:, -1:].reshape(shape)
                        batch_y_tiled = np.tile(batch_y, [1, 1, n_features])
                        batch_y_inverse = self.nsw_data_loader.scaler.inverse_transform(batch_y_tiled.reshape(shape[0] * shape[1], n_features))[:, -1:].reshape(shape)
                    else:
                        outputs_inverse = self.nsw_data_loader.scaler.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                        batch_y_inverse = self.nsw_data_loader.scaler.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
        
                batch_time = batch_time.detach().cpu().numpy()
                pred_inverse = None
                true_inverse = None
                if self.config.inverse:
                    pred_inverse = outputs_inverse  # outputs.detach().cpu().numpy()  # .squeeze()
                    true_inverse = batch_y_inverse  # batch_y.detach().cpu().numpy()  # .squeeze()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()
                ts = batch_time

                preds.append(pred)
                trues.append(true)
                if self.config.inverse:
                    preds_inverse.append(pred_inverse)
                    trues_inverse.append(true_inverse)
                timestamps.append(ts)
                inputx.append(batch_x.detach().cpu().numpy())
                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                    # if self.config.scalee:
                    #     shape = input.shape
                    #     input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    #visual(gt, pd, os.path.join(folder_path, '_' + str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        if self.config.inverse:
            preds_inverse = np.array(preds_inverse)
            trues_inverse = np.array(trues_inverse)
        timestamps = np.array(timestamps)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        if self.config.scale:
            preds_inverse = preds_inverse.reshape(-1, preds_inverse.shape[-2], preds_inverse.shape[-1])
            trues_inverse = trues_inverse.reshape(-1, trues_inverse.shape[-2], trues_inverse.shape[-1])
        timestamps = timestamps.reshape(-1, timestamps.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        if self.config.scale:
            mae_inverse, mse_inverse, rmse_inverse, mape_inverse, mspe_inverse, rse_inverse, corr_inverse = metric(preds_inverse, trues_inverse)
            print('Inverse Metrics - mse:{}, mae:{}, rmse:{}, rse:{}\n'.format(mse_inverse, mae_inverse, rmse_inverse, rse_inverse))
            
        print('mse:{}, mae:{}, rmse:{}, rse:{}'.format(mse, mae, rmse, rse))

        f = open(os.path.join(folder_path, "result.txt"), 'a')
        f.write(task_id + "  \n")
        if self.config.scale:
            f.write('mse:{}, mae:{}, rse:{}, mse_inverse:{}, rmse_inverse:{}, mae_inverse:{}, rse_inverse:{}'.format(mse, mae, rse, mse_inverse, rmse_inverse, mae_inverse, rse_inverse))
        else:
            f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + '_metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr.mean()]))
        np.save(folder_path + '_pred.npy', preds)
        np.save(folder_path + '_true.npy', trues)
        np.save(folder_path + '_ts.npy', timestamps)
        # np.save(folder_path + '_x.npy', inputx)
        if self.config.scale:
            np.save(folder_path + '_pred_inverse.npy', preds_inverse)
            np.save(folder_path + '_true_inverse.npy', trues_inverse)
        return

class SarimaxModel(Base_Model):
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

        
        if self.config.log_transform_target:
            self.training_data[self.config.target_col] = np.log(self.training_data[self.config.target_col])
            self.validation_data[self.config.target_col] = np.log(self.validation_data[self.config.target_col])
            self.test_data[self.config.target_col] = np.log(self.test_data[self.config.target_col])

    def test_model(self, model_fit:SARIMAXResultsWrapper | str | None, test_mode: bool = False):
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
            if self.config.log_transform_target:
                forecast = np.exp(forecast)
            # Store predictions and actuals
            all_predictions.extend(forecast.values)
            actuals = test_df.iloc[i : i + self.config.forecast_horizon].values
            if self.config.log_transform_target:
                actuals = np.exp(actuals)
            all_actuals.extend(actuals)
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
            'origin_index': all_origins,
            'timestamp': all_timestamps,
            'y_actual': all_actuals,
            'y_pred': all_predictions,
            'rmse': [rmse] * len(all_timestamps),
            'mse': [mse] * len(all_timestamps),
            'mae': [mae] * len(all_timestamps)
        })
        if test_mode:
            os.makedirs(os.path.join(self.nsw_data_loader.output_dir, "sarimax_results"), exist_ok=True)
            results_df.to_csv(os.path.join(self.nsw_data_loader.output_dir, "sarimax_results", f"test_results_{self.config.task_id}.csv"), index=False)
            print(f"Saved detailed rolling forecast results to sarimax_results/test_results_{self.config.task_id}.csv")
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
                                        _, _, _, _, mae, rmse, mse = self.test_model(model_fit)
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
        train_val_data = pd.concat([self.training_data, self.validation_data])
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
        os.makedirs(os.path.join(self.nsw_data_loader.output_dir, "sarimax_results"), exist_ok=True)
        fitting_progress_log_df.to_csv(os.path.join(self.nsw_data_loader.output_dir, "sarimax_results", f"fitting_log_{self.config.task_id}.csv"), index=False)
        os.makedirs(os.path.join(self.nsw_data_loader.output_dir, "sarimax_models"), exist_ok=True)
        self.best_model.save(os.path.join(self.nsw_data_loader.output_dir, "sarimax_models", f"model_{self.config.task_id}.pkl"))
        print("Training complete. Saved best model and fitting progress log.")
      
        
if __name__ == "__main__":
    # sarimax_config = SARIMAXConfig(
    #     task_id="sarimax_test",
    #     forecast_horizon=48,
    #     lookback_window=1440, # For SARIMAX, this is the number of most recent time steps to use for training.
    #     date_col='DATETIME',
    #     target_col='TOTALDEMAND',
    #     log_transform_target=True,
    #     feature_cols=['demand_1_week_ago', 'demand_1_year_ago', 'TEMPERATURE','TEMP_SQUARED', 'IS_WEEKEND'],
    #     scale=False,
    #     p=[2, 3, 4],
    #     d=[0],
    #     q=[0],
    #     P=[1],
    #     D=[1],
    #     Q=[1],
    #     seasonality_period=48,
    #     enforce_stationarity=True,
    #     enforce_invertibility=True
    # )
    # sarimax_model = SarimaxModel(sarimax_config)
    # sarimax_model.train_model()
    # all_origins, all_timestamps, all_actuals, all_predictions, mae, rmse, mse = sarimax_model.test_model(None, test_mode=True)

    # @dataclass
    # class Config:
    #     task_id: str
    #     forecast_horizon: int
    #     lookback_window: int | None = None
    #     target_col: str = 'TOTALDEMAND'
    #     log_transform_target: bool = False
    #     feature_cols: list[str] = field(default_factory=list)
    #     scale: bool = True # if scale, then make sure to set inverse to True as well
    #     seed: int | None = None

    # @dataclass
    # class TransformersConfig(Config):
    #     model:str
    #     date_col: str = 'DATETIME'
    #     variate: str = 'MS' # 'S' for single variate, 'MS' for multiple predictors but single output, 'M' for multiple predictors and multiple outputs
    #     patch_size: int = 16
    #     stride: int = 1
    #     d_model: int = 128
    #     num_attention_heads: int = 4
    #     num_encoder_layers: int = 3
    #     dim_ff: int = 256
    #     dropout_ff: float = 0.1
    #     dropout_head_fc: float = 0.1
    #     use_gpu: bool = True
    #     time_encoding: str = 'timeF'
    #     shuffle_flag: bool = True
    #     training_epochs: int = 10
    #     batch_size: int = 64
    #     learning_rate: float = 0.0001
    #     output_attention: bool = False
    #     lradj: str = 'TST' # options are 'type1', 'type2', 'type3', 'TST', 'constant', '1', '2', '3', '4', '5', '6'. See tools.py for details on each type of learning rate adjustment strategy.
    #     patience: int = 5 # for early stopping
    #     pct_start: float = 0.3 # for OneCycleLR scheduler, the percentage of the cycle spent increasing the learning rate.
    #     @property
    #     def enc_in(self) -> int:
    #         return len(self.feature_cols) if self.feature_cols is not None else 1
    #     @property
    #     def c_out(self) -> int:
    #         return 1

    patchtst_config = TransformersConfig(
        task_id="patchtst_test",
        model="PatchTST",
        forecast_horizon=48,
        lookback_window=336,
        log_transform_target=True,
        feature_cols=['TEMPERATURE', 'TEMP_SQUARED','IS_WEEKEND', 'demand_1_year_ago'],
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
        training_epochs=2,
        batch_size=64,
        learning_rate=0.0001,
        output_attention=False,
        lradj='TST',
        patience=10,
    )
    patch_tst_model = PatchTSTModel(patchtst_config)
    patch_tst_model.train_model(task_id=patchtst_config.task_id)
    patch_tst_model.test_model(task_id=patchtst_config.task_id, test=1)

    pass