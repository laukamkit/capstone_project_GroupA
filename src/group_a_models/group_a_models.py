from model_configs import SARIMAXConfig
from Base_Model import Base_Model
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from time import time
from copy import deepcopy


class SarimaxModel(Base_Model):
    def __init__(self, config: SARIMAXConfig):
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
        super().__init__(config)
        self.config: SARIMAXConfig = config

        self.training_data = self.train_scaled if self.config.scale else self.train
        self.validation_data = self.val_scaled if self.config.scale else self.validation
        self.test_data = self.test_scaled if self.config.scale else self.test
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
    sarimax_config = SARIMAXConfig(
        task_id="sarimax_test",
        forecast_horizon=48,
        lookback_window=1440, # For SARIMAX, this is the number of most recent time steps to use for training.
        date_col='DATETIME',
        target_col='TOTALDEMAND',
        log_transform_target=True,
        feature_cols=['demand_1_week_ago', 'demand_1_year_ago', 'TEMPERATURE','TEMP_SQUARED', 'IS_WEEKEND'],
        scale=False,
        p=[2, 3, 4],
        d=[0],
        q=[0],
        P=[1],
        D=[1],
        Q=[1],
        seasonality_period=48,
        enforce_stationarity=True,
        enforce_invertibility=True
    )
    sarimax_model = SarimaxModel(sarimax_config)
    sarimax_model.train_model()
    all_origins, all_timestamps, all_actuals, all_predictions, mae, rmse, mse = sarimax_model.test_model(None, test_mode=True)
    pass