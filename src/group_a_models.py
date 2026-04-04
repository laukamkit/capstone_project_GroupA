from nsw_data_loader import NSWDataLoader
from nsw_dataset import WindowedNSWDataSet
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from model_configs import *
import os
from datetime import datetime

class GroupAModels:
    def __init__(self, config: Config):
        self.config: Config = config
        self.nsw_data_loader = NSWDataLoader()
        self.train, self.validation, self.test, self.train_scaled, self.val_scaled, self.test_scaled = self.nsw_data_loader.load_data()

    def train_model(self):
        raise NotImplementedError
    
    def test_model(self):
        raise NotImplementedError
    
    def data_loader(self):
        raise NotImplementedError
            

class SarimaxModel(GroupAModels):
    def __init__(self, config: SARIMAXConfig):
        self.order = config.order
        self.seasonal_order = config.seasonal_order
        self.enforce_stationarity = config.enforce_stationarity
        self.enforce_invertibility = config.enforce_invertibility
        super().__init__(config)
        self.config: SARIMAXConfig = config

    def _prepare_sarimax_data(self, df):
        """Extract and align endog/exog from a dataframe for SARIMAX."""
        combined = pd.concat([df[self.config.target_col], df[self.config.feature_cols]], axis=1).dropna()
        endog = combined[self.config.target_col].copy()
        exog = combined[self.config.feature_cols].copy()
        endog.index.freq = self.config.time_freq
        exog.index.freq = self.config.time_freq
        return endog, exog

    def _rolling_apply_loop(self, current_model, endog, exog, max_windows=None):
        """
        Roll through endog/exog in forecast_horizon steps using .apply() to advance state.
        Returns a list of (timestamp, y_actual, y_pred, rmse, mse, mae) tuples.
        """
        H = self.config.forecast_horizon
        results = []
        n_windows = len(endog) // H
        if max_windows is not None:
            n_windows = min(n_windows, max_windows)

        val_pred_vs_actual_df = pd.DataFrame(columns=['datetime', 'actual', 'predicted'])
        val_metrics_df = pd.DataFrame(columns=['window', 'date_start', 'date_end', 'rmse', 'mse', 'mae'])
        for i in range(n_windows):
            start = i * H
            end = start + H
            future_exog = exog.iloc[start:end]
            actual = endog.iloc[start:end]

            forecast = current_model.get_forecast(steps=H, exog=future_exog)
            y_pred = forecast.predicted_mean
            y_pred.index = actual.index  # align index for metric calc

            rmse = np.sqrt(mean_squared_error(actual, y_pred))
            mse = mean_squared_error(actual, y_pred)
            mae = np.mean(np.abs(actual.values - y_pred.values))
            rows = [] # need to fix this because it's not looping.
            for idx, actuals, preds, rmse, mse, mae in results:
                for t, actual, pred in zip(idx, actuals, preds):
                    rows.append({'datetime': t, 'actual': actual, 'predicted': pred})
            val_pred_vs_actual_df = pd.concat([val_pred_vs_actual_df, pd.DataFrame(rows)], ignore_index=True)
            val_temp_metrics_df = pd.DataFrame([{'window': i, 'date_start': actual.index.min(), 'date_end': actual.index.max(), 'rmse': rmse, 'mse': mse, 'mae': mae}])
            val_metrics_df = pd.concat([val_metrics_df, val_temp_metrics_df], ignore_index=True)
            # Advance model state by revealing the true observed values
            current_model:SARIMAXResults = current_model.apply(endog=actual, exog=future_exog)

        return val_pred_vs_actual_df, val_metrics_df, current_model

    def train_model(self, **kwargs):
        training_data = self.train_scaled if self.config.scale else self.train
        y_train_clean, X_train_clean = self._prepare_sarimax_data(training_data)

        model: SARIMAX = SARIMAX(
            endog=y_train_clean,
            exog=X_train_clean,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
        )
        self.fitted_model: SARIMAXResults = model.fit(
            disp=False, low_memory=True,
            maxiter=kwargs.get('maxiter', 20),
            method=kwargs.get('method', 'lbfgs'),
        )

        # Rolling validation using .apply() — no DataLoader needed
        val_data = self.val_scaled if self.config.scale else self.validation
        val_endog, val_exog = self._prepare_sarimax_data(val_data)
        val_pred_vs_actual_df, val_metrics_df, _ = self._rolling_apply_loop(
            self.fitted_model, val_endog, val_exog, max_windows=self.config.val_iter
        )

        os.makedirs(os.path.join(self.nsw_data_loader.output_dir, "sarimax_results"), exist_ok=True)
        val_pred_vs_actual_df.to_csv(os.path.join(self.nsw_data_loader.output_dir, "sarimax_results", f"validation_pred_vs_actual_{self.config.forecast_horizon}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"), index=False)
        val_metrics_df.to_csv(os.path.join(self.nsw_data_loader.output_dir, "sarimax_results", f"validation_metrics_{self.config.forecast_horizon}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"), index=False)
        os.makedirs(os.path.join(self.nsw_data_loader.output_dir, "sarimax_models"), exist_ok=True)
        self.fitted_model.save(os.path.join(self.nsw_data_loader.output_dir, "sarimax_models", f"sarimax_model_{self.config.forecast_horizon}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"))

    def test_model(self, model_file_name: str):
        self.fitted_model = SARIMAXResults.load(os.path.join(self.nsw_data_loader.output_dir, "sarimax_models", model_file_name))
        if self.fitted_model is None:
            raise ValueError("Model must be trained before testing.")

        # Fast-forward model state through the entire validation set first
        val_data = self.val_scaled if self.config.scale else self.validation
        val_endog, val_exog = self._prepare_sarimax_data(val_data)
        print("Fast-forwarding model state through validation set...")
        model_at_test_start = self.fitted_model.apply(endog=val_endog, exog=val_exog)

        # Rolling forecast over test set
        test_data = self.test_scaled if self.config.scale else self.test
        test_endog, test_exog = self._prepare_sarimax_data(test_data)
        print(f"Running rolling forecast over test set ({len(test_endog) // self.config.forecast_horizon} windows)...")
        test_results, _ = self._rolling_apply_loop(model_at_test_start, test_endog, test_exog)

        for i, r in enumerate(test_results):
            if i % 100 == 0:
                print(f"Tested {i} windows... RMSE so far: {np.mean([x[3] for x in test_results[:i+1]]):.4f}")

        results = pd.DataFrame(test_results, columns=['timestamp', 'y_actual', 'y_pred', 'rmse', 'mse', 'mae'])
        os.makedirs(os.path.join(self.nsw_data_loader.output_dir, "sarimax_results"), exist_ok=True)
        results.to_csv(os.path.join(self.nsw_data_loader.output_dir, "sarimax_results", f"test_results_{self.config.forecast_horizon}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"), index=False)
        
        
if __name__ == "__main__":
    sarimax_config = SARIMAXConfig(
        task_id="sarimax_test",
        forecast_horizon=24,
        lookback_window=336,
        date_col='DATETIME',
        target_col='TOTALDEMAND',
        feature_cols=['TEMPERATURE'],
        is_training=True,
        scale=True,
        order=(1, 1, 0),
        seasonal_order=(0, 1, 0, 48),
        enforce_stationarity=False,
        enforce_invertibility=False,
        val_iter=20
    )
    sarimax_model = SarimaxModel(sarimax_config)
    sarimax_model.train_model(maxiter=20)