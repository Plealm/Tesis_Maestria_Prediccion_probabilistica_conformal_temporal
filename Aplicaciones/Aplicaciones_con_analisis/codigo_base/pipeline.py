import numpy as np
import pandas as pd
import warnings
import gc
import time
import copy
import traceback
from tqdm import tqdm
from typing import Dict, List, Tuple

# Importamos los modelos (asumiendo que están en modelos.py)
from modelos import (CircularBlockBootstrapModel, SieveBootstrapModel, LSPM, LSPMW, 
                     DeepARModel, AREPD, MondrianCPSModel, AdaptiveVolatilityMondrianCPS,
                     EnCQR_LSTM_Model)
from metricas import crps

warnings.filterwarnings("ignore")

def clear_all_sessions():
    """Limpia memoria de forma agresiva."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except:
        pass


class TimeBalancedOptimizerWithTQDM:
    """
    Optimizador robusto que evita errores de Pickle y muestra errores reales.
    """
    
    def __init__(self, random_state: int = 42, verbose: bool = False):
        self.random_state = random_state
        self.verbose = verbose
        self.rng = np.random.default_rng(random_state)

    def _get_efficient_grid(self, model_name: str, n_train: int) -> List[Dict]:
        """Retorna un Grid de búsqueda pequeño pero efectivo."""
        
        if model_name == 'Block Bootstrapping':
            return [{'block_length': 12}, {'block_length': 24}, {'block_length': 48}]
        elif model_name == 'Sieve Bootstrap':
            return [{'order': 12}, {'order': 24}, {'order': 48}]
        elif model_name == 'LSPM':
            return [{}] 
        elif model_name == 'LSPMW':
            return [{'rho': 0.95}, {'rho': 0.99}]
        elif model_name == 'AREPD':
            return [{'n_lags': 24, 'rho': 0.95}, {'n_lags': 24, 'rho': 0.90}]
        elif model_name == 'MCPS':
            return [{'n_lags': 12, 'n_bins': 5}, {'n_lags': 24, 'n_bins': 10}]
        elif model_name == 'AV-MCPS':
            return [{'n_lags': 12, 'n_pred_bins': 5, 'n_vol_bins': 3}]
        elif model_name == 'DeepAR':
            return [
                {'hidden_size': 24, 'n_lags': 24, 'epochs': 10, 'lr': 0.01}, 
                {'hidden_size': 40, 'n_lags': 24, 'epochs': 15, 'lr': 0.005}
            ]
        elif model_name == 'EnCQR-LSTM':
            return [{'n_lags': 24, 'units': 24, 'epochs': 10}, {'n_lags': 24, 'units': 48, 'epochs': 15}]
        
        return [{}]

    def _safe_copy_model(self, model):
        """Copia segura del modelo evitando errores de pickle."""
        try:
            return copy.deepcopy(model)
        except Exception:
            cls = type(model)
            init_kwargs = {}
            if hasattr(model, 'random_state'): init_kwargs['random_state'] = model.random_state
            if hasattr(model, 'verbose'): init_kwargs['verbose'] = model.verbose
            if hasattr(model, 'n_boot'): init_kwargs['n_boot'] = model.n_boot
            
            try:
                new_model = cls(**init_kwargs)
            except:
                new_model = cls()
            
            unsafe_keys = ['xgb', 'base_model', 'model', 'tf', 'torch', '_trained_model', '_frozen_model']
            for k, v in model.__dict__.items():
                if k not in unsafe_keys and not k.startswith('_') and not callable(v):
                    try:
                        setattr(new_model, k, copy.deepcopy(v))
                    except:
                        setattr(new_model, k, v)
            
            if hasattr(model, 'tf'): new_model.tf = model.tf
            if hasattr(model, 'torch'): new_model.torch = model.torch
            if hasattr(model, 'xgb'): new_model.xgb = model.xgb
                
            return new_model

    def _evaluate_with_convergence(self, model, train_data, val_data):
        scores = []
        if len(train_data) < 10: return np.inf

        for i in range(len(val_data)):
            history = np.concatenate([train_data, val_data[:i]]) if i > 0 else train_data
            true_val = val_data[i]
            
            try:
                if isinstance(model, (CircularBlockBootstrapModel, SieveBootstrapModel)):
                    input_data = history
                else:
                    input_data = pd.DataFrame({'valor': history})

                pred_samples = model.fit_predict(input_data)
                pred_flat = np.asarray(pred_samples).flatten()
                
                score = crps(pred_flat, true_val)
                if not np.isnan(score) and not np.isinf(score):
                    scores.append(score)
            except Exception:
                continue
            
            if len(scores) > 10 and i > 15: break 
            
        return np.mean(scores) if scores else np.inf

    def optimize_all_models(self, models: Dict, train_data: np.ndarray, val_data: np.ndarray) -> Dict:
        optimized_params = {}
        if self.verbose: print(f"\n⚡ OPTIMIZACIÓN EFICIENTE ({len(models)} Modelos)")

        model_items = list(models.items())
        pbar = tqdm(model_items, desc="Optimizando", unit="modelo")
        
        for name, model in pbar:
            pbar.set_description(f"Optimizando {name}")
            param_grid = self._get_efficient_grid(name, len(train_data))
            
            if len(param_grid) <= 1 and not param_grid[0]:
                optimized_params[name] = {}
                continue
            
            best_score = np.inf
            best_params = {}
            
            for params in param_grid:
                model_copy = None
                try:
                    model_copy = self._safe_copy_model(model)
                    for key, val in params.items():
                        if hasattr(model_copy, key): setattr(model_copy, key, val)
                    
                    score = self._evaluate_with_convergence(model_copy, train_data, val_data)
                    
                    if score < best_score:
                        best_score = score
                        best_params = params
                except Exception:
                    pass
                finally:
                    if model_copy:
                        try: del model_copy
                        except: pass
                    clear_all_sessions()
            
            optimized_params[name] = best_params
            if hasattr(model, 'best_params'): model.best_params = best_params
            for key, val in best_params.items():
                if hasattr(model, key): setattr(model, key, val)

        return optimized_params


class PipelineProcessedData:
    """
    Pipeline genérico que recibe una Serie de Tiempo YA PROCESADA (pandas Series).
    No descarga datos. Espera recibir los datos limpios y transformados.
    """
    
    N_TEST_HOURS = 24
    
    def __init__(self, n_boot: int = 1000, seed: int = 42, verbose: bool = False,
                 val_ratio: float = 0.15):
        self.n_boot = n_boot
        self.seed = seed
        self.verbose = verbose
        self.val_ratio = val_ratio
        self.rng = np.random.default_rng(seed)
    
    def split_data(self, values, timestamps):
        """Divide los datos recibidos en Train/Val/Test."""
        n_total = len(values)
        n_test = self.N_TEST_HOURS
        n_val = int((n_total - n_test) * self.val_ratio)
        n_train = n_total - n_val - n_test
        
        if n_train < 50:
             raise ValueError(f"Datos insuficientes para entrenar: n_train={n_train}")
        
        return {
            'train': {'values': values[:n_train], 'timestamps': timestamps[:n_train]},
            'val': {'values': values[n_train:n_train+n_val], 'timestamps': timestamps[n_train:n_train+n_val]},
            'test': {'values': values[n_train+n_val:], 'timestamps': timestamps[n_train+n_val:]},
            'metadata': {'n_train': n_train, 'n_val': n_val}
        }
    
    def _setup_models(self, seed: int) -> Dict:
        """Inicializa los modelos."""
        return {
            'Block Bootstrapping': CircularBlockBootstrapModel(n_boot=self.n_boot, random_state=seed, verbose=self.verbose),
            'Sieve Bootstrap': SieveBootstrapModel(n_boot=self.n_boot, random_state=seed, verbose=self.verbose),
            'LSPM': LSPM(random_state=seed, verbose=self.verbose),
            'LSPMW': LSPMW(rho=0.95, random_state=seed, verbose=self.verbose),
            'AREPD': AREPD(n_lags=24, rho=0.95, random_state=seed, verbose=self.verbose),
            'MCPS': MondrianCPSModel(n_lags=24, n_bins=10, random_state=seed, verbose=self.verbose),
            'AV-MCPS': AdaptiveVolatilityMondrianCPS(n_lags=24, random_state=seed, verbose=self.verbose),
            'DeepAR': DeepARModel(hidden_size=24, n_lags=24, epochs=10, num_samples=self.n_boot, random_state=seed, verbose=self.verbose),
            'EnCQR-LSTM': EnCQR_LSTM_Model(n_lags=24, units=24, epochs=10, num_samples=self.n_boot, random_state=seed, verbose=self.verbose)
        }
    
    def _optimize_and_freeze_models(self, models, train_data, val_data):
        optimizer = TimeBalancedOptimizerWithTQDM(random_state=self.seed, verbose=self.verbose)
        optimized_params = optimizer.optimize_all_models(models, train_data, val_data)
        
        for name, params in optimized_params.items():
            if params and name in models:
                model = models[name]
                if hasattr(model, 'best_params'): model.best_params = params
                for k, v in params.items():
                    if hasattr(model, k): setattr(model, k, v)
        
        if self.verbose: print("\n🔒 Fase 2: Congelamiento (Entrenamiento Final)")
        train_val_combined = np.concatenate([train_data, val_data])
        
        pbar = tqdm(models.items(), desc="Congelando modelos")
        for name, model in pbar:
            try:
                if hasattr(model, 'freeze_hyperparameters'):
                    model.freeze_hyperparameters(train_val_combined)
            except Exception as e:
                if self.verbose: tqdm.write(f"  ⚠️ Error congelando {name}: {e}")
        
        return models
    
    def run_evaluation(self, input_series: pd.Series, save_predictions=False):
        """
        Ejecuta el pipeline sobre la serie entregada como argumento.
        
        Args:
            input_series (pd.Series): Serie procesada (ej. Trend_Removed) con DateTimeIndex.
            save_predictions (bool): Si True, retorna diccionario con todas las muestras.
        """
        if not isinstance(input_series, pd.Series):
            raise ValueError("input_series debe ser un pandas Series.")
        if not isinstance(input_series.index, pd.DatetimeIndex):
            raise ValueError("input_series debe tener un DatetimeIndex.")

        print("="*60)
        print(f"📥 PROCESSED DATA PIPELINE (Datos Recibidos: {len(input_series)} puntos)")
        print("="*60)
        
        # Extraer valores y timestamps del objeto Series
        values = input_series.values
        timestamps = input_series.index
        
        # Split
        split = self.split_data(values, timestamps)
        
        # Inicializar y Optimizar
        models = self._setup_models(self.seed)
        models = self._optimize_and_freeze_models(models, split['train']['values'], split['val']['values'])
        
        if self.verbose: print(f"\n🔮 Fase 3: Predicción ({self.N_TEST_HOURS} pasos)")
        
        results_rows = []
        predictions_dict = {} if save_predictions else None
        
        test_vals = split['test']['values']
        history = np.concatenate([split['train']['values'], split['val']['values']])
        
        for t in tqdm(range(len(test_vals)), desc="Prediciendo Steps"):
            curr_hist = np.concatenate([history, test_vals[:t]])
            true_val = test_vals[t]
            row = {'Paso': t+1, 'Valor_Observado': true_val}
            
            if save_predictions:
                predictions_dict[t] = {'timestamp': split['test']['timestamps'][t], 'true_value': true_val, 'predictions': {}}
            
            for name, model in models.items():
                try:
                    if isinstance(model, (CircularBlockBootstrapModel, SieveBootstrapModel)):
                        preds = model.fit_predict(curr_hist)
                    else:
                        preds = model.fit_predict(pd.DataFrame({'valor': curr_hist}))
                    
                    preds = np.asarray(preds).flatten()
                    
                    if len(preds) == 0 or np.isnan(preds).all():
                        row[name] = np.nan
                    else:
                        preds = np.nan_to_num(preds, nan=np.nanmean(preds))
                        row[name] = crps(preds, true_val)
                        
                    if save_predictions: predictions_dict[t]['predictions'][name] = preds
                except Exception:
                    row[name] = np.nan
            
            results_rows.append(row)
            if t % 6 == 0: clear_all_sessions()
            
        df_results = pd.DataFrame(results_rows)
        return df_results, predictions_dict
