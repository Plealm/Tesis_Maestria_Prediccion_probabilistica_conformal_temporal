import numpy as np
import pandas as pd
import warnings
import gc
import time
import copy
import traceback
from tqdm import tqdm
from typing import Dict, List, Tuple
from gluonts.dataset.repository import get_dataset

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
        """Retorna un Grid de búsqueda pequeño pero efectivo para los 9 modelos."""
        
        # 1. Block Bootstrapping
        if model_name == 'Block Bootstrapping':
            return [{'block_length': 12}, {'block_length': 24}, {'block_length': 48}]
        
        # 2. Sieve Bootstrap
        elif model_name == 'Sieve Bootstrap':
            return [{'order': 12}, {'order': 24}, {'order': 48}]
        
        # 3. LSPM (Standard) - Generalmente no paramétrico
        elif model_name == 'LSPM':
            return [{}] 
        
        # 4. LSPMW (Weighted)
        elif model_name == 'LSPMW':
            return [{'rho': 0.95}, {'rho': 0.99}]
        
        # 5. AREPD
        elif model_name == 'AREPD':
            return [{'n_lags': 24, 'rho': 0.95}, {'n_lags': 24, 'rho': 0.90}]
        
        # 6. MCPS
        elif model_name == 'MCPS':
            return [{'n_lags': 12, 'n_bins': 5}, {'n_lags': 24, 'n_bins': 10}]
        
        # 7. AV-MCPS
        elif model_name == 'AV-MCPS':
            return [{'n_lags': 12, 'n_pred_bins': 5, 'n_vol_bins': 3}]
        
        # 8. DeepAR
        elif model_name == 'DeepAR':
            return [
                {'hidden_size': 24, 'n_lags': 24, 'epochs': 10, 'lr': 0.01}, 
                {'hidden_size': 40, 'n_lags': 24, 'epochs': 15, 'lr': 0.005}
            ]
        
        # 9. EnCQR-LSTM
        elif model_name == 'EnCQR-LSTM':
            return [{'n_lags': 24, 'units': 24, 'epochs': 10}, {'n_lags': 24, 'units': 48, 'epochs': 15}]
        
        return [{}]

    def _safe_copy_model(self, model):
        """
        Copia segura del modelo evitando errores de pickle con módulos C++.
        Restaura referencias a módulos (tf, torch) para que __del__ funcione.
        """
        try:
            return copy.deepcopy(model)
        except Exception:
            # Fallback manual
            cls = type(model)
            init_kwargs = {}
            if hasattr(model, 'random_state'): init_kwargs['random_state'] = model.random_state
            if hasattr(model, 'verbose'): init_kwargs['verbose'] = model.verbose
            if hasattr(model, 'n_boot'): init_kwargs['n_boot'] = model.n_boot
            
            try:
                new_model = cls(**init_kwargs)
            except:
                new_model = cls()
            
            # Copiar atributos seguros
            unsafe_keys = ['xgb', 'base_model', 'model', 'tf', 'torch', '_trained_model', '_frozen_model']
            for k, v in model.__dict__.items():
                if k not in unsafe_keys and not k.startswith('_') and not callable(v):
                    try:
                        setattr(new_model, k, copy.deepcopy(v))
                    except:
                        setattr(new_model, k, v)
            
            # --- CORRECCIÓN CRÍTICA ---
            # Restaurar referencias a módulos para que el destructor (__del__) no falle
            # al intentar limpiar sesión (self.tf.keras...)
            if hasattr(model, 'tf'):
                new_model.tf = model.tf
            if hasattr(model, 'torch'):
                new_model.torch = model.torch
            if hasattr(model, 'xgb'):
                new_model.xgb = model.xgb
                
            return new_model

    def _evaluate_with_convergence(self, model, train_data, val_data):
        """Evalúa CRPS en validación."""
        scores = []
        errors_shown = 0
        
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
            except Exception as e:
                if self.verbose and errors_shown < 1:
                    tqdm.write(f"  ⚠️ Error eval {type(model).__name__}: {str(e)}")
                    errors_shown += 1
                continue
            
            if len(scores) > 10 and i > 15: break 
            
        return np.mean(scores) if scores else np.inf

    def optimize_all_models(self, models: Dict, train_data: np.ndarray, val_data: np.ndarray) -> Dict:
        optimized_params = {}
        
        if self.verbose:
            print(f"\n⚡ OPTIMIZACIÓN EFICIENTE ({len(models)} Modelos)")

        model_items = list(models.items())
        pbar = tqdm(model_items, desc="Optimizando", unit="modelo")
        
        for name, model in pbar:
            pbar.set_description(f"Optimizando {name}")
            model_start = time.time()
            
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
                    
                except Exception as e:
                    if self.verbose:
                        tqdm.write(f"  ❌ Error config {name} {params}: {e}")
                finally:
                    # Limpieza explícita
                    if model_copy:
                        try:
                            del model_copy
                        except: 
                            pass
                    clear_all_sessions()
            
            optimized_params[name] = best_params
            
            if hasattr(model, 'best_params'): model.best_params = best_params
            for key, val in best_params.items():
                if hasattr(model, key): setattr(model, key, val)

            elapsed = time.time() - model_start
            if self.verbose:
                msg = f"Mejor CRPS={best_score:.4f}" if best_score != np.inf else "Falló (CRPS=inf)"
                tqdm.write(f"  ✓ {name:<20}: {msg} [{elapsed:.1f}s]")

        return optimized_params


class PipelineElectricity:
    """
    Pipeline para el dataset Electricity de GluonTS.
    """
    
    N_TEST_HOURS = 24
    
    def __init__(self, n_boot: int = 1000, seed: int = 42, verbose: bool = False,
                 val_ratio: float = 0.15, max_data_points: int = 5000):
        self.n_boot = n_boot
        self.seed = seed
        self.verbose = verbose
        self.val_ratio = val_ratio
        self.max_data_points = max_data_points
        self.rng = np.random.default_rng(seed)
    
    def load_electricity_data(self, series_index: int = 0):
        if self.verbose: print("📥 Descargando dataset Electricity...")
        electricity = get_dataset("electricity")
        series_list = list(electricity.train)
        if series_index >= len(series_list): raise ValueError("Index error")
        
        elec_series = series_list[series_index]
        values_full = elec_series['target']
        start_timestamp = elec_series['start'].to_timestamp()
        timestamps_full = pd.date_range(start=start_timestamp, periods=len(values_full), freq=elec_series['start'].freq)
        
        if len(values_full) > self.max_data_points:
            return values_full[-self.max_data_points:], timestamps_full[-self.max_data_points:]
        return values_full, timestamps_full
    
    def split_data(self, values, timestamps):
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
        """Inicializa los 9 modelos exactos."""
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
        
        # Aplicar params optimizados
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
    
    def run_evaluation(self, series_index=0, save_predictions=False):
        print("="*60)
        print(f"🔌 ELECTRICITY DATASET - Serie {series_index}")
        print("="*60)
        
        try:
            values, timestamps = self.load_electricity_data(series_index)
            split = self.split_data(values, timestamps)
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            return pd.DataFrame(), None
        
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
                except Exception as e:
                    row[name] = np.nan
            
            results_rows.append(row)
            if t % 6 == 0: clear_all_sessions()
            
        df_results = pd.DataFrame(results_rows)
        return df_results, predictions_dict
    


# --- AGREGAR ESTO AL FINAL DE pipeline.py ---

class PipelineTraffic:
    """
    Pipeline para el dataset Traffic de GluonTS.
    """
    
    N_TEST_HOURS = 24
    
    def __init__(self, n_boot: int = 1000, seed: int = 42, verbose: bool = False,
                 val_ratio: float = 0.15, max_data_points: int = 5000):
        self.n_boot = n_boot
        self.seed = seed
        self.verbose = verbose
        self.val_ratio = val_ratio
        self.max_data_points = max_data_points
        self.rng = np.random.default_rng(seed)
    
    def load_traffic_data(self, series_index: int = 0):
        if self.verbose: print("Cargando dataset: Traffic...")
        traffic = get_dataset("traffic")
        series_list = list(traffic.train)
        if series_index >= len(series_list): raise ValueError("Index error")
        
        # Traffic dataset handling
        traf_series = series_list[series_index]
        values_full = traf_series['target']
        start_timestamp = traf_series['start'].to_timestamp()
        timestamps_full = pd.date_range(start=start_timestamp, periods=len(values_full), freq=traf_series['start'].freq)
        
        if len(values_full) > self.max_data_points:
            return values_full[-self.max_data_points:], timestamps_full[-self.max_data_points:]
        return values_full, timestamps_full
    
    def split_data(self, values, timestamps):
        # Misma lógica de split
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
    
    # Reutilizamos los métodos de configuración y optimización internos
    # copiando la lógica para asegurar independencia
    def _setup_models(self, seed: int) -> Dict:
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
    
    def run_evaluation(self, series_index=0, save_predictions=False):
        print("="*60)
        print(f"🚦 TRAFFIC DATASET - Serie {series_index}")
        print("="*60)
        
        try:
            values, timestamps = self.load_traffic_data(series_index)
            split = self.split_data(values, timestamps)
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            return pd.DataFrame(), None
        
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
                except Exception as e:
                    row[name] = np.nan
            
            results_rows.append(row)
            if t % 6 == 0: clear_all_sessions()
            
        df_results = pd.DataFrame(results_rows)
        return df_results, predictions_dict
    

class PipelineExchange:
    """
    Pipeline para el dataset Exchange Rate de GluonTS.
    Configuración específica:
    - Datos totales: Últimos 1825 puntos (aprox 5 años).
    - Test: Últimos 30 puntos (30 días).
    """
    
    N_TEST_DAYS = 30
    TOTAL_DATA_POINTS = 1825
    
    def __init__(self, n_boot: int = 1000, seed: int = 42, verbose: bool = False,
                 val_ratio: float = 0.15):
        self.n_boot = n_boot
        self.seed = seed
        self.verbose = verbose
        self.val_ratio = val_ratio
        self.rng = np.random.default_rng(seed)
    
    def load_exchange_data(self, series_index: int = 0):
        if self.verbose: print("💱 Descargando dataset Exchange Rate...")
        exchange = get_dataset("exchange_rate")
        series_list = list(exchange.train)
        
        if series_index >= len(series_list): 
            raise ValueError(f"Index error: El dataset solo tiene {len(series_list)} series.")
        
        # Extraer serie
        exch_series = series_list[series_index]
        values_full = exch_series['target']
        start_timestamp = exch_series['start'].to_timestamp()
        
        # Generar índice temporal
        timestamps_full = pd.date_range(
            start=start_timestamp, 
            periods=len(values_full), 
            freq=exch_series['start'].freq
        )
        
        # REGLA: Tomar solo los últimos 1825 datos
        if len(values_full) > self.TOTAL_DATA_POINTS:
            if self.verbose: 
                print(f"   Cortando datos: Tomando los últimos {self.TOTAL_DATA_POINTS} registros de {len(values_full)}.")
            values_full = values_full[-self.TOTAL_DATA_POINTS:]
            timestamps_full = timestamps_full[-self.TOTAL_DATA_POINTS:]
        else:
            if self.verbose:
                print(f"   Nota: El dataset es menor a {self.TOTAL_DATA_POINTS}, usando todo ({len(values_full)}).")
            
        return values_full, timestamps_full
    
    def split_data(self, values, timestamps):
        """
        Divide en Train/Val/Test asegurando que Test sean los últimos 30 días.
        """
        n_total = len(values)
        n_test = self.N_TEST_DAYS
        
        # Validación dinámica basada en el ratio sobre los datos restantes
        n_rest = n_total - n_test
        n_val = int(n_rest * self.val_ratio)
        n_train = n_rest - n_val
        
        if n_train < 50:
             raise ValueError(f"Datos insuficientes para entrenar: n_train={n_train}")
        
        return {
            'train': {'values': values[:n_train], 'timestamps': timestamps[:n_train]},
            'val': {'values': values[n_train:n_train+n_val], 'timestamps': timestamps[n_train:n_train+n_val]},
            'test': {'values': values[n_train+n_val:], 'timestamps': timestamps[n_train+n_val:]},
            'metadata': {'n_train': n_train, 'n_val': n_val, 'n_test': n_test}
        }
    
    def _setup_models(self, seed: int) -> Dict:
        """
        Inicializa los modelos.
        NOTA: Se cambia n_lags a 30 (días) por defecto para adaptarse mejor a datos diarios.
        """
        return {
            'Block Bootstrapping': CircularBlockBootstrapModel(n_boot=self.n_boot, random_state=seed, verbose=self.verbose),
            'Sieve Bootstrap': SieveBootstrapModel(n_boot=self.n_boot, random_state=seed, verbose=self.verbose),
            'LSPM': LSPM(random_state=seed, verbose=self.verbose),
            'LSPMW': LSPMW(rho=0.95, random_state=seed, verbose=self.verbose),
            'AREPD': AREPD(n_lags=30, rho=0.95, random_state=seed, verbose=self.verbose),
            'MCPS': MondrianCPSModel(n_lags=30, n_bins=10, random_state=seed, verbose=self.verbose),
            'AV-MCPS': AdaptiveVolatilityMondrianCPS(n_lags=30, random_state=seed, verbose=self.verbose),
            'DeepAR': DeepARModel(hidden_size=30, n_lags=30, epochs=10, num_samples=self.n_boot, random_state=seed, verbose=self.verbose),
            'EnCQR-LSTM': EnCQR_LSTM_Model(n_lags=30, units=30, epochs=10, num_samples=self.n_boot, random_state=seed, verbose=self.verbose)
        }
    
    def _optimize_and_freeze_models(self, models, train_data, val_data):
        # Reutilizamos el optimizador existente
        optimizer = TimeBalancedOptimizerWithTQDM(random_state=self.seed, verbose=self.verbose)
        optimized_params = optimizer.optimize_all_models(models, train_data, val_data)
        
        # Aplicar params optimizados
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
    
    def run_evaluation(self, series_index=0, save_predictions=False):
        print("="*60)
        print(f"💱 EXCHANGE RATE DATASET - Serie {series_index}")
        print("="*60)
        
        try:
            values, timestamps = self.load_exchange_data(series_index)
            split = self.split_data(values, timestamps)
            
            if self.verbose:
                print(f"   Datos Totales usados: {len(values)}")
                print(f"   Train: {len(split['train']['values'])} | Val: {len(split['val']['values'])} | Test: {len(split['test']['values'])}")
                
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame(), None
        
        models = self._setup_models(self.seed)
        models = self._optimize_and_freeze_models(models, split['train']['values'], split['val']['values'])
        
        if self.verbose: print(f"\n🔮 Fase 3: Predicción ({self.N_TEST_DAYS} pasos)")
        
        results_rows = []
        predictions_dict = {} if save_predictions else None
        
        test_vals = split['test']['values']
        history = np.concatenate([split['train']['values'], split['val']['values']])
        
        # Loop de predicción paso a paso
        for t in tqdm(range(len(test_vals)), desc="Prediciendo Días"):
            curr_hist = np.concatenate([history, test_vals[:t]])
            true_val = test_vals[t]
            row = {'Paso': t+1, 'Valor_Observado': true_val}
            
            if save_predictions:
                predictions_dict[t] = {'timestamp': split['test']['timestamps'][t], 'true_value': true_val, 'predictions': {}}
            
            for name, model in models.items():
                try:
                    # Adaptación de input según el modelo
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
                        
                    if save_predictions: 
                        predictions_dict[t]['predictions'][name] = preds
                        
                except Exception as e:
                    # if self.verbose: tqdm.write(f"Error {name} step {t}: {e}")
                    row[name] = np.nan
            
            results_rows.append(row)
            # Limpiar memoria cada 5 pasos para evitar overflow
            if t % 5 == 0: clear_all_sessions()
            
        df_results = pd.DataFrame(results_rows)
        return df_results, predictions_dict