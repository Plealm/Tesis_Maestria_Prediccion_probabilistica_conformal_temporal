import numpy as np
import pandas as pd
import warnings
import gc
import os
import time
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import Dict, List, Tuple, Union, Any

warnings.filterwarnings("ignore")

from simulacion import ARMASimulation
from modelos import (CircularBlockBootstrapModel, SieveBootstrapModel, LSPM, LSPMW, 
                     DeepARModel, AREPD, MondrianCPSModel, AdaptiveVolatilityMondrianCPS,
                     EnCQR_LSTM_Model, TimeBalancedOptimizer)
from metricas import crps, ecrps
from simulacion import ARIMASimulation, SETARSimulation, ARMASimulation
from figuras import PlotManager

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


class Pipeline140SinSesgos_ARMA:
    """
    Pipeline para ARMA sin sesgos temporales.
    Comparación de Densidad Predictiva vs Densidad Teórica mediante ECRPS.
    """
    N_TEST_STEPS = 12
    N_VALIDATION = 40
    N_TRAIN = 200
    
    ARMA_CONFIGS = [
        {'nombre': 'AR(1)', 'phi': [0.9], 'theta': []},
        {'nombre': 'AR(2)', 'phi': [0.5, -0.3], 'theta': []},
        {'nombre': 'MA(1)', 'phi': [], 'theta': [0.7]},
        {'nombre': 'MA(2)', 'phi': [], 'theta': [0.4, 0.2]},
        {'nombre': 'ARMA(1,1)', 'phi': [0.6], 'theta': [0.3]},
        {'nombre': 'ARMA(2,2)', 'phi': [0.4, -0.2], 'theta': [0.5, 0.1]},
        {'nombre': 'ARMA(2,1)', 'phi': [0.7, 0.2], 'theta': [0.5]}
    ]
    DISTRIBUTIONS = ['normal', 'uniform', 'exponential', 't-student', 'mixture']
    VARIANCES = [0.2, 0.5, 1.0, 3.0]

    def __init__(self, n_boot: int = 1000, seed: int = 42, verbose: bool = False):
        self.n_boot = n_boot
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)

    def generate_all_scenarios(self) -> list:
        scenarios = []
        # Generar combinaciones basadas en los atributos actuales de la instancia
        for i, arma in enumerate(self.ARMA_CONFIGS):
            for dist in self.DISTRIBUTIONS:
                for var in self.VARIANCES:
                    scenarios.append((arma.copy(), dist, var, self.seed + i))
        return scenarios

    def _run_scenario_wrapper(self, args):
        return self.run_single_scenario(*args)

    def run_single_scenario(self, arma_cfg, dist, var, scenario_seed):
        simulator = ARMASimulation(
            phi=arma_cfg['phi'], theta=arma_cfg['theta'],
            noise_dist=dist, sigma=np.sqrt(var), seed=scenario_seed
        )
        total_len = self.N_TRAIN + self.N_VALIDATION + self.N_TEST_STEPS
        series, errors = simulator.simulate(n=total_len, burn_in=100)
        
        train_data = series[:self.N_TRAIN]
        val_data = series[self.N_TRAIN : self.N_TRAIN + self.N_VALIDATION]
        
        models = {
            'Block Bootstrapping': CircularBlockBootstrapModel(n_boot=self.n_boot, random_state=scenario_seed),
            'Sieve Bootstrap': SieveBootstrapModel(n_boot=self.n_boot, random_state=scenario_seed),
            'LSPM': LSPM(random_state=scenario_seed),
            'LSPMW': LSPMW(rho=0.95, random_state=scenario_seed),
            'AREPD': AREPD(n_lags=5, rho=0.93, random_state=scenario_seed),
            'MCPS': MondrianCPSModel(n_lags=10, random_state=scenario_seed),
            'AV-MCPS': AdaptiveVolatilityMondrianCPS(n_lags=12, random_state=scenario_seed),
            'DeepAR': DeepARModel(hidden_size=20, n_lags=10, epochs=25, num_samples=self.n_boot, random_state=scenario_seed),
            'EnCQR-LSTM': EnCQR_LSTM_Model(n_lags=15, B=3, units=24, epochs=20, num_samples=self.n_boot, random_state=scenario_seed)
        }

        optimizer = TimeBalancedOptimizer(random_state=scenario_seed, verbose=self.verbose)
        best_params = optimizer.optimize_all_models(models, train_data, val_data)
        
        train_val_full = series[:self.N_TRAIN + self.N_VALIDATION]
        for name, model in models.items():
            if name in best_params:
                for k, v in best_params[name].items():
                    if hasattr(model, k): setattr(model, k, v)
            if hasattr(model, 'freeze_hyperparameters'):
                model.freeze_hyperparameters(train_val_full)

        results_rows = []
        plot_data = {}

        for t in range(self.N_TEST_STEPS):
            idx = self.N_TRAIN + self.N_VALIDATION + t
            h_series = series[:idx]
            h_errors = errors[:idx]
            
            true_samples = simulator.get_true_next_step_samples(h_series, h_errors, n_samples=1000)
            plot_data[t] = {'true_distribution': true_samples, 'model_predictions': {}}
            # Usar 'Paso' para compatibilidad con run_analysis
            row = {'Paso': t + 1, 'Config': arma_cfg['nombre'], 'Dist': dist, 'Var': var}
            
            for name, model in models.items():
                try:
                    if "Bootstrap" in name: pred = model.fit_predict(h_series)
                    else: pred = model.fit_predict(pd.DataFrame({'valor': h_series}))
                    pred_array = np.asarray(pred).flatten()
                    plot_data[t]['model_predictions'][name] = pred_array
                    row[name] = ecrps(pred_array, true_samples)
                except:
                    row[name] = np.nan
            results_rows.append(row)

        scen_name = f"{arma_cfg['nombre']}_{dist}_V{var}_S{scenario_seed}"
        df_res = pd.DataFrame(results_rows)
        for m_name in models.keys():
            path = f"reportes/{scen_name}/{m_name.replace(' ', '_')}.png"
            PlotManager.plot_individual_model_evolution(scen_name, m_name, plot_data, df_res, path)

        clear_all_sessions()
        return results_rows

    # CORRECCIÓN: run_all ahora acepta los argumentos del wrapper
    def run_all(self, excel_filename="resultados.xlsx", batch_size=10, n_jobs=2):
        tasks = self.generate_all_scenarios()
        print(f"🚀 Ejecutando {len(tasks)} escenarios en lotes de {batch_size}...")
        
        all_results = []
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            print(f"  -> Procesando lote {i//batch_size + 1}...")
            results = Parallel(n_jobs=n_jobs)(delayed(self._run_scenario_wrapper)(t) for t in batch)
            for r in results:
                all_results.extend(r)
            
            # Guardado intermedio por seguridad
            temp_df = pd.DataFrame(all_results)
            temp_df.to_excel(excel_filename, index=False)

        final_df = pd.DataFrame(all_results)
        return final_df

class Pipeline140SinSesgos_ARIMA:
    """
    Pipeline ARIMA:
    1) Comparación de Densidad Predictiva vs Densidad Teórica mediante ECRPS.
    2) Generación de 9 gráficos (12 pasos) por escenario con colores únicos.
    """
    
    N_TEST_STEPS = 12
    N_VALIDATION = 40
    N_TRAIN = 200
    
    SCENARIO_BUDGET = 300  
    OPTIMIZATION_BUDGET = 120  
    FREEZE_BUDGET = 30         
    TEST_BUDGET = 150          

    ARIMA_CONFIGS = [
        {'nombre': 'ARIMA(0,1,0)', 'phi': [], 'theta': []},
        {'nombre': 'ARIMA(1,1,0)', 'phi': [0.6], 'theta': []},
        {'nombre': 'ARIMA(2,1,0)', 'phi': [0.5, -0.2], 'theta': []},
        {'nombre': 'ARIMA(0,1,1)', 'phi': [], 'theta': [0.5]},
        {'nombre': 'ARIMA(0,1,2)', 'phi': [], 'theta': [0.4, 0.25]},
        {'nombre': 'ARIMA(1,1,1)', 'phi': [0.7], 'theta': [-0.3]},
        {'nombre': 'ARIMA(2,1,2)', 'phi': [0.6, 0.2], 'theta': [0.4, -0.1]}
    ]
    DISTRIBUTIONS = ['normal', 'uniform', 'exponential', 't-student', 'mixture']
    VARIANCES = [0.2, 0.5, 1.0, 3.0]

    def __init__(self, n_boot: int = 1000, seed: int = 42, verbose: bool = False):
        self.n_boot = n_boot
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)

    def generate_all_scenarios(self) -> list:
        scenarios = []
        scenario_id = 0
        for arima_cfg in self.ARIMA_CONFIGS:
            for dist in self.DISTRIBUTIONS:
                for var in self.VARIANCES:
                    scenarios.append((arima_cfg.copy(), dist, var, self.seed + scenario_id))
                    scenario_id += 1
        return scenarios

    def _run_scenario_wrapper(self, args):
        return self.run_single_scenario(*args)

    def run_single_scenario(self, arima_config, dist, var, scenario_seed):
        scenario_start = time.time()
        
        # 1. Simulación ARIMA (Serie Integrada + Errores subyacentes)
        simulator = ARIMASimulation(
            phi=arima_config['phi'], theta=arima_config['theta'],
            noise_dist=dist, sigma=np.sqrt(var), seed=scenario_seed
        )
        total_len = self.N_TRAIN + self.N_VALIDATION + self.N_TEST_STEPS
        series, errors = simulator.simulate(n=total_len, burn_in=100)
        
        train_data = series[:self.N_TRAIN]
        val_data = series[self.N_TRAIN : self.N_TRAIN + self.N_VALIDATION]
        
        # 2. Configuración de Modelos Ligeros
        models = self._setup_models(scenario_seed)

        # 3. Optimización y Congelamiento (Budget: 150s)
        prep_start = time.time()
        optimizer = TimeBalancedOptimizer(random_state=scenario_seed, verbose=self.verbose)
        best_params = optimizer.optimize_all_models(models, train_data, val_data)
        
        train_val_full = series[:self.N_TRAIN + self.N_VALIDATION]
        for name, model in models.items():
            if name in best_params:
                for k, v in best_params[name].items():
                    if hasattr(model, k): setattr(model, k, v)
            if hasattr(model, 'freeze_hyperparameters'):
                model.freeze_hyperparameters(train_val_full)
        prep_elapsed = time.time() - prep_start

        # 4. Testing Rolling Window con ECRPS (Budget: 150s)
        test_start = time.time()
        results_rows = []
        plot_data = {}
        time_per_step = self.TEST_BUDGET / self.N_TEST_STEPS

        for t in range(self.N_TEST_STEPS):
            step_start = time.time()
            idx = self.N_TRAIN + self.N_VALIDATION + t
            h_series = series[:idx]
            h_errors = errors[:idx]
            
            # Obtener Ground Truth Teórico (Dist. de Y_{n+1})
            true_samples = simulator.get_true_next_step_samples(h_series, h_errors, n_samples=1000)
            plot_data[t] = {'true_distribution': true_samples, 'model_predictions': {}}
            
            row = {'Paso': t + 1, 'Config': arima_config['nombre'], 'Dist': dist, 'Var': var}
            
            for name, model in models.items():
                try:
                    # Check timeout por paso
                    if (time.time() - step_start) > time_per_step and t > 0:
                        row[name] = np.nan
                        continue

                    # Inferencia
                    if "Bootstrap" in name: pred = model.fit_predict(h_series)
                    else: pred = model.fit_predict(pd.DataFrame({'valor': h_series}))
                    
                    pred_array = np.asarray(pred).flatten()
                    plot_data[t]['model_predictions'][name] = pred_array
                    
                    # Métrica ECRPS (Densidad vs Densidad)
                    row[name] = ecrps(pred_array, true_samples)
                except:
                    row[name] = np.nan
            
            results_rows.append(row)
        
        test_elapsed = time.time() - test_start

        # 5. Generación de Gráficos (9 imágenes por modelo)
        scen_id = f"{arima_config['nombre']}_{dist}_V{var}_S{scenario_seed}"
        df_res = pd.DataFrame(results_rows)
        
        for m_name in models.keys():
            path = f"reportes_arima/{scen_id}/{m_name.replace(' ', '_')}.png"
            PlotManager.plot_individual_model_evolution(scen_id, m_name, plot_data, df_res, path)

        total_elapsed = time.time() - scenario_start
        if self.verbose:
            print(f"✅ Escenario {arima_config['nombre']} fin en {total_elapsed:.1f}s")

        clear_all_sessions()
        return results_rows

    def _setup_models(self, seed):
        """Versión optimizada para velocidad."""
        return {
            'Block Bootstrapping': CircularBlockBootstrapModel(n_boot=self.n_boot, random_state=seed),
            'Sieve Bootstrap': SieveBootstrapModel(n_boot=self.n_boot, random_state=seed),
            'LSPM': LSPM(random_state=seed),
            'LSPMW': LSPMW(rho=0.95, random_state=seed),
            'AREPD': AREPD(n_lags=5, rho=0.93, random_state=seed),
            'MCPS': MondrianCPSModel(n_lags=10, random_state=seed),
            'AV-MCPS': AdaptiveVolatilityMondrianCPS(n_lags=12, random_state=seed),
            'DeepAR': DeepARModel(hidden_size=16, n_lags=5, epochs=20, num_samples=self.n_boot, 
                                  random_state=seed, early_stopping_patience=3),
            'EnCQR-LSTM': EnCQR_LSTM_Model(n_lags=15, B=3, units=24, epochs=15, num_samples=self.n_boot, 
                                          random_state=seed)
        }

    def run_all(self, excel_filename="resultados_arima_ecrps.xlsx", batch_size=10, max_workers=4):
        tasks = self.generate_all_scenarios()
        print(f"🚀 Iniciando Pipeline ARIMA: {len(tasks)} escenarios.")
        
        all_results = []
        num_batches = (len(tasks) + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(tasks))
            batch = tasks[start_idx:end_idx]
            
            print(f"📦 Procesando Lote {i+1}/{num_batches}...")
            results = Parallel(n_jobs=max_workers, backend='loky')(
                delayed(self._run_scenario_wrapper)(t) for t in batch
            )
            
            for r in results: all_results.extend(r)
            pd.DataFrame(all_results).to_excel(excel_filename, index=False)
            clear_all_sessions()
            gc.collect()

        return pd.DataFrame(all_results)

class Pipeline140SinSesgos_SETAR:
    """
    Pipeline SETAR:
    1) Comparación de Densidad Predictiva vs Densidad Teórica mediante ECRPS.
    2) Generación de 9 gráficos (12 niveles) por escenario con colores únicos.
    """
    
    N_TEST_STEPS = 12
    N_VALIDATION = 40
    N_TRAIN = 200
    
    SCENARIO_BUDGET = 300  
    OPTIMIZATION_BUDGET = 120  
    FREEZE_BUDGET = 30         
    TEST_BUDGET = 150          

    SETAR_CONFIGS = [
        {'nombre': 'SETAR-1', 'phi_regime1': [0.6], 'phi_regime2': [-0.5], 'threshold': 0.0, 'delay': 1, 'description': 'SETAR(2;1,1) d=1, r=0'},
        {'nombre': 'SETAR-2', 'phi_regime1': [0.7], 'phi_regime2': [-0.7], 'threshold': 0.0, 'delay': 2, 'description': 'SETAR(2;1,1) d=2, r=0'},
        {'nombre': 'SETAR-3', 'phi_regime1': [0.5, -0.2], 'phi_regime2': [-0.3, 0.1], 'threshold': 0.5, 'delay': 1, 'description': 'SETAR(2;2,2) d=1, r=0.5'},
        {'nombre': 'SETAR-4', 'phi_regime1': [0.8, -0.15], 'phi_regime2': [-0.6, 0.2], 'threshold': 1.0, 'delay': 2, 'description': 'SETAR(2;2,2) d=2, r=1.0'},
        {'nombre': 'SETAR-5', 'phi_regime1': [0.4, -0.1, 0.05], 'phi_regime2': [-0.3, 0.1, -0.05], 'threshold': 0.0, 'delay': 1, 'description': 'SETAR(2;3,3) d=1, r=0'},
        {'nombre': 'SETAR-6', 'phi_regime1': [0.5, -0.3, 0.1], 'phi_regime2': [-0.4, 0.2, -0.05], 'threshold': 0.5, 'delay': 2, 'description': 'SETAR(2;3,3) d=2, r=0.5'},
        {'nombre': 'SETAR-7', 'phi_regime1': [0.3, 0.1], 'phi_regime2': [-0.2, -0.1], 'threshold': 0.8, 'delay': 3, 'description': 'SETAR(2;2,2) d=3, r=0.8'}
    ]
    
    DISTRIBUTIONS = ['normal', 'uniform', 'exponential', 't-student', 'mixture']
    VARIANCES = [0.2, 0.5, 1.0, 3.0]

    def __init__(self, n_boot: int = 1000, seed: int = 42, verbose: bool = False):
        self.n_boot = n_boot
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)

    def generate_all_scenarios(self) -> list:
        scenarios = []
        scenario_id = 0
        for setar_cfg in self.SETAR_CONFIGS:
            for dist in self.DISTRIBUTIONS:
                for var in self.VARIANCES:
                    scenarios.append((setar_cfg.copy(), dist, var, self.seed + scenario_id))
                    scenario_id += 1
        return scenarios

    def _run_scenario_wrapper(self, args):
        return self.run_single_scenario(*args)

    def run_single_scenario(self, setar_config, dist, var, scenario_seed):
        scenario_start = time.time()
        
        # 1. Simulación SETAR
        simulator = SETARSimulation(
            model_type=setar_config['nombre'],
            phi_regime1=setar_config['phi_regime1'],
            phi_regime2=setar_config['phi_regime2'],
            threshold=setar_config['threshold'],
            delay=setar_config['delay'],
            noise_dist=dist,
            sigma=np.sqrt(var),
            seed=scenario_seed
        )
        total_len = self.N_TRAIN + self.N_VALIDATION + self.N_TEST_STEPS
        series, errors = simulator.simulate(n=total_len, burn_in=100)
        
        train_data = series[:self.N_TRAIN]
        val_data = series[self.N_TRAIN : self.N_TRAIN + self.N_VALIDATION]
        
        # 2. Configuración de Modelos
        models = self._setup_models(scenario_seed)

        # 3. Optimización y Congelamiento
        prep_start = time.time()
        optimizer = TimeBalancedOptimizer(random_state=scenario_seed, verbose=self.verbose)
        best_params = optimizer.optimize_all_models(models, train_data, val_data)
        
        train_val_full = series[:self.N_TRAIN + self.N_VALIDATION]
        for name, model in models.items():
            if name in best_params:
                for k, v in best_params[name].items():
                    if hasattr(model, k): setattr(model, k, v)
            if hasattr(model, 'freeze_hyperparameters'):
                model.freeze_hyperparameters(train_val_full)
        prep_elapsed = time.time() - prep_start

        # 4. Testing Rolling Window con ECRPS
        test_start = time.time()
        results_rows = []
        plot_data = {}
        time_per_step = self.TEST_BUDGET / self.N_TEST_STEPS

        for t in range(self.N_TEST_STEPS):
            step_start = time.time()
            idx = self.N_TRAIN + self.N_VALIDATION + t
            h_series = series[:idx]
            h_errors = errors[:idx]
            
            # Obtener Ground Truth Teórico (Depende del régimen actual en SETAR)
            true_samples = simulator.get_true_next_step_samples(h_series, h_errors, n_samples=1000)
            plot_data[t] = {'true_distribution': true_samples, 'model_predictions': {}}
            
            # Columna 'Paso' para compatibilidad con run_analysis
            row = {
                'Paso': t + 1, 
                'Config': setar_config['nombre'], 
                'Descripción': setar_config['description'],
                'Dist': dist, 
                'Var': var
            }
            
            for name, model in models.items():
                try:
                    if (time.time() - step_start) > time_per_step and t > 0:
                        row[name] = np.nan
                        continue

                    if "Bootstrap" in name: pred = model.fit_predict(h_series)
                    else: pred = model.fit_predict(pd.DataFrame({'valor': h_series}))
                    
                    pred_array = np.asarray(pred).flatten()
                    plot_data[t]['model_predictions'][name] = pred_array
                    
                    # ECRPS: Comparación de densidades
                    row[name] = ecrps(pred_array, true_samples)
                except:
                    row[name] = np.nan
            
            results_rows.append(row)
        
        test_elapsed = time.time() - test_start

        # 5. Generación de Gráficos (9 imágenes por modelo)
        scen_id = f"{setar_config['nombre']}_{dist}_V{var}_S{scenario_seed}"
        df_res = pd.DataFrame(results_rows)
        
        for m_name in models.keys():
            path = f"reportes_setar/{scen_id}/{m_name.replace(' ', '_')}.png"
            PlotManager.plot_individual_model_evolution(scen_id, m_name, plot_data, df_res, path)

        total_elapsed = time.time() - scenario_start
        if self.verbose:
            print(f"✅ Escenario SETAR {setar_config['nombre']} fin en {total_elapsed:.1f}s")

        clear_all_sessions()
        return results_rows

    def _setup_models(self, seed):
        return {
            'Block Bootstrapping': CircularBlockBootstrapModel(n_boot=self.n_boot, random_state=seed),
            'Sieve Bootstrap': SieveBootstrapModel(n_boot=self.n_boot, random_state=seed),
            'LSPM': LSPM(random_state=seed),
            'LSPMW': LSPMW(rho=0.95, random_state=seed),
            'AREPD': AREPD(n_lags=5, rho=0.93, random_state=seed),
            'MCPS': MondrianCPSModel(n_lags=10, random_state=seed),
            'AV-MCPS': AdaptiveVolatilityMondrianCPS(n_lags=12, random_state=seed),
            'DeepAR': DeepARModel(hidden_size=16, n_lags=5, epochs=20, num_samples=self.n_boot, 
                                  random_state=seed, early_stopping_patience=3),
            'EnCQR-LSTM': EnCQR_LSTM_Model(n_lags=15, B=3, units=24, epochs=15, num_samples=self.n_boot, 
                                          random_state=seed)
        }

    def run_all(self, excel_filename="resultados_setar_ecrps.xlsx", batch_size=10, max_workers=4):
        tasks = self.generate_all_scenarios()
        print(f"🚀 Iniciando Pipeline SETAR: {len(tasks)} escenarios.")
        
        all_results = []
        num_batches = (len(tasks) + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(tasks))
            batch = tasks[start_idx:end_idx]
            
            print(f"📦 Procesando Lote {i+1}/{num_batches}...")
            results = Parallel(n_jobs=max_workers, backend='loky')(
                delayed(self._run_scenario_wrapper)(t) for t in batch
            )
            
            for r in results: all_results.extend(r)
            pd.DataFrame(all_results).to_excel(excel_filename, index=False)
            clear_all_sessions()
            gc.collect()

        return pd.DataFrame(all_results)

# ============================================================
# ¿ Es importante diferenciar e integrar en ARIMA?
# ===========================================================

class TransformadorDiferenciacionIntegracion:
    """
    Maneja la transformación diferenciación ↔ integración.
    
    Para ARIMA(p,d,q):
    - d=1: ΔY_t = Y_t - Y_{t-1}
    - Integración: Y_t = Y_{t-1} + ΔY_t
    """
    
    def __init__(self, d: int = 1, verbose: bool = False):
        """
        Args:
            d: Orden de diferenciación (típicamente 1)
            verbose: Mostrar información
        """
        if d not in [1, 2]:
            raise ValueError("Solo se soporta d=1 o d=2")
        self.d = d
        self.verbose = verbose
    
    def diferenciar_serie(self, serie: np.ndarray) -> np.ndarray:
        """
        Aplica diferenciación de orden d.
        
        Args:
            serie: Serie original Y_t
        
        Returns:
            Serie diferenciada ΔY_t
        """
        if self.d == 1:
            # Primera diferencia: ΔY_t = Y_t - Y_{t-1}
            serie_diff = np.diff(serie)
        elif self.d == 2:
            # Segunda diferencia: Δ²Y_t = ΔY_t - ΔY_{t-1}
            serie_diff = np.diff(np.diff(serie))
        else:
            serie_diff = serie
        
        if self.verbose:
            print(f"  Diferenciación d={self.d}: {len(serie)} → {len(serie_diff)} puntos")
        
        return serie_diff
    
    def integrar_predicciones(self, predicciones_diff: np.ndarray,
                              ultimo_valor_observado: float) -> np.ndarray:
        """
        Integra predicciones desde espacio diferenciado.
        
        Para d=1:
            Y_{t+1} = Y_t + ΔY_{t+1}
        
        Args:
            predicciones_diff: Muestras de ΔY_{t+1}
            ultimo_valor_observado: Y_t (último valor conocido)
        
        Returns:
            Muestras de Y_{t+1}
        """
        if self.d == 1:
            # Y_{t+1} = Y_t + ΔY_{t+1}
            predicciones_integradas = ultimo_valor_observado + predicciones_diff
        elif self.d == 2:
            # Para d=2 se necesitaría también Y_{t-1}, por simplicidad solo d=1
            raise NotImplementedError("Integración para d=2 no implementada")
        else:
            predicciones_integradas = predicciones_diff
        
        if self.verbose:
            print(f"  Integración: Y_t={ultimo_valor_observado:.4f}, "
                  f"ΔY_t ∈ [{np.min(predicciones_diff):.4f}, {np.max(predicciones_diff):.4f}] → "
                  f"Y_{{t+1}} ∈ [{np.min(predicciones_integradas):.4f}, {np.max(predicciones_integradas):.4f}]")
        
        return predicciones_integradas


class Pipeline140SinSesgos_ARIMA_ConDiferenciacion:
    """
    Pipeline ARIMA optimizado para generar el formato de Excel solicitado.
    Evalúa cada escenario en dos modalidades: SIN_DIFF y CON_DIFF.
    """
    
    N_TEST_STEPS = 12
    N_VALIDATION = 40
    N_TRAIN = 200
    
    # Configuraciones ARIMA (d=1 por defecto para estos nombres)
    ARIMA_CONFIGS = [
        {'nombre': 'RW', 'phi': [], 'theta': []}, # Random Walk es ARIMA(0,1,0)
        {'nombre': 'AR(1)', 'phi': [0.6], 'theta': []},
        {'nombre': 'AR(2)', 'phi': [0.5, -0.2], 'theta': []},
        {'nombre': 'MA(1)', 'phi': [], 'theta': [0.5]},
        {'nombre': 'MA(2)', 'phi': [], 'theta': [0.4, 0.25]},
        {'nombre': 'ARMA(1,1)', 'phi': [0.7], 'theta': [-0.3]},
        {'nombre': 'ARMA(2,2)', 'phi': [0.6, 0.2], 'theta': [0.4, -0.1]}
    ]
    DISTRIBUTIONS = ['normal', 'uniform', 'exponential', 't-student', 'mixture']
    VARIANCES = [0.2, 0.5, 1.0, 3.0]

    def __init__(self, n_boot: int = 1000, seed: int = 42, verbose: bool = False):
        self.n_boot = n_boot
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)

    def _setup_models(self, seed: int):
        return {
            'AREPD': AREPD(n_lags=5, rho=0.9, random_state=seed),
            'AV-MCPS': AdaptiveVolatilityMondrianCPS(n_lags=15, random_state=seed),
            'Block Bootstrapping': CircularBlockBootstrapModel(n_boot=self.n_boot, random_state=seed),
            'DeepAR': DeepARModel(hidden_size=15, n_lags=5, epochs=25, num_samples=self.n_boot, random_state=seed),
            'EnCQR-LSTM': EnCQR_LSTM_Model(n_lags=20, B=3, units=24, epochs=15, num_samples=self.n_boot, random_state=seed),
            'LSPM': LSPM(random_state=seed),
            'LSPMW': LSPMW(rho=0.95, random_state=seed),
            'MCPS': MondrianCPSModel(n_lags=10, random_state=seed),
            'Sieve Bootstrap': SieveBootstrapModel(n_boot=self.n_boot, random_state=seed)
        }

    def _run_modalidad(self, simulator, series_levels, errors, arima_config, dist, var, modalidad, scenario_seed):
        """Ejecuta una modalidad específica (SIN_DIFF o CON_DIFF)"""
        
        # 1. Preparar datos según modalidad
        if modalidad == "CON_DIFF":
            # Los modelos ven incrementos ΔY_t
            series_to_models = np.diff(series_levels, prepend=series_levels[0])
        else:
            # Los modelos ven niveles Y_t
            series_to_models = series_levels

        train_data = series_to_models[:self.N_TRAIN]
        val_data = series_to_models[self.N_TRAIN : self.N_TRAIN + self.N_VALIDATION]
        
        # 2. Setup y Optimización
        models = self._setup_models(scenario_seed)
        optimizer = TimeBalancedOptimizer(random_state=scenario_seed, verbose=False)
        best_params = optimizer.optimize_all_models(models, train_data, val_data)
        
        train_val_full = series_to_models[:self.N_TRAIN + self.N_VALIDATION]
        for name, model in models.items():
            if name in best_params:
                for k, v in best_params[name].items():
                    if hasattr(model, k): setattr(model, k, v)
            if hasattr(model, 'freeze_hyperparameters'):
                model.freeze_hyperparameters(train_val_full)

        # 3. Testing Rolling Window
        results_rows = []
        p, q = len(arima_config['phi']), len(arima_config['theta'])
        d = 1 # Por defecto en esta simulación

        for t in range(self.N_TEST_STEPS):
            idx = self.N_TRAIN + self.N_VALIDATION + t
            h_series_levels = series_levels[:idx]
            h_errors = errors[:idx]
            h_to_model = series_to_models[:idx]
            
            # Densidad Teórica (Siguiente paso real)
            true_samples = simulator.get_true_next_step_samples(h_series_levels, h_errors, n_samples=1000)
            
            # Fila base con el formato de la imagen
            row = {
                'Paso': t + 1,
                'Proceso': f"ARMA_I({p},{d},{q})",
                'p': p,
                'd': d,
                'q': q,
                'ARMA_base': arima_config['nombre'],
                'Distribución': dist,
                'Varianza': var,
                'Modalidad': modalidad,
                'Valor_Observado': series_levels[idx] # El valor real que ocurrió
            }
            
            for name, model in models.items():
                try:
                    if "Bootstrap" in name: pred = model.fit_predict(h_to_model)
                    else: pred = model.fit_predict(pd.DataFrame({'valor': h_to_model}))
                    
                    pred_array = np.asarray(pred).flatten()
                    
                    # Si predijo incremento, sumar al último nivel para comparar densidades en niveles
                    if modalidad == "CON_DIFF":
                        pred_array = series_levels[idx-1] + pred_array
                    
                    row[name] = ecrps(pred_array, true_samples)
                except:
                    row[name] = np.nan
            
            results_rows.append(row)
            
        return results_rows

    def _run_scenario_wrapper(self, args):
        arima_cfg, dist, var, seed = args
        
        # Simulación (Niveles)
        simulator = ARIMASimulation(
            phi=arima_cfg['phi'], theta=arima_cfg['theta'],
            noise_dist=dist, sigma=np.sqrt(var), seed=seed
        )
        total_len = self.N_TRAIN + self.N_VALIDATION + self.N_TEST_STEPS
        series_levels, errors = simulator.simulate(n=total_len, burn_in=100)
        
        # Ejecutar ambas modalidades
        res_sin = self._run_modalidad(simulator, series_levels, errors, arima_cfg, dist, var, "SIN_DIFF", seed)
        res_con = self._run_modalidad(simulator, series_levels, errors, arima_cfg, dist, var, "CON_DIFF", seed + 1)
        
        clear_all_sessions()
        return res_sin + res_con

    def generate_all_scenarios(self) -> list:
        scenarios = []
        s_id = 0
        for arima_cfg in self.ARIMA_CONFIGS:
            for dist in self.DISTRIBUTIONS:
                for var in self.VARIANCES:
                    scenarios.append((arima_cfg.copy(), dist, var, self.seed + s_id))
                    s_id += 1
        return scenarios

    def run_all(self, excel_filename="resultados_arima_completo.xlsx", batch_size=5, n_jobs=2):
        tasks = self.generate_all_scenarios()
        print(f"🚀 Ejecutando {len(tasks)} escenarios ARIMA (Doble Modalidad)...")
        
        all_results = []
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            print(f"  -> Procesando lote {i//batch_size + 1}...")
            results = Parallel(n_jobs=n_jobs)(delayed(self._run_scenario_wrapper)(t) for t in batch)
            for r in results:
                all_results.extend(r)
            
            pd.DataFrame(all_results).to_excel(excel_filename, index=False)

        return pd.DataFrame(all_results)



# ============================================================================
# ¿ Hasta que qué orden de integración d es viable simular ARIMA(p,d,q)?
# ============================================================================


class ARIMAMultiDSimulation:
    """
    Simulador ARIMA(p,d,q) con orden de integración d variable (1 a 10).
    
    Genera: Y_t donde ∇^d Y_t ~ ARMA(p,q)
    Es decir: (1-B)^d Y_t = φ(B) θ(B)^(-1) ε_t
    """
    
    def __init__(self, phi: List[float], theta: List[float], d: int,
                 noise_dist: str = 'normal', sigma: float = 1.0,
                 seed: int = 42, verbose: bool = False):
        """
        Args:
            phi: Coeficientes AR del componente ARMA
            theta: Coeficientes MA del componente ARMA
            d: Orden de integración (1 a 10)
            noise_dist: Distribución del ruido
            sigma: Desviación estándar del ruido
            seed: Semilla aleatoria
            verbose: Mostrar información
        """
        self.phi = np.array(phi) if phi else np.array([])
        self.theta = np.array(theta) if theta else np.array([])
        self.d = d
        self.noise_dist = noise_dist
        self.sigma = sigma
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        
        if d < 1 or d > 10:
            raise ValueError(f"d debe estar entre 1 y 10, recibido: {d}")
    
    def _generate_errors(self, n: int) -> np.ndarray:
        """Genera errores según la distribución especificada (igual que ARMASimulation)."""
        if self.noise_dist == 'normal':
            return self.rng.normal(0, self.sigma, n)
        elif self.noise_dist == 'uniform':
            limit = np.sqrt(3) * self.sigma
            return self.rng.uniform(-limit, limit, size=n)
        elif self.noise_dist == 'exponential':
            return self.rng.exponential(scale=self.sigma, size=n) - self.sigma
        elif self.noise_dist == 't-student':
            from scipy.stats import t
            df = 5
            scale_factor = self.sigma * np.sqrt((df - 2) / df)
            return t.rvs(df, scale=scale_factor, size=n, random_state=self.rng)
        elif self.noise_dist == 'mixture':
            n1 = int(n * 0.75)
            n2 = n - n1
            variance_of_means = 0.75 * (-0.25 * self.sigma * 2)**2 + 0.25 * (0.75 * self.sigma * 2)**2
            if self.sigma**2 < variance_of_means:
                raise ValueError("La varianza de la mezcla no puede ser la sigma deseada.")
            component_std = np.sqrt(self.sigma**2 - variance_of_means)
            comp1 = self.rng.normal(-0.25 * self.sigma * 2, component_std, n1)
            comp2 = self.rng.normal(0.75 * self.sigma * 2, component_std, n2)
            mixture = np.concatenate([comp1, comp2])
            self.rng.shuffle(mixture)
            return mixture - np.mean(mixture)
        else:
            raise ValueError(f"Distribución de ruido no soportada: {self.noise_dist}")
    
    def simulate(self, n: int, burn_in: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simula serie ARIMA(p,d,q).
        
        Proceso:
        1. Simula W_t ~ ARMA(p,q) estacionario
        2. Integra d veces: Y_t = Σ^d W_t
        
        Returns:
            (serie_integrada, errores)
        """
        total_n = n + burn_in
        p = len(self.phi)
        q = len(self.theta)
        max_lag = max(p, q, 1)
        
        # Paso 1: Simular ARMA(p,q) estacionario
        errors = self._generate_errors(total_n + max_lag)
        w_series = np.zeros(total_n + max_lag)
        
        # Inicialización
        initial_values = self.rng.normal(0, self.sigma, max_lag)
        w_series[:max_lag] = initial_values
        
        # Generar ARMA
        for t in range(max_lag, total_n + max_lag):
            ar_part = 0.0
            if p > 0:
                ar_part = np.dot(self.phi, w_series[t-p:t][::-1])
            
            ma_part = 0.0
            if q > 0:
                ma_part = np.dot(self.theta, errors[t-q:t][::-1])
            
            w_series[t] = ar_part + ma_part + errors[t]
        
        # Remover inicialización
        w_series = w_series[max_lag:]
        errors = errors[max_lag:]
        
        # Paso 2: Integrar d veces
        y_series = w_series.copy()
        for _ in range(self.d):
            y_series = np.cumsum(y_series)
        
        # Remover burn-in
        y_series = y_series[burn_in:]
        errors = errors[burn_in:]
        
        if self.verbose:
            print(f"  Simulado ARIMA(p={p},d={self.d},q={q})")
            print(f"  Serie final: n={len(y_series)}, rango=[{y_series.min():.2f}, {y_series.max():.2f}]")
        
        return y_series, errors
    
    def get_true_next_step_samples(self, series_history: np.ndarray,
                                   errors_history: np.ndarray,
                                   n_samples: int = 5000) -> np.ndarray:
        """
        Genera muestras de la distribución verdadera del siguiente paso.
        
        Para ARIMA(p,d,q):
        - Diferencia d veces la historia para obtener W_t
        - Predice siguiente W usando ARMA
        - Integra de vuelta para obtener Y
        """
        # Diferenciar d veces para obtener el proceso ARMA
        w_history = series_history.copy()
        last_values = []
        
        for _ in range(self.d):
            last_values.append(w_history[-1])
            w_history = np.diff(w_history)
        
        p = len(self.phi)
        q = len(self.theta)
        
        # Predicción en el espacio ARMA
        ar_pred = 0.0
        if p > 0 and len(w_history) >= p:
            ar_pred = np.dot(self.phi, w_history[-p:][::-1])
        
        ma_pred = 0.0
        if q > 0 and len(errors_history) >= q:
            ma_pred = np.dot(self.theta, errors_history[-q:][::-1])
        
        # Generar muestras de W_{t+1}
        noise_samples = self._generate_errors(n_samples)
        w_next_samples = ar_pred + ma_pred + noise_samples
        
        # Integrar d veces de vuelta a Y
        y_next_samples = w_next_samples.copy()
        for i in range(self.d - 1, -1, -1):
            y_next_samples = last_values[i] + y_next_samples
        
        return y_next_samples


class PipelineARIMA_MultiD_DobleModalidad:
    """
    Pipeline Multi-D CORREGIDO para ARIMA(p,d,q) con múltiples órdenes de integración.
    
    CORRECCIONES FUNDAMENTALES (basado en PipelineARIMA_MultiD_SieveOnly):
    1. Usa ARIMASimulation (no ARIMAMultiDSimulation) para d=1
    2. Implementa integración manual para d>1 (como Pipeline140)
    3. Densidades predictivas calculadas en el espacio correcto
    4. Integración coherente para predicciones
    5. Evalúa TODOS los modelos (no solo Sieve Bootstrap)
    """
    
    N_TEST_STEPS = 12
    N_VALIDATION_FOR_OPT = 40
    N_TRAIN_INITIAL = 200

    ARMA_CONFIGS = [
        {'nombre': 'RW', 'phi': [], 'theta': []},
        {'nombre': 'AR(1)', 'phi': [0.6], 'theta': []},
        {'nombre': 'AR(2)', 'phi': [0.5, -0.2], 'theta': []},
        {'nombre': 'MA(1)', 'phi': [], 'theta': [0.5]},
        {'nombre': 'MA(2)', 'phi': [], 'theta': [0.4, 0.25]},
        {'nombre': 'ARMA(1,1)', 'phi': [0.7], 'theta': [-0.3]},
        {'nombre': 'ARMA(2,2)', 'phi': [0.6, 0.2], 'theta': [0.4, -0.1]}
    ]
    
    DISTRIBUTIONS = ['normal', 'uniform', 'exponential', 't-student', 'mixture']
    VARIANCES = [0.2, 0.5, 1.0, 3.0]
    D_VALUES = [1, 2, 3, 4, 5, 6, 7, 10]
    
    def __init__(self, n_boot: int = 1000, seed: int = 42, verbose: bool = False):
        self.n_boot = n_boot
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)

    def _setup_models(self, seed: int):
        """Configura TODOS los modelos (igual que otras pipelines)."""
        return {
            'AREPD': AREPD(n_lags=5, rho=0.9, random_state=seed),
            'AV-MCPS': AdaptiveVolatilityMondrianCPS(n_lags=15, random_state=seed),
            'Block Bootstrapping': CircularBlockBootstrapModel(n_boot=self.n_boot, random_state=seed),
            'DeepAR': DeepARModel(hidden_size=15, n_lags=5, epochs=25, num_samples=self.n_boot, random_state=seed),
            'EnCQR-LSTM': EnCQR_LSTM_Model(n_lags=20, B=3, units=24, epochs=15, num_samples=self.n_boot, random_state=seed),
            'LSPM': LSPM(random_state=seed),
            'LSPMW': LSPMW(rho=0.95, random_state=seed),
            'MCPS': MondrianCPSModel(n_lags=10, random_state=seed),
            'Sieve Bootstrap': SieveBootstrapModel(n_boot=self.n_boot, random_state=seed)
        }

    def _simulate_arima_manual(self, arma_config: dict, d_value: int, 
                              dist: str, var: float, seed: int, n: int):
        """
        Simula ARIMA EXACTAMENTE como Pipeline140SinSesgos_ARIMA_ConDiferenciacion.
        
        Proceso:
        1. Simula W_t ~ ARMA(p,q) usando ARIMASimulation
        2. Integra manualmente d veces: Y_t = S^d(W_t)
        
        IMPORTANTE: Para d=1, esto es IDÉNTICO a ARIMASimulation directamente.
        """
        from simulacion import ARIMASimulation
        
        # Simular usando ARIMASimulation (siempre con d=1 internamente)
        simulator = ARIMASimulation(
            phi=arma_config['phi'],
            theta=arma_config['theta'],
            noise_dist=dist,
            sigma=np.sqrt(var),
            seed=seed
        )
        
        # Para ARIMASimulation, la serie ya viene con 1 integración
        # Si d=1, usamos directamente. Si d>1, integramos (d-1) veces adicionales
        series_base, errors = simulator.simulate(n=n, burn_in=100)
        
        # Si d=1, ya está integrada correctamente
        if d_value == 1:
            y_series = series_base.copy()
        else:
            # Para d>1, integrar (d-1) veces adicionales
            y_series = series_base.copy()
            for _ in range(d_value - 1):
                y_series = np.cumsum(y_series)
        
        return y_series, series_base, errors, simulator

    def _get_true_density_from_simulator(self, simulator, series_history: np.ndarray,
                                        errors_history: np.ndarray, 
                                        n_samples: int = 1000) -> np.ndarray:
        """
        Obtiene densidad verdadera usando EXACTAMENTE el método de ARIMASimulation.
        
        IDÉNTICO a Pipeline140SinSesgos_ARIMA_ConDiferenciacion.
        """
        return simulator.get_true_next_step_samples(
            series_history, errors_history, n_samples=n_samples
        )

    def _integrate_d_times_for_prediction(self, w_next_samples: np.ndarray,
                                         y_series: np.ndarray, 
                                         current_idx: int,
                                         d_value: int) -> np.ndarray:
        """
        Integra predicciones desde espacio ARMA(d=1) a ARIMA(d>1).
        
        BASADO EN: TransformadorDiferenciacionIntegracion del código original
        
        Para d=1: Y_{t+1} = Y_t + W_{t+1}
        Para d>1: Usar fórmula recursiva
        """
        if d_value == 1:
            # Caso simple: Y_{t+1} = Y_t + ΔY_t donde ΔY_t = W_{t+1}
            return y_series[current_idx - 1] + w_next_samples
        else:
            # Para d>1, necesitamos aplicar integración múltiple
            # Guardamos los últimos d valores de Y
            y_last_values = []
            temp_y = y_series[:current_idx].copy()
            
            for level in range(d_value):
                y_last_values.append(temp_y[-1])
                if level < d_value - 1:
                    temp_y = np.diff(temp_y)
            
            # Integrar desde W_{t+1} hasta Y_{t+1}
            y_next_samples = w_next_samples.copy()
            for level in range(d_value - 1, -1, -1):
                y_next_samples = y_last_values[level] + y_next_samples
            
            return y_next_samples

    def _run_single_modalidad(self, arma_config: dict, d_value: int,
                             dist: str, var: float, scenario_seed: int,
                             y_series: np.ndarray, series_base: np.ndarray,
                             errors: np.ndarray, test_start_idx: int,
                             usar_diferenciacion: bool, simulator) -> list:
        """
        Ejecuta una modalidad EXACTAMENTE como Pipeline140SinSesgos_ARIMA_ConDiferenciacion.
        Pero ahora evalúa TODOS los modelos (no solo Sieve Bootstrap).
        
        MODALIDADES:
        - SIN_DIFF: Modelos ven Y_t (serie integrada de orden d)
        - CON_DIFF: Modelos ven ∇Y_t (serie diferenciada 1 vez)
        """
        modalidad_str = "CON_DIFF" if usar_diferenciacion else "SIN_DIFF"
        
        # Preparar serie según modalidad (IGUAL que Pipeline140)
        if usar_diferenciacion:
            # Los modelos ven incrementos ΔY_t
            series_to_models = np.diff(y_series, prepend=y_series[0])
        else:
            # Los modelos ven niveles Y_t
            series_to_models = y_series.copy()
        
        train_calib_data = series_to_models[:test_start_idx]
        
        # Crear TODOS los modelos
        models = self._setup_models(scenario_seed)
        
        # Optimización (TimeBalancedOptimizer como Pipeline140)
        optimizer = TimeBalancedOptimizer(random_state=self.seed, verbose=self.verbose)
        
        split = min(self.N_VALIDATION_FOR_OPT, len(train_calib_data) // 3)
        best_params = optimizer.optimize_all_models(
            models, 
            train_calib_data[:-split], 
            train_calib_data[-split:]
        )
        
        # Aplicar hiperparámetros óptimos
        for name, model in models.items():
            if name in best_params:
                for k, v in best_params[name].items():
                    if hasattr(model, k): 
                        setattr(model, k, v)
            if hasattr(model, 'freeze_hyperparameters'):
                model.freeze_hyperparameters(train_calib_data)

        # Testing rolling window
        results_rows = []
        p = len(arma_config['phi'])
        q = len(arma_config['theta'])

        for t in range(self.N_TEST_STEPS):
            curr_idx = test_start_idx + t
            h_series_levels = y_series[:curr_idx]
            h_to_model = series_to_models[:curr_idx]
            
            # DENSIDAD VERDADERA: Usar el simulador base (ARIMASimulation)
            # Esto da la densidad de Y_{t+1} donde Y tiene 1 integración
            # Si d=1, es directa. Si d>1, necesitamos integrar
            
            if d_value == 1:
                # Para d=1, usar directamente get_true_next_step_samples
                true_samples_base = self._get_true_density_from_simulator(
                    simulator, series_base[:curr_idx], errors[:curr_idx]
                )
                true_samples = true_samples_base
            else:
                # Para d>1, obtener densidad base y luego integrar
                true_samples_base = self._get_true_density_from_simulator(
                    simulator, series_base[:curr_idx], errors[:curr_idx]
                )
                # Integrar las muestras (d-1) veces adicionales
                true_samples = self._integrate_d_times_for_prediction(
                    true_samples_base, y_series, curr_idx, d_value
                )
            
            # Fila de resultados
            row = {
                'Paso': t + 1,
                'Proceso': f"ARMA_I({p},{d_value},{q})",
                'p': p,
                'd': d_value,
                'q': q,
                'ARMA_base': arma_config['nombre'],
                'Distribución': dist,
                'Varianza': var,
                'Modalidad': modalidad_str,
                'Valor_Observado': y_series[curr_idx]
            }
            
            # Evaluar TODOS los modelos
            for name, model in models.items():
                try:
                    if "Bootstrap" in name:
                        pred = model.fit_predict(h_to_model)
                    else:
                        pred = model.fit_predict(pd.DataFrame({'valor': h_to_model}))
                    
                    pred_array = np.asarray(pred).flatten()
                    
                    # Integrar predicciones si es necesario (IGUAL que Pipeline140)
                    if usar_diferenciacion:
                        # pred_array son incrementos ΔY_{t+1}
                        # Y_{t+1} = Y_t + ΔY_{t+1}
                        pred_array = y_series[curr_idx - 1] + pred_array
                    
                    # Calcular ECRPS
                    row[name] = ecrps(pred_array, true_samples)
                except Exception as e:
                    if self.verbose:
                        print(f"Error en {name}: {e}")
                    row[name] = np.nan
            
            results_rows.append(row)

        return results_rows

    def _run_scenario_wrapper(self, args):
        """Wrapper para procesamiento paralelo."""
        arma_cfg, d_val, dist, var, seed = args
        
        total_n = self.N_TRAIN_INITIAL + self.N_TEST_STEPS
        
        # Simular ARIMA manualmente (como Pipeline140)
        y_series, series_base, errors, simulator = self._simulate_arima_manual(
            arma_cfg, d_val, dist, var, seed, total_n
        )
        
        # Ejecutar ambas modalidades
        res_sin_diff = self._run_single_modalidad(
            arma_cfg, d_val, dist, var, seed,
            y_series, series_base, errors,
            self.N_TRAIN_INITIAL, False, simulator
        )
        
        res_con_diff = self._run_single_modalidad(
            arma_cfg, d_val, dist, var, seed + 1,
            y_series, series_base, errors,
            self.N_TRAIN_INITIAL, True, simulator
        )
        
        clear_all_sessions()
        return res_sin_diff + res_con_diff

    def run_all(self, excel_filename: str = "RESULTADOS_MULTID_ECRPS_CORREGIDO.xlsx", 
                batch_size: int = 10, n_jobs: int = 3):
        """
        Ejecuta todas las simulaciones con la misma interfaz que la versión original.
        """
        print("="*80)
        print("🚀 PIPELINE MULTI-D CORREGIDO: ARIMA_I(p,d,q) - TODOS LOS MODELOS")
        print("="*80)
        
        # Generar tareas
        tasks = []
        s_id = 0
        for d in self.D_VALUES:
            for cfg in self.ARMA_CONFIGS:
                for dist in self.DISTRIBUTIONS:
                    for var in self.VARIANCES:
                        tasks.append((cfg.copy(), d, dist, var, self.seed + s_id))
                        s_id += 1
        
        print(f"📊 Total de escenarios: {len(tasks)}")
        print(f"   - Valores de d: {self.D_VALUES}")
        print(f"   - ARMA configs: {len(self.ARMA_CONFIGS)}")
        print(f"   - Distribuciones: {len(self.DISTRIBUTIONS)}")
        print(f"   - Varianzas: {len(self.VARIANCES)}")
        print(f"   - Modalidades por escenario: 2 (SIN_DIFF, CON_DIFF)")
        print(f"   - Modelos: TODOS (9 modelos)")
        print(f"   - Simulador base: ARIMASimulation (consistente con Pipeline140)")
        print(f"   - Total filas esperadas: {len(tasks) * 2 * self.N_TEST_STEPS}")
        
        # Procesamiento por lotes
        all_results = []
        num_batches = (len(tasks) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(tasks))
            batch = tasks[start_idx:end_idx]
            
            print(f"📦 Procesando lote {i+1}/{num_batches}...")
            
            results = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(self._run_scenario_wrapper)(t) for t in batch
            )
            
            for r in results: 
                all_results.extend(r)
            
            # Guardar progreso
            pd.DataFrame(all_results).to_excel(excel_filename, index=False)
            print(f"   ✅ {len(all_results)} filas guardadas")
            
            clear_all_sessions()
            gc.collect()
        
        print(f"✅ Simulación completa: {excel_filename}")
        return pd.DataFrame(all_results)


    
# ============================================================================
# ¿La cantidad de datos afecta a la calidad de las densidades predictivas?
# ============================================================================
import numpy as np
import pandas as pd
import gc
from joblib import Parallel, delayed

class Pipeline140_TamanosCrecientes:
    """
    ✅ PIPELINE CORREGIDO - Tamaños Crecientes
    
    CORRECCIÓN APLICADA:
    - Ahora mantiene proporción FIJA 83%/17% en OPTIMIZACIÓN
    - Pero usa TODO el histórico (train+calib) para freeze_hyperparameters
    - Esto garantiza que tamaños diferentes tomen tiempos diferentes
    
    ESTRUCTURA:
    - Proporción fija: 83% train / 17% calib (SOLO para optimización)
    - 5 tamaños totales diferentes
    - 12 pasos de predicción (fijos)
    - 3 tipos de procesos: ARMA (7 configs), ARIMA (7 configs), SETAR (7 configs)
    - 5 distribuciones × 4 varianzas
    """
    
    N_TEST_STEPS = 12  # Siempre 12 pasos de predicción
    
    # 21 Configuraciones (7 ARMA + 7 ARIMA + 7 SETAR)
    ARMA_CONFIGS = [
        {'nombre': 'AR(1)', 'phi': [0.9], 'theta': []},
        {'nombre': 'AR(2)', 'phi': [0.5, -0.3], 'theta': []},
        {'nombre': 'MA(1)', 'phi': [], 'theta': [0.7]},
        {'nombre': 'MA(2)', 'phi': [], 'theta': [0.4, 0.2]},
        {'nombre': 'ARMA(1,1)', 'phi': [0.6], 'theta': [0.3]},
        {'nombre': 'ARMA(2,2)', 'phi': [0.4, -0.2], 'theta': [0.5, 0.1]},
        {'nombre': 'ARMA(2,1)', 'phi': [0.7, 0.2], 'theta': [0.5]}
    ]
    
    ARIMA_CONFIGS = [
        {'nombre': 'ARIMA(0,1,0)', 'phi': [], 'theta': []},
        {'nombre': 'ARIMA(1,1,0)', 'phi': [0.6], 'theta': []},
        {'nombre': 'ARIMA(2,1,0)', 'phi': [0.5, -0.2], 'theta': []},
        {'nombre': 'ARIMA(0,1,1)', 'phi': [], 'theta': [0.5]},
        {'nombre': 'ARIMA(0,1,2)', 'phi': [], 'theta': [0.4, 0.25]},
        {'nombre': 'ARIMA(1,1,1)', 'phi': [0.7], 'theta': [-0.3]},
        {'nombre': 'ARIMA(2,1,2)', 'phi': [0.6, 0.2], 'theta': [0.4, -0.1]}
    ]
    
    SETAR_CONFIGS = [
        {'nombre': 'SETAR-1', 'phi_regime1': [0.6], 'phi_regime2': [-0.5], 'threshold': 0.0, 'delay': 1},
        {'nombre': 'SETAR-2', 'phi_regime1': [0.7], 'phi_regime2': [-0.7], 'threshold': 0.0, 'delay': 2},
        {'nombre': 'SETAR-3', 'phi_regime1': [0.5, -0.2], 'phi_regime2': [-0.3, 0.1], 'threshold': 0.5, 'delay': 1},
        {'nombre': 'SETAR-4', 'phi_regime1': [0.8, -0.15], 'phi_regime2': [-0.6, 0.2], 'threshold': 1.0, 'delay': 2},
        {'nombre': 'SETAR-5', 'phi_regime1': [0.4, -0.1, 0.05], 'phi_regime2': [-0.3, 0.1, -0.05], 'threshold': 0.0, 'delay': 1},
        {'nombre': 'SETAR-6', 'phi_regime1': [0.5, -0.3, 0.1], 'phi_regime2': [-0.4, 0.2, -0.05], 'threshold': 0.5, 'delay': 2},
        {'nombre': 'SETAR-7', 'phi_regime1': [0.3, 0.1], 'phi_regime2': [-0.2, -0.1], 'threshold': 0.8, 'delay': 3}
    ]
    
    DISTRIBUTIONS = ['normal', 'uniform', 'exponential', 't-student', 'mixture']
    VARIANCES = [0.2, 0.5, 1.0, 3.0]
    
    # ✅ 5 Tamaños con proporción fija 83% / 17%
    SIZE_COMBINATIONS = [
        {'tag': 'N=120', 'n_total': 120, 'n_train': 100, 'n_calib': 20},
        {'tag': 'N=240', 'n_total': 240, 'n_train': 199, 'n_calib': 41},
        {'tag': 'N=360', 'n_total': 360, 'n_train': 299, 'n_calib': 61},
        {'tag': 'N=600', 'n_total': 600, 'n_train': 498, 'n_calib': 102},
        {'tag': 'N=1200', 'n_total': 1200, 'n_train': 996, 'n_calib': 204}
    ]

    def __init__(self, n_boot: int = 1000, seed: int = 42, verbose: bool = False, proceso_tipo: str = 'ARMA'):
        self.n_boot = n_boot
        self.seed = seed
        self.verbose = verbose
        self.proceso_tipo = proceso_tipo.upper()
        self.rng = np.random.default_rng(seed)

    def _setup_models(self, seed: int):
        """Configuración de modelos"""
        return {
            'Block Bootstrapping': CircularBlockBootstrapModel(n_boot=self.n_boot, random_state=seed),
            'Sieve Bootstrap': SieveBootstrapModel(n_boot=self.n_boot, random_state=seed),
            'LSPM': LSPM(random_state=seed),
            'LSPMW': LSPMW(rho=0.95, random_state=seed),
            'AREPD': AREPD(n_lags=5, rho=0.93, random_state=seed),
            'MCPS': MondrianCPSModel(n_lags=10, random_state=seed),
            'AV-MCPS': AdaptiveVolatilityMondrianCPS(n_lags=12, random_state=seed),
            'DeepAR': DeepARModel(hidden_size=20, n_lags=10, epochs=25, num_samples=self.n_boot, random_state=seed),
            'EnCQR-LSTM': EnCQR_LSTM_Model(n_lags=15, B=3, units=24, epochs=20, num_samples=self.n_boot, random_state=seed)
        }

    def _get_configs_for_process_type(self):
        """Obtiene las configuraciones según el tipo de proceso"""
        if self.proceso_tipo == 'ARMA':
            return self.ARMA_CONFIGS
        elif self.proceso_tipo == 'ARIMA':
            return self.ARIMA_CONFIGS
        elif self.proceso_tipo == 'SETAR':
            return self.SETAR_CONFIGS
        else:
            raise ValueError(f"Tipo de proceso desconocido: {self.proceso_tipo}")

    def _create_simulator(self, config: dict, dist: str, var: float, seed: int):
        """Crea simulador según tipo de proceso"""
        sigma = np.sqrt(var)
        
        if self.proceso_tipo == 'ARMA':
            return ARMASimulation(
                phi=config['phi'], 
                theta=config['theta'],
                noise_dist=dist, 
                sigma=sigma, 
                seed=seed
            )
        elif self.proceso_tipo == 'ARIMA':
            return ARIMASimulation(
                phi=config['phi'], 
                theta=config['theta'],
                noise_dist=dist, 
                sigma=sigma, 
                seed=seed
            )
        else:  # SETAR
            return SETARSimulation(
                phi_regime1=config['phi_regime1'],
                phi_regime2=config['phi_regime2'],
                threshold=config['threshold'],
                delay=config['delay'],
                noise_dist=dist,
                sigma=sigma,
                seed=seed
            )

    def run_single_scenario(self, config: dict, dist: str, var: float, 
                           n_train: int, n_calib: int, size_tag: str, 
                           scenario_seed: int) -> List[Dict]:
        """
        ✅ CORREGIDO: Ahora usa consistentemente los datos
        
        CAMBIOS:
        1. Optimización usa solo n_train para entrenar
        2. freeze_hyperparameters() usa TODO (train+calib)
        3. Esto hace que tamaños diferentes tomen tiempos diferentes
        """
        
        n_total = n_train + n_calib
        
        # 1. Simulación
        simulator = self._create_simulator(config, dist, var, scenario_seed)
        total_len = n_total + self.N_TEST_STEPS
        series, errors = simulator.simulate(n=total_len, burn_in=100)
        
        # ✅ CORRECCIÓN: Usar solo n_train para optimización
        train_data = series[:n_train]
        val_data = series[n_train:n_total]  # Solo n_calib datos
        
        if self.verbose:
            print(f"   📊 Train: {len(train_data)}, Calib: {len(val_data)}, Test steps: {self.N_TEST_STEPS}")
        
        # 2. Optimización de hiperparámetros (usa train_data y val_data)
        models = self._setup_models(scenario_seed)
        optimizer = TimeBalancedOptimizer(random_state=scenario_seed, verbose=self.verbose)
        best_params = optimizer.optimize_all_models(models, train_data, val_data)
        
        # 3. Aplicar mejores hiperparámetros y congelar con TODOS los datos
        train_val_full = series[:n_total]  # ✅ ESTO crece con el tamaño
        
        for name, model in models.items():
            if name in best_params:
                for k, v in best_params[name].items():
                    if hasattr(model, k): 
                        setattr(model, k, v)
            
            # ✅ CLAVE: freeze_hyperparameters() usa TODO el histórico
            # Esto hace que N=1200 tome más tiempo que N=120
            if hasattr(model, 'freeze_hyperparameters'):
                model.freeze_hyperparameters(train_val_full)

        # 4. Testing Rolling Window
        results_rows = []

        for t in range(self.N_TEST_STEPS):
            idx = n_total + t
            h_series = series[:idx]
            h_errors = errors[:idx]
            
            # Densidad teórica
            true_samples = simulator.get_true_next_step_samples(h_series, h_errors, n_samples=1000)
            
            # Fila de resultados
            row = {
                'Paso': t + 1,
                'Tipo_Proceso': self.proceso_tipo,
                'Proceso': config['nombre'],
                'Distribución': dist,
                'Varianza': var,
                'N_Train': n_train,
                'N_Calib': n_calib,
                'N_Total': n_total,  # ✅ Ahora calculado correctamente
                'Size': size_tag
            }
            
            # ✅ Usar fit_predict()
            for name, model in models.items():
                try:
                    if "Bootstrap" in name:
                        pred = model.fit_predict(h_series)
                    else:
                        pred = model.fit_predict(pd.DataFrame({'valor': h_series}))
                    
                    pred_array = np.asarray(pred).flatten()
                    row[name] = ecrps(pred_array, true_samples)
                    
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️ Error {name} en paso {t+1}: {e}")
                    row[name] = np.nan
            
            results_rows.append(row)

        clear_all_sessions()
        return results_rows

    def _run_scenario_wrapper(self, args: Tuple) -> List[Dict]:
        """Wrapper para paralelización"""
        return self.run_single_scenario(*args)

    def generate_all_scenarios(self) -> List[Tuple]:
        """
        ✅ Genera escenarios para UN tipo de proceso
        """
        scenarios = []
        configs = self._get_configs_for_process_type()
        
        # Debug info
        if self.verbose or True:
            print(f"\n🔍 Generando escenarios para {self.proceso_tipo}:")
            print(f"   • Configs: {len(configs)}")
            print(f"   • Tamaños: {len(self.SIZE_COMBINATIONS)}")
            print(f"   • Distribuciones: {len(self.DISTRIBUTIONS)}")
            print(f"   • Varianzas: {len(self.VARIANCES)}")
            esperados = len(configs) * len(self.SIZE_COMBINATIONS) * len(self.DISTRIBUTIONS) * len(self.VARIANCES)
            print(f"   • ESPERADOS: {esperados} escenarios")
            print(f"   • Filas esperadas: {esperados * self.N_TEST_STEPS}\n")
        
        s_id = 0
        for cfg in configs:
            for size in self.SIZE_COMBINATIONS:
                for dist in self.DISTRIBUTIONS:
                    for var in self.VARIANCES:
                        scenarios.append((
                            cfg.copy(),
                            dist,
                            var,
                            size['n_train'],
                            size['n_calib'],
                            size['tag'],
                            self.seed + s_id
                        ))
                        s_id += 1
        
        print(f"✅ Generados {len(scenarios)} escenarios\n")
        return scenarios

    def run_all(self, excel_filename: str = None, batch_size: int = 20, 
                max_workers: int = None, save_frequency: int = 3) -> pd.DataFrame:
        """
        Ejecuta todos los escenarios con paralelización
        """
        
        # Auto-detecta workers
        if max_workers is None:
            cpu_count = os.cpu_count() or 4
            max_workers = max(10, min(int(cpu_count * 0.75), cpu_count - 2))
        
        if excel_filename is None:
            excel_filename = f"RESULTADOS_TAMANOS_{self.proceso_tipo}.xlsx"
        
        tasks = self.generate_all_scenarios()
        num_batches = (len(tasks) + batch_size - 1) // batch_size
        
        print(f"\n{'='*60}")
        print(f"🚀 PIPELINE TAMAÑOS CRECIENTES - {self.proceso_tipo}")
        print(f"{'='*60}")
        print(f"📊 Total escenarios: {len(tasks)}")
        print(f"📦 Batches: {num_batches} (tamaño {batch_size})")
        print(f"👷 Workers: {max_workers} de {os.cpu_count()} cores")
        print(f"💾 Guardado cada {save_frequency} batches")
        print(f"{'='*60}\n")
        
        all_results = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(tasks))
            batch = tasks[start_idx:end_idx]
            
            print(f"🔄 Batch {i+1}/{num_batches}... ", end='', flush=True)
            
            batch_results = Parallel(n_jobs=max_workers, backend='loky', verbose=0)(
                delayed(self._run_scenario_wrapper)(t) for t in batch
            )
            
            for result_list in batch_results:
                all_results.extend(result_list)
            
            print(f"✅ {len(all_results)} filas")
            
            if (i + 1) % save_frequency == 0 or (i + 1) == num_batches:
                pd.DataFrame(all_results).to_excel(excel_filename, index=False)
                print(f"💾 Checkpoint: {excel_filename}")
            
            del batch_results, batch
            clear_all_sessions()
            gc.collect()
        
        df_final = pd.DataFrame(all_results)
        df_final.to_excel(excel_filename, index=False)
        
        print(f"\n🎉 Completado: {len(all_results)} filas → {excel_filename}\n")
        return df_final

class Pipeline_ARIMA_Fast:
    """
    ⚡ PIPELINE OPTIMIZADO - SOLO ARIMA - ULTRA RÁPIDO
    
    OPTIMIZACIONES APLICADAS:
    ✅ Solo procesos ARIMA (eliminado ARMA y SETAR)
    ✅ Modelos más rápidos priorizados
    ✅ Menos epochs en DeepAR y EnCQR
    ✅ Optimización de hiperparámetros más agresiva
    ✅ Cache de simulaciones
    ✅ Paralelización mejorada
    
    ESTRUCTURA:
    - Proporción fija: 83% train / 17% calib
    - 5 tamaños totales diferentes
    - 12 pasos de predicción
    - 7 configuraciones ARIMA
    - 5 distribuciones × 4 varianzas
    - Total: 700 escenarios (vs 2100 original)
    """
    
    N_TEST_STEPS = 12
    
    # Solo ARIMA configs
    ARIMA_CONFIGS = [
        {'nombre': 'ARIMA(0,1,0)', 'phi': [], 'theta': []},
        {'nombre': 'ARIMA(1,1,0)', 'phi': [0.6], 'theta': []},
        {'nombre': 'ARIMA(2,1,0)', 'phi': [0.5, -0.2], 'theta': []},
        {'nombre': 'ARIMA(0,1,1)', 'phi': [], 'theta': [0.5]},
        {'nombre': 'ARIMA(0,1,2)', 'phi': [], 'theta': [0.4, 0.25]},
        {'nombre': 'ARIMA(1,1,1)', 'phi': [0.7], 'theta': [-0.3]},
        {'nombre': 'ARIMA(2,1,2)', 'phi': [0.6, 0.2], 'theta': [0.4, -0.1]}
    ]
    
    DISTRIBUTIONS = ['normal', 'uniform', 'exponential', 't-student', 'mixture']
    VARIANCES = [0.2, 0.5, 1.0, 3.0]
    
    SIZE_COMBINATIONS = [
        {'tag': 'N=120', 'n_total': 120, 'n_train': 100, 'n_calib': 20},
        {'tag': 'N=240', 'n_total': 240, 'n_train': 199, 'n_calib': 41},
        {'tag': 'N=360', 'n_total': 360, 'n_train': 299, 'n_calib': 61},
        {'tag': 'N=600', 'n_total': 600, 'n_train': 498, 'n_calib': 102},
        {'tag': 'N=1200', 'n_total': 1200, 'n_train': 996, 'n_calib': 204}
    ]

    def __init__(self, n_boot: int = 500, seed: int = 42, verbose: bool = False):
        """
        Args:
            n_boot: Reducido a 500 (era 1000) para más velocidad
        """
        self.n_boot = n_boot
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        self._simulation_cache = {}  # ⚡ Cache de simulaciones

    def _setup_models(self, seed: int):
        """⚡ Modelos optimizados para velocidad"""
        return {
            # Modelos rápidos
            'Block Bootstrap': CircularBlockBootstrapModel(n_boot=self.n_boot, random_state=seed),
            'Sieve Bootstrap': SieveBootstrapModel(n_boot=self.n_boot, random_state=seed),
            'LSPM': LSPM(random_state=seed),
            'LSPMW': LSPMW(rho=0.95, random_state=seed),
            'AREPD': AREPD(n_lags=4, rho=0.93, random_state=seed),  # Reducido de 5 a 4 lags
            
            # Modelos medianos (optimizados)
            'MCPS': MondrianCPSModel(n_lags=8, random_state=seed),  # Reducido de 10 a 8
            'AV-MCPS': AdaptiveVolatilityMondrianCPS(n_lags=10, random_state=seed),  # Reducido de 12 a 10
            
            # Modelos lentos (muy optimizados)
            'DeepAR': DeepARModel(
                hidden_size=16,  # Reducido de 20 a 16
                n_lags=8,        # Reducido de 10 a 8
                epochs=15,       # Reducido de 25 a 15
                num_samples=self.n_boot, 
                random_state=seed
            ),
            'EnCQR-LSTM': EnCQR_LSTM_Model(
                n_lags=12,       # Reducido de 15 a 12
                B=2,             # Reducido de 3 a 2
                units=20,        # Reducido de 24 a 20
                epochs=12,       # Reducido de 20 a 12
                num_samples=self.n_boot, 
                random_state=seed
            )
        }

    def _get_cache_key(self, config: dict, dist: str, var: float, n_total: int, seed: int) -> str:
        """Genera key único para cache"""
        phi_str = ','.join(map(str, config['phi']))
        theta_str = ','.join(map(str, config['theta']))
        return f"ARIMA_{phi_str}_{theta_str}_{dist}_{var}_{n_total}_{seed}"

    def _create_simulator(self, config: dict, dist: str, var: float, seed: int):
        """Crea simulador ARIMA"""
        sigma = np.sqrt(var)
        return ARIMASimulation(
            phi=config['phi'], 
            theta=config['theta'],
            noise_dist=dist, 
            sigma=sigma, 
            seed=seed
        )

    def run_single_scenario(self, config: dict, dist: str, var: float, 
                           n_train: int, n_calib: int, size_tag: str, 
                           scenario_seed: int) -> List[Dict]:
        """⚡ Versión optimizada con cache"""
        
        n_total = n_train + n_calib
        cache_key = self._get_cache_key(config, dist, var, n_total, scenario_seed)
        
        # ⚡ Revisar cache
        if cache_key in self._simulation_cache:
            series, errors = self._simulation_cache[cache_key]
        else:
            simulator = self._create_simulator(config, dist, var, scenario_seed)
            total_len = n_total + self.N_TEST_STEPS
            series, errors = simulator.simulate(n=total_len, burn_in=100)
            self._simulation_cache[cache_key] = (series, errors)
        
        train_data = series[:n_train]
        val_data = series[n_train:n_total]
        
        if self.verbose:
            print(f"   📊 Train: {len(train_data)}, Calib: {len(val_data)}")
        
        # Optimización rápida (CORRECCIÓN: removido max_trials)
        models = self._setup_models(scenario_seed)
        optimizer = TimeBalancedOptimizer(
            random_state=scenario_seed, 
            verbose=self.verbose
        )
        best_params = optimizer.optimize_all_models(models, train_data, val_data)
        
        # Aplicar hiperparámetros
        train_val_full = series[:n_total]
        
        for name, model in models.items():
            if name in best_params:
                for k, v in best_params[name].items():
                    if hasattr(model, k): 
                        setattr(model, k, v)
            
            if hasattr(model, 'freeze_hyperparameters'):
                model.freeze_hyperparameters(train_val_full)

        # Testing rolling window
        results_rows = []
        simulator = self._create_simulator(config, dist, var, scenario_seed)

        for t in range(self.N_TEST_STEPS):
            idx = n_total + t
            h_series = series[:idx]
            h_errors = errors[:idx]
            
            # ⚡ Reducir samples teóricos para velocidad
            true_samples = simulator.get_true_next_step_samples(
                h_series, h_errors, n_samples=500  # Reducido de 1000
            )
            
            row = {
                'Paso': t + 1,
                'Proceso': config['nombre'],
                'Distribución': dist,
                'Varianza': var,
                'N_Train': n_train,
                'N_Calib': n_calib,
                'N_Total': n_total,
                'Size': size_tag
            }
            
            for name, model in models.items():
                try:
                    if "Bootstrap" in name:
                        pred = model.fit_predict(h_series)
                    else:
                        pred = model.fit_predict(pd.DataFrame({'valor': h_series}))
                    
                    pred_array = np.asarray(pred).flatten()
                    row[name] = ecrps(pred_array, true_samples)
                    
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️ Error {name}: {e}")
                    row[name] = np.nan
            
            results_rows.append(row)

        clear_all_sessions()
        return results_rows

    def _run_scenario_wrapper(self, args: Tuple) -> List[Dict]:
        """Wrapper para paralelización"""
        return self.run_single_scenario(*args)

    def generate_all_scenarios(self) -> List[Tuple]:
        """Genera escenarios ARIMA"""
        scenarios = []
        
        print(f"\n🔍 Generando escenarios ARIMA:")
        print(f"   • Configs: {len(self.ARIMA_CONFIGS)}")
        print(f"   • Tamaños: {len(self.SIZE_COMBINATIONS)}")
        print(f"   • Distribuciones: {len(self.DISTRIBUTIONS)}")
        print(f"   • Varianzas: {len(self.VARIANCES)}")
        esperados = len(self.ARIMA_CONFIGS) * len(self.SIZE_COMBINATIONS) * len(self.DISTRIBUTIONS) * len(self.VARIANCES)
        print(f"   • TOTAL: {esperados} escenarios (3x menos que versión completa)")
        print(f"   • Filas: {esperados * self.N_TEST_STEPS}\n")
        
        s_id = 0
        for cfg in self.ARIMA_CONFIGS:
            for size in self.SIZE_COMBINATIONS:
                for dist in self.DISTRIBUTIONS:
                    for var in self.VARIANCES:
                        scenarios.append((
                            cfg.copy(),
                            dist,
                            var,
                            size['n_train'],
                            size['n_calib'],
                            size['tag'],
                            self.seed + s_id
                        ))
                        s_id += 1
        
        print(f"✅ Generados {len(scenarios)} escenarios\n")
        return scenarios

    def run_all(self, excel_filename: str = "RESULTADOS_ARIMA_FAST.xlsx", 
                batch_size: int = 30, max_workers: int = None, 
                save_frequency: int = 2) -> pd.DataFrame:
        """
        ⚡ Ejecución ultra-rápida
        
        Cambios de velocidad:
        - batch_size aumentado: 20 → 30
        - save_frequency reducido: 3 → 2
        - max_workers más agresivo
        """
        
        if max_workers is None:
            cpu_count = os.cpu_count() or 4
            max_workers = max(12, min(int(cpu_count * 0.85), cpu_count - 1))  # Más agresivo
        
        tasks = self.generate_all_scenarios()
        num_batches = (len(tasks) + batch_size - 1) // batch_size
        
        print(f"\n{'='*60}")
        print(f"⚡ PIPELINE ARIMA ULTRA-RÁPIDO")
        print(f"{'='*60}")
        print(f"📊 Escenarios: {len(tasks)} (700 vs 2100 original = 67% menos)")
        print(f"📦 Batches: {num_batches} (tamaño {batch_size})")
        print(f"👷 Workers: {max_workers} de {os.cpu_count()} cores")
        print(f"🔥 n_boot: {self.n_boot} (500 vs 1000 = 50% menos)")
        print(f"⚡ Epochs reducidos: DeepAR 15, EnCQR 12")
        print(f"💾 Guardado cada {save_frequency} batches")
        print(f"{'='*60}\n")
        
        all_results = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(tasks))
            batch = tasks[start_idx:end_idx]
            
            print(f"🔄 Batch {i+1}/{num_batches}... ", end='', flush=True)
            
            batch_results = Parallel(n_jobs=max_workers, backend='loky', verbose=0)(
                delayed(self._run_scenario_wrapper)(t) for t in batch
            )
            
            for result_list in batch_results:
                all_results.extend(result_list)
            
            print(f"✅ {len(all_results)} filas")
            
            if (i + 1) % save_frequency == 0 or (i + 1) == num_batches:
                pd.DataFrame(all_results).to_excel(excel_filename, index=False)
                print(f"💾 Checkpoint: {excel_filename}")
            
            # ⚡ Limpieza agresiva
            del batch_results, batch
            if (i + 1) % 5 == 0:  # Limpiar cache cada 5 batches
                self._simulation_cache.clear()
            clear_all_sessions()
            gc.collect()
        
        df_final = pd.DataFrame(all_results)
        df_final.to_excel(excel_filename, index=False)
        
        print(f"\n🎉 Completado: {len(all_results)} filas → {excel_filename}")
        print(f"⚡ Velocidad estimada: 3-4x más rápido que versión original\n")
        return df_final

class Pipeline_SETAR_Fast:
    """
    ⚡ PIPELINE OPTIMIZADO - SOLO SETAR - ULTRA RÁPIDO
    
    OPTIMIZACIONES APLICADAS:
    ✅ Solo procesos SETAR (eliminado ARMA y ARIMA)
    ✅ Modelos más rápidos priorizados
    ✅ Menos epochs en DeepAR y EnCQR
    ✅ Optimización de hiperparámetros más agresiva
    ✅ Cache de simulaciones
    ✅ Paralelización mejorada
    
    ESTRUCTURA:
    - Proporción fija: 83% train / 17% calib
    - 5 tamaños totales diferentes
    - 12 pasos de predicción
    - 7 configuraciones SETAR
    - 5 distribuciones × 4 varianzas
    - Total: 700 escenarios (vs 2100 original)
    """
    
    N_TEST_STEPS = 12
    
    # Solo SETAR configs
    SETAR_CONFIGS = [
        {'nombre': 'SETAR(1,1)', 'p1': 1, 'p2': 1, 'threshold': 0.0},
        {'nombre': 'SETAR(1,2)', 'p1': 1, 'p2': 2, 'threshold': 0.0},
        {'nombre': 'SETAR(2,1)', 'p1': 2, 'p2': 1, 'threshold': 0.0},
        {'nombre': 'SETAR(2,2)', 'p1': 2, 'p2': 2, 'threshold': 0.0},
        {'nombre': 'SETAR(1,3)', 'p1': 1, 'p2': 3, 'threshold': 0.0},
        {'nombre': 'SETAR(3,1)', 'p1': 3, 'p2': 1, 'threshold': 0.0},
        {'nombre': 'SETAR(2,3)', 'p1': 2, 'p2': 3, 'threshold': 0.0}
    ]
    
    DISTRIBUTIONS = ['normal', 'uniform', 'exponential', 't-student', 'mixture']
    VARIANCES = [0.2, 0.5, 1.0, 3.0]
    
    SIZE_COMBINATIONS = [
        {'tag': 'N=120', 'n_total': 120, 'n_train': 100, 'n_calib': 20},
        {'tag': 'N=240', 'n_total': 240, 'n_train': 199, 'n_calib': 41},
        {'tag': 'N=360', 'n_total': 360, 'n_train': 299, 'n_calib': 61},
        {'tag': 'N=600', 'n_total': 600, 'n_train': 498, 'n_calib': 102},
        {'tag': 'N=1200', 'n_total': 1200, 'n_train': 996, 'n_calib': 204}
    ]

    def __init__(self, n_boot: int = 500, seed: int = 42, verbose: bool = False):
        """
        Args:
            n_boot: Reducido a 500 (era 1000) para más velocidad
        """
        self.n_boot = n_boot
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        self._simulation_cache = {}  # ⚡ Cache de simulaciones

    def _setup_models(self, seed: int):
        """⚡ Modelos optimizados para velocidad"""
        return {
            # Modelos rápidos
            'Block Bootstrap': CircularBlockBootstrapModel(n_boot=self.n_boot, random_state=seed),
            'Sieve Bootstrap': SieveBootstrapModel(n_boot=self.n_boot, random_state=seed),
            'LSPM': LSPM(random_state=seed),
            'LSPMW': LSPMW(rho=0.95, random_state=seed),
            'AREPD': AREPD(n_lags=4, rho=0.93, random_state=seed),  # Reducido de 5 a 4 lags
            
            # Modelos medianos (optimizados)
            'MCPS': MondrianCPSModel(n_lags=8, random_state=seed),  # Reducido de 10 a 8
            'AV-MCPS': AdaptiveVolatilityMondrianCPS(n_lags=10, random_state=seed),  # Reducido de 12 a 10
            
            # Modelos lentos (muy optimizados)
            'DeepAR': DeepARModel(
                hidden_size=16,  # Reducido de 20 a 16
                n_lags=8,        # Reducido de 10 a 8
                epochs=15,       # Reducido de 25 a 15
                num_samples=self.n_boot, 
                random_state=seed
            ),
            'EnCQR-LSTM': EnCQR_LSTM_Model(
                n_lags=12,       # Reducido de 15 a 12
                B=2,             # Reducido de 3 a 2
                units=20,        # Reducido de 24 a 20
                epochs=12,       # Reducido de 20 a 12
                num_samples=self.n_boot, 
                random_state=seed
            )
        }

    def _get_cache_key(self, config: dict, dist: str, var: float, n_total: int, seed: int) -> str:
        """Genera key único para cache"""
        return f"SETAR_{config['p1']}_{config['p2']}_{config['threshold']}_{dist}_{var}_{n_total}_{seed}"

    def _create_simulator(self, config: dict, dist: str, var: float, seed: int):
        """Crea simulador SETAR"""
        sigma = np.sqrt(var)
        
        # Generar coeficientes autorregresivos simples para cada régimen
        rng = np.random.default_rng(seed)
        
        # Régimen 1: coeficientes que suman < 1 para estabilidad
        phi1 = []
        for _ in range(config['p1']):
            coef = rng.uniform(0.3, 0.7)
            phi1.append(coef)
        # Normalizar para estabilidad
        phi1 = [c / sum(phi1) * 0.8 for c in phi1]
        
        # Régimen 2: coeficientes diferentes
        phi2 = []
        for _ in range(config['p2']):
            coef = rng.uniform(0.2, 0.6)
            phi2.append(coef)
        # Normalizar para estabilidad
        phi2 = [c / sum(phi2) * 0.75 for c in phi2]
        
        return SETARSimulation(
            phi1=phi1,
            phi2=phi2,
            threshold=config['threshold'],
            noise_dist=dist,
            sigma=sigma,
            seed=seed
        )

    def run_single_scenario(self, config: dict, dist: str, var: float, 
                           n_train: int, n_calib: int, size_tag: str, 
                           scenario_seed: int) -> List[Dict]:
        """⚡ Versión optimizada con cache"""
        
        n_total = n_train + n_calib
        cache_key = self._get_cache_key(config, dist, var, n_total, scenario_seed)
        
        # ⚡ Revisar cache
        if cache_key in self._simulation_cache:
            series, errors = self._simulation_cache[cache_key]
        else:
            simulator = self._create_simulator(config, dist, var, scenario_seed)
            total_len = n_total + self.N_TEST_STEPS
            series, errors = simulator.simulate(n=total_len, burn_in=100)
            self._simulation_cache[cache_key] = (series, errors)
        
        train_data = series[:n_train]
        val_data = series[n_train:n_total]
        
        if self.verbose:
            print(f"   📊 Train: {len(train_data)}, Calib: {len(val_data)}")
        
        # Optimización rápida (sin max_trials)
        models = self._setup_models(scenario_seed)
        optimizer = TimeBalancedOptimizer(
            random_state=scenario_seed, 
            verbose=self.verbose
        )
        best_params = optimizer.optimize_all_models(models, train_data, val_data)
        
        # Aplicar hiperparámetros
        train_val_full = series[:n_total]
        
        for name, model in models.items():
            if name in best_params:
                for k, v in best_params[name].items():
                    if hasattr(model, k): 
                        setattr(model, k, v)
            
            if hasattr(model, 'freeze_hyperparameters'):
                model.freeze_hyperparameters(train_val_full)

        # Testing rolling window
        results_rows = []
        simulator = self._create_simulator(config, dist, var, scenario_seed)

        for t in range(self.N_TEST_STEPS):
            idx = n_total + t
            h_series = series[:idx]
            h_errors = errors[:idx]
            
            # ⚡ Reducir samples teóricos para velocidad
            true_samples = simulator.get_true_next_step_samples(
                h_series, h_errors, n_samples=500  # Reducido de 1000
            )
            
            row = {
                'Paso': t + 1,
                'Proceso': config['nombre'],
                'Distribución': dist,
                'Varianza': var,
                'N_Train': n_train,
                'N_Calib': n_calib,
                'N_Total': n_total,
                'Size': size_tag
            }
            
            for name, model in models.items():
                try:
                    if "Bootstrap" in name:
                        pred = model.fit_predict(h_series)
                    else:
                        pred = model.fit_predict(pd.DataFrame({'valor': h_series}))
                    
                    pred_array = np.asarray(pred).flatten()
                    row[name] = ecrps(pred_array, true_samples)
                    
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️ Error {name}: {e}")
                    row[name] = np.nan
            
            results_rows.append(row)

        clear_all_sessions()
        return results_rows

    def _run_scenario_wrapper(self, args: Tuple) -> List[Dict]:
        """Wrapper para paralelización"""
        return self.run_single_scenario(*args)

    def generate_all_scenarios(self) -> List[Tuple]:
        """Genera escenarios SETAR"""
        scenarios = []
        
        print(f"\n🔍 Generando escenarios SETAR:")
        print(f"   • Configs: {len(self.SETAR_CONFIGS)}")
        print(f"   • Tamaños: {len(self.SIZE_COMBINATIONS)}")
        print(f"   • Distribuciones: {len(self.DISTRIBUTIONS)}")
        print(f"   • Varianzas: {len(self.VARIANCES)}")
        esperados = len(self.SETAR_CONFIGS) * len(self.SIZE_COMBINATIONS) * len(self.DISTRIBUTIONS) * len(self.VARIANCES)
        print(f"   • TOTAL: {esperados} escenarios (3x menos que versión completa)")
        print(f"   • Filas: {esperados * self.N_TEST_STEPS}\n")
        
        s_id = 0
        for cfg in self.SETAR_CONFIGS:
            for size in self.SIZE_COMBINATIONS:
                for dist in self.DISTRIBUTIONS:
                    for var in self.VARIANCES:
                        scenarios.append((
                            cfg.copy(),
                            dist,
                            var,
                            size['n_train'],
                            size['n_calib'],
                            size['tag'],
                            self.seed + s_id
                        ))
                        s_id += 1
        
        print(f"✅ Generados {len(scenarios)} escenarios\n")
        return scenarios

    def run_all(self, excel_filename: str = "RESULTADOS_SETAR_FAST.xlsx", 
                batch_size: int = 30, max_workers: int = None, 
                save_frequency: int = 2) -> pd.DataFrame:
        """
        ⚡ Ejecución ultra-rápida
        
        Cambios de velocidad:
        - batch_size aumentado: 20 → 30
        - save_frequency reducido: 3 → 2
        - max_workers más agresivo
        """
        
        if max_workers is None:
            cpu_count = os.cpu_count() or 4
            max_workers = max(12, min(int(cpu_count * 0.85), cpu_count - 1))  # Más agresivo
        
        tasks = self.generate_all_scenarios()
        num_batches = (len(tasks) + batch_size - 1) // batch_size
        
        print(f"\n{'='*60}")
        print(f"⚡ PIPELINE SETAR ULTRA-RÁPIDO")
        print(f"{'='*60}")
        print(f"📊 Escenarios: {len(tasks)} (700 vs 2100 original = 67% menos)")
        print(f"📦 Batches: {num_batches} (tamaño {batch_size})")
        print(f"👷 Workers: {max_workers} de {os.cpu_count()} cores")
        print(f"🔥 n_boot: {self.n_boot} (500 vs 1000 = 50% menos)")
        print(f"⚡ Epochs reducidos: DeepAR 15, EnCQR 12")
        print(f"💾 Guardado cada {save_frequency} batches")
        print(f"{'='*60}\n")
        
        all_results = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(tasks))
            batch = tasks[start_idx:end_idx]
            
            print(f"🔄 Batch {i+1}/{num_batches}... ", end='', flush=True)
            
            batch_results = Parallel(n_jobs=max_workers, backend='loky', verbose=0)(
                delayed(self._run_scenario_wrapper)(t) for t in batch
            )
            
            for result_list in batch_results:
                all_results.extend(result_list)
            
            print(f"✅ {len(all_results)} filas")
            
            if (i + 1) % save_frequency == 0 or (i + 1) == num_batches:
                pd.DataFrame(all_results).to_excel(excel_filename, index=False)
                print(f"💾 Checkpoint: {excel_filename}")
            
            # ⚡ Limpieza agresiva
            del batch_results, batch
            if (i + 1) % 5 == 0:  # Limpiar cache cada 5 batches
                self._simulation_cache.clear()
            clear_all_sessions()
            gc.collect()
        
        df_final = pd.DataFrame(all_results)
        df_final.to_excel(excel_filename, index=False)
        
        print(f"\n🎉 Completado: {len(all_results)} filas → {excel_filename}")
        print(f"⚡ Velocidad estimada: 3-4x más rápido que versión original\n")
        return df_final



# ============================================================================
# ¿La proporción de datos afecta a la calidad de las densidades predictivas?
# ============================================================================
import numpy as np
import pandas as pd
import gc
import os
import psutil
from joblib import Parallel, delayed
from typing import Dict, List, Tuple


import numpy as np
import pandas as pd
import warnings
import gc
import os
import time
from joblib import Parallel, delayed
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

from simulacion import ARMASimulation, ARIMASimulation, SETARSimulation
from modelos import (CircularBlockBootstrapModel, SieveBootstrapModel, LSPM, LSPMW, 
                     DeepARModel, AREPD, MondrianCPSModel, AdaptiveVolatilityMondrianCPS,
                     EnCQR_LSTM_Model, TimeBalancedOptimizer)
from metricas import ecrps

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


class Pipeline240_ProporcionesVariables:
    """
    ✅ PIPELINE ULTRA-OPTIMIZADO - Proporciones Variables (N=240)
    
    GARANTÍAS:
    - 25,200 filas exactas (2,100 escenarios × 12 pasos)
    - Máxima velocidad con paralelización optimizada
    - Gestión eficiente de memoria
    """
    
    N_TOTAL = 240
    N_TEST_STEPS = 12
    
    SIZE_COMBINATIONS = [
        {'prop_tag': '10%', 'n_train': 216, 'n_calib': 24, 'prop_val': 0.10},
        {'prop_tag': '20%', 'n_train': 192, 'n_calib': 48, 'prop_val': 0.20},
        {'prop_tag': '30%', 'n_train': 168, 'n_calib': 72, 'prop_val': 0.30},
        {'prop_tag': '40%', 'n_train': 144, 'n_calib': 96, 'prop_val': 0.40},
        {'prop_tag': '50%', 'n_train': 120, 'n_calib': 120, 'prop_val': 0.50}
    ]
    
    CONFIGS = {
        'ARMA': [
            {'nombre': 'AR(1)', 'phi': [0.9], 'theta': []},
            {'nombre': 'AR(2)', 'phi': [0.5, -0.3], 'theta': []},
            {'nombre': 'MA(1)', 'phi': [], 'theta': [0.7]},
            {'nombre': 'MA(2)', 'phi': [], 'theta': [0.4, 0.2]},
            {'nombre': 'ARMA(1,1)', 'phi': [0.6], 'theta': [0.3]},
            {'nombre': 'ARMA(2,2)', 'phi': [0.4, -0.2], 'theta': [0.5, 0.1]},
            {'nombre': 'ARMA(2,1)', 'phi': [0.7, 0.2], 'theta': [0.5]}
        ],
        'ARIMA': [
            {'nombre': 'ARIMA(0,1,0)', 'phi': [], 'theta': []},
            {'nombre': 'ARIMA(1,1,0)', 'phi': [0.6], 'theta': []},
            {'nombre': 'ARIMA(2,1,0)', 'phi': [0.5, -0.2], 'theta': []},
            {'nombre': 'ARIMA(0,1,1)', 'phi': [], 'theta': [0.5]},
            {'nombre': 'ARIMA(0,1,2)', 'phi': [], 'theta': [0.4, 0.25]},
            {'nombre': 'ARIMA(1,1,1)', 'phi': [0.7], 'theta': [-0.3]},
            {'nombre': 'ARIMA(2,1,2)', 'phi': [0.6, 0.2], 'theta': [0.4, -0.1]}
        ],
        'SETAR': [
            {'nombre': 'SETAR-1', 'phi_regime1': [0.6], 'phi_regime2': [-0.5], 'threshold': 0.0, 'delay': 1},
            {'nombre': 'SETAR-2', 'phi_regime1': [0.7], 'phi_regime2': [-0.7], 'threshold': 0.0, 'delay': 2},
            {'nombre': 'SETAR-3', 'phi_regime1': [0.5, -0.2], 'phi_regime2': [-0.3, 0.1], 'threshold': 0.5, 'delay': 1},
            {'nombre': 'SETAR-4', 'phi_regime1': [0.8, -0.15], 'phi_regime2': [-0.6, 0.2], 'threshold': 1.0, 'delay': 2},
            {'nombre': 'SETAR-5', 'phi_regime1': [0.4, -0.1, 0.05], 'phi_regime2': [-0.3, 0.1, -0.05], 'threshold': 0.0, 'delay': 1},
            {'nombre': 'SETAR-6', 'phi_regime1': [0.5, -0.3, 0.1], 'phi_regime2': [-0.4, 0.2, -0.05], 'threshold': 0.5, 'delay': 2},
            {'nombre': 'SETAR-7', 'phi_regime1': [0.3, 0.1], 'phi_regime2': [-0.2, -0.1], 'threshold': 0.8, 'delay': 3}
        ]
    }
    
    DISTRIBUTIONS = ['normal', 'uniform', 'exponential', 't-student', 'mixture']
    VARIANCES = [0.2, 0.5, 1.0, 3.0]

    def __init__(self, n_boot: int = 1000, seed: int = 42, verbose: bool = False, proceso_tipo: str = 'ARMA'):
        self.n_boot = n_boot
        self.seed = seed
        self.verbose = verbose
        self.proceso_tipo = proceso_tipo.upper()
        self.rng = np.random.default_rng(seed)
        
        # ✅ Validación de configuración
        if self.proceso_tipo not in self.CONFIGS:
            raise ValueError(f"proceso_tipo debe ser 'ARMA', 'ARIMA' o 'SETAR', no '{proceso_tipo}'")

    def _setup_models(self, seed: int):
        """Configuración de modelos con parámetros optimizados"""
        return {
            'Block Bootstrapping': CircularBlockBootstrapModel(n_boot=self.n_boot, random_state=seed),
            'Sieve Bootstrap': SieveBootstrapModel(n_boot=self.n_boot, random_state=seed),
            'LSPM': LSPM(random_state=seed),
            'LSPMW': LSPMW(rho=0.95, random_state=seed),
            'AREPD': AREPD(n_lags=5, rho=0.93, random_state=seed),
            'MCPS': MondrianCPSModel(n_lags=10, random_state=seed),
            'AV-MCPS': AdaptiveVolatilityMondrianCPS(n_lags=12, random_state=seed),
            'DeepAR': DeepARModel(hidden_size=20, n_lags=10, epochs=25, num_samples=self.n_boot, random_state=seed),
            'EnCQR-LSTM': EnCQR_LSTM_Model(n_lags=15, B=3, units=24, epochs=20, num_samples=self.n_boot, random_state=seed)
        }

    def _create_simulator(self, config: dict, dist: str, var: float, seed: int):
        """Crea simulador según tipo de proceso"""
        sigma = np.sqrt(var)
        
        if self.proceso_tipo == 'ARMA':
            return ARMASimulation(
                phi=config['phi'], 
                theta=config['theta'],
                noise_dist=dist, 
                sigma=sigma, 
                seed=seed
            )
        elif self.proceso_tipo == 'ARIMA':
            return ARIMASimulation(
                phi=config['phi'], 
                theta=config['theta'],
                noise_dist=dist, 
                sigma=sigma, 
                seed=seed
            )
        else:  # SETAR
            return SETARSimulation(
                phi_regime1=config['phi_regime1'],
                phi_regime2=config['phi_regime2'],
                threshold=config['threshold'],
                delay=config['delay'],
                noise_dist=dist,
                sigma=sigma,
                seed=seed
            )

    def run_single_scenario(self, config: dict, dist: str, var: float, 
                           n_train: int, n_calib: int, prop_tag: str, 
                           scenario_seed: int) -> List[Dict]:
        """
        ✅ Ejecuta un escenario completo - GARANTIZA 12 filas por escenario
        """
        try:
            # 1. Simulación
            simulator = self._create_simulator(config, dist, var, scenario_seed)
            total_len = self.N_TOTAL + self.N_TEST_STEPS
            series, errors = simulator.simulate(n=total_len, burn_in=100)
            
            train_data = series[:n_train]
            val_data = series[n_train:self.N_TOTAL]
            
            # 2. Optimización de hiperparámetros
            models = self._setup_models(scenario_seed)
            optimizer = TimeBalancedOptimizer(random_state=scenario_seed, verbose=False)
            best_params = optimizer.optimize_all_models(models, train_data, val_data)
            
            # 3. Aplicar mejores hiperparámetros
            train_val_full = series[:self.N_TOTAL]
            for name, model in models.items():
                if name in best_params:
                    for k, v in best_params[name].items():
                        if hasattr(model, k): 
                            setattr(model, k, v)
                
                if hasattr(model, 'freeze_hyperparameters'):
                    model.freeze_hyperparameters(train_val_full)

            # 4. Testing Rolling Window - GARANTIZA 12 filas
            results_rows = []

            for t in range(self.N_TEST_STEPS):
                idx = self.N_TOTAL + t
                h_series = series[:idx]
                h_errors = errors[:idx]
                
                # Densidad teórica
                true_samples = simulator.get_true_next_step_samples(h_series, h_errors, n_samples=1000)
                
                # Fila base
                row = {
                    'Paso': t + 1,
                    'Tipo_Proceso': self.proceso_tipo,
                    'Proceso': config['nombre'],
                    'Distribución': dist,
                    'Varianza': var,
                    'N_Train': n_train,
                    'N_Calib': n_calib,
                    'Prop_Calib': prop_tag
                }
                
                # Evaluar modelos
                for name, model in models.items():
                    try:
                        if "Bootstrap" in name:
                            pred = model.fit_predict(h_series)
                        else:
                            pred = model.fit_predict(pd.DataFrame({'valor': h_series}))
                        
                        pred_array = np.asarray(pred).flatten()
                        row[name] = ecrps(pred_array, true_samples)
                        
                    except Exception as e:
                        if self.verbose:
                            print(f"⚠️ Error {name} en paso {t+1}: {e}")
                        row[name] = np.nan
                
                results_rows.append(row)

            # ✅ VALIDACIÓN: Garantizar 12 filas
            if len(results_rows) != self.N_TEST_STEPS:
                raise ValueError(f"Error: se generaron {len(results_rows)} filas en lugar de {self.N_TEST_STEPS}")
            
            clear_all_sessions()
            return results_rows
            
        except Exception as e:
            print(f"❌ ERROR en escenario {config['nombre']}-{dist}-{var}: {e}")
            # Devolver 12 filas con NaN en caso de error total
            return [{
                'Paso': t + 1,
                'Tipo_Proceso': self.proceso_tipo,
                'Proceso': config['nombre'],
                'Distribución': dist,
                'Varianza': var,
                'N_Train': n_train,
                'N_Calib': n_calib,
                'Prop_Calib': prop_tag,
                **{modelo: np.nan for modelo in ['Block Bootstrapping', 'Sieve Bootstrap', 'LSPM', 
                                                   'LSPMW', 'AREPD', 'MCPS', 'AV-MCPS', 'DeepAR', 'EnCQR-LSTM']}
            } for t in range(self.N_TEST_STEPS)]

    def _run_scenario_wrapper(self, args: Tuple) -> List[Dict]:
        """Wrapper para paralelización"""
        return self.run_single_scenario(*args)

    def generate_all_scenarios(self) -> List[Tuple]:
        """
        ✅ Genera exactamente 2,100 escenarios
        """
        scenarios = []
        configs = self.CONFIGS[self.proceso_tipo]
        
        s_id = 0
        for cfg in configs:
            for size in self.SIZE_COMBINATIONS:
                for dist in self.DISTRIBUTIONS:
                    for var in self.VARIANCES:
                        scenarios.append((
                            cfg.copy(),
                            dist,
                            var,
                            size['n_train'],
                            size['n_calib'],
                            size['prop_tag'],
                            self.seed + s_id
                        ))
                        s_id += 1
        
        # ✅ VALIDACIÓN CRÍTICA
        expected = len(configs) * len(self.SIZE_COMBINATIONS) * len(self.DISTRIBUTIONS) * len(self.VARIANCES)
        assert len(scenarios) == expected, f"Error: {len(scenarios)} escenarios != {expected} esperados"
        
        print(f"\n{'='*60}")
        print(f"📊 CONFIGURACIÓN VALIDADA:")
        print(f"   • Configs {self.proceso_tipo}: {len(configs)}")
        print(f"   • Proporciones: {len(self.SIZE_COMBINATIONS)}")
        print(f"   • Distribuciones: {len(self.DISTRIBUTIONS)}")
        print(f"   • Varianzas: {len(self.VARIANCES)}")
        print(f"   • Escenarios totales: {len(scenarios)}")
        print(f"   • Filas esperadas: {len(scenarios) * self.N_TEST_STEPS} (25,200)")
        print(f"{'='*60}\n")
        
        return scenarios

    def run_all(self, excel_filename: str = None, batch_size: int = 30, 
                max_workers: int = None, save_frequency: int = 2) -> pd.DataFrame:
        """
        ✅ Ejecuta todos los escenarios - GARANTIZA 25,200 filas
        
        OPTIMIZACIONES:
        - batch_size=30: Balance óptimo velocidad/memoria
        - save_frequency=2: Guardado cada 60 escenarios (720 filas)
        - max_workers auto-ajustado al 80% de cores
        """
        
        # ✅ Auto-detecta workers (80% de cores disponibles)
        if max_workers is None:
            cpu_count = os.cpu_count() or 4
            max_workers = max(8, int(cpu_count * 0.8))
        
        if excel_filename is None:
            excel_filename = f"RESULTADOS_PROPORCIONES_240_{self.proceso_tipo}.xlsx"
        
        tasks = self.generate_all_scenarios()
        num_batches = (len(tasks) + batch_size - 1) // batch_size
        
        print(f"🚀 INICIANDO PIPELINE - {self.proceso_tipo}")
        print(f"   📦 Batches: {num_batches} (tamaño {batch_size})")
        print(f"   👷 Workers: {max_workers}/{os.cpu_count()} cores")
        print(f"   💾 Guardado cada {save_frequency} batches ({batch_size * save_frequency * 12} filas)\n")
        
        all_results = []
        checkpoint_counter = 0
        total_filas = 0
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(tasks))
            batch = tasks[start_idx:end_idx]
            
            print(f"🔄 Batch {i+1}/{num_batches} ({len(batch)} escenarios)... ", end='', flush=True)
            
            # ✅ Procesamiento paralelo optimizado
            batch_results = Parallel(n_jobs=max_workers, backend='loky', verbose=0)(
                delayed(self._run_scenario_wrapper)(t) for t in batch
            )
            
            # Acumular resultados
            for result_list in batch_results:
                all_results.extend(result_list)
            
            total_filas = len(all_results)
            print(f"✅ {total_filas} filas")
            
            # ✅ Guardado incremental con validación
            if (i + 1) % save_frequency == 0 or (i + 1) == num_batches:
                
                if checkpoint_counter == 0:
                    df_checkpoint = pd.DataFrame(all_results)
                    df_checkpoint.to_excel(excel_filename, index=False)
                    rows_saved = len(df_checkpoint)
                    del df_checkpoint
                else:
                    df_new = pd.DataFrame(all_results)
                    df_prev = pd.read_excel(excel_filename)
                    df_combined = pd.concat([df_prev, df_new], ignore_index=True)
                    df_combined.to_excel(excel_filename, index=False)
                    rows_saved = len(df_combined)
                    del df_prev, df_new, df_combined
                
                print(f"   💾 Checkpoint {checkpoint_counter + 1}: {rows_saved} filas guardadas")
                
                all_results.clear()
                checkpoint_counter += 1
                gc.collect()
            
            # Limpieza
            del batch_results, batch
            clear_all_sessions()
            gc.collect()
        
        # ✅ VALIDACIÓN FINAL CRÍTICA
        df_final = pd.read_excel(excel_filename)
        expected_rows = len(tasks) * self.N_TEST_STEPS
        
        print(f"\n{'='*60}")
        print(f"🎉 PIPELINE COMPLETADO")
        print(f"   📊 Filas obtenidas: {len(df_final)}")
        print(f"   ✅ Filas esperadas: {expected_rows}")
        
        if len(df_final) != expected_rows:
            print(f"   ⚠️  WARNING: Diferencia de {expected_rows - len(df_final)} filas")
        else:
            print(f"   ✅ VALIDACIÓN EXITOSA: 25,200 filas")
        
        print(f"   📁 Archivo: {excel_filename}")
        print(f"{'='*60}\n")
        
        all_results.clear()
        del all_results
        gc.collect()
        
        return df_final


class Pipeline_SETAR_Fast:
    """
    ⚡ PIPELINE OPTIMIZADO - SOLO SETAR - ULTRA RÁPIDO
    
    OPTIMIZACIONES APLICADAS:
    ✅ Solo procesos SETAR (eliminado ARMA y ARIMA)
    ✅ Modelos más rápidos priorizados
    ✅ Menos epochs en DeepAR y EnCQR
    ✅ Optimización de hiperparámetros más agresiva
    ✅ Cache de simulaciones
    ✅ Paralelización mejorada
    
    ESTRUCTURA:
    - Proporción fija: 83% train / 17% calib
    - 5 tamaños totales diferentes
    - 12 pasos de predicción
    - 7 configuraciones SETAR
    - 5 distribuciones × 4 varianzas
    - Total: 700 escenarios (vs 2100 original)
    """
    
    N_TEST_STEPS = 12
    
    # Solo SETAR configs
    SETAR_CONFIGS = [
        {'nombre': 'SETAR(1,1)', 'p1': 1, 'p2': 1, 'threshold': 0.0},
        {'nombre': 'SETAR(1,2)', 'p1': 1, 'p2': 2, 'threshold': 0.0},
        {'nombre': 'SETAR(2,1)', 'p1': 2, 'p2': 1, 'threshold': 0.0},
        {'nombre': 'SETAR(2,2)', 'p1': 2, 'p2': 2, 'threshold': 0.0},
        {'nombre': 'SETAR(1,3)', 'p1': 1, 'p2': 3, 'threshold': 0.0},
        {'nombre': 'SETAR(3,1)', 'p1': 3, 'p2': 1, 'threshold': 0.0},
        {'nombre': 'SETAR(2,3)', 'p1': 2, 'p2': 3, 'threshold': 0.0}
    ]
    
    DISTRIBUTIONS = ['normal', 'uniform', 'exponential', 't-student', 'mixture']
    VARIANCES = [0.2, 0.5, 1.0, 3.0]
    
    SIZE_COMBINATIONS = [
        {'tag': 'N=120', 'n_total': 120, 'n_train': 100, 'n_calib': 20},
        {'tag': 'N=240', 'n_total': 240, 'n_train': 199, 'n_calib': 41},
        {'tag': 'N=360', 'n_total': 360, 'n_train': 299, 'n_calib': 61},
        {'tag': 'N=600', 'n_total': 600, 'n_train': 498, 'n_calib': 102},
        {'tag': 'N=1200', 'n_total': 1200, 'n_train': 996, 'n_calib': 204}
    ]

    def __init__(self, n_boot: int = 500, seed: int = 42, verbose: bool = False):
        """
        Args:
            n_boot: Reducido a 500 (era 1000) para más velocidad
        """
        self.n_boot = n_boot
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        self._simulation_cache = {}  # ⚡ Cache de simulaciones

    def _setup_models(self, seed: int):
        """⚡ Modelos optimizados para velocidad"""
        return {
            # Modelos rápidos
            'Block Bootstrap': CircularBlockBootstrapModel(n_boot=self.n_boot, random_state=seed),
            'Sieve Bootstrap': SieveBootstrapModel(n_boot=self.n_boot, random_state=seed),
            'LSPM': LSPM(random_state=seed),
            'LSPMW': LSPMW(rho=0.95, random_state=seed),
            'AREPD': AREPD(n_lags=4, rho=0.93, random_state=seed),  # Reducido de 5 a 4 lags
            
            # Modelos medianos (optimizados)
            'MCPS': MondrianCPSModel(n_lags=8, random_state=seed),  # Reducido de 10 a 8
            'AV-MCPS': AdaptiveVolatilityMondrianCPS(n_lags=10, random_state=seed),  # Reducido de 12 a 10
            
            # Modelos lentos (muy optimizados)
            'DeepAR': DeepARModel(
                hidden_size=16,  # Reducido de 20 a 16
                n_lags=8,        # Reducido de 10 a 8
                epochs=15,       # Reducido de 25 a 15
                num_samples=self.n_boot, 
                random_state=seed
            ),
            'EnCQR-LSTM': EnCQR_LSTM_Model(
                n_lags=12,       # Reducido de 15 a 12
                B=2,             # Reducido de 3 a 2
                units=20,        # Reducido de 24 a 20
                epochs=12,       # Reducido de 20 a 12
                num_samples=self.n_boot, 
                random_state=seed
            )
        }

    def _get_cache_key(self, config: dict, dist: str, var: float, n_total: int, seed: int) -> str:
        """Genera key único para cache"""
        return f"SETAR_{config['p1']}_{config['p2']}_{config['threshold']}_{dist}_{var}_{n_total}_{seed}"

    def _create_simulator(self, config: dict, dist: str, var: float, seed: int):
        """Crea simulador SETAR"""
        sigma = np.sqrt(var)
        
        # Generar coeficientes autorregresivos simples para cada régimen
        rng = np.random.default_rng(seed)
        
        # Régimen 1: coeficientes que suman < 1 para estabilidad
        phi_regime1 = []
        for _ in range(config['p1']):
            coef = rng.uniform(0.3, 0.7)
            phi_regime1.append(coef)
        # Normalizar para estabilidad
        phi_regime1 = [c / sum(phi_regime1) * 0.8 for c in phi_regime1]
        
        # Régimen 2: coeficientes diferentes
        phi_regime2 = []
        for _ in range(config['p2']):
            coef = rng.uniform(0.2, 0.6)
            phi_regime2.append(coef)
        # Normalizar para estabilidad
        phi_regime2 = [c / sum(phi_regime2) * 0.75 for c in phi_regime2]
        
        return SETARSimulation(
            phi_regime1=phi_regime1,
            phi_regime2=phi_regime2,
            threshold=config['threshold'],
            delay=1,  # Delay fijo en 1
            noise_dist=dist,
            sigma=sigma,
            seed=seed
        )

    def run_single_scenario(self, config: dict, dist: str, var: float, 
                           n_train: int, n_calib: int, size_tag: str, 
                           scenario_seed: int) -> List[Dict]:
        """⚡ Versión optimizada con cache"""
        
        n_total = n_train + n_calib
        cache_key = self._get_cache_key(config, dist, var, n_total, scenario_seed)
        
        # ⚡ Revisar cache
        if cache_key in self._simulation_cache:
            series, errors = self._simulation_cache[cache_key]
        else:
            simulator = self._create_simulator(config, dist, var, scenario_seed)
            total_len = n_total + self.N_TEST_STEPS
            series, errors = simulator.simulate(n=total_len, burn_in=100)
            self._simulation_cache[cache_key] = (series, errors)
        
        train_data = series[:n_train]
        val_data = series[n_train:n_total]
        
        if self.verbose:
            print(f"   📊 Train: {len(train_data)}, Calib: {len(val_data)}")
        
        # Optimización rápida (sin max_trials)
        models = self._setup_models(scenario_seed)
        optimizer = TimeBalancedOptimizer(
            random_state=scenario_seed, 
            verbose=self.verbose
        )
        best_params = optimizer.optimize_all_models(models, train_data, val_data)
        
        # Aplicar hiperparámetros
        train_val_full = series[:n_total]
        
        for name, model in models.items():
            if name in best_params:
                for k, v in best_params[name].items():
                    if hasattr(model, k): 
                        setattr(model, k, v)
            
            if hasattr(model, 'freeze_hyperparameters'):
                model.freeze_hyperparameters(train_val_full)

        # Testing rolling window
        results_rows = []
        simulator = self._create_simulator(config, dist, var, scenario_seed)

        for t in range(self.N_TEST_STEPS):
            idx = n_total + t
            h_series = series[:idx]
            h_errors = errors[:idx]
            
            # ⚡ Reducir samples teóricos para velocidad
            true_samples = simulator.get_true_next_step_samples(
                h_series, h_errors, n_samples=500  # Reducido de 1000
            )
            
            row = {
                'Paso': t + 1,
                'Proceso': config['nombre'],
                'Distribución': dist,
                'Varianza': var,
                'N_Train': n_train,
                'N_Calib': n_calib,
                'N_Total': n_total,
                'Size': size_tag
            }
            
            for name, model in models.items():
                try:
                    if "Bootstrap" in name:
                        pred = model.fit_predict(h_series)
                    else:
                        pred = model.fit_predict(pd.DataFrame({'valor': h_series}))
                    
                    pred_array = np.asarray(pred).flatten()
                    row[name] = ecrps(pred_array, true_samples)
                    
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️ Error {name}: {e}")
                    row[name] = np.nan
            
            results_rows.append(row)

        clear_all_sessions()
        return results_rows

    def _run_scenario_wrapper(self, args: Tuple) -> List[Dict]:
        """Wrapper para paralelización"""
        return self.run_single_scenario(*args)

    def generate_all_scenarios(self) -> List[Tuple]:
        """Genera escenarios SETAR"""
        scenarios = []
        
        print(f"\n🔍 Generando escenarios SETAR:")
        print(f"   • Configs: {len(self.SETAR_CONFIGS)}")
        print(f"   • Tamaños: {len(self.SIZE_COMBINATIONS)}")
        print(f"   • Distribuciones: {len(self.DISTRIBUTIONS)}")
        print(f"   • Varianzas: {len(self.VARIANCES)}")
        esperados = len(self.SETAR_CONFIGS) * len(self.SIZE_COMBINATIONS) * len(self.DISTRIBUTIONS) * len(self.VARIANCES)
        print(f"   • TOTAL: {esperados} escenarios (3x menos que versión completa)")
        print(f"   • Filas: {esperados * self.N_TEST_STEPS}\n")
        
        s_id = 0
        for cfg in self.SETAR_CONFIGS:
            for size in self.SIZE_COMBINATIONS:
                for dist in self.DISTRIBUTIONS:
                    for var in self.VARIANCES:
                        scenarios.append((
                            cfg.copy(),
                            dist,
                            var,
                            size['n_train'],
                            size['n_calib'],
                            size['tag'],
                            self.seed + s_id
                        ))
                        s_id += 1
        
        print(f"✅ Generados {len(scenarios)} escenarios\n")
        return scenarios

    def run_all(self, excel_filename: str = "RESULTADOS_SETAR_FAST.xlsx", 
                batch_size: int = 30, max_workers: int = None, 
                save_frequency: int = 2) -> pd.DataFrame:
        """
        ⚡ Ejecución ultra-rápida
        
        Cambios de velocidad:
        - batch_size aumentado: 20 → 30
        - save_frequency reducido: 3 → 2
        - max_workers más agresivo
        """
        
        if max_workers is None:
            cpu_count = os.cpu_count() or 4
            max_workers = max(12, min(int(cpu_count * 0.85), cpu_count - 1))  # Más agresivo
        
        tasks = self.generate_all_scenarios()
        num_batches = (len(tasks) + batch_size - 1) // batch_size
        
        print(f"\n{'='*60}")
        print(f"⚡ PIPELINE SETAR ULTRA-RÁPIDO")
        print(f"{'='*60}")
        print(f"📊 Escenarios: {len(tasks)} (700 vs 2100 original = 67% menos)")
        print(f"📦 Batches: {num_batches} (tamaño {batch_size})")
        print(f"👷 Workers: {max_workers} de {os.cpu_count()} cores")
        print(f"🔥 n_boot: {self.n_boot} (500 vs 1000 = 50% menos)")
        print(f"⚡ Epochs reducidos: DeepAR 15, EnCQR 12")
        print(f"💾 Guardado cada {save_frequency} batches")
        print(f"{'='*60}\n")
        
        all_results = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(tasks))
            batch = tasks[start_idx:end_idx]
            
            print(f"🔄 Batch {i+1}/{num_batches}... ", end='', flush=True)
            
            batch_results = Parallel(n_jobs=max_workers, backend='loky', verbose=0)(
                delayed(self._run_scenario_wrapper)(t) for t in batch
            )
            
            for result_list in batch_results:
                all_results.extend(result_list)
            
            print(f"✅ {len(all_results)} filas")
            
            if (i + 1) % save_frequency == 0 or (i + 1) == num_batches:
                pd.DataFrame(all_results).to_excel(excel_filename, index=False)
                print(f"💾 Checkpoint: {excel_filename}")
            
            # ⚡ Limpieza agresiva
            del batch_results, batch
            if (i + 1) % 5 == 0:  # Limpiar cache cada 5 batches
                self._simulation_cache.clear()
            clear_all_sessions()
            gc.collect()
        
        df_final = pd.DataFrame(all_results)
        df_final.to_excel(excel_filename, index=False)
        
        print(f"\n🎉 Completado: {len(all_results)} filas → {excel_filename}")
        print(f"⚡ Velocidad estimada: 3-4x más rápido que versión original\n")
        return df_final



# ============================================================================
# LSPMW
# ============================================================================

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft
from joblib import Parallel, delayed
import time
import gc

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft
from joblib import Parallel, delayed
import time
import gc

class UnifiedPipeline_LSPM_LSPMW:
    """
    Pipeline Unificado para ARMA, ARIMA y SETAR.
    Compara LSPM vs LSPMW usando ECRPS y Test de Diebold-Mariano Modificado por grupos.
    """
    
    N_TEST_STEPS = 12
    N_VALIDATION = 40
    N_TRAIN = 200
    
    # Configuraciones ARMA
    ARMA_CONFIGS = [
        {'nombre': 'AR(1)', 'phi': [0.9], 'theta': []},
        {'nombre': 'AR(2)', 'phi': [0.5, -0.3], 'theta': []},
        {'nombre': 'MA(1)', 'phi': [], 'theta': [0.7]},
        {'nombre': 'MA(2)', 'phi': [], 'theta': [0.4, 0.2]},
        {'nombre': 'ARMA(1,1)', 'phi': [0.6], 'theta': [0.3]},
        {'nombre': 'ARMA(2,2)', 'phi': [0.4, -0.2], 'theta': [0.5, 0.1]},
        {'nombre': 'ARMA(2,1)', 'phi': [0.7, 0.2], 'theta': [0.5]}
    ]
    
    # Configuraciones ARIMA
    ARIMA_CONFIGS = [
        {'nombre': 'ARIMA(0,1,0)', 'phi': [], 'theta': []},
        {'nombre': 'ARIMA(1,1,0)', 'phi': [0.6], 'theta': []},
        {'nombre': 'ARIMA(2,1,0)', 'phi': [0.5, -0.2], 'theta': []},
        {'nombre': 'ARIMA(0,1,1)', 'phi': [], 'theta': [0.5]},
        {'nombre': 'ARIMA(0,1,2)', 'phi': [], 'theta': [0.4, 0.25]},
        {'nombre': 'ARIMA(1,1,1)', 'phi': [0.7], 'theta': [-0.3]},
        {'nombre': 'ARIMA(2,1,2)', 'phi': [0.6, 0.2], 'theta': [0.4, -0.1]}
    ]
    
    # Configuraciones SETAR
    SETAR_CONFIGS = [
        {'nombre': 'SETAR-1', 'phi_regime1': [0.6], 'phi_regime2': [-0.5], 'threshold': 0.0, 'delay': 1},
        {'nombre': 'SETAR-2', 'phi_regime1': [0.7], 'phi_regime2': [-0.7], 'threshold': 0.0, 'delay': 2},
        {'nombre': 'SETAR-3', 'phi_regime1': [0.5, -0.2], 'phi_regime2': [-0.3, 0.1], 'threshold': 0.5, 'delay': 1},
        {'nombre': 'SETAR-4', 'phi_regime1': [0.8, -0.15], 'phi_regime2': [-0.6, 0.2], 'threshold': 1.0, 'delay': 2},
        {'nombre': 'SETAR-5', 'phi_regime1': [0.4, -0.1, 0.05], 'phi_regime2': [-0.3, 0.1, -0.05], 'threshold': 0.0, 'delay': 1},
        {'nombre': 'SETAR-6', 'phi_regime1': [0.5, -0.3, 0.1], 'phi_regime2': [-0.4, 0.2, -0.05], 'threshold': 0.5, 'delay': 2},
        {'nombre': 'SETAR-7', 'phi_regime1': [0.3, 0.1], 'phi_regime2': [-0.2, -0.1], 'threshold': 0.8, 'delay': 3}
    ]
    
    DISTRIBUTIONS = ['normal', 'uniform', 'exponential', 't-student', 'mixture']
    VARIANCES = [0.2, 0.5, 1.0, 3.0]

    def __init__(self, seed: int = 42, verbose: bool = False):
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)

    def generate_all_scenarios(self, model_type='ARMA'):
        """Genera escenarios según el tipo de modelo."""
        scenarios = []
        scenario_id = 0
        
        if model_type == 'ARMA':
            configs = self.ARMA_CONFIGS
        elif model_type == 'ARIMA':
            configs = self.ARIMA_CONFIGS
        elif model_type == 'SETAR':
            configs = self.SETAR_CONFIGS
        else:
            raise ValueError(f"Tipo de modelo no válido: {model_type}")
        
        for cfg in configs:
            for dist in self.DISTRIBUTIONS:
                for var in self.VARIANCES:
                    scenarios.append((cfg.copy(), dist, var, self.seed + scenario_id, model_type))
                    scenario_id += 1
        return scenarios

    def modified_diebold_mariano_test(self, errors1, errors2, h=1):
        """
        Test Diebold-Mariano con fixed-smoothing asymptotics (Coroneo & Iacone, 2020)
        
        Parameters:
        -----------
        errors1, errors2 : array-like
            Errores de pronóstico (ECRPS) de los dos modelos
        h : int
            Horizonte de pronóstico (forecast horizon)
        
        Returns:
        --------
        hln_dm_stat : float
            Estadístico con fixed-m asymptotics
        p_value : float
            P-valor usando distribución t-Student con 2m grados de libertad
        dm_stat : float
            Estadístico DM original (para referencia)
        """
        d = errors1 - errors2
        d_bar = np.mean(d)
        T = len(d)
        
        if T < 2:
            return np.nan, np.nan, np.nan
        
        u = d - d_bar
        m = max(1, int(np.floor(T**(1/3))))
        
        # FFT de las desviaciones
        fft_u = fft(u)
        periodogram = np.abs(fft_u)**2 / (2 * np.pi * T)
        
        if m >= len(periodogram) - 1:
            m = len(periodogram) - 2
        
        sigma_hat_sq = 2 * np.pi * np.mean(periodogram[1:m+1])
        
        if sigma_hat_sq <= 0:
            sigma_hat_sq = np.var(d, ddof=1) / T
            if sigma_hat_sq <= 0:
                return 0, 1.0, 0
        
        dm_stat = np.sqrt(T) * d_bar / np.sqrt(sigma_hat_sq)
        df = 2 * m
        hln_dm_stat = dm_stat
        p_value = 2 * (1 - stats.t.cdf(abs(hln_dm_stat), df))
        
        return hln_dm_stat, p_value, dm_stat

    def _run_scenario_wrapper(self, args):
        return self.run_single_scenario(*args)

    def run_single_scenario(self, config, dist, var, scenario_seed, model_type):
        """Ejecuta un escenario individual para cualquier tipo de modelo."""
        
        # 1. Simulación según tipo de modelo
        if model_type == 'ARMA':
            simulator = ARMASimulation(
                phi=config['phi'], theta=config['theta'],
                noise_dist=dist, sigma=np.sqrt(var), seed=scenario_seed
            )
        elif model_type == 'ARIMA':
            simulator = ARIMASimulation(
                phi=config['phi'], theta=config['theta'],
                noise_dist=dist, sigma=np.sqrt(var), seed=scenario_seed
            )
        elif model_type == 'SETAR':
            simulator = SETARSimulation(
                model_type=config['nombre'],
                phi_regime1=config['phi_regime1'],
                phi_regime2=config['phi_regime2'],
                threshold=config['threshold'],
                delay=config['delay'],
                noise_dist=dist,
                sigma=np.sqrt(var),
                seed=scenario_seed
            )
        
        total_len = self.N_TRAIN + self.N_VALIDATION + self.N_TEST_STEPS
        series, errors = simulator.simulate(n=total_len, burn_in=100)
        
        train_data = series[:self.N_TRAIN]
        val_data = series[self.N_TRAIN : self.N_TRAIN + self.N_VALIDATION]
        
        # 2. Configurar solo LSPM y LSPMW
        models = {
            'LSPM': LSPM(random_state=scenario_seed),
            'LSPMW': LSPMW(rho=0.95, random_state=scenario_seed)
        }
        
        # 3. Optimización
        optimizer = TimeBalancedOptimizer(random_state=scenario_seed, verbose=self.verbose)
        best_params = optimizer.optimize_all_models(models, train_data, val_data)
        
        train_val_full = series[:self.N_TRAIN + self.N_VALIDATION]
        for name, model in models.items():
            if name in best_params:
                for k, v in best_params[name].items():
                    if hasattr(model, k): 
                        setattr(model, k, v)
            if hasattr(model, 'freeze_hyperparameters'):
                model.freeze_hyperparameters(train_val_full)
        
        # 4. Testing Rolling Window - Generar filas por cada paso
        results_rows = []
        
        for t in range(self.N_TEST_STEPS):
            idx = self.N_TRAIN + self.N_VALIDATION + t
            h_series = series[:idx]
            h_errors = errors[:idx]
            
            true_samples = simulator.get_true_next_step_samples(h_series, h_errors, n_samples=1000)
            
            row = {
                'Tipo': model_type,
                'Config': config['nombre'],
                'Dist': dist,
                'Var': var,
                'Paso': t + 1,
                'LSPM_ECRPS': np.nan,
                'LSPMW_ECRPS': np.nan
            }
            
            for name, model in models.items():
                try:
                    pred = model.fit_predict(pd.DataFrame({'valor': h_series}))
                    pred_array = np.asarray(pred).flatten()
                    ecrps_value = ecrps(pred_array, true_samples)
                    
                    if name == 'LSPM':
                        row['LSPM_ECRPS'] = ecrps_value
                    else:
                        row['LSPMW_ECRPS'] = ecrps_value
                except:
                    pass
            
            results_rows.append(row)
        
        clear_all_sessions()
        return results_rows

    def run_all(self, model_types=['ARMA', 'ARIMA', 'SETAR'], 
                excel_filename="resultados_lspm_vs_lspmw.xlsx", 
                batch_size=10, max_workers=4):
        """
        Ejecuta todos los escenarios para los tipos de modelos especificados.
        """
        all_results = []
        
        for model_type in model_types:
            tasks = self.generate_all_scenarios(model_type)
            print(f"\n🚀 Procesando {model_type}: {len(tasks)} escenarios")
            
            num_batches = (len(tasks) + batch_size - 1) // batch_size
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(tasks))
                batch = tasks[start_idx:end_idx]
                
                print(f"  📦 Lote {i+1}/{num_batches}...")
                results = Parallel(n_jobs=max_workers, backend='loky')(
                    delayed(self._run_scenario_wrapper)(t) for t in batch
                )
                
                # Aplanar resultados (cada escenario devuelve 12 filas)
                for r in results:
                    all_results.extend(r)
                
                # Guardado intermedio
                pd.DataFrame(all_results).to_excel(excel_filename, index=False)
                clear_all_sessions()
                gc.collect()
        
        # Crear DataFrame completo
        df_raw = pd.DataFrame(all_results)
        
        # Aplicar Test Diebold-Mariano por grupos
        df_summary = self.generate_dm_summary_table(df_raw)
        
        # Guardar Excel con dos hojas
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            df_raw.to_excel(writer, sheet_name='Datos_Brutos', index=False)
            df_summary.to_excel(writer, sheet_name='Resumen_DM', index=False)
        
        print(f"\n✅ Archivo generado: {excel_filename}")
        print(f"   - Datos brutos: {len(df_raw)} filas")
        print(f"   - Total escenarios: {len(df_raw) // 12}")
        
        return df_raw, df_summary

    def generate_dm_summary_table(self, df_raw):
        """
        Genera tabla resumen con Test Diebold-Mariano por grupos.
        """
        results = []
        
        # 1. Test para cada tipo de modelo (ARMA, ARIMA, SETAR)
        for model_type in ['ARMA', 'ARIMA', 'SETAR']:
            df_type = df_raw[df_raw['Tipo'] == model_type].copy()
            
            if len(df_type) == 0:
                continue
            
            # Filtrar NaN
            df_type = df_type.dropna(subset=['LSPM_ECRPS', 'LSPMW_ECRPS'])
            
            if len(df_type) < 2:
                continue
            
            lspm_errors = df_type['LSPM_ECRPS'].values
            lspmw_errors = df_type['LSPMW_ECRPS'].values
            
            # Test DM
            dm_stat, p_value, _ = self.modified_diebold_mariano_test(lspm_errors, lspmw_errors)
            
            # Determinar conclusión
            if p_value < 0.05:
                if dm_stat < 0:
                    conclusion = 'LSPM es significativamente mejor'
                else:
                    conclusion = 'LSPMW es significativamente mejor'
            else:
                conclusion = 'No hay diferencia significativa'
            
            # Agregar fila para LSPM
            results.append({
                'Modelo': 'LSPM',
                f'ECRPS Promedio {model_type}': np.mean(lspm_errors),
                f'ECRPS Mediano {model_type}': np.median(lspm_errors),
                f'Significativo en {model_type}': 'Sí' if p_value < 0.05 else 'No',
                f'Conclusión {model_type}': conclusion,
                f'DM Stat {model_type}': dm_stat,
                f'P-Value {model_type}': p_value
            })
        
        # Reorganizar para que sea una tabla con LSPM y LSPMW en filas
        # y cada grupo de modelo (ARMA, ARIMA, SETAR) en columnas
        
        # Primero extraer datos por tipo
        summary_data = {'Modelo': ['LSPM', 'LSPMW']}
        
        for model_type in ['ARMA', 'ARIMA', 'SETAR']:
            df_type = df_raw[df_raw['Tipo'] == model_type].copy()
            df_type = df_type.dropna(subset=['LSPM_ECRPS', 'LSPMW_ECRPS'])
            
            if len(df_type) < 2:
                summary_data[f'ECRPS Promedio {model_type}'] = [np.nan, np.nan]
                summary_data[f'ECRPS Mediano {model_type}'] = [np.nan, np.nan]
                summary_data[f'Significativo en {model_type}'] = ['N/A', 'N/A']
                summary_data[f'Conclusión {model_type}'] = ['N/A', 'N/A']
                continue
            
            lspm_errors = df_type['LSPM_ECRPS'].values
            lspmw_errors = df_type['LSPMW_ECRPS'].values
            
            dm_stat, p_value, _ = self.modified_diebold_mariano_test(lspm_errors, lspmw_errors)
            
            significativo = 'Sí' if p_value < 0.05 else 'No'
            
            if p_value < 0.05:
                if dm_stat < 0:
                    conclusion_lspm = f'Mejor (DM={dm_stat:.3f}, p={p_value:.4f})'
                    conclusion_lspmw = f'Peor (DM={dm_stat:.3f}, p={p_value:.4f})'
                else:
                    conclusion_lspm = f'Peor (DM={dm_stat:.3f}, p={p_value:.4f})'
                    conclusion_lspmw = f'Mejor (DM={dm_stat:.3f}, p={p_value:.4f})'
            else:
                conclusion_lspm = f'Empate (DM={dm_stat:.3f}, p={p_value:.4f})'
                conclusion_lspmw = f'Empate (DM={dm_stat:.3f}, p={p_value:.4f})'
            
            summary_data[f'ECRPS Promedio {model_type}'] = [
                np.mean(lspm_errors),
                np.mean(lspmw_errors)
            ]
            summary_data[f'ECRPS Mediano {model_type}'] = [
                np.median(lspm_errors),
                np.median(lspmw_errors)
            ]
            summary_data[f'Significativo en {model_type}'] = [
                significativo,
                significativo
            ]
            summary_data[f'Conclusión {model_type}'] = [
                conclusion_lspm,
                conclusion_lspmw
            ]
        
        # Agregar columna "GENERAL" con todos los datos combinados
        df_all = df_raw.dropna(subset=['LSPM_ECRPS', 'LSPMW_ECRPS'])
        
        if len(df_all) >= 2:
            lspm_all = df_all['LSPM_ECRPS'].values
            lspmw_all = df_all['LSPMW_ECRPS'].values
            
            dm_stat_all, p_value_all, _ = self.modified_diebold_mariano_test(lspm_all, lspmw_all)
            
            significativo_all = 'Sí' if p_value_all < 0.05 else 'No'
            
            if p_value_all < 0.05:
                if dm_stat_all < 0:
                    conclusion_lspm_all = f'Mejor (DM={dm_stat_all:.3f}, p={p_value_all:.4f})'
                    conclusion_lspmw_all = f'Peor (DM={dm_stat_all:.3f}, p={p_value_all:.4f})'
                else:
                    conclusion_lspm_all = f'Peor (DM={dm_stat_all:.3f}, p={p_value_all:.4f})'
                    conclusion_lspmw_all = f'Mejor (DM={dm_stat_all:.3f}, p={p_value_all:.4f})'
            else:
                conclusion_lspm_all = f'Empate (DM={dm_stat_all:.3f}, p={p_value_all:.4f})'
                conclusion_lspmw_all = f'Empate (DM={dm_stat_all:.3f}, p={p_value_all:.4f})'
            
            summary_data['ECRPS Promedio GENERAL'] = [
                np.mean(lspm_all),
                np.mean(lspmw_all)
            ]
            summary_data['ECRPS Mediano GENERAL'] = [
                np.median(lspm_all),
                np.median(lspmw_all)
            ]
            summary_data['Significativo en GENERAL'] = [
                significativo_all,
                significativo_all
            ]
            summary_data['Conclusión GENERAL'] = [
                conclusion_lspm_all,
                conclusion_lspmw_all
            ]
        
        return pd.DataFrame(summary_data)

class Pipeline140SinSesgos_ARIMA_ConDiferenciacion_LSPMW:
    """
    Pipeline ARIMA optimizado para evaluar únicamente LSPMW.
    Evalúa cada escenario en dos modalidades: SIN_DIFF y CON_DIFF.
    """
    
    N_TEST_STEPS = 12
    N_VALIDATION = 40
    N_TRAIN = 200
    
    # Configuraciones ARIMA (d=1 por defecto para estos nombres)
    ARIMA_CONFIGS = [
        {'nombre': 'RW', 'phi': [], 'theta': []}, # Random Walk es ARIMA(0,1,0)
        {'nombre': 'AR(1)', 'phi': [0.6], 'theta': []},
        {'nombre': 'AR(2)', 'phi': [0.5, -0.2], 'theta': []},
        {'nombre': 'MA(1)', 'phi': [], 'theta': [0.5]},
        {'nombre': 'MA(2)', 'phi': [], 'theta': [0.4, 0.25]},
        {'nombre': 'ARMA(1,1)', 'phi': [0.7], 'theta': [-0.3]},
        {'nombre': 'ARMA(2,2)', 'phi': [0.6, 0.2], 'theta': [0.4, -0.1]}
    ]
    DISTRIBUTIONS = ['normal', 'uniform', 'exponential', 't-student', 'mixture']
    VARIANCES = [0.2, 0.5, 1.0, 3.0]

    def __init__(self, n_boot: int = 1000, seed: int = 42, verbose: bool = False):
        self.n_boot = n_boot
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)

    def _setup_model(self, seed: int):
        """Solo crea LSPMW"""
        return LSPMW(rho=0.95, random_state=seed)

    def _run_modalidad(self, simulator, series_levels, errors, arima_config, dist, var, modalidad, scenario_seed):
        """Ejecuta una modalidad específica (SIN_DIFF o CON_DIFF)"""
        
        # 1. Preparar datos según modalidad
        if modalidad == "CON_DIFF":
            # El modelo ve incrementos ΔY_t
            series_to_model = np.diff(series_levels, prepend=series_levels[0])
        else:
            # El modelo ve niveles Y_t
            series_to_model = series_levels

        train_data = series_to_model[:self.N_TRAIN]
        val_data = series_to_model[self.N_TRAIN : self.N_TRAIN + self.N_VALIDATION]
        
        # 2. Setup y Optimización
        model = self._setup_model(scenario_seed)
        optimizer = TimeBalancedOptimizer(random_state=scenario_seed, verbose=False)
        
        # Optimizar solo LSPMW
        models_dict = {'LSPMW': model}
        best_params = optimizer.optimize_all_models(models_dict, train_data, val_data)
        
        train_val_full = series_to_model[:self.N_TRAIN + self.N_VALIDATION]
        if 'LSPMW' in best_params:
            for k, v in best_params['LSPMW'].items():
                if hasattr(model, k): 
                    setattr(model, k, v)
        
        if hasattr(model, 'freeze_hyperparameters'):
            model.freeze_hyperparameters(train_val_full)

        # 3. Testing Rolling Window
        results_rows = []
        p, q = len(arima_config['phi']), len(arima_config['theta'])
        d = 1 # Por defecto en esta simulación

        for t in range(self.N_TEST_STEPS):
            idx = self.N_TRAIN + self.N_VALIDATION + t
            h_series_levels = series_levels[:idx]
            h_errors = errors[:idx]
            h_to_model = series_to_model[:idx]
            
            # Densidad Teórica (Siguiente paso real)
            true_samples = simulator.get_true_next_step_samples(h_series_levels, h_errors, n_samples=1000)
            
            # Fila base con el formato de la imagen
            row = {
                'Paso': t + 1,
                'Proceso': f"ARMA_I({p},{d},{q})",
                'p': p,
                'd': d,
                'q': q,
                'ARMA_base': arima_config['nombre'],
                'Distribución': dist,
                'Varianza': var,
                'Modalidad': modalidad,
                'Valor_Observado': series_levels[idx] # El valor real que ocurrió
            }
            
            try:
                pred = model.fit_predict(pd.DataFrame({'valor': h_to_model}))
                pred_array = np.asarray(pred).flatten()
                
                # Si predijo incremento, sumar al último nivel para comparar densidades en niveles
                if modalidad == "CON_DIFF":
                    pred_array = series_levels[idx-1] + pred_array
                
                row['LSPMW'] = ecrps(pred_array, true_samples)
            except:
                row['LSPMW'] = np.nan
            
            results_rows.append(row)
            
        return results_rows

    def _run_scenario_wrapper(self, args):
        arima_cfg, dist, var, seed = args
        
        # Simulación (Niveles)
        simulator = ARIMASimulation(
            phi=arima_cfg['phi'], theta=arima_cfg['theta'],
            noise_dist=dist, sigma=np.sqrt(var), seed=seed
        )
        total_len = self.N_TRAIN + self.N_VALIDATION + self.N_TEST_STEPS
        series_levels, errors = simulator.simulate(n=total_len, burn_in=100)
        
        # Ejecutar ambas modalidades
        res_sin = self._run_modalidad(simulator, series_levels, errors, arima_cfg, dist, var, "SIN_DIFF", seed)
        res_con = self._run_modalidad(simulator, series_levels, errors, arima_cfg, dist, var, "CON_DIFF", seed + 1)
        
        clear_all_sessions()
        return res_sin + res_con

    def generate_all_scenarios(self) -> list:
        scenarios = []
        s_id = 0
        for arima_cfg in self.ARIMA_CONFIGS:
            for dist in self.DISTRIBUTIONS:
                for var in self.VARIANCES:
                    scenarios.append((arima_cfg.copy(), dist, var, self.seed + s_id))
                    s_id += 1
        return scenarios

    def run_all(self, excel_filename="resultados_arima_lspmw.xlsx", batch_size=5, n_jobs=2):
        tasks = self.generate_all_scenarios()
        print(f"🚀 Ejecutando {len(tasks)} escenarios ARIMA (Solo LSPMW - Doble Modalidad)...")
        
        all_results = []
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            print(f"  -> Procesando lote {i//batch_size + 1}...")
            results = Parallel(n_jobs=n_jobs)(delayed(self._run_scenario_wrapper)(t) for t in batch)
            for r in results:
                all_results.extend(r)
            
            pd.DataFrame(all_results).to_excel(excel_filename, index=False)

        return pd.DataFrame(all_results)

class PipelineARIMA_MultiD_LSPMW_Only:
    """
    Pipeline Multi-D CORREGIDO para ARIMA(p,d,q) con múltiples órdenes de integración.
    VERSION OPTIMIZADA: Solo evalúa LSPMW.
    
    CORRECCIONES FUNDAMENTALES:
    1. Usa ARIMASimulation (no ARIMAMultiDSimulation) para d=1
    2. Implementa integración manual para d>1
    3. Densidades predictivas calculadas en el espacio correcto
    4. Integración coherente para predicciones
    5. Evalúa solo LSPMW (más rápido)
    """
    
    N_TEST_STEPS = 12
    N_VALIDATION_FOR_OPT = 40
    N_TRAIN_INITIAL = 200

    ARMA_CONFIGS = [
        {'nombre': 'RW', 'phi': [], 'theta': []},
        {'nombre': 'AR(1)', 'phi': [0.6], 'theta': []},
        {'nombre': 'AR(2)', 'phi': [0.5, -0.2], 'theta': []},
        {'nombre': 'MA(1)', 'phi': [], 'theta': [0.5]},
        {'nombre': 'MA(2)', 'phi': [], 'theta': [0.4, 0.25]},
        {'nombre': 'ARMA(1,1)', 'phi': [0.7], 'theta': [-0.3]},
        {'nombre': 'ARMA(2,2)', 'phi': [0.6, 0.2], 'theta': [0.4, -0.1]}
    ]
    
    DISTRIBUTIONS = ['normal', 'uniform', 'exponential', 't-student', 'mixture']
    VARIANCES = [0.2, 0.5, 1.0, 3.0]
    D_VALUES = [1, 2, 3, 4, 5, 6, 7, 10]
    
    def __init__(self, n_boot: int = 1000, seed: int = 42, verbose: bool = False):
        self.n_boot = n_boot
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)

    def _setup_model(self, seed: int):
        """Solo crea LSPMW."""
        return LSPMW(rho=0.95, random_state=seed)

    def _simulate_arima_manual(self, arma_config: dict, d_value: int, 
                              dist: str, var: float, seed: int, n: int):
        """
        Simula ARIMA EXACTAMENTE como Pipeline140SinSesgos_ARIMA_ConDiferenciacion.
        
        Proceso:
        1. Simula W_t ~ ARMA(p,q) usando ARIMASimulation
        2. Integra manualmente d veces: Y_t = S^d(W_t)
        
        IMPORTANTE: Para d=1, esto es IDÉNTICO a ARIMASimulation directamente.
        """
        from simulacion import ARIMASimulation
        
        # Simular usando ARIMASimulation (siempre con d=1 internamente)
        simulator = ARIMASimulation(
            phi=arma_config['phi'],
            theta=arma_config['theta'],
            noise_dist=dist,
            sigma=np.sqrt(var),
            seed=seed
        )
        
        # Para ARIMASimulation, la serie ya viene con 1 integración
        # Si d=1, usamos directamente. Si d>1, integramos (d-1) veces adicionales
        series_base, errors = simulator.simulate(n=n, burn_in=100)
        
        # Si d=1, ya está integrada correctamente
        if d_value == 1:
            y_series = series_base.copy()
        else:
            # Para d>1, integrar (d-1) veces adicionales
            y_series = series_base.copy()
            for _ in range(d_value - 1):
                y_series = np.cumsum(y_series)
        
        return y_series, series_base, errors, simulator

    def _get_true_density_from_simulator(self, simulator, series_history: np.ndarray,
                                        errors_history: np.ndarray, 
                                        n_samples: int = 1000) -> np.ndarray:
        """
        Obtiene densidad verdadera usando EXACTAMENTE el método de ARIMASimulation.
        
        IDÉNTICO a Pipeline140SinSesgos_ARIMA_ConDiferenciacion.
        """
        return simulator.get_true_next_step_samples(
            series_history, errors_history, n_samples=n_samples
        )

    def _integrate_d_times_for_prediction(self, w_next_samples: np.ndarray,
                                         y_series: np.ndarray, 
                                         current_idx: int,
                                         d_value: int) -> np.ndarray:
        """
        Integra predicciones desde espacio ARMA(d=1) a ARIMA(d>1).
        
        Para d=1: Y_{t+1} = Y_t + W_{t+1}
        Para d>1: Usar fórmula recursiva
        """
        if d_value == 1:
            # Caso simple: Y_{t+1} = Y_t + ΔY_t donde ΔY_t = W_{t+1}
            return y_series[current_idx - 1] + w_next_samples
        else:
            # Para d>1, necesitamos aplicar integración múltiple
            # Guardamos los últimos d valores de Y
            y_last_values = []
            temp_y = y_series[:current_idx].copy()
            
            for level in range(d_value):
                y_last_values.append(temp_y[-1])
                if level < d_value - 1:
                    temp_y = np.diff(temp_y)
            
            # Integrar desde W_{t+1} hasta Y_{t+1}
            y_next_samples = w_next_samples.copy()
            for level in range(d_value - 1, -1, -1):
                y_next_samples = y_last_values[level] + y_next_samples
            
            return y_next_samples

    def _run_single_modalidad(self, arma_config: dict, d_value: int,
                             dist: str, var: float, scenario_seed: int,
                             y_series: np.ndarray, series_base: np.ndarray,
                             errors: np.ndarray, test_start_idx: int,
                             usar_diferenciacion: bool, simulator) -> list:
        """
        Ejecuta una modalidad EXACTAMENTE como Pipeline140SinSesgos_ARIMA_ConDiferenciacion.
        Solo evalúa LSPMW.
        
        MODALIDADES:
        - SIN_DIFF: Modelo ve Y_t (serie integrada de orden d)
        - CON_DIFF: Modelo ve ∇Y_t (serie diferenciada 1 vez)
        """
        modalidad_str = "CON_DIFF" if usar_diferenciacion else "SIN_DIFF"
        
        # Preparar serie según modalidad
        if usar_diferenciacion:
            # El modelo ve incrementos ΔY_t
            series_to_model = np.diff(y_series, prepend=y_series[0])
        else:
            # El modelo ve niveles Y_t
            series_to_model = y_series.copy()
        
        train_calib_data = series_to_model[:test_start_idx]
        
        # Crear solo LSPMW
        model = self._setup_model(scenario_seed)
        
        # Optimización
        optimizer = TimeBalancedOptimizer(random_state=self.seed, verbose=self.verbose)
        
        split = min(self.N_VALIDATION_FOR_OPT, len(train_calib_data) // 3)
        models_dict = {'LSPMW': model}
        best_params = optimizer.optimize_all_models(
            models_dict, 
            train_calib_data[:-split], 
            train_calib_data[-split:]
        )
        
        # Aplicar hiperparámetros óptimos
        if 'LSPMW' in best_params:
            for k, v in best_params['LSPMW'].items():
                if hasattr(model, k): 
                    setattr(model, k, v)
        
        if hasattr(model, 'freeze_hyperparameters'):
            model.freeze_hyperparameters(train_calib_data)

        # Testing rolling window
        results_rows = []
        p = len(arma_config['phi'])
        q = len(arma_config['theta'])

        for t in range(self.N_TEST_STEPS):
            curr_idx = test_start_idx + t
            h_series_levels = y_series[:curr_idx]
            h_to_model = series_to_model[:curr_idx]
            
            # DENSIDAD VERDADERA
            if d_value == 1:
                true_samples_base = self._get_true_density_from_simulator(
                    simulator, series_base[:curr_idx], errors[:curr_idx]
                )
                true_samples = true_samples_base
            else:
                # Para d>1, obtener densidad base y luego integrar
                true_samples_base = self._get_true_density_from_simulator(
                    simulator, series_base[:curr_idx], errors[:curr_idx]
                )
                # Integrar las muestras (d-1) veces adicionales
                true_samples = self._integrate_d_times_for_prediction(
                    true_samples_base, y_series, curr_idx, d_value
                )
            
            # Fila de resultados
            row = {
                'Paso': t + 1,
                'Proceso': f"ARMA_I({p},{d_value},{q})",
                'p': p,
                'd': d_value,
                'q': q,
                'ARMA_base': arma_config['nombre'],
                'Distribución': dist,
                'Varianza': var,
                'Modalidad': modalidad_str,
                'Valor_Observado': y_series[curr_idx]
            }
            
            # Evaluar LSPMW
            try:
                pred = model.fit_predict(pd.DataFrame({'valor': h_to_model}))
                pred_array = np.asarray(pred).flatten()
                
                # Integrar predicciones si es necesario
                if usar_diferenciacion:
                    # pred_array son incrementos ΔY_{t+1}
                    # Y_{t+1} = Y_t + ΔY_{t+1}
                    pred_array = y_series[curr_idx - 1] + pred_array
                
                # Calcular ECRPS
                row['LSPMW'] = ecrps(pred_array, true_samples)
            except Exception as e:
                if self.verbose:
                    print(f"Error en LSPMW: {e}")
                row['LSPMW'] = np.nan
            
            results_rows.append(row)

        return results_rows

    def _run_scenario_wrapper(self, args):
        """Wrapper para procesamiento paralelo."""
        arma_cfg, d_val, dist, var, seed = args
        
        total_n = self.N_TRAIN_INITIAL + self.N_TEST_STEPS
        
        # Simular ARIMA manualmente
        y_series, series_base, errors, simulator = self._simulate_arima_manual(
            arma_cfg, d_val, dist, var, seed, total_n
        )
        
        # Ejecutar ambas modalidades
        res_sin_diff = self._run_single_modalidad(
            arma_cfg, d_val, dist, var, seed,
            y_series, series_base, errors,
            self.N_TRAIN_INITIAL, False, simulator
        )
        
        res_con_diff = self._run_single_modalidad(
            arma_cfg, d_val, dist, var, seed + 1,
            y_series, series_base, errors,
            self.N_TRAIN_INITIAL, True, simulator
        )
        
        clear_all_sessions()
        return res_sin_diff + res_con_diff

    def run_all(self, excel_filename: str = "RESULTADOS_MULTID_LSPMW_ONLY.xlsx", 
                batch_size: int = 10, n_jobs: int = 3):
        """
        Ejecuta todas las simulaciones.
        Devuelve df_resultados, df_resumen para mantener compatibilidad.
        """
        print("="*80)
        print("🚀 PIPELINE MULTI-D: ARIMA_I(p,d,q) - SOLO LSPMW")
        print("="*80)
        
        # Generar tareas
        tasks = []
        s_id = 0
        for d in self.D_VALUES:
            for cfg in self.ARMA_CONFIGS:
                for dist in self.DISTRIBUTIONS:
                    for var in self.VARIANCES:
                        tasks.append((cfg.copy(), d, dist, var, self.seed + s_id))
                        s_id += 1
        
        print(f"📊 Total de escenarios: {len(tasks)}")
        print(f"   - Valores de d: {self.D_VALUES}")
        print(f"   - ARMA configs: {len(self.ARMA_CONFIGS)}")
        print(f"   - Distribuciones: {len(self.DISTRIBUTIONS)}")
        print(f"   - Varianzas: {len(self.VARIANCES)}")
        print(f"   - Modalidades por escenario: 2 (SIN_DIFF, CON_DIFF)")
        print(f"   - Modelo: LSPMW únicamente")
        print(f"   - Total filas esperadas: {len(tasks) * 2 * self.N_TEST_STEPS}")
        
        # Procesamiento por lotes
        all_results = []
        num_batches = (len(tasks) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(tasks))
            batch = tasks[start_idx:end_idx]
            
            print(f"📦 Procesando lote {i+1}/{num_batches}...")
            
            results = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(self._run_scenario_wrapper)(t) for t in batch
            )
            
            for r in results: 
                all_results.extend(r)
            
            # Guardar progreso
            pd.DataFrame(all_results).to_excel(excel_filename, index=False)
            print(f"   ✅ {len(all_results)} filas guardadas")
            
            clear_all_sessions()
            gc.collect()
        
        df_resultados = pd.DataFrame(all_results)
        
        # Crear resumen agregado
        df_resumen = df_resultados.groupby(
            ['Proceso', 'ARMA_base', 'Distribución', 'Varianza', 'Modalidad', 'd']
        ).agg({
            'LSPMW': ['mean', 'std', 'count']
        }).reset_index()
        
        # Aplanar nombres de columnas
        df_resumen.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                              for col in df_resumen.columns.values]
        
        print(f"✅ Simulación completa: {excel_filename}")
        return df_resultados, df_resumen
    

class Pipeline140_TamanosCrecientes_LSPMW_Only:
    """
    ✅ PIPELINE CORREGIDO - Tamaños Crecientes
    VERSION OPTIMIZADA: Solo evalúa LSPMW
    
    CORRECCIÓN APLICADA:
    - Mantiene proporción FIJA 83%/17% en OPTIMIZACIÓN
    - Usa TODO el histórico (train+calib) para freeze_hyperparameters
    - Esto garantiza que tamaños diferentes tomen tiempos diferentes
    
    ESTRUCTURA:
    - Proporción fija: 83% train / 17% calib (SOLO para optimización)
    - 5 tamaños totales diferentes
    - 12 pasos de predicción (fijos)
    - 3 tipos de procesos: ARMA (7 configs), ARIMA (7 configs), SETAR (7 configs)
    - 5 distribuciones × 4 varianzas
    - Solo evalúa LSPMW
    """
    
    N_TEST_STEPS = 12  # Siempre 12 pasos de predicción
    
    # 21 Configuraciones (7 ARMA + 7 ARIMA + 7 SETAR)
    ARMA_CONFIGS = [
        {'nombre': 'AR(1)', 'phi': [0.9], 'theta': []},
        {'nombre': 'AR(2)', 'phi': [0.5, -0.3], 'theta': []},
        {'nombre': 'MA(1)', 'phi': [], 'theta': [0.7]},
        {'nombre': 'MA(2)', 'phi': [], 'theta': [0.4, 0.2]},
        {'nombre': 'ARMA(1,1)', 'phi': [0.6], 'theta': [0.3]},
        {'nombre': 'ARMA(2,2)', 'phi': [0.4, -0.2], 'theta': [0.5, 0.1]},
        {'nombre': 'ARMA(2,1)', 'phi': [0.7, 0.2], 'theta': [0.5]}
    ]
    
    ARIMA_CONFIGS = [
        {'nombre': 'ARIMA(0,1,0)', 'phi': [], 'theta': []},
        {'nombre': 'ARIMA(1,1,0)', 'phi': [0.6], 'theta': []},
        {'nombre': 'ARIMA(2,1,0)', 'phi': [0.5, -0.2], 'theta': []},
        {'nombre': 'ARIMA(0,1,1)', 'phi': [], 'theta': [0.5]},
        {'nombre': 'ARIMA(0,1,2)', 'phi': [], 'theta': [0.4, 0.25]},
        {'nombre': 'ARIMA(1,1,1)', 'phi': [0.7], 'theta': [-0.3]},
        {'nombre': 'ARIMA(2,1,2)', 'phi': [0.6, 0.2], 'theta': [0.4, -0.1]}
    ]
    
    SETAR_CONFIGS = [
        {'nombre': 'SETAR-1', 'phi_regime1': [0.6], 'phi_regime2': [-0.5], 'threshold': 0.0, 'delay': 1},
        {'nombre': 'SETAR-2', 'phi_regime1': [0.7], 'phi_regime2': [-0.7], 'threshold': 0.0, 'delay': 2},
        {'nombre': 'SETAR-3', 'phi_regime1': [0.5, -0.2], 'phi_regime2': [-0.3, 0.1], 'threshold': 0.5, 'delay': 1},
        {'nombre': 'SETAR-4', 'phi_regime1': [0.8, -0.15], 'phi_regime2': [-0.6, 0.2], 'threshold': 1.0, 'delay': 2},
        {'nombre': 'SETAR-5', 'phi_regime1': [0.4, -0.1, 0.05], 'phi_regime2': [-0.3, 0.1, -0.05], 'threshold': 0.0, 'delay': 1},
        {'nombre': 'SETAR-6', 'phi_regime1': [0.5, -0.3, 0.1], 'phi_regime2': [-0.4, 0.2, -0.05], 'threshold': 0.5, 'delay': 2},
        {'nombre': 'SETAR-7', 'phi_regime1': [0.3, 0.1], 'phi_regime2': [-0.2, -0.1], 'threshold': 0.8, 'delay': 3}
    ]
    
    DISTRIBUTIONS = ['normal', 'uniform', 'exponential', 't-student', 'mixture']
    VARIANCES = [0.2, 0.5, 1.0, 3.0]
    
    # ✅ 5 Tamaños con proporción fija 83% / 17%
    SIZE_COMBINATIONS = [
        {'tag': 'N=120', 'n_total': 120, 'n_train': 100, 'n_calib': 20},
        {'tag': 'N=240', 'n_total': 240, 'n_train': 199, 'n_calib': 41},
        {'tag': 'N=360', 'n_total': 360, 'n_train': 299, 'n_calib': 61},
        {'tag': 'N=600', 'n_total': 600, 'n_train': 498, 'n_calib': 102},
        {'tag': 'N=1200', 'n_total': 1200, 'n_train': 996, 'n_calib': 204}
    ]

    def __init__(self, n_boot: int = 1000, seed: int = 42, verbose: bool = False, proceso_tipo: str = 'ARMA'):
        self.n_boot = n_boot
        self.seed = seed
        self.verbose = verbose
        self.proceso_tipo = proceso_tipo.upper()
        self.rng = np.random.default_rng(seed)

    def _setup_model(self, seed: int):
        """Solo crea LSPMW"""
        return LSPMW(rho=0.95, random_state=seed)

    def _get_configs_for_process_type(self):
        """Obtiene las configuraciones según el tipo de proceso"""
        if self.proceso_tipo == 'ARMA':
            return self.ARMA_CONFIGS
        elif self.proceso_tipo == 'ARIMA':
            return self.ARIMA_CONFIGS
        elif self.proceso_tipo == 'SETAR':
            return self.SETAR_CONFIGS
        else:
            raise ValueError(f"Tipo de proceso desconocido: {self.proceso_tipo}")

    def _create_simulator(self, config: dict, dist: str, var: float, seed: int):
        """Crea simulador según tipo de proceso"""
        sigma = np.sqrt(var)
        
        if self.proceso_tipo == 'ARMA':
            return ARMASimulation(
                phi=config['phi'], 
                theta=config['theta'],
                noise_dist=dist, 
                sigma=sigma, 
                seed=seed
            )
        elif self.proceso_tipo == 'ARIMA':
            return ARIMASimulation(
                phi=config['phi'], 
                theta=config['theta'],
                noise_dist=dist, 
                sigma=sigma, 
                seed=seed
            )
        else:  # SETAR
            return SETARSimulation(
                phi_regime1=config['phi_regime1'],
                phi_regime2=config['phi_regime2'],
                threshold=config['threshold'],
                delay=config['delay'],
                noise_dist=dist,
                sigma=sigma,
                seed=seed
            )

    def run_single_scenario(self, config: dict, dist: str, var: float, 
                           n_train: int, n_calib: int, size_tag: str, 
                           scenario_seed: int) -> List[Dict]:
        """
        ✅ CORREGIDO: Ahora usa consistentemente los datos
        Solo evalúa LSPMW
        
        CAMBIOS:
        1. Optimización usa solo n_train para entrenar
        2. freeze_hyperparameters() usa TODO (train+calib)
        3. Esto hace que tamaños diferentes tomen tiempos diferentes
        """
        
        n_total = n_train + n_calib
        
        # 1. Simulación
        simulator = self._create_simulator(config, dist, var, scenario_seed)
        total_len = n_total + self.N_TEST_STEPS
        series, errors = simulator.simulate(n=total_len, burn_in=100)
        
        # ✅ CORRECCIÓN: Usar solo n_train para optimización
        train_data = series[:n_train]
        val_data = series[n_train:n_total]  # Solo n_calib datos
        
        if self.verbose:
            print(f"   📊 Train: {len(train_data)}, Calib: {len(val_data)}, Test steps: {self.N_TEST_STEPS}")
        
        # 2. Optimización de hiperparámetros (solo LSPMW)
        model = self._setup_model(scenario_seed)
        optimizer = TimeBalancedOptimizer(random_state=scenario_seed, verbose=self.verbose)
        
        models_dict = {'LSPMW': model}
        best_params = optimizer.optimize_all_models(models_dict, train_data, val_data)
        
        # 3. Aplicar mejores hiperparámetros y congelar con TODOS los datos
        train_val_full = series[:n_total]  # ✅ ESTO crece con el tamaño
        
        if 'LSPMW' in best_params:
            for k, v in best_params['LSPMW'].items():
                if hasattr(model, k): 
                    setattr(model, k, v)
        
        # ✅ CLAVE: freeze_hyperparameters() usa TODO el histórico
        # Esto hace que N=1200 tome más tiempo que N=120
        if hasattr(model, 'freeze_hyperparameters'):
            model.freeze_hyperparameters(train_val_full)

        # 4. Testing Rolling Window
        results_rows = []

        for t in range(self.N_TEST_STEPS):
            idx = n_total + t
            h_series = series[:idx]
            h_errors = errors[:idx]
            
            # Densidad teórica
            true_samples = simulator.get_true_next_step_samples(h_series, h_errors, n_samples=1000)
            
            # Fila de resultados
            row = {
                'Paso': t + 1,
                'Tipo_Proceso': self.proceso_tipo,
                'Proceso': config['nombre'],
                'Distribución': dist,
                'Varianza': var,
                'N_Train': n_train,
                'N_Calib': n_calib,
                'N_Total': n_total,
                'Size': size_tag
            }
            
            # ✅ Evaluar solo LSPMW
            try:
                pred = model.fit_predict(pd.DataFrame({'valor': h_series}))
                pred_array = np.asarray(pred).flatten()
                row['LSPMW'] = ecrps(pred_array, true_samples)
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Error LSPMW en paso {t+1}: {e}")
                row['LSPMW'] = np.nan
            
            results_rows.append(row)

        clear_all_sessions()
        return results_rows

    def _run_scenario_wrapper(self, args: Tuple) -> List[Dict]:
        """Wrapper para paralelización"""
        return self.run_single_scenario(*args)

    def generate_all_scenarios(self) -> List[Tuple]:
        """
        ✅ Genera escenarios para UN tipo de proceso
        """
        scenarios = []
        configs = self._get_configs_for_process_type()
        
        # Debug info
        if self.verbose or True:
            print(f"\n🔍 Generando escenarios para {self.proceso_tipo}:")
            print(f"   • Configs: {len(configs)}")
            print(f"   • Tamaños: {len(self.SIZE_COMBINATIONS)}")
            print(f"   • Distribuciones: {len(self.DISTRIBUTIONS)}")
            print(f"   • Varianzas: {len(self.VARIANCES)}")
            esperados = len(configs) * len(self.SIZE_COMBINATIONS) * len(self.DISTRIBUTIONS) * len(self.VARIANCES)
            print(f"   • ESPERADOS: {esperados} escenarios")
            print(f"   • Filas esperadas: {esperados * self.N_TEST_STEPS}\n")
        
        s_id = 0
        for cfg in configs:
            for size in self.SIZE_COMBINATIONS:
                for dist in self.DISTRIBUTIONS:
                    for var in self.VARIANCES:
                        scenarios.append((
                            cfg.copy(),
                            dist,
                            var,
                            size['n_train'],
                            size['n_calib'],
                            size['tag'],
                            self.seed + s_id
                        ))
                        s_id += 1
        
        print(f"✅ Generados {len(scenarios)} escenarios\n")
        return scenarios

    def run_all(self, excel_filename: str = None, batch_size: int = 20, 
                n_jobs: int = None, save_frequency: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Ejecuta todos los escenarios con paralelización.
        Devuelve df_resultados, df_resumen para compatibilidad.
        """
        
        # Auto-detecta workers
        if n_jobs is None:
            cpu_count = os.cpu_count() or 4
            n_jobs = max(10, min(int(cpu_count * 0.75), cpu_count - 2))
        
        if excel_filename is None:
            excel_filename = f"RESULTADOS_TAMANOS_{self.proceso_tipo}_LSPMW.xlsx"
        
        tasks = self.generate_all_scenarios()
        num_batches = (len(tasks) + batch_size - 1) // batch_size
        
        print(f"\n{'='*60}")
        print(f"🚀 PIPELINE TAMAÑOS CRECIENTES - {self.proceso_tipo} - SOLO LSPMW")
        print(f"{'='*60}")
        print(f"📊 Total escenarios: {len(tasks)}")
        print(f"📦 Batches: {num_batches} (tamaño {batch_size})")
        print(f"👷 Workers: {n_jobs} de {os.cpu_count()} cores")
        print(f"💾 Guardado cada {save_frequency} batches")
        print(f"{'='*60}\n")
        
        all_results = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(tasks))
            batch = tasks[start_idx:end_idx]
            
            print(f"🔄 Batch {i+1}/{num_batches}... ", end='', flush=True)
            
            batch_results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
                delayed(self._run_scenario_wrapper)(t) for t in batch
            )
            
            for result_list in batch_results:
                all_results.extend(result_list)
            
            print(f"✅ {len(all_results)} filas")
            
            if (i + 1) % save_frequency == 0 or (i + 1) == num_batches:
                pd.DataFrame(all_results).to_excel(excel_filename, index=False)
                print(f"💾 Checkpoint: {excel_filename}")
            
            del batch_results, batch
            clear_all_sessions()
            gc.collect()
        
        df_resultados = pd.DataFrame(all_results)
        df_resultados.to_excel(excel_filename, index=False)
        
        # Crear resumen agregado
        df_resumen = df_resultados.groupby(
            ['Tipo_Proceso', 'Proceso', 'Distribución', 'Varianza', 'Size', 'N_Total']
        ).agg({
            'LSPMW': ['mean', 'std', 'count']
        }).reset_index()
        
        # Aplanar nombres de columnas
        df_resumen.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                              for col in df_resumen.columns.values]
        
        print(f"\n🎉 Completado: {len(all_results)} filas → {excel_filename}\n")
        return df_resultados, df_resumen
    
# ===========================================================
# Temporalidad
# ===========================================================

class SingleScenarioTester:
    """
    Clase para probar un único escenario (ARMA, ARIMA o SETAR) y generar
    un reporte detallado con ECRPS y tiempos de ejecución por predicción.
    """
    
    N_TEST_STEPS = 12
    N_VALIDATION = 40
    N_TRAIN = 200
    
    def __init__(self, n_boot: int = 1000, seed: int = 42, verbose: bool = True):
        self.n_boot = n_boot
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
    
    def test_arma_scenario(self, phi: list, theta: list, noise_dist: str, 
                          variance: float, excel_filename: str = None):
        """
        Prueba un escenario ARMA individual.
        
        Args:
            phi: Coeficientes AR
            theta: Coeficientes MA
            noise_dist: Distribución del ruido ('normal', 'uniform', etc.)
            variance: Varianza del ruido
            excel_filename: Nombre del archivo Excel de salida
        """
        config_name = self._get_arma_name(phi, theta)
        if excel_filename is None:
            excel_filename = f"test_ARMA_{config_name}_{noise_dist}_V{variance}.xlsx"
        
        simulator = ARMASimulation(
            phi=phi, theta=theta,
            noise_dist=noise_dist, sigma=np.sqrt(variance), seed=self.seed
        )
        
        return self._run_test(simulator, config_name, noise_dist, variance, 
                             excel_filename, "ARMA")
    
    def test_arima_scenario(self, phi: list, theta: list, noise_dist: str, 
                           variance: float, excel_filename: str = None):
        """
        Prueba un escenario ARIMA individual.
        
        Args:
            phi: Coeficientes AR
            theta: Coeficientes MA
            noise_dist: Distribución del ruido
            variance: Varianza del ruido
            excel_filename: Nombre del archivo Excel de salida
        """
        config_name = self._get_arima_name(phi, theta)
        if excel_filename is None:
            excel_filename = f"test_ARIMA_{config_name}_{noise_dist}_V{variance}.xlsx"
        
        simulator = ARIMASimulation(
            phi=phi, theta=theta,
            noise_dist=noise_dist, sigma=np.sqrt(variance), seed=self.seed
        )
        
        return self._run_test(simulator, config_name, noise_dist, variance, 
                             excel_filename, "ARIMA")
    
    def test_setar_scenario(self, phi_regime1: list, phi_regime2: list, 
                           threshold: float, delay: int, noise_dist: str, 
                           variance: float, excel_filename: str = None):
        """
        Prueba un escenario SETAR individual.
        
        Args:
            phi_regime1: Coeficientes del régimen 1
            phi_regime2: Coeficientes del régimen 2
            threshold: Umbral de cambio de régimen
            delay: Retardo para el umbral
            noise_dist: Distribución del ruido
            variance: Varianza del ruido
            excel_filename: Nombre del archivo Excel de salida
        """
        config_name = f"SETAR(2;{len(phi_regime1)},{len(phi_regime2)})_d{delay}_r{threshold}"
        if excel_filename is None:
            excel_filename = f"test_SETAR_{config_name.replace('.', '_')}_{noise_dist}_V{variance}.xlsx"
        
        simulator = SETARSimulation(
            model_type=config_name,
            phi_regime1=phi_regime1,
            phi_regime2=phi_regime2,
            threshold=threshold,
            delay=delay,
            noise_dist=noise_dist,
            sigma=np.sqrt(variance),
            seed=self.seed
        )
        
        return self._run_test(simulator, config_name, noise_dist, variance, 
                             excel_filename, "SETAR")
    
    def _run_test(self, simulator, config_name: str, noise_dist: str, 
                  variance: float, excel_filename: str, model_type: str):
        """Ejecuta el test completo para un escenario."""
        
        print(f"\n{'='*70}")
        print(f"🧪 Probando escenario {model_type}: {config_name}")
        print(f"   Distribución: {noise_dist} | Varianza: {variance}")
        print(f"{'='*70}\n")
        
        # 1. Generar serie temporal
        total_len = self.N_TRAIN + self.N_VALIDATION + self.N_TEST_STEPS
        series, errors = simulator.simulate(n=total_len, burn_in=100)
        
        train_data = series[:self.N_TRAIN]
        val_data = series[self.N_TRAIN : self.N_TRAIN + self.N_VALIDATION]
        
        # 2. Configurar modelos
        models = self._setup_models()
        
        # 3. Optimización de hiperparámetros
        print("⚙️  Optimizando hiperparámetros...")
        opt_start = time.time()
        optimizer = TimeBalancedOptimizer(random_state=self.seed, verbose=self.verbose)
        best_params = optimizer.optimize_all_models(models, train_data, val_data)
        opt_time = time.time() - opt_start
        print(f"✅ Optimización completada en {opt_time:.2f}s\n")
        
        # 4. Aplicar parámetros óptimos y congelar modelos
        train_val_full = series[:self.N_TRAIN + self.N_VALIDATION]
        model_total_times = {}
        
        for name, model in models.items():
            freeze_start = time.time()
            if name in best_params:
                for k, v in best_params[name].items():
                    if hasattr(model, k): 
                        setattr(model, k, v)
            if hasattr(model, 'freeze_hyperparameters'):
                model.freeze_hyperparameters(train_val_full)
            model_total_times[name] = time.time() - freeze_start
        
        # 5. Testing con métricas detalladas
        print("🔬 Iniciando pruebas en ventana deslizante...\n")
        results = []
        
        for t in range(self.N_TEST_STEPS):
            print(f"   Paso {t+1}/{self.N_TEST_STEPS}...", end=" ")
            step_start = time.time()
            
            idx = self.N_TRAIN + self.N_VALIDATION + t
            h_series = series[:idx]
            h_errors = errors[:idx]
            
            # Obtener distribución teórica verdadera
            true_samples = simulator.get_true_next_step_samples(
                h_series, h_errors, n_samples=1000
            )
            
            row = {
                'Paso': t + 1,
                'Configuración': config_name,
                'Tipo': model_type,
                'Distribución': noise_dist,
                'Varianza': variance,
                'Valor_Real': series[idx] if idx < len(series) else np.nan
            }
            
            # Evaluar cada modelo
            for name, model in models.items():
                model_step_start = time.time()
                
                try:
                    # Predicción
                    if "Bootstrap" in name:
                        pred = model.fit_predict(h_series)
                    else:
                        pred = model.fit_predict(pd.DataFrame({'valor': h_series}))
                    
                    pred_array = np.asarray(pred).flatten()
                    
                    # Calcular ECRPS
                    ecrps_value = ecrps(pred_array, true_samples)
                    
                    # Guardar métricas
                    row[f'{name}_ECRPS'] = ecrps_value
                    row[f'{name}_Tiempo_s'] = time.time() - model_step_start
                    
                    # Acumular tiempo total del modelo
                    model_total_times[name] += row[f'{name}_Tiempo_s']
                    
                except Exception as e:
                    if self.verbose:
                        print(f"\n   ⚠️  Error en {name}: {str(e)}")
                    row[f'{name}_ECRPS'] = np.nan
                    row[f'{name}_Tiempo_s'] = np.nan
            
            results.append(row)
            print(f"✓ ({time.time() - step_start:.2f}s)")
        
        # 6. Agregar fila con tiempos totales
        total_row = {
            'Paso': 'TOTAL',
            'Configuración': config_name,
            'Tipo': model_type,
            'Distribución': noise_dist,
            'Varianza': variance,
            'Valor_Real': np.nan
        }
        
        for name in models.keys():
            total_row[f'{name}_ECRPS'] = np.nan
            total_row[f'{name}_Tiempo_s'] = model_total_times.get(name, np.nan)
        
        results.append(total_row)
        
        # 7. Crear DataFrame y guardar
        df = pd.DataFrame(results)
        
        # Crear directorio si no existe
        os.makedirs('test_results', exist_ok=True)
        filepath = os.path.join('test_results', excel_filename)
        
        # Guardar con formato
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Resultados', index=False)
            
            # Agregar hoja de resumen
            summary = self._create_summary(df, model_total_times, opt_time)
            summary.to_excel(writer, sheet_name='Resumen', index=False)
        
        print(f"\n✅ Resultados guardados en: {filepath}")
        print(f"\n{'='*70}\n")
        
        clear_all_sessions()
        return df
    
    def _create_summary(self, df, model_times, opt_time):
        """Crea un resumen estadístico de los resultados."""
        models = [col.replace('_ECRPS', '') for col in df.columns if col.endswith('_ECRPS')]
        
        summary_data = []
        for model in models:
            ecrps_col = f'{model}_ECRPS'
            
            # Filtrar solo pasos de predicción (excluir 'TOTAL')
            pred_data = df[df['Paso'] != 'TOTAL'][ecrps_col]
            
            summary_data.append({
                'Modelo': model,
                'ECRPS_Promedio': pred_data.mean(),
                'ECRPS_Mediana': pred_data.median(),
                'ECRPS_Std': pred_data.std(),
                'ECRPS_Min': pred_data.min(),
                'ECRPS_Max': pred_data.max(),
                'Tiempo_Total_s': model_times.get(model, np.nan),
                'Tiempo_Promedio_por_Paso_s': model_times.get(model, np.nan) / self.N_TEST_STEPS
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('ECRPS_Promedio')
        
        # Agregar fila con tiempo de optimización
        opt_row = pd.DataFrame([{
            'Modelo': 'Optimización Hiperparámetros',
            'ECRPS_Promedio': np.nan,
            'ECRPS_Mediana': np.nan,
            'ECRPS_Std': np.nan,
            'ECRPS_Min': np.nan,
            'ECRPS_Max': np.nan,
            'Tiempo_Total_s': opt_time,
            'Tiempo_Promedio_por_Paso_s': np.nan
        }])
        
        summary_df = pd.concat([opt_row, summary_df], ignore_index=True)
        
        return summary_df
    
    def _setup_models(self):
        """Configura todos los modelos a probar."""
        return {
            'Block Bootstrapping': CircularBlockBootstrapModel(
                n_boot=self.n_boot, random_state=self.seed
            ),
            'Sieve Bootstrap': SieveBootstrapModel(
                n_boot=self.n_boot, random_state=self.seed
            ),
            'LSPM': LSPM(random_state=self.seed),
            'LSPMW': LSPMW(rho=0.95, random_state=self.seed),
            'AREPD': AREPD(n_lags=5, rho=0.93, random_state=self.seed),
            'MCPS': MondrianCPSModel(n_lags=10, random_state=self.seed),
            'AV-MCPS': AdaptiveVolatilityMondrianCPS(
                n_lags=12, random_state=self.seed
            ),
            'DeepAR': DeepARModel(
                hidden_size=16, n_lags=5, epochs=20, 
                num_samples=self.n_boot, random_state=self.seed,
                early_stopping_patience=3
            ),
            'EnCQR-LSTM': EnCQR_LSTM_Model(
                n_lags=15, B=3, units=24, epochs=15, 
                num_samples=self.n_boot, random_state=self.seed
            )
        }
    
    def _get_arma_name(self, phi, theta):
        """Genera nombre para configuración ARMA."""
        p = len(phi)
        q = len(theta)
        if p > 0 and q > 0:
            return f"ARMA({p},{q})"
        elif p > 0:
            return f"AR({p})"
        else:
            return f"MA({q})"
    
    def _get_arima_name(self, phi, theta):
        """Genera nombre para configuración ARIMA."""
        p = len(phi)
        q = len(theta)
        return f"ARIMA({p},1,{q})"

