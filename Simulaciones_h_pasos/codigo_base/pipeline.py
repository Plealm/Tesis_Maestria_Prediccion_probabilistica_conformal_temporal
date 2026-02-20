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


class PipelineARMA_100Trayectorias:
    """
    Pipeline ARMA que genera 100 trayectorias estocásticas para predicción 
    multi-paso (h-steps ahead) mediante muestreo recursivo.
    
    Compara distribuciones de modelos (100 trayectorias) contra la distribución 
    teórica del proceso (usando get_true_next_step_samples) usando ECRPS.
    """
    N_TEST_STEPS = 12
    N_VALIDATION = 40
    N_TRAIN = 200
    N_TRAJECTORIES_MODEL = 100    # Trayectorias por modelo
    N_TRAJECTORIES_TRUE = 1000    # Muestras para la "Verdad Teórica" por paso (para compatibilidad)
    N_SAMPLES_TRUE = 1000         # Alias de N_TRAJECTORIES_TRUE

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

    def _setup_models(self, seed):
        """Inicializa los modelos con configuración estándar."""
        return {
            'LSPM': LSPM(random_state=seed),
            'DeepAR': DeepARModel(
                hidden_size=20, n_lags=10, epochs=25, 
                num_samples=self.n_boot,
                random_state=seed, early_stopping_patience=4
            ),
            'Sieve Bootstrap': SieveBootstrapModel(
                n_boot=self.n_boot, random_state=seed
            ),
            'MCPS': MondrianCPSModel(n_lags=10, random_state=seed)
        }

    def _generate_true_distribution_recursive(self, simulator, history_series, history_errors, steps):
        """
        Genera la 'Verdad Teórica' usando el método get_true_next_step_samples
        de forma recursiva para cada horizonte h.
        
        Args:
            simulator: Instancia de ARMASimulation con parámetros reales
            history_series: Serie histórica observada
            history_errors: Errores históricos del proceso
            steps: Número de pasos a proyectar
            
        Returns:
            np.ndarray: Matriz (N_SAMPLES_TRUE, steps) con distribución teórica por horizonte
        """
        true_forecasts = np.zeros((self.N_TRAJECTORIES_TRUE, steps))
        
        # Para cada muestra, generamos una trayectoria completa usando el método teórico
        for sample_idx in range(self.N_TRAJECTORIES_TRUE):
            current_series = history_series.copy()
            current_errors = history_errors.copy()
            
            for h in range(steps):
                # Obtener UNA muestra de la distribución teórica del siguiente paso
                next_step_samples = simulator.get_true_next_step_samples(
                    current_series, current_errors, n_samples=1
                )
                sampled_value = next_step_samples[0]
                
                # Guardar predicción
                true_forecasts[sample_idx, h] = sampled_value
                
                # Actualizar historial para el siguiente paso
                current_series = np.append(current_series, sampled_value)
                
                # Calcular el error implícito del paso actual
                # Para ARMA: error = innovación que generó este valor
                # Aproximación: usar residuo si no hay método específico
                if hasattr(simulator, '_compute_error_from_observation'):
                    error = simulator._compute_error_from_observation(
                        sampled_value, current_series, current_errors
                    )
                else:
                    # Fallback: estimar error como residuo del modelo ajustado
                    # Para AR puro: error ≈ y_t - predicción AR
                    # Para MA: necesitamos inferirlo (más complejo)
                    p = len(simulator.phi) if hasattr(simulator, 'phi') else 0
                    if p > 0:
                        ar_pred = sum(simulator.phi[i] * current_series[-(i+2)] 
                                    for i in range(min(p, len(current_series)-1)))
                        error = sampled_value - ar_pred
                    else:
                        # Para MA puro o si no hay phi, asumir error ~ valor
                        error = sampled_value
                
                current_errors = np.append(current_errors, error)
        
        return true_forecasts

    def run_single_scenario(self, arma_config: dict, dist: str, var: float, rep: int) -> tuple:
        """
        Ejecuta un escenario completo: simula datos, optimiza modelos, genera trayectorias
        y calcula ECRPS contra distribución teórica.
        
        Args:
            arma_config: Configuración ARMA (phi, theta, nombre)
            dist: Distribución del ruido
            var: Varianza del ruido
            rep: ID del escenario para semilla
            
        Returns:
            tuple: (results_rows, plot_data, scenario_name, df_results)
                - results_rows: Lista de diccionarios con resultados por paso
                - plot_data: Dict con distribuciones por paso para graficar
                - scenario_name: Nombre identificador del escenario
                - df_results: DataFrame con resultados para PlotManager
        """
        scenario_seed = self.seed + rep
        scenario_name = f"{arma_config['nombre']}_{dist}_V{var}_S{scenario_seed}"
        
        if self.verbose:
            print(f"\n🔄 Procesando: {scenario_name}")
        
        # 1. SIMULACIÓN: Generar serie temporal completa con errores
        simulator = ARMASimulation(
            phi=arma_config['phi'], theta=arma_config['theta'],
            noise_dist=dist, sigma=np.sqrt(var), seed=scenario_seed
        )
        total_len = self.N_TRAIN + self.N_VALIDATION + self.N_TEST_STEPS
        full_series, full_errors = simulator.simulate(n=total_len, burn_in=100)
        
        train_series = full_series[:self.N_TRAIN]
        val_series = full_series[self.N_TRAIN:self.N_TRAIN + self.N_VALIDATION]
        train_val_combined = np.concatenate([train_series, val_series])
        
        # 2. OPTIMIZACIÓN: Entrenar y optimizar modelos
        models = self._setup_models(scenario_seed)
        optimizer = TimeBalancedOptimizer(random_state=scenario_seed, verbose=self.verbose)
        
        best_params = optimizer.optimize_all_models(models, train_series, val_series)
        
        # Aplicar mejores hiperparámetros y congelar
        for name, model in models.items():
            if name in best_params and best_params[name]:
                for k, v in best_params[name].items():
                    if hasattr(model, k): 
                        setattr(model, k, v)
            if hasattr(model, 'freeze_hyperparameters'):
                model.freeze_hyperparameters(train_val_combined)

        # 3. GENERACIÓN DE TRAYECTORIAS: 100 por modelo mediante aleatorización recursiva
        model_forecasts = {
            name: np.zeros((self.N_TRAJECTORIES_MODEL, self.N_TEST_STEPS)) 
            for name in models.keys()
        }
        
        for name, model in models.items():
            if self.verbose:
                print(f"  Generando {self.N_TRAJECTORIES_MODEL} trayectorias para {name}...")
            
            for i in range(self.N_TRAJECTORIES_MODEL):
                current_history = train_val_combined.copy()
                
                for h in range(self.N_TEST_STEPS):
                    if name == 'Sieve Bootstrap':
                        pred_dist = model.fit_predict(current_history)
                    else:
                        pred_dist = model.fit_predict(pd.DataFrame({'valor': current_history}))
                    
                    sampled_val = self.rng.choice(np.asarray(pred_dist).flatten())
                    model_forecasts[name][i, h] = sampled_val
                    current_history = np.append(current_history, sampled_val)
            
            clear_all_sessions()

        # 4. GENERAR DISTRIBUCIÓN TEÓRICA USANDO get_true_next_step_samples
        train_val_errors = full_errors[:self.N_TRAIN + self.N_VALIDATION]
        
        if self.verbose:
            print(f"  Generando distribución teórica (método exacto con errores)...")
        
        true_dist_paths = self._generate_true_distribution_recursive(
            simulator, train_val_combined, train_val_errors, self.N_TEST_STEPS
        )

        # 5. PREPARAR DATOS PARA GRÁFICOS (formato compatible con PlotManager)
        plot_data = {}
        for h in range(self.N_TEST_STEPS):
            plot_data[h] = {
                'true_distribution': true_dist_paths[:, h],
                'model_predictions': {
                    name: model_forecasts[name][:, h]
                    for name in models.keys()
                }
            }

        # 6. CÁLCULO DE MÉTRICAS: ECRPS por cada horizonte h
        results_rows = []
        for h in range(self.N_TEST_STEPS):
            # FORMATO COMPATIBLE CON run_analysis():
            # Usar 'Paso_H', 'Proceso', 'Distribución', 'Varianza'
            row = {
                'Paso': h + 1,           # Para PlotManager
                'Paso_H': h + 1,         # Para run_analysis
                'Config': arma_config['nombre'],
                'Proceso': arma_config['nombre'],      # Para run_analysis
                'Dist': dist,
                'Distribución': dist,    # Para run_analysis
                'Var': var,
                'Varianza': var          # Para run_analysis
            }
            
            true_samples_h = true_dist_paths[:, h]
            
            for name in models.keys():
                model_samples_h = model_forecasts[name][:, h]
                row[name] = ecrps(model_samples_h, true_samples_h)
            
            results_rows.append(row)
        
        # 7. CREAR DATAFRAME PARA PLOTMANAGER
        df_results = pd.DataFrame(results_rows)
        
        # 8. GENERAR GRÁFICOS USANDO PLOTMANAGER
        for model_name in models.keys():
            output_path = f"reportes/{scenario_name}/{model_name.replace(' ', '_')}.png"
            PlotManager.plot_individual_model_evolution(
                scenario_name, model_name, plot_data, df_results, output_path
            )
        
        return results_rows, plot_data, scenario_name, df_results

    def run_all(self, excel_filename="resultados_trayectorias_ARMA.xlsx", 
                n_jobs=4, batch_size=10):
        """
        Ejecuta todos los escenarios en paralelo y guarda resultados en Excel.
        
        Args:
            excel_filename: Nombre del archivo Excel de salida
            n_jobs: Número de procesos paralelos
            batch_size: Tamaño de lote para procesamiento y guardado intermedio
            
        Returns:
            pd.DataFrame: DataFrame con todos los resultados
        """
        # Generar lista de escenarios
        scenarios = []
        scenario_id = 0
        for arma_cfg in self.ARMA_CONFIGS:
            for dist in self.DISTRIBUTIONS:
                for var in self.VARIANCES:
                    scenarios.append((arma_cfg.copy(), dist, var, scenario_id))
                    scenario_id += 1
        
        print("="*80)
        print(f"🚀 INICIANDO EVALUACIÓN DE {len(scenarios)} ESCENARIOS")
        print("="*80)
        print(f"📊 Configuración:")
        print(f"   - Trayectorias por modelo: {self.N_TRAJECTORIES_MODEL}")
        print(f"   - Muestras teóricas por paso: {self.N_TRAJECTORIES_TRUE}")
        print(f"   - Pasos de predicción: {self.N_TEST_STEPS}")
        print(f"   - Procesos ARMA: {len(self.ARMA_CONFIGS)}")
        print(f"   - Distribuciones: {len(self.DISTRIBUTIONS)}")
        print(f"   - Varianzas: {len(self.VARIANCES)}")
        print(f"   - Procesos paralelos: {n_jobs}")
        print(f"   - Tamaño de lote: {batch_size}")
        print()
        
        # Procesamiento en lotes con guardado intermedio
        all_results = []
        all_predictions = {}
        
        for i in range(0, len(scenarios), batch_size):
            batch = scenarios[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(scenarios) + batch_size - 1) // batch_size
            
            print(f"📦 Procesando lote {batch_num}/{total_batches}...")
            
            results = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(self.run_single_scenario)(*s) for s in batch
            )
            
            # Consolidar resultados del lote
            for idx, (rows, preds, name, _) in enumerate(results):
                all_results.extend(rows)
                all_predictions[name] = preds
            
            # Guardado intermedio
            temp_df = pd.DataFrame(all_results)
            temp_df.to_excel(excel_filename, index=False)
            print(f"   ✅ Lote {batch_num} completado. Guardado intermedio en {excel_filename}")
        
        # Guardado final
        final_df = pd.DataFrame(all_results)
        final_df.to_excel(excel_filename, index=False)
        
        print()
        print("="*80)
        print("✅ PROCESO COMPLETADO")
        print("="*80)
        print(f"📁 Resultados guardados en: {excel_filename}")
        print(f"📊 Total de filas: {len(final_df)}")
        print(f"📈 Columnas: {list(final_df.columns)}")
        print()
        
        # Almacenar predicciones para uso posterior
        self._predictions_cache = all_predictions
        
        return final_df

    def get_predictions_dict(self):
        """
        Retorna el diccionario de predicciones del último run_all().
        Útil para generar gráficos después de la ejecución.
        """
        if hasattr(self, '_predictions_cache'):
            return self._predictions_cache
        else:
            raise ValueError("No hay predicciones disponibles. Ejecuta run_all() primero.")
        

class PipelineARIMA_100Trayectorias:
    """
    Pipeline ARIMA que genera 100 trayectorias estocásticas para predicción 
    multi-paso (h-steps ahead) mediante muestreo recursivo.
    
    Compara distribuciones de modelos (100 trayectorias) contra la distribución 
    teórica del proceso (usando get_true_next_step_samples) usando ECRPS.
    """
    N_TEST_STEPS = 12
    N_VALIDATION = 40
    N_TRAIN = 200
    N_TRAJECTORIES_MODEL = 100    # Trayectorias por modelo
    N_TRAJECTORIES_TRUE = 1000    # Muestras para la "Verdad Teórica" por paso
    N_SAMPLES_TRUE = 1000         # Alias de N_TRAJECTORIES_TRUE

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

    def _setup_models(self, seed):
        """Inicializa los modelos con configuración estándar."""
        return {
            'LSPM': LSPM(random_state=seed),
            'DeepAR': DeepARModel(
                hidden_size=20, n_lags=10, epochs=25, 
                num_samples=self.n_boot,
                random_state=seed, early_stopping_patience=4
            ),
            'Sieve Bootstrap': SieveBootstrapModel(
                n_boot=self.n_boot, random_state=seed
            ),
            'MCPS': MondrianCPSModel(n_lags=10, random_state=seed)
        }

    def _generate_true_distribution_recursive(self, simulator, history_series, history_errors, steps):
        """
        Genera la 'Verdad Teórica' usando el método get_true_next_step_samples
        de forma recursiva para cada horizonte h.
        
        Args:
            simulator: Instancia de ARIMASimulation con parámetros reales
            history_series: Serie histórica observada (integrada)
            history_errors: Errores históricos del proceso diferenciado
            steps: Número de pasos a proyectar
            
        Returns:
            np.ndarray: Matriz (N_TRAJECTORIES_TRUE, steps) con distribución teórica por horizonte
        """
        true_forecasts = np.zeros((self.N_TRAJECTORIES_TRUE, steps))
        
        # Para cada muestra, generamos una trayectoria completa usando el método teórico
        for sample_idx in range(self.N_TRAJECTORIES_TRUE):
            current_series = history_series.copy()
            current_errors = history_errors.copy()
            
            for h in range(steps):
                # Obtener UNA muestra de la distribución teórica del siguiente paso
                next_step_samples = simulator.get_true_next_step_samples(
                    current_series, current_errors, n_samples=1
                )
                sampled_value = next_step_samples[0]
                
                # Guardar predicción
                true_forecasts[sample_idx, h] = sampled_value
                
                # Actualizar historial para el siguiente paso
                current_series = np.append(current_series, sampled_value)
                
                # Calcular el error implícito del paso actual
                # Para ARIMA: error en la serie diferenciada
                if hasattr(simulator, '_compute_error_from_observation'):
                    error = simulator._compute_error_from_observation(
                        sampled_value, current_series, current_errors
                    )
                else:
                    # Fallback: estimar error como residuo del modelo ajustado
                    # Calcular la diferencia de primer orden
                    if len(current_series) >= 2:
                        diff_value = sampled_value - current_series[-2]
                    else:
                        diff_value = sampled_value
                    
                    # Para la parte AR del modelo diferenciado
                    p = len(simulator.phi) if hasattr(simulator, 'phi') else 0
                    if p > 0 and len(current_errors) >= p:
                        # Calcular serie diferenciada histórica
                        diff_series = np.diff(current_series[:-1]) if len(current_series) > 2 else np.array([0])
                        ar_pred = sum(simulator.phi[i] * diff_series[-(i+1)] 
                                    for i in range(min(p, len(diff_series))))
                        error = diff_value - ar_pred
                    else:
                        # Para MA puro o sin suficiente historia
                        error = diff_value
                
                current_errors = np.append(current_errors, error)
        
        return true_forecasts

    def run_single_scenario(self, arima_config: dict, dist: str, var: float, rep: int) -> tuple:
        """
        Ejecuta un escenario completo: simula datos ARIMA, optimiza modelos, genera trayectorias
        y calcula ECRPS contra distribución teórica.
        
        Args:
            arima_config: Configuración ARIMA (phi, theta, nombre)
            dist: Distribución del ruido
            var: Varianza del ruido
            rep: ID del escenario para semilla
            
        Returns:
            tuple: (results_rows, plot_data, scenario_name, df_results)
                - results_rows: Lista de diccionarios con resultados por paso
                - plot_data: Dict con distribuciones por paso para graficar
                - scenario_name: Nombre identificador del escenario
                - df_results: DataFrame con resultados para PlotManager
        """
        scenario_seed = self.seed + rep
        scenario_name = f"{arima_config['nombre']}_{dist}_V{var}_S{scenario_seed}"
        
        if self.verbose:
            print(f"\n🔄 Procesando: {scenario_name}")
        
        # 1. SIMULACIÓN ARIMA: Generar serie temporal integrada con errores
        simulator = ARIMASimulation(
            phi=arima_config['phi'], theta=arima_config['theta'],
            noise_dist=dist, sigma=np.sqrt(var), seed=scenario_seed
        )
        total_len = self.N_TRAIN + self.N_VALIDATION + self.N_TEST_STEPS
        full_series, full_errors = simulator.simulate(n=total_len, burn_in=100)
        
        train_series = full_series[:self.N_TRAIN]
        val_series = full_series[self.N_TRAIN:self.N_TRAIN + self.N_VALIDATION]
        train_val_combined = np.concatenate([train_series, val_series])
        
        # 2. OPTIMIZACIÓN: Entrenar y optimizar modelos
        models = self._setup_models(scenario_seed)
        optimizer = TimeBalancedOptimizer(random_state=scenario_seed, verbose=self.verbose)
        
        best_params = optimizer.optimize_all_models(models, train_series, val_series)
        
        # Aplicar mejores hiperparámetros y congelar
        for name, model in models.items():
            if name in best_params and best_params[name]:
                for k, v in best_params[name].items():
                    if hasattr(model, k): 
                        setattr(model, k, v)
            if hasattr(model, 'freeze_hyperparameters'):
                model.freeze_hyperparameters(train_val_combined)

        # 3. GENERACIÓN DE TRAYECTORIAS: 100 por modelo mediante aleatorización recursiva
        model_forecasts = {
            name: np.zeros((self.N_TRAJECTORIES_MODEL, self.N_TEST_STEPS)) 
            for name in models.keys()
        }
        
        for name, model in models.items():
            if self.verbose:
                print(f"  Generando {self.N_TRAJECTORIES_MODEL} trayectorias para {name}...")
            
            for i in range(self.N_TRAJECTORIES_MODEL):
                current_history = train_val_combined.copy()
                
                for h in range(self.N_TEST_STEPS):
                    if name == 'Sieve Bootstrap':
                        pred_dist = model.fit_predict(current_history)
                    else:
                        pred_dist = model.fit_predict(pd.DataFrame({'valor': current_history}))
                    
                    sampled_val = self.rng.choice(np.asarray(pred_dist).flatten())
                    model_forecasts[name][i, h] = sampled_val
                    current_history = np.append(current_history, sampled_val)
            
            clear_all_sessions()

        # 4. GENERAR DISTRIBUCIÓN TEÓRICA USANDO get_true_next_step_samples
        train_val_errors = full_errors[:self.N_TRAIN + self.N_VALIDATION]
        
        if self.verbose:
            print(f"  Generando distribución teórica (método exacto con errores)...")
        
        true_dist_paths = self._generate_true_distribution_recursive(
            simulator, train_val_combined, train_val_errors, self.N_TEST_STEPS
        )

        # 5. PREPARAR DATOS PARA GRÁFICOS (formato compatible con PlotManager)
        plot_data = {}
        for h in range(self.N_TEST_STEPS):
            plot_data[h] = {
                'true_distribution': true_dist_paths[:, h],
                'model_predictions': {
                    name: model_forecasts[name][:, h]
                    for name in models.keys()
                }
            }

        # 6. CÁLCULO DE MÉTRICAS: ECRPS por cada horizonte h
        results_rows = []
        for h in range(self.N_TEST_STEPS):
            # FORMATO COMPATIBLE CON run_analysis():
            # Usar 'Paso_H', 'Proceso', 'Distribución', 'Varianza'
            row = {
                'Paso': h + 1,           # Para PlotManager
                'Paso_H': h + 1,         # Para run_analysis
                'Config': arima_config['nombre'],
                'Proceso': arima_config['nombre'],      # Para run_analysis
                'Dist': dist,
                'Distribución': dist,    # Para run_analysis
                'Var': var,
                'Varianza': var          # Para run_analysis
            }
            
            true_samples_h = true_dist_paths[:, h]
            
            for name in models.keys():
                model_samples_h = model_forecasts[name][:, h]
                row[name] = ecrps(model_samples_h, true_samples_h)
            
            results_rows.append(row)
        
        # 7. CREAR DATAFRAME PARA PLOTMANAGER
        df_results = pd.DataFrame(results_rows)
        
        # 8. GENERAR GRÁFICOS USANDO PLOTMANAGER
        for model_name in models.keys():
            output_path = f"reportes_arima/{scenario_name}/{model_name.replace(' ', '_')}.png"
            PlotManager.plot_individual_model_evolution(
                scenario_name, model_name, plot_data, df_results, output_path
            )
        
        return results_rows, plot_data, scenario_name, df_results

    def run_all(self, excel_filename="resultados_trayectorias_ARIMA.xlsx", 
                n_jobs=4, batch_size=10):
        """
        Ejecuta todos los escenarios en paralelo y guarda resultados en Excel.
        
        Args:
            excel_filename: Nombre del archivo Excel de salida
            n_jobs: Número de procesos paralelos
            batch_size: Tamaño de lote para procesamiento y guardado intermedio
            
        Returns:
            pd.DataFrame: DataFrame con todos los resultados
        """
        # Generar lista de escenarios
        scenarios = []
        scenario_id = 0
        for arima_cfg in self.ARIMA_CONFIGS:
            for dist in self.DISTRIBUTIONS:
                for var in self.VARIANCES:
                    scenarios.append((arima_cfg.copy(), dist, var, scenario_id))
                    scenario_id += 1
        
        print("="*80)
        print(f"🚀 INICIANDO EVALUACIÓN DE {len(scenarios)} ESCENARIOS ARIMA")
        print("="*80)
        print(f"📊 Configuración:")
        print(f"   - Trayectorias por modelo: {self.N_TRAJECTORIES_MODEL}")
        print(f"   - Muestras teóricas por paso: {self.N_TRAJECTORIES_TRUE}")
        print(f"   - Pasos de predicción: {self.N_TEST_STEPS}")
        print(f"   - Procesos ARIMA: {len(self.ARIMA_CONFIGS)}")
        print(f"   - Distribuciones: {len(self.DISTRIBUTIONS)}")
        print(f"   - Varianzas: {len(self.VARIANCES)}")
        print(f"   - Procesos paralelos: {n_jobs}")
        print(f"   - Tamaño de lote: {batch_size}")
        print()
        
        # Procesamiento en lotes con guardado intermedio
        all_results = []
        all_predictions = {}
        
        for i in range(0, len(scenarios), batch_size):
            batch = scenarios[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(scenarios) + batch_size - 1) // batch_size
            
            print(f"📦 Procesando lote {batch_num}/{total_batches}...")
            
            results = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(self.run_single_scenario)(*s) for s in batch
            )
            
            # Consolidar resultados del lote
            for idx, (rows, preds, name, _) in enumerate(results):
                all_results.extend(rows)
                all_predictions[name] = preds
            
            # Guardado intermedio
            temp_df = pd.DataFrame(all_results)
            temp_df.to_excel(excel_filename, index=False)
            print(f"   ✅ Lote {batch_num} completado. Guardado intermedio en {excel_filename}")
        
        # Guardado final
        final_df = pd.DataFrame(all_results)
        final_df.to_excel(excel_filename, index=False)
        
        print()
        print("="*80)
        print("✅ PROCESO COMPLETADO")
        print("="*80)
        print(f"📁 Resultados guardados en: {excel_filename}")
        print(f"📊 Total de filas: {len(final_df)}")
        print(f"📈 Columnas: {list(final_df.columns)}")
        print()
        
        # Almacenar predicciones para uso posterior
        self._predictions_cache = all_predictions
        
        return final_df

    def get_predictions_dict(self):
        """
        Retorna el diccionario de predicciones del último run_all().
        Útil para generar gráficos después de la ejecución.
        """
        if hasattr(self, '_predictions_cache'):
            return self._predictions_cache
        else:
            raise ValueError("No hay predicciones disponibles. Ejecuta run_all() primero.")
        
class PipelineSETAR_100Trayectorias:
    """
    Pipeline SETAR que genera 100 trayectorias estocásticas para predicción 
    multi-paso (h-steps ahead) mediante muestreo recursivo.
    
    Compara distribuciones de modelos (100 trayectorias) contra la distribución 
    teórica del proceso (usando get_true_next_step_samples) usando ECRPS.
    """
    N_TEST_STEPS = 12
    N_VALIDATION = 40
    N_TRAIN = 200
    N_TRAJECTORIES_MODEL = 100    # Trayectorias por modelo
    N_TRAJECTORIES_TRUE = 1000    # Muestras para la "Verdad Teórica" por paso
    N_SAMPLES_TRUE = 1000         # Alias de N_TRAJECTORIES_TRUE

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

    def _setup_models(self, seed):
        """Inicializa los modelos con configuración estándar."""
        return {
            'LSPM': LSPM(random_state=seed),
            'DeepAR': DeepARModel(
                hidden_size=20, n_lags=10, epochs=25, 
                num_samples=self.n_boot,
                random_state=seed, early_stopping_patience=4
            ),
            'Sieve Bootstrap': SieveBootstrapModel(
                n_boot=self.n_boot, random_state=seed
            ),
            'MCPS': MondrianCPSModel(n_lags=10, random_state=seed)
        }

    def _generate_true_distribution_recursive(self, simulator, history_series, history_errors, steps):
        """
        Genera la 'Verdad Teórica' usando el método get_true_next_step_samples
        de forma recursiva para cada horizonte h.
        
        Args:
            simulator: Instancia de SETARSimulation con parámetros reales
            history_series: Serie histórica observada
            history_errors: Errores históricos del proceso
            steps: Número de pasos a proyectar
            
        Returns:
            np.ndarray: Matriz (N_TRAJECTORIES_TRUE, steps) con distribución teórica por horizonte
        """
        true_forecasts = np.zeros((self.N_TRAJECTORIES_TRUE, steps))
        
        # Para cada muestra, generamos una trayectoria completa usando el método teórico
        for sample_idx in range(self.N_TRAJECTORIES_TRUE):
            current_series = history_series.copy()
            current_errors = history_errors.copy()
            
            for h in range(steps):
                # Obtener UNA muestra de la distribución teórica del siguiente paso
                next_step_samples = simulator.get_true_next_step_samples(
                    current_series, current_errors, n_samples=1
                )
                sampled_value = next_step_samples[0]
                
                # Guardar predicción
                true_forecasts[sample_idx, h] = sampled_value
                
                # Actualizar historial para el siguiente paso
                current_series = np.append(current_series, sampled_value)
                
                # Calcular el error implícito del paso actual
                # Para SETAR: error depende del régimen activo
                if hasattr(simulator, '_compute_error_from_observation'):
                    error = simulator._compute_error_from_observation(
                        sampled_value, current_series, current_errors
                    )
                else:
                    # Fallback: estimar error como residuo del modelo ajustado
                    # Determinar régimen actual basado en threshold y delay
                    delay = simulator.delay
                    threshold = simulator.threshold
                    
                    if len(current_series) > delay:
                        regime_indicator = current_series[-(delay+1)]
                        if regime_indicator <= threshold:
                            phi = simulator.phi_regime1
                        else:
                            phi = simulator.phi_regime2
                    else:
                        phi = simulator.phi_regime1  # Default al régimen 1
                    
                    # Calcular predicción AR del régimen activo
                    p = len(phi)
                    if p > 0 and len(current_series) > p:
                        ar_pred = sum(phi[i] * current_series[-(i+2)] 
                                    for i in range(min(p, len(current_series)-1)))
                        error = sampled_value - ar_pred
                    else:
                        error = sampled_value
                
                current_errors = np.append(current_errors, error)
        
        return true_forecasts

    def run_single_scenario(self, setar_config: dict, dist: str, var: float, rep: int) -> tuple:
        """
        Ejecuta un escenario completo: simula datos SETAR, optimiza modelos, genera trayectorias
        y calcula ECRPS contra distribución teórica.
        
        Args:
            setar_config: Configuración SETAR (phi_regime1, phi_regime2, threshold, delay, nombre)
            dist: Distribución del ruido
            var: Varianza del ruido
            rep: ID del escenario para semilla
            
        Returns:
            tuple: (results_rows, plot_data, scenario_name, df_results)
                - results_rows: Lista de diccionarios con resultados por paso
                - plot_data: Dict con distribuciones por paso para graficar
                - scenario_name: Nombre identificador del escenario
                - df_results: DataFrame con resultados para PlotManager
        """
        scenario_seed = self.seed + rep
        scenario_name = f"{setar_config['nombre']}_{dist}_V{var}_S{scenario_seed}"
        
        if self.verbose:
            print(f"\n🔄 Procesando: {scenario_name}")
        
        # 1. SIMULACIÓN SETAR: Generar serie temporal con errores
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
        full_series, full_errors = simulator.simulate(n=total_len, burn_in=100)
        
        train_series = full_series[:self.N_TRAIN]
        val_series = full_series[self.N_TRAIN:self.N_TRAIN + self.N_VALIDATION]
        train_val_combined = np.concatenate([train_series, val_series])
        
        # 2. OPTIMIZACIÓN: Entrenar y optimizar modelos
        models = self._setup_models(scenario_seed)
        optimizer = TimeBalancedOptimizer(random_state=scenario_seed, verbose=self.verbose)
        
        best_params = optimizer.optimize_all_models(models, train_series, val_series)
        
        # Aplicar mejores hiperparámetros y congelar
        for name, model in models.items():
            if name in best_params and best_params[name]:
                for k, v in best_params[name].items():
                    if hasattr(model, k): 
                        setattr(model, k, v)
            if hasattr(model, 'freeze_hyperparameters'):
                model.freeze_hyperparameters(train_val_combined)

        # 3. GENERACIÓN DE TRAYECTORIAS: 100 por modelo mediante aleatorización recursiva
        model_forecasts = {
            name: np.zeros((self.N_TRAJECTORIES_MODEL, self.N_TEST_STEPS)) 
            for name in models.keys()
        }
        
        for name, model in models.items():
            if self.verbose:
                print(f"  Generando {self.N_TRAJECTORIES_MODEL} trayectorias para {name}...")
            
            for i in range(self.N_TRAJECTORIES_MODEL):
                current_history = train_val_combined.copy()
                
                for h in range(self.N_TEST_STEPS):
                    if name == 'Sieve Bootstrap':
                        pred_dist = model.fit_predict(current_history)
                    else:
                        pred_dist = model.fit_predict(pd.DataFrame({'valor': current_history}))
                    
                    sampled_val = self.rng.choice(np.asarray(pred_dist).flatten())
                    model_forecasts[name][i, h] = sampled_val
                    current_history = np.append(current_history, sampled_val)
            
            clear_all_sessions()

        # 4. GENERAR DISTRIBUCIÓN TEÓRICA USANDO get_true_next_step_samples
        train_val_errors = full_errors[:self.N_TRAIN + self.N_VALIDATION]
        
        if self.verbose:
            print(f"  Generando distribución teórica (método exacto con errores)...")
        
        true_dist_paths = self._generate_true_distribution_recursive(
            simulator, train_val_combined, train_val_errors, self.N_TEST_STEPS
        )

        # 5. PREPARAR DATOS PARA GRÁFICOS (formato compatible con PlotManager)
        plot_data = {}
        for h in range(self.N_TEST_STEPS):
            plot_data[h] = {
                'true_distribution': true_dist_paths[:, h],
                'model_predictions': {
                    name: model_forecasts[name][:, h]
                    for name in models.keys()
                }
            }

        # 6. CÁLCULO DE MÉTRICAS: ECRPS por cada horizonte h
        results_rows = []
        for h in range(self.N_TEST_STEPS):
            # FORMATO COMPATIBLE CON run_analysis():
            row = {
                'Paso': h + 1,           # Para PlotManager
                'Paso_H': h + 1,         # Para run_analysis
                'Config': setar_config['nombre'],
                'Proceso': setar_config['nombre'],      # Para run_analysis
                'Descripción': setar_config['description'],
                'Dist': dist,
                'Distribución': dist,    # Para run_analysis
                'Var': var,
                'Varianza': var          # Para run_analysis
            }
            
            true_samples_h = true_dist_paths[:, h]
            
            for name in models.keys():
                model_samples_h = model_forecasts[name][:, h]
                row[name] = ecrps(model_samples_h, true_samples_h)
            
            results_rows.append(row)
        
        # 7. CREAR DATAFRAME PARA PLOTMANAGER
        df_results = pd.DataFrame(results_rows)
        
        # 8. GENERAR GRÁFICOS USANDO PLOTMANAGER
        for model_name in models.keys():
            output_path = f"reportes_setar/{scenario_name}/{model_name.replace(' ', '_')}.png"
            PlotManager.plot_individual_model_evolution(
                scenario_name, model_name, plot_data, df_results, output_path
            )
        
        return results_rows, plot_data, scenario_name, df_results

    def run_all(self, excel_filename="resultados_trayectorias_SETAR.xlsx", 
                n_jobs=4, batch_size=10):
        """
        Ejecuta todos los escenarios en paralelo y guarda resultados en Excel.
        
        Args:
            excel_filename: Nombre del archivo Excel de salida
            n_jobs: Número de procesos paralelos
            batch_size: Tamaño de lote para procesamiento y guardado intermedio
            
        Returns:
            pd.DataFrame: DataFrame con todos los resultados
        """
        # Generar lista de escenarios
        scenarios = []
        scenario_id = 0
        for setar_cfg in self.SETAR_CONFIGS:
            for dist in self.DISTRIBUTIONS:
                for var in self.VARIANCES:
                    scenarios.append((setar_cfg.copy(), dist, var, scenario_id))
                    scenario_id += 1
        
        print("="*80)
        print(f"🚀 INICIANDO EVALUACIÓN DE {len(scenarios)} ESCENARIOS SETAR")
        print("="*80)
        print(f"📊 Configuración:")
        print(f"   - Trayectorias por modelo: {self.N_TRAJECTORIES_MODEL}")
        print(f"   - Muestras teóricas por paso: {self.N_TRAJECTORIES_TRUE}")
        print(f"   - Pasos de predicción: {self.N_TEST_STEPS}")
        print(f"   - Procesos SETAR: {len(self.SETAR_CONFIGS)}")
        print(f"   - Distribuciones: {len(self.DISTRIBUTIONS)}")
        print(f"   - Varianzas: {len(self.VARIANCES)}")
        print(f"   - Procesos paralelos: {n_jobs}")
        print(f"   - Tamaño de lote: {batch_size}")
        print()
        
        # Procesamiento en lotes con guardado intermedio
        all_results = []
        all_predictions = {}
        
        for i in range(0, len(scenarios), batch_size):
            batch = scenarios[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(scenarios) + batch_size - 1) // batch_size
            
            print(f"📦 Procesando lote {batch_num}/{total_batches}...")
            
            results = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(self.run_single_scenario)(*s) for s in batch
            )
            
            # Consolidar resultados del lote
            for idx, (rows, preds, name, _) in enumerate(results):
                all_results.extend(rows)
                all_predictions[name] = preds
            
            # Guardado intermedio
            temp_df = pd.DataFrame(all_results)
            temp_df.to_excel(excel_filename, index=False)
            print(f"   ✅ Lote {batch_num} completado. Guardado intermedio en {excel_filename}")
        
        # Guardado final
        final_df = pd.DataFrame(all_results)
        final_df.to_excel(excel_filename, index=False)
        
        print()
        print("="*80)
        print("✅ PROCESO COMPLETADO")
        print("="*80)
        print(f"📁 Resultados guardados en: {excel_filename}")
        print(f"📊 Total de filas: {len(final_df)}")
        print(f"📈 Columnas: {list(final_df.columns)}")
        print()
        
        # Almacenar predicciones para uso posterior
        self._predictions_cache = all_predictions
        
        return final_df

    def get_predictions_dict(self):
        """
        Retorna el diccionario de predicciones del último run_all().
        Útil para generar gráficos después de la ejecución.
        """
        if hasattr(self, '_predictions_cache'):
            return self._predictions_cache
        else:
            raise ValueError("No hay predicciones disponibles. Ejecuta run_all() primero.")
        

class PipelineUnified_LSPMW:
    """
    Pipeline unificado que ejecuta ARMA, ARIMA y SETAR en paralelo, 
    evaluando únicamente el modelo LSPMW contra la distribución teórica.
    
    Genera 100 trayectorias por escenario y compara con distribución teórica
    usando ECRPS en predicción multi-paso (h-steps ahead).
    """
    N_TEST_STEPS = 12
    N_VALIDATION = 40
    N_TRAIN = 200
    N_TRAJECTORIES_MODEL = 100    # Trayectorias del modelo LSPMW
    N_TRAJECTORIES_TRUE = 1000    # Muestras para distribución teórica
    N_SAMPLES_TRUE = 1000

    # Configuraciones de todos los procesos
    ARMA_CONFIGS = [
        {'tipo': 'ARMA', 'nombre': 'AR(1)', 'phi': [0.9], 'theta': []},
        {'tipo': 'ARMA', 'nombre': 'AR(2)', 'phi': [0.5, -0.3], 'theta': []},
        {'tipo': 'ARMA', 'nombre': 'MA(1)', 'phi': [], 'theta': [0.7]},
        {'tipo': 'ARMA', 'nombre': 'MA(2)', 'phi': [], 'theta': [0.4, 0.2]},
        {'tipo': 'ARMA', 'nombre': 'ARMA(1,1)', 'phi': [0.6], 'theta': [0.3]},
        {'tipo': 'ARMA', 'nombre': 'ARMA(2,2)', 'phi': [0.4, -0.2], 'theta': [0.5, 0.1]},
        {'tipo': 'ARMA', 'nombre': 'ARMA(2,1)', 'phi': [0.7, 0.2], 'theta': [0.5]}
    ]
    
    ARIMA_CONFIGS = [
        {'tipo': 'ARIMA', 'nombre': 'ARIMA(0,1,0)', 'phi': [], 'theta': []},
        {'tipo': 'ARIMA', 'nombre': 'ARIMA(1,1,0)', 'phi': [0.6], 'theta': []},
        {'tipo': 'ARIMA', 'nombre': 'ARIMA(2,1,0)', 'phi': [0.5, -0.2], 'theta': []},
        {'tipo': 'ARIMA', 'nombre': 'ARIMA(0,1,1)', 'phi': [], 'theta': [0.5]},
        {'tipo': 'ARIMA', 'nombre': 'ARIMA(0,1,2)', 'phi': [], 'theta': [0.4, 0.25]},
        {'tipo': 'ARIMA', 'nombre': 'ARIMA(1,1,1)', 'phi': [0.7], 'theta': [-0.3]},
        {'tipo': 'ARIMA', 'nombre': 'ARIMA(2,1,2)', 'phi': [0.6, 0.2], 'theta': [0.4, -0.1]}
    ]
    
    SETAR_CONFIGS = [
        {'tipo': 'SETAR', 'nombre': 'SETAR-1', 'phi_regime1': [0.6], 'phi_regime2': [-0.5], 
         'threshold': 0.0, 'delay': 1, 'description': 'SETAR(2;1,1) d=1, r=0'},
        {'tipo': 'SETAR', 'nombre': 'SETAR-2', 'phi_regime1': [0.7], 'phi_regime2': [-0.7], 
         'threshold': 0.0, 'delay': 2, 'description': 'SETAR(2;1,1) d=2, r=0'},
        {'tipo': 'SETAR', 'nombre': 'SETAR-3', 'phi_regime1': [0.5, -0.2], 'phi_regime2': [-0.3, 0.1], 
         'threshold': 0.5, 'delay': 1, 'description': 'SETAR(2;2,2) d=1, r=0.5'},
        {'tipo': 'SETAR', 'nombre': 'SETAR-4', 'phi_regime1': [0.8, -0.15], 'phi_regime2': [-0.6, 0.2], 
         'threshold': 1.0, 'delay': 2, 'description': 'SETAR(2;2,2) d=2, r=1.0'},
        {'tipo': 'SETAR', 'nombre': 'SETAR-5', 'phi_regime1': [0.4, -0.1, 0.05], 'phi_regime2': [-0.3, 0.1, -0.05], 
         'threshold': 0.0, 'delay': 1, 'description': 'SETAR(2;3,3) d=1, r=0'},
        {'tipo': 'SETAR', 'nombre': 'SETAR-6', 'phi_regime1': [0.5, -0.3, 0.1], 'phi_regime2': [-0.4, 0.2, -0.05], 
         'threshold': 0.5, 'delay': 2, 'description': 'SETAR(2;3,3) d=2, r=0.5'},
        {'tipo': 'SETAR', 'nombre': 'SETAR-7', 'phi_regime1': [0.3, 0.1], 'phi_regime2': [-0.2, -0.1], 
         'threshold': 0.8, 'delay': 3, 'description': 'SETAR(2;2,2) d=3, r=0.8'}
    ]
    
    DISTRIBUTIONS = ['normal', 'uniform', 'exponential', 't-student', 'mixture']
    VARIANCES = [0.2, 0.5, 1.0, 3.0]

    def __init__(self, seed: int = 42, verbose: bool = False):
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)

    def _create_simulator(self, config: dict, dist: str, var: float, seed: int):
        """Crea el simulador apropiado según el tipo de proceso."""
        if config['tipo'] == 'ARMA':
            return ARMASimulation(
                phi=config['phi'], theta=config['theta'],
                noise_dist=dist, sigma=np.sqrt(var), seed=seed
            )
        elif config['tipo'] == 'ARIMA':
            return ARIMASimulation(
                phi=config['phi'], theta=config['theta'],
                noise_dist=dist, sigma=np.sqrt(var), seed=seed
            )
        elif config['tipo'] == 'SETAR':
            return SETARSimulation(
                model_type=config['nombre'],
                phi_regime1=config['phi_regime1'],
                phi_regime2=config['phi_regime2'],
                threshold=config['threshold'],
                delay=config['delay'],
                noise_dist=dist,
                sigma=np.sqrt(var),
                seed=seed
            )
        else:
            raise ValueError(f"Tipo de proceso desconocido: {config['tipo']}")

    def _generate_true_distribution_recursive(self, simulator, history_series, history_errors, steps):
        """
        Genera distribución teórica recursiva usando get_true_next_step_samples.
        Funciona para ARMA, ARIMA y SETAR.
        """
        true_forecasts = np.zeros((self.N_TRAJECTORIES_TRUE, steps))
        
        for sample_idx in range(self.N_TRAJECTORIES_TRUE):
            current_series = history_series.copy()
            current_errors = history_errors.copy()
            
            for h in range(steps):
                next_step_samples = simulator.get_true_next_step_samples(
                    current_series, current_errors, n_samples=1
                )
                sampled_value = next_step_samples[0]
                true_forecasts[sample_idx, h] = sampled_value
                
                current_series = np.append(current_series, sampled_value)
                
                # Calcular error implícito (simplificado)
                if hasattr(simulator, '_compute_error_from_observation'):
                    error = simulator._compute_error_from_observation(
                        sampled_value, current_series, current_errors
                    )
                else:
                    # Fallback básico
                    if len(current_series) >= 2:
                        error = sampled_value - current_series[-2]
                    else:
                        error = sampled_value
                
                current_errors = np.append(current_errors, error)
        
        return true_forecasts

    def run_single_scenario(self, config: dict, dist: str, var: float, rep: int) -> tuple:
        """
        Ejecuta un escenario: simula datos, entrena LSPM, genera trayectorias
        y calcula ECRPS contra distribución teórica.
        """
        scenario_seed = self.seed + rep
        scenario_name = f"{config['nombre']}_{dist}_V{var}_S{scenario_seed}"
        
        if self.verbose:
            print(f"\n🔄 [{config['tipo']}] Procesando: {scenario_name}")
        
        # 1. SIMULACIÓN
        simulator = self._create_simulator(config, dist, var, scenario_seed)
        total_len = self.N_TRAIN + self.N_VALIDATION + self.N_TEST_STEPS
        full_series, full_errors = simulator.simulate(n=total_len, burn_in=100)
        
        train_series = full_series[:self.N_TRAIN]
        val_series = full_series[self.N_TRAIN:self.N_TRAIN + self.N_VALIDATION]
        train_val_combined = np.concatenate([train_series, val_series])
        
        # 2. OPTIMIZAR Y ENTRENAR LSPMW
        lspmw_model = LSPMW(random_state=scenario_seed)
        
        # LSPMW requiere optimización específica usando train/validation
        # Usa TimeBalancedOptimizer para encontrar mejores hiperparámetros
        optimizer = TimeBalancedOptimizer(random_state=scenario_seed, verbose=self.verbose)
        
        models_dict = {'LSPMW': lspmw_model}
        best_params = optimizer.optimize_all_models(models_dict, train_series, val_series)
        
        # Aplicar mejores hiperparámetros encontrados
        if 'LSPMW' in best_params and best_params['LSPMW']:
            for k, v in best_params['LSPMW'].items():
                if hasattr(lspmw_model, k):
                    setattr(lspmw_model, k, v)
        
        # Congelar hiperparámetros con datos completos (train + validation)
        if hasattr(lspmw_model, 'freeze_hyperparameters'):
            lspmw_model.freeze_hyperparameters(train_val_combined)
        
        # 3. GENERAR TRAYECTORIAS LSPMW
        if self.verbose:
            print(f"  Generando {self.N_TRAJECTORIES_MODEL} trayectorias LSPMW...")
        
        lspmw_forecasts = np.zeros((self.N_TRAJECTORIES_MODEL, self.N_TEST_STEPS))
        
        for i in range(self.N_TRAJECTORIES_MODEL):
            current_history = train_val_combined.copy()
            
            for h in range(self.N_TEST_STEPS):
                pred_dist = lspmw_model.fit_predict(pd.DataFrame({'valor': current_history}))
                sampled_val = self.rng.choice(np.asarray(pred_dist).flatten())
                lspmw_forecasts[i, h] = sampled_val
                current_history = np.append(current_history, sampled_val)
        
        clear_all_sessions()
        
        # 4. GENERAR DISTRIBUCIÓN TEÓRICA
        train_val_errors = full_errors[:self.N_TRAIN + self.N_VALIDATION]
        
        if self.verbose:
            print(f"  Generando distribución teórica...")
        
        true_dist_paths = self._generate_true_distribution_recursive(
            simulator, train_val_combined, train_val_errors, self.N_TEST_STEPS
        )
        
        # 5. CALCULAR ECRPS POR HORIZONTE
        results_rows = []
        for h in range(self.N_TEST_STEPS):
            row = {
                'Tipo_Proceso': config['tipo'],
                'Paso': h + 1,
                'Paso_H': h + 1,
                'Config': config['nombre'],
                'Proceso': config['nombre'],
                'Dist': dist,
                'Distribución': dist,
                'Var': var,
                'Varianza': var,
                'Semilla': scenario_seed
            }
            
            if config['tipo'] == 'SETAR' and 'description' in config:
                row['Descripción'] = config['description']
            
            true_samples_h = true_dist_paths[:, h]
            model_samples_h = lspmw_forecasts[:, h]
            row['LSPMW'] = ecrps(model_samples_h, true_samples_h)
            
            results_rows.append(row)
        
        # 6. PREPARAR DATOS PARA GRÁFICOS
        plot_data = {}
        for h in range(self.N_TEST_STEPS):
            plot_data[h] = {
                'true_distribution': true_dist_paths[:, h],
                'model_predictions': {
                    'LSPMW': lspmw_forecasts[:, h]
                }
            }
        
        df_results = pd.DataFrame(results_rows)
        
        # 7. GENERAR GRÁFICO
        output_path = f"reportes_unified_lspmw/{config['tipo']}/{scenario_name}/LSPMW.png"
        PlotManager.plot_individual_model_evolution(
            scenario_name, 'LSPMW', plot_data, df_results, output_path
        )
        
        return results_rows, plot_data, scenario_name, df_results

    def run_all(self, excel_filename="resultados_unified_LSPMW.xlsx", 
                n_jobs=4, batch_size=10):
        """
        Ejecuta todos los escenarios (ARMA + ARIMA + SETAR) en paralelo
        evaluando solo LSPMW.
        """
        # Combinar todas las configuraciones
        all_configs = self.ARMA_CONFIGS + self.ARIMA_CONFIGS + self.SETAR_CONFIGS
        
        # Generar lista de escenarios
        scenarios = []
        scenario_id = 0
        for config in all_configs:
            for dist in self.DISTRIBUTIONS:
                for var in self.VARIANCES:
                    scenarios.append((config.copy(), dist, var, scenario_id))
                    scenario_id += 1
        
        print("="*80)
        print(f"🚀 INICIANDO EVALUACIÓN UNIFICADA - SOLO LSPMW")
        print("="*80)
        print(f"📊 Configuración:")
        print(f"   - Procesos ARMA: {len(self.ARMA_CONFIGS)}")
        print(f"   - Procesos ARIMA: {len(self.ARIMA_CONFIGS)}")
        print(f"   - Procesos SETAR: {len(self.SETAR_CONFIGS)}")
        print(f"   - Total procesos: {len(all_configs)}")
        print(f"   - Distribuciones: {len(self.DISTRIBUTIONS)}")
        print(f"   - Varianzas: {len(self.VARIANCES)}")
        print(f"   - Total escenarios: {len(scenarios)}")
        print(f"   - Trayectorias LSPMW: {self.N_TRAJECTORIES_MODEL}")
        print(f"   - Muestras teóricas: {self.N_TRAJECTORIES_TRUE}")
        print(f"   - Pasos predicción: {self.N_TEST_STEPS}")
        print(f"   - Procesos paralelos: {n_jobs}")
        print(f"   - Tamaño de lote: {batch_size}")
        print()
        
        # Procesamiento en lotes
        all_results = []
        all_predictions = {}
        
        for i in range(0, len(scenarios), batch_size):
            batch = scenarios[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(scenarios) + batch_size - 1) // batch_size
            
            print(f"📦 Procesando lote {batch_num}/{total_batches}...")
            
            results = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(self.run_single_scenario)(*s) for s in batch
            )
            
            # Consolidar resultados
            for idx, (rows, preds, name, _) in enumerate(results):
                all_results.extend(rows)
                all_predictions[name] = preds
            
            # Guardado intermedio
            temp_df = pd.DataFrame(all_results)
            temp_df.to_excel(excel_filename, index=False)
            print(f"   ✅ Lote {batch_num} completado. Guardado en {excel_filename}")
        
        # Guardado final
        final_df = pd.DataFrame(all_results)
        final_df.to_excel(excel_filename, index=False)
        
        print()
        print("="*80)
        print("✅ PROCESO COMPLETADO")
        print("="*80)
        print(f"📁 Resultados: {excel_filename}")
        print(f"📊 Total filas: {len(final_df)}")
        print(f"📈 Columnas: {list(final_df.columns)}")
        print()
        print("📋 Resumen por tipo de proceso:")
        print(final_df.groupby('Tipo_Proceso')['LSPMW'].agg(['count', 'mean', 'std']))
        print()
        
        self._predictions_cache = all_predictions
        return final_df

    def get_predictions_dict(self):
        """Retorna predicciones del último run_all()."""
        if hasattr(self, '_predictions_cache'):
            return self._predictions_cache
        else:
            raise ValueError("No hay predicciones disponibles. Ejecuta run_all() primero.")


