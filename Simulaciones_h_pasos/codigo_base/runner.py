import warnings
import time
import os
from tqdm import tqdm
warnings.filterwarnings("ignore")

# Configuración de threads para optimización
n_threads = str(os.cpu_count())
os.environ["OMP_NUM_THREADS"] = n_threads
os.environ["OPENBLAS_NUM_THREADS"] = n_threads
os.environ["MKL_NUM_THREADS"] = n_threads
os.environ["NUMEXPR_NUM_THREADS"] = n_threads
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pipeline import PipelineARMA_100Trayectorias, PipelineARIMA_100Trayectorias, PipelineSETAR_100Trayectorias
from figuras import PlotManager
import pandas as pd
import numpy as np


def run_analysis(df_final):
    """Análisis exhaustivo de resultados."""
    print("\n" + "="*80)
    print("ANÁLISIS EXHAUSTIVO DE RESULTADOS - 100 TRAYECTORIAS")
    print("="*80)
    
    # Columnas de modelos disponibles
    model_cols = ['LSPM', 'DeepAR', 'Sieve Bootstrap', 'MCPS']
    model_cols = [c for c in model_cols if c in df_final.columns]
    
    if len(df_final) == 0:
        print("⚠️ No hay datos suficientes para el análisis.")
        return

    # 1. RANKING GLOBAL POR MODELO
    print("\n🏆 1. RANKING GLOBAL (Media ECRPS)")
    print("-" * 80)
    
    means = {}
    for model in model_cols:
        val = df_final[model].mean()
        means[model] = val
    
    sorted_models = sorted(means.keys(), key=lambda x: means[x])
    
    print(f"{'Rank':<6} {'Modelo':<25} {'ECRPS Medio':<15} {'Std':<15}")
    print("-" * 70)
    for i, m in enumerate(sorted_models):
        std = df_final[m].std()
        print(f"{i+1:<6} {m:<25} {means[m]:.6f}      {std:.6f}")

    # 2. VICTORIAS POR PASO
    print("\n🎯 2. VICTORIAS (Mejor modelo por cada paso-escenario)")
    print("-" * 80)
    wins = {m: 0 for m in model_cols}
    total = 0
    
    for _, row in df_final.iterrows():
        scores = {m: row[m] for m in model_cols if not pd.isna(row[m])}
        if scores:
            winner = min(scores, key=scores.get)
            wins[winner] += 1
            total += 1
            
    for m in sorted(wins, key=wins.get, reverse=True):
        if total > 0:
            pct = (wins[m] / total) * 100
            print(f"  {m:<25}: {wins[m]:4d} victorias ({pct:.1f}%)")

    # 3. ANÁLISIS POR HORIZONTE DE PREDICCIÓN
    print("\n📈 3. RENDIMIENTO POR HORIZONTE (Paso H)")
    print("-" * 80)
    
    if 'Paso_H' in df_final.columns:
        horizons = sorted(df_final['Paso_H'].unique())
        print(f"\n{'Paso':<6} ", end="")
        for m in model_cols:
            print(f"{m:<15}", end="")
        print("\n" + "-"*80)
        
        for h in horizons:
            df_h = df_final[df_final['Paso_H'] == h]
            print(f"{h:<6} ", end="")
            for model in model_cols:
                mean_ecrps = df_h[model].mean()
                print(f"{mean_ecrps:<15.6f}", end="")
            print()

    # 4. ANÁLISIS POR TIPO DE PROCESO
    print("\n🔄 4. RENDIMIENTO POR PROCESO ARMA")
    print("-" * 80)
    
    if 'Proceso' in df_final.columns:
        for proceso in sorted(df_final['Proceso'].unique()):
            df_proc = df_final[df_final['Proceso'] == proceso]
            print(f"\n  {proceso}:")
            for model in model_cols:
                mean_ecrps = df_proc[model].mean()
                wins_proc = sum(1 for _, row in df_proc.iterrows() 
                               if row[model] == min(row[model_cols]))
                print(f"    {model:<20}: {mean_ecrps:.6f} ({wins_proc} victorias)")

    # 5. ANÁLISIS POR DISTRIBUCIÓN
    print("\n📊 5. RENDIMIENTO POR DISTRIBUCIÓN")
    print("-" * 80)
    
    if 'Distribución' in df_final.columns:
        for dist in sorted(df_final['Distribución'].unique()):
            df_dist = df_final[df_final['Distribución'] == dist]
            print(f"\n  {dist}:")
            for model in model_cols:
                mean_ecrps = df_dist[model].mean()
                print(f"    {model:<20}: {mean_ecrps:.6f}")

    # 6. ANÁLISIS POR VARIANZA
    print("\n📉 6. RENDIMIENTO POR NIVEL DE VARIANZA")
    print("-" * 80)
    
    if 'Varianza' in df_final.columns:
        for var in sorted(df_final['Varianza'].unique()):
            df_var = df_final[df_final['Varianza'] == var]
            print(f"\n  Varianza = {var}:")
            for model in model_cols:
                mean_ecrps = df_var[model].mean()
                print(f"    {model:<20}: {mean_ecrps:.6f}")

    print("\n" + "="*80)
    print("FIN DEL ANÁLISIS")
    print("="*80)


def main_full_140():
    """Ejecución completa de 140 escenarios con 100 trayectorias cada uno."""
    start_time = time.time()
    
    print("="*80)
    print("SIMULACIÓN COMPLETA - 140 ESCENARIOS CON 100 TRAYECTORIAS")
    print("="*80)
    
    # Crear pipeline
    pipeline = PipelineARMA_100Trayectorias(
        n_boot=1000,
        seed=42,
        verbose=False
    )
    
    # Ejecutar todos los escenarios
    df_final = pipeline.run_all(
        excel_filename="resultados_140_trayectorias_FINAL.xlsx",
        n_jobs=4
    )
    
    # Análisis de resultados
    run_analysis(df_final)
    
    # Generar gráficos resumidos
    print("\n📊 Generando gráficos resumidos...")
    model_cols = ['LSPM', 'DeepAR', 'Sieve Bootstrap', 'MCPS']
    
    # Gráfico de evolución por horizonte
    results_by_step = {}
    for h in range(1, 13):
        step_data = df_final[df_final['Paso_H'] == h][model_cols].mean().to_frame().T
        results_by_step[h] = step_data
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s ({elapsed/3600:.2f} horas)")
    
    return df_final


def main_two_scenarios():
    """
    Ejecuta DOS escenarios para pruebas intermedias.
    Genera gráficos detallados de densidades paso a paso.
    """
    start_time = time.time()
    
    print("="*80)
    print("EVALUACIÓN CON 2 ESCENARIOS (100 TRAYECTORIAS CADA UNO)")
    print("="*80)
    
    # Crear pipeline
    pipeline = PipelineARMA_100Trayectorias(n_boot=1000, seed=42, verbose=True)
    
    # Configurar 2 escenarios diferentes
    pipeline.ARMA_CONFIGS = [
        {'nombre': 'AR(1)', 'phi': [0.9], 'theta': []},
        {'nombre': 'MA(1)', 'phi': [], 'theta': [0.7]}
    ]
    pipeline.DISTRIBUTIONS = ['normal']
    pipeline.VARIANCES = [1.0]
    
    print(f"\n📊 Configuración:")
    print(f"   - Escenarios: 2 (AR(1) y MA(1))")
    print(f"   - Distribución: Normal")
    print(f"   - Varianza: 1.0")
    print(f"   - Trayectorias por modelo: {pipeline.N_TRAJECTORIES_MODEL}")
    print(f"   - Trayectorias teóricas: {pipeline.N_TRAJECTORIES_TRUE}")
    print()
    
    # Ejecutar
    df_final = pipeline.run_all(
        excel_filename="resultados_2_escenarios_trayectorias.xlsx",
        n_jobs=2
    )
    
    # Análisis
    run_analysis(df_final)
    
    # Generar gráficos detallados por escenario
    print("\n📊 Generando gráficos de densidades...")
    
    try:
        predictions_dict = pipeline.get_predictions_dict()
        model_cols = ['LSPM', 'DeepAR', 'Sieve Bootstrap', 'MCPS']
        
        os.makedirs("graficos_densidades", exist_ok=True)
        
        for scenario_name, preds_by_step in predictions_dict.items():
            # Extraer componentes del nombre del escenario
            parts = scenario_name.split('_')
            proceso = parts[0]
            dist = parts[1] if len(parts) > 1 else 'normal'
            var_str = parts[2] if len(parts) > 2 else 'var1.0'
            var = float(var_str.replace('var', ''))
            
            # Filtrar resultados para este escenario
            df_scenario = df_final[
                (df_final['Proceso'] == proceso) & 
                (df_final['Distribución'] == dist) & 
                (df_final['Varianza'] == var)
            ].copy()
            
            if len(df_scenario) == 0:
                print(f"  ⚠️ No se encontraron resultados para {scenario_name}")
                continue
            
            # Generar gráfico vertical
            PlotManager.plot_scenario_densities(
                scenario_name=scenario_name,
                predictions_dict=preds_by_step,
                df_results=df_scenario,
                model_names=model_cols,
                save_path=f"graficos_densidades/densidades_{scenario_name}.png"
            )
        
        print(f"\n✅ Gráficos guardados en: graficos_densidades/")
        
    except Exception as e:
        print(f"\n⚠️ Error al generar gráficos: {e}")
        print("   Los resultados en Excel están disponibles.")
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s")
    
    return df_final



def main_full_140_arima():
    """Ejecución completa de 140 escenarios ARIMA con 100 trayectorias cada uno."""
    start_time = time.time()
    
    print("="*80)
    print("SIMULACIÓN COMPLETA ARIMA - 140 ESCENARIOS CON 100 TRAYECTORIAS")
    print("="*80)
    
    # Crear pipeline
    pipeline = PipelineARIMA_100Trayectorias(
        n_boot=1000,
        seed=42,
        verbose=False
    )
    
    # Ejecutar todos los escenarios
    df_final = pipeline.run_all(
        excel_filename="resultados_140_trayectorias_ARIMA_FINAL.xlsx",
        n_jobs=4
    )
    
    # Análisis de resultados
    run_analysis(df_final)
    
    # Generar gráficos resumidos
    print("\n📊 Generando gráficos resumidos...")
    model_cols = ['LSPM', 'DeepAR', 'Sieve Bootstrap', 'MCPS']
    
    # Gráfico de evolución por horizonte
    results_by_step = {}
    for h in range(1, 13):
        step_data = df_final[df_final['Paso_H'] == h][model_cols].mean().to_frame().T
        results_by_step[h] = step_data

    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s ({elapsed/3600:.2f} horas)")
    
    return df_final


def main_two_scenarios_arima():
    """
    Ejecuta DOS escenarios ARIMA para pruebas intermedias.
    Genera gráficos detallados de densidades paso a paso.
    """
    start_time = time.time()
    
    print("="*80)
    print("EVALUACIÓN CON 2 ESCENARIOS ARIMA (100 TRAYECTORIAS CADA UNO)")
    print("="*80)
    
    # Crear pipeline
    pipeline = PipelineARIMA_100Trayectorias(n_boot=1000, seed=42, verbose=True)
    
    # Configurar 2 escenarios diferentes
    pipeline.ARIMA_CONFIGS = [
        {'nombre': 'ARIMA(1,1,0)', 'phi': [0.6], 'theta': []},
        {'nombre': 'ARIMA(0,1,1)', 'phi': [], 'theta': [0.5]}
    ]
    pipeline.DISTRIBUTIONS = ['normal']
    pipeline.VARIANCES = [1.0]
    
    print(f"\n📊 Configuración:")
    print(f"   - Escenarios: 2 (ARIMA(1,1,0) y ARIMA(0,1,1))")
    print(f"   - Distribución: Normal")
    print(f"   - Varianza: 1.0")
    print(f"   - Trayectorias por modelo: {pipeline.N_TRAJECTORIES_MODEL}")
    print(f"   - Trayectorias teóricas: {pipeline.N_TRAJECTORIES_TRUE}")
    print()
    
    # Ejecutar
    df_final = pipeline.run_all(
        excel_filename="resultados_2_escenarios_trayectorias_arima.xlsx",
        n_jobs=2
    )
    
    # Análisis
    run_analysis(df_final)
    
    # Generar gráficos detallados por escenario
    print("\n📊 Generando gráficos de densidades...")
    
    try:
        predictions_dict = pipeline.get_predictions_dict()
        model_cols = ['LSPM', 'DeepAR', 'Sieve Bootstrap', 'MCPS']
        
        os.makedirs("graficos_densidades_arima", exist_ok=True)
        
        for scenario_name, preds_by_step in predictions_dict.items():
            # Extraer componentes del nombre del escenario
            parts = scenario_name.split('_')
            proceso = parts[0]  # e.g., 'ARIMA(1,1,0)'
            dist = parts[1] if len(parts) > 1 else 'normal'
            
            # Manejar el formato V{var} en el nombre del escenario
            var_str = [p for p in parts if p.startswith('V')]
            if var_str:
                var = float(var_str[0][1:])  # Quitar la 'V' y convertir a float
            else:
                var = 1.0
            
            # Filtrar resultados para este escenario
            df_scenario = df_final[
                (df_final['Proceso'] == proceso) & 
                (df_final['Distribución'] == dist) & 
                (df_final['Varianza'] == var)
            ].copy()
            
            if len(df_scenario) == 0:
                print(f"  ⚠️ No se encontraron resultados para {scenario_name}")
                continue
            
            # Generar gráfico vertical
            PlotManager.plot_scenario_densities(
                scenario_name=scenario_name,
                predictions_dict=preds_by_step,
                df_results=df_scenario,
                model_names=model_cols,
                save_path=f"graficos_densidades_arima/densidades_{scenario_name}.png"
            )
        
        print(f"\n✅ Gráficos guardados en: graficos_densidades_arima/")
        
    except Exception as e:
        print(f"\n⚠️ Error al generar gráficos: {e}")
        print("   Los resultados en Excel están disponibles.")
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s")
    
    return df_final

def main_full_140_setar():
    """Ejecución completa de 140 escenarios SETAR con 100 trayectorias cada uno."""
    start_time = time.time()
    
    print("="*80)
    print("SIMULACIÓN COMPLETA SETAR - 140 ESCENARIOS CON 100 TRAYECTORIAS")
    print("="*80)
    
    # Crear pipeline
    pipeline = PipelineSETAR_100Trayectorias(
        n_boot=1000,
        seed=42,
        verbose=False
    )
    
    # Ejecutar todos los escenarios
    df_final = pipeline.run_all(
        excel_filename="resultados_140_trayectorias_SETAR_FINAL.xlsx",
        n_jobs=4
    )
    
    # Análisis de resultados
    run_analysis(df_final)
    
    # Generar gráficos resumidos
    print("\n📊 Generando gráficos resumidos...")
    model_cols = ['LSPM', 'DeepAR', 'Sieve Bootstrap', 'MCPS']
    
    # Gráfico de evolución por horizonte
    results_by_step = {}
    for h in range(1, 13):
        step_data = df_final[df_final['Paso_H'] == h][model_cols].mean().to_frame().T
        results_by_step[h] = step_data
    
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s ({elapsed/3600:.2f} horas)")
    
    return df_final


def main_two_scenarios_setar():
    """
    Ejecuta DOS escenarios SETAR para pruebas intermedias.
    Genera gráficos detallados de densidades paso a paso.
    """
    start_time = time.time()
    
    print("="*80)
    print("EVALUACIÓN CON 2 ESCENARIOS SETAR (100 TRAYECTORIAS CADA UNO)")
    print("="*80)
    
    # Crear pipeline
    pipeline = PipelineSETAR_100Trayectorias(n_boot=1000, seed=42, verbose=True)
    
    # Configurar 2 escenarios diferentes: uno simple y uno complejo
    pipeline.SETAR_CONFIGS = [
        {'nombre': 'SETAR-1', 'phi_regime1': [0.6], 'phi_regime2': [-0.5], 
         'threshold': 0.0, 'delay': 1, 'description': 'SETAR(2;1,1) d=1, r=0'},
        {'nombre': 'SETAR-3', 'phi_regime1': [0.5, -0.2], 'phi_regime2': [-0.3, 0.1], 
         'threshold': 0.5, 'delay': 1, 'description': 'SETAR(2;2,2) d=1, r=0.5'}
    ]
    pipeline.DISTRIBUTIONS = ['normal']
    pipeline.VARIANCES = [1.0]
    
    print(f"\n📊 Configuración:")
    print(f"   - Escenarios: 2 (SETAR-1 simple y SETAR-3 moderado)")
    print(f"   - SETAR-1: AR(1) en ambos regímenes, threshold=0.0, delay=1")
    print(f"   - SETAR-3: AR(2) en ambos regímenes, threshold=0.5, delay=1")
    print(f"   - Distribución: Normal")
    print(f"   - Varianza: 1.0")
    print(f"   - Trayectorias por modelo: {pipeline.N_TRAJECTORIES_MODEL}")
    print(f"   - Trayectorias teóricas: {pipeline.N_TRAJECTORIES_TRUE}")
    print()
    
    # Ejecutar
    df_final = pipeline.run_all(
        excel_filename="resultados_2_escenarios_trayectorias_setar.xlsx",
        n_jobs=2
    )
    
    # Análisis
    run_analysis(df_final)
    
    # Generar gráficos detallados por escenario
    print("\n📊 Generando gráficos de densidades...")
    
    try:
        predictions_dict = pipeline.get_predictions_dict()
        model_cols = ['LSPM', 'DeepAR', 'Sieve Bootstrap', 'MCPS']
        
        os.makedirs("graficos_densidades_setar", exist_ok=True)
        
        for scenario_name, preds_by_step in predictions_dict.items():
            # Extraer componentes del nombre del escenario
            parts = scenario_name.split('_')
            proceso = parts[0]  # e.g., 'SETAR-1'
            dist = parts[1] if len(parts) > 1 else 'normal'
            
            # Manejar el formato V{var} en el nombre del escenario
            var_str = [p for p in parts if p.startswith('V')]
            if var_str:
                var = float(var_str[0][1:])  # Quitar la 'V' y convertir a float
            else:
                var = 1.0
            
            # Filtrar resultados para este escenario
            df_scenario = df_final[
                (df_final['Proceso'] == proceso) & 
                (df_final['Distribución'] == dist) & 
                (df_final['Varianza'] == var)
            ].copy()
            
            if len(df_scenario) == 0:
                print(f"  ⚠️ No se encontraron resultados para {scenario_name}")
                continue
            
            # Generar gráfico vertical
            PlotManager.plot_scenario_densities(
                scenario_name=scenario_name,
                predictions_dict=preds_by_step,
                df_results=df_scenario,
                model_names=model_cols,
                save_path=f"graficos_densidades_setar/densidades_{scenario_name}.png"
            )
        
        print(f"\n✅ Gráficos guardados en: graficos_densidades_setar/")
        
    except Exception as e:
        print(f"\n⚠️ Error al generar gráficos: {e}")
        print("   Los resultados en Excel están disponibles.")
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s")
    
    return df_final

