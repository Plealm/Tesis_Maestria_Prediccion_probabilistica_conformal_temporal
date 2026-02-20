import warnings
import time
import os
from tqdm import tqdm
warnings.filterwarnings("ignore")

n_threads = str(os.cpu_count())
os.environ["OMP_NUM_THREADS"] = n_threads
os.environ["OPENBLAS_NUM_THREADS"] = n_threads
os.environ["MKL_NUM_THREADS"] = n_threads
os.environ["NUMEXPR_NUM_THREADS"] = n_threads
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pipeline import Pipeline140SinSesgos_ARMA, Pipeline140SinSesgos_ARIMA, Pipeline140SinSesgos_SETAR, Pipeline140_TamanosCrecientes, Pipeline240_ProporcionesVariables
import pandas as pd
import numpy as np


def run_analysis(df_final):
    """Función común para análisis exhaustivo de resultados."""
    print("\n" + "="*80)
    print("ANÁLISIS EXHAUSTIVO DE RESULTADOS")
    print("="*80)
    
    model_cols = ['AREPD', 'AV-MCPS', 'Block Bootstrapping', 'DeepAR', 
                  'EnCQR-LSTM', 'LSPM', 'LSPMW', 'MCPS', 'Sieve Bootstrap']
    
    model_cols = [c for c in model_cols if c in df_final.columns]
    
    if 'Paso' in df_final.columns:
        df_steps = df_final[df_final['Paso'] != 'Promedio'].copy()
    else:
        df_steps = df_final.copy()
    
    if len(df_steps) == 0:
        print("⚠️ No hay datos suficientes para el análisis.")
        return

    # 1. RANKING GLOBAL
    print("\n🏆 1. RANKING GLOBAL (Media CRPS)")
    print("-" * 80)
    
    means = {}
    for model in model_cols:
        val = df_steps[model].mean()
        means[model] = val
    
    sorted_models = sorted(means.keys(), key=lambda x: means[x])
    
    print(f"{'Rank':<6} {'Modelo':<25} {'CRPS Medio':<15}")
    print("-" * 60)
    for i, m in enumerate(sorted_models):
        print(f"{i+1:<6} {m:<25} {means[m]:.6f}")

    # 2. MEJOR POR ESCENARIO
    print("\n🎯 2. VICTORIAS (Mejor modelo por paso)")
    print("-" * 80)
    wins = {m: 0 for m in model_cols}
    total = 0
    
    for _, row in df_steps.iterrows():
        scores = {m: row[m] for m in model_cols if not pd.isna(row[m])}
        if scores:
            winner = min(scores, key=scores.get)
            wins[winner] += 1
            total += 1
            
    for m in sorted(wins, key=wins.get, reverse=True):
        if total > 0:
            pct = (wins[m] / total) * 100
            print(f"  {m:<25}: {wins[m]:3d} victorias ({pct:.1f}%)")

    print("\n" + "="*80)
    print("FIN DEL ANÁLISIS")
    print("="*80)

# ============================================================================
# ============================================================================
#  SIMULACION PRINCIPAL
# ============================================================================
# ============================================================================


# ============================================================================
#  SIMULACION ARMA
# ============================================================================

def main_full_140():
    """Ejecución completa de 140 escenarios con gestión de memoria."""
    start_time = time.time()
    
    print("="*80)
    print("INICIANDO SIMULACIÓN DE 140 ESCENARIOS")
    print("="*80)
    
    pipeline = Pipeline140SinSesgos_ARMA(
        n_boot=1000,
        seed=42,
        verbose=False
    )
    
    df_final = pipeline.run_all(
        excel_filename="resultados_140_FINAL_FIXED.xlsx",
        batch_size=10 
    )
    
    run_analysis(df_final)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s ({elapsed/3600:.2f} horas)")
    
    return df_final


def main_two_scenarios():
    """
    FIX: Ejecuta solo 2 escenarios DENTRO del wrapper de paralelización.
    No hace monkey patching problemático.
    """
    start_time = time.time()
    
    print("="*80)
    print("EVALUACIÓN CON SOLO 2 ESCENARIOS")
    print("="*80)
    
    # Crear pipeline con configuración especial
    pipeline = Pipeline140SinSesgos_ARMA(n_boot=1000, seed=42, verbose=True)
    
    # Configurar solo 2 escenarios manualmente
    pipeline.ARMA_CONFIGS = [
        {'nombre': 'AR(1)', 'phi': [0.9], 'theta': []},
        {'nombre': 'MA(1)', 'phi': [], 'theta': [0.7]}
    ]
    pipeline.DISTRIBUTIONS = ['normal']
    pipeline.VARIANCES = [1.0]
    
    # Ahora generate_all_scenarios() solo generará 2 escenarios
    df_final = pipeline.run_all(
        excel_filename="resultados_2_ESCENARIOS.xlsx",
        batch_size=2
    )
    
    run_analysis(df_final)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s")
    
    return df_final

# ============================================================================
#  SIMULACION ARIMA
# ============================================================================

def main_full_140_ARIMA():
    """
    Ejecución completa de 140 escenarios ARIMA con gestión de memoria.
    """
    start_time = time.time()
    
    print("="*80)
    print("INICIANDO SIMULACIÓN DE 140 ESCENARIOS ARIMA")
    print("="*80)
    
    pipeline = Pipeline140SinSesgos_ARIMA(
        n_boot=1000,
        seed=42,
        verbose=False
    )
    
    df_final = pipeline.run_all(
        excel_filename="resultados_140_ARIMA_FINAL.xlsx",
        batch_size=10 
    )
    
    run_analysis(df_final)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s ({elapsed/3600:.2f} horas)")
    
    return df_final


def main_two_scenarios_ARIMA():
    """
    Ejecuta solo 2 escenarios ARIMA para pruebas rápidas.
    """
    start_time = time.time()
    
    print("="*80)
    print("EVALUACIÓN CON SOLO 2 ESCENARIOS ARIMA")
    print("="*80)
    
    # Crear pipeline con configuración especial
    pipeline = Pipeline140SinSesgos_ARIMA(n_boot=1000, seed=42, verbose=True)
    
    # Configurar solo 2 escenarios manualmente
    pipeline.ARIMA_CONFIGS = [
        {'nombre': 'ARIMA(1,1,0)', 'phi': [0.7], 'theta': []},
        {'nombre': 'ARIMA(0,1,1)', 'phi': [], 'theta': [0.6]}
    ]
    pipeline.DISTRIBUTIONS = ['normal']
    pipeline.VARIANCES = [1.0]
    
    # Ahora generate_all_scenarios() solo generará 2 escenarios
    df_final = pipeline.run_all(
        excel_filename="resultados_2_ESCENARIOS_ARIMA.xlsx",
        batch_size=2
    )
    
    run_analysis(df_final)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s")
    
    return df_final

# ============================================================================
#  SIMULACION SETAR
# ============================================================================


def main_full_140_SETAR():
    """
    Ejecución completa de 140 escenarios SETAR con gestión de memoria.
    """
    start_time = time.time()
    
    print("="*80)
    print("INICIANDO SIMULACIÓN DE 140 ESCENARIOS SETAR")
    print("="*80)
    
    pipeline = Pipeline140SinSesgos_SETAR(
        n_boot=1000,
        seed=42,
        verbose=False
    )
    
    df_final = pipeline.run_all(
        excel_filename="resultados_140_SETAR_FINAL.xlsx",
        batch_size=10 
    )
    
    run_analysis(df_final)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s ({elapsed/3600:.2f} horas)")
    
    return df_final


def main_two_scenarios_SETAR():
    """
    Ejecuta solo 2 escenarios SETAR para pruebas rápidas.
    """
    start_time = time.time()
    
    print("="*80)
    print("EVALUACIÓN CON SOLO 2 ESCENARIOS SETAR")
    print("="*80)
    
    # Crear pipeline con configuración especial
    pipeline = Pipeline140SinSesgos_SETAR(n_boot=1000, seed=42, verbose=True)
    
    # Configurar solo 2 escenarios manualmente
    pipeline.SETAR_CONFIGS = [
        {
            'nombre': 'SETAR-1',
            'phi_regime1': [0.6],
            'phi_regime2': [-0.5],
            'threshold': 0.0,
            'delay': 1,
            'description': 'SETAR(2;1,1) - AR(1) con d=1'
        },
        {
            'nombre': 'SETAR-3',
            'phi_regime1': [0.5, -0.2],
            'phi_regime2': [-0.3, 0.1],
            'threshold': 0.5,
            'delay': 1,
            'description': 'SETAR(2;2,2) - AR(2) con d=1'
        }
    ]
    pipeline.DISTRIBUTIONS = ['normal']
    pipeline.VARIANCES = [1.0]
    
    # Ahora generate_all_scenarios() solo generará 2 escenarios
    df_final = pipeline.run_all(
        excel_filename="resultados_2_ESCENARIOS_SETAR.xlsx",
        batch_size=2
    )
    
    run_analysis(df_final)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s")
    
    return df_final

# ============================================================================
# ============================================================================
#  SIMULACION Diferenciado
# ============================================================================
# ============================================================================


def main_two_scenarios_diferenciado():
    """
    Ejecuta 2 escenarios ARIMA CON diferenciación adicional para pruebas rápidas.
    """
    start_time = time.time()
    
    print("="*80)
    print("EVALUACIÓN CON 2 ESCENARIOS ARIMA - AMBAS MODALIDADES")
    print("="*80)
    
    from pipeline import Pipeline140SinSesgos_ARIMA_ConDiferenciacion
    
    # Crear pipeline (evalúa automáticamente SIN_DIFF y CON_DIFF)
    pipeline = Pipeline140SinSesgos_ARIMA_ConDiferenciacion(
        n_boot=1000, 
        seed=42, 
        verbose=True
        # ❌ ELIMINAR: usar_diferenciacion=True
    )
    
    # Configurar solo 2 escenarios
    pipeline.ARIMA_CONFIGS = [
        {'nombre': 'ARIMA(1,1,0)', 'phi': [0.7], 'theta': []},
        {'nombre': 'ARIMA(0,1,1)', 'phi': [], 'theta': [0.6]}
    ]
    pipeline.DISTRIBUTIONS = ['normal']
    pipeline.VARIANCES = [1.0]
    
    df_final = pipeline.run_all(
        excel_filename="resultados_2_ESCENARIOS_ARIMA_AMBAS_MODALIDADES.xlsx",
        batch_size=2
    )
    
    run_analysis(df_final)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s")
    
    return df_final


def main_full_140_diferenciado():
    """
    Ejecución completa de 140 escenarios ARIMA evaluando ambas modalidades.
    
    Este pipeline evalúa cada escenario en DOS modalidades:
    - SIN_DIFF: Los modelos ven niveles Y_t
    - CON_DIFF: Los modelos ven incrementos ΔY_t
    
    Esto permite comparar si trabajar en espacio de incrementos mejora
    el desempeño de los métodos de predicción conformal.
    """
    start_time = time.time()
    
    print("="*80)
    print("INICIANDO SIMULACIÓN DE 140 ESCENARIOS ARIMA - AMBAS MODALIDADES")
    print("="*80)
    
    from pipeline import Pipeline140SinSesgos_ARIMA_ConDiferenciacion
    
    # Crear pipeline (evalúa automáticamente SIN_DIFF y CON_DIFF)
    pipeline = Pipeline140SinSesgos_ARIMA_ConDiferenciacion(
        n_boot=1000,
        seed=42,
        verbose=False
        # ❌ ELIMINAR: usar_diferenciacion=True
    )
    
    df_final = pipeline.run_all(
        excel_filename="resultados_140_ARIMA_AMBAS_MODALIDADES.xlsx",
        batch_size=10
    )
    
    print("\n" + "="*80)
    print("ANÁLISIS DE RESULTADOS - AMBAS MODALIDADES")
    print("="*80)
    
    run_analysis(df_final)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s ({elapsed/3600:.2f} horas)")
    
    return df_final

# ============================================================================
# ============================================================================
#  SIMULACION Diferenciado diferentes niveles de diferenciación
# ============================================================================
# ============================================================================

def analisis_completo_doble_modalidad(df_final):
    """
    Análisis exhaustivo para resultados con doble modalidad (SIN_DIFF vs CON_DIFF).
    
    Compara:
    1. Desempeño por cada valor de d
    2. SIN_DIFF vs CON_DIFF: ¿cuál funciona mejor?
    3. Tendencias según d aumenta
    4. Mejor d por modelo y modalidad
    """
    print("\n" + "="*80)
    print("ANÁLISIS EXHAUSTIVO - DOBLE MODALIDAD (SIN_DIFF vs CON_DIFF)")
    print("="*80)
    
    model_cols = ['AREPD', 'AV-MCPS', 'Block Bootstrapping', 'DeepAR', 
                  'EnCQR-LSTM', 'LSPM', 'LSPMW', 'MCPS', 'Sieve Bootstrap']
    model_cols = [c for c in model_cols if c in df_final.columns]
    
    if 'Paso' in df_final.columns:
        df_steps = df_final[df_final['Paso'] != 'Promedio'].copy()
    else:
        df_steps = df_final.copy()
    
    if len(df_steps) == 0:
        print("⚠️ No hay datos suficientes para el análisis.")
        return
    
    # Asegurar tipos correctos
    if 'd' in df_steps.columns:
        df_steps['d'] = pd.to_numeric(df_steps['d'], errors='coerce')
    
    d_values = sorted(df_steps['d'].unique())
    modalidades = sorted(df_steps['Modalidad'].unique()) if 'Modalidad' in df_steps.columns else []
    
    # =================================================================
    # 1. COMPARACIÓN GLOBAL: SIN_DIFF vs CON_DIFF
    # =================================================================
    print("\n" + "="*80)
    print("🔍 1. COMPARACIÓN GLOBAL POR MODALIDAD")
    print("="*80)
    
    for modalidad in modalidades:
        df_mod = df_steps[df_steps['Modalidad'] == modalidad]
        
        if len(df_mod) == 0:
            continue
        
        print(f"\n{'='*60}")
        print(f"MODALIDAD: {modalidad}")
        print(f"{'='*60}")
        
        means = {}
        for model in model_cols:
            if model in df_mod.columns:
                val = df_mod[model].mean()
                if not pd.isna(val):
                    means[model] = val
        
        if not means:
            print("  (Sin datos válidos)")
            continue
        
        sorted_models = sorted(means.keys(), key=lambda x: means[x])
        
        print(f"{'Rank':<6} {'Modelo':<25} {'CRPS Medio':<15}")
        print("-" * 60)
        for i, m in enumerate(sorted_models):
            print(f"{i+1:<6} {m:<25} {means[m]:.6f}")
    
    # =================================================================
    # 2. RANKING POR CADA d Y MODALIDAD
    # =================================================================
    print("\n" + "="*80)
    print("📊 2. RANKING POR CADA VALOR DE d (AMBAS MODALIDADES)")
    print("="*80)
    
    for d_val in d_values:
        df_d = df_steps[df_steps['d'] == d_val]
        
        if len(df_d) == 0:
            continue
        
        print(f"\n{'='*70}")
        print(f"d = {d_val}")
        print(f"{'='*70}")
        
        for modalidad in modalidades:
            df_d_mod = df_d[df_d['Modalidad'] == modalidad]
            
            if len(df_d_mod) == 0:
                continue
            
            print(f"\n  --- {modalidad} ---")
            
            means = {}
            for model in model_cols:
                if model in df_d_mod.columns:
                    val = df_d_mod[model].mean()
                    if not pd.isna(val):
                        means[model] = val
            
            if not means:
                print("    (Sin datos válidos)")
                continue
            
            sorted_models = sorted(means.keys(), key=lambda x: means[x])
            
            print(f"  {'Rank':<6} {'Modelo':<25} {'CRPS':<12}")
            print("  " + "-" * 50)
            for i, m in enumerate(sorted_models[:5]):  # Top 5
                print(f"  {i+1:<6} {m:<25} {means[m]:.6f}")
    
    # =================================================================
    # 3. VICTORIAS POR MODELO EN CADA MODALIDAD
    # =================================================================
    print("\n" + "="*80)
    print("🎯 3. VICTORIAS POR MODALIDAD (Mejor modelo por paso)")
    print("="*80)
    
    for modalidad in modalidades:
        df_mod = df_steps[df_steps['Modalidad'] == modalidad]
        
        if len(df_mod) == 0:
            continue
        
        print(f"\n{'='*60}")
        print(f"MODALIDAD: {modalidad}")
        print(f"{'='*60}")
        
        wins = {m: 0 for m in model_cols}
        total = 0
        
        for _, row in df_mod.iterrows():
            scores = {m: row[m] for m in model_cols if not pd.isna(row[m])}
            if scores:
                winner = min(scores, key=scores.get)
                wins[winner] += 1
                total += 1
        
        if total == 0:
            print("  (Sin datos válidos)")
            continue
        
        for m in sorted(wins, key=wins.get, reverse=True):
            if wins[m] > 0:
                pct = (wins[m] / total) * 100
                print(f"  {m:<25}: {wins[m]:4d} victorias ({pct:.1f}%)")
    
    # =================================================================
    # 4. TENDENCIAS: Desempeño según d (por modalidad)
    # =================================================================
    print("\n" + "="*80)
    print("📈 4. TENDENCIAS: Desempeño según d aumenta")
    print("="*80)
    
    for modalidad in modalidades:
        print(f"\n{'='*70}")
        print(f"MODALIDAD: {modalidad}")
        print(f"{'='*70}")
        
        print(f"\n{'Modelo':<25} ", end="")
        for d_val in d_values:
            print(f"d={d_val:<3}", end="  ")
        print()
        print("-" * (25 + 7 * len(d_values)))
        
        for model in model_cols:
            print(f"{model:<25} ", end="")
            for d_val in d_values:
                df_d_mod = df_steps[(df_steps['d'] == d_val) & 
                                    (df_steps['Modalidad'] == modalidad)]
                if model in df_d_mod.columns:
                    val = df_d_mod[model].mean()
                    if not pd.isna(val):
                        print(f"{val:.4f}", end="  ")
                    else:
                        print("  ---  ", end="  ")
                else:
                    print("  ---  ", end="  ")
            print()
    
    # =================================================================
    # 5. COMPARACIÓN DIRECTA: SIN_DIFF vs CON_DIFF por modelo
    # =================================================================
    print("\n" + "="*80)
    print("⚖️  5. COMPARACIÓN DIRECTA: SIN_DIFF vs CON_DIFF")
    print("="*80)
    
    if len(modalidades) == 2:
        mod_sin = [m for m in modalidades if 'SIN' in m][0]
        mod_con = [m for m in modalidades if 'CON' in m][0]
        
        print(f"\n{'Modelo':<25} {mod_sin:<12} {mod_con:<12} {'Diferencia':<12} {'Mejor':<10}")
        print("-" * 75)
        
        for model in model_cols:
            df_sin = df_steps[df_steps['Modalidad'] == mod_sin]
            df_con = df_steps[df_steps['Modalidad'] == mod_con]
            
            if model in df_sin.columns and model in df_con.columns:
                val_sin = df_sin[model].mean()
                val_con = df_con[model].mean()
                
                if not pd.isna(val_sin) and not pd.isna(val_con):
                    diff = val_con - val_sin
                    mejor = mod_sin if val_sin < val_con else mod_con
                    
                    print(f"{model:<25} {val_sin:.6f}   {val_con:.6f}   "
                          f"{diff:+.6f}   {mejor:<10}")
    
    # =================================================================
    # 6. MEJOR d POR MODELO Y MODALIDAD
    # =================================================================
    print("\n" + "="*80)
    print("🎲 6. MEJOR VALOR DE d PARA CADA MODELO Y MODALIDAD")
    print("="*80)
    
    for modalidad in modalidades:
        print(f"\n{'='*60}")
        print(f"MODALIDAD: {modalidad}")
        print(f"{'='*60}")
        
        print(f"\n{'Modelo':<25} {'Mejor d':<10} {'CRPS en ese d':<15}")
        print("-" * 60)
        
        for model in model_cols:
            best_d = None
            best_crps = float('inf')
            
            for d_val in d_values:
                df_d_mod = df_steps[(df_steps['d'] == d_val) & 
                                    (df_steps['Modalidad'] == modalidad)]
                if model in df_d_mod.columns:
                    val = df_d_mod[model].mean()
                    if not pd.isna(val) and val < best_crps:
                        best_crps = val
                        best_d = d_val
            
            if best_d is not None:
                print(f"{model:<25} {best_d:<10} {best_crps:.6f}")
    
    # =================================================================
    # 7. RESUMEN EJECUTIVO
    # =================================================================
    print("\n" + "="*80)
    print("📋 7. RESUMEN EJECUTIVO")
    print("="*80)
    
    # Mejor modalidad global
    if len(modalidades) == 2:
        crps_sin = df_steps[df_steps['Modalidad'] == mod_sin][model_cols].mean().mean()
        crps_con = df_steps[df_steps['Modalidad'] == mod_con][model_cols].mean().mean()
        
        print(f"\n✓ MEJOR MODALIDAD GLOBAL:")
        print(f"  • {mod_sin}: CRPS promedio = {crps_sin:.6f}")
        print(f"  • {mod_con}: CRPS promedio = {crps_con:.6f}")
        
        if crps_sin < crps_con:
            print(f"  → GANADOR: {mod_sin} (diferencia: {crps_con - crps_sin:.6f})")
        else:
            print(f"  → GANADOR: {mod_con} (diferencia: {crps_sin - crps_con:.6f})")
    
    # Mejor modelo global
    global_means = {}
    for model in model_cols:
        if model in df_steps.columns:
            val = df_steps[model].mean()
            if not pd.isna(val):
                global_means[model] = val
    
    if global_means:
        best_model = min(global_means, key=global_means.get)
        print(f"\n✓ MEJOR MODELO GLOBAL:")
        print(f"  → {best_model}: CRPS = {global_means[best_model]:.6f}")
    
    print("\n" + "="*80)
    print("FIN DEL ANÁLISIS")
    print("="*80)


def main_full_2800():
    """
    Ejecución completa: 2,800 filas (1,400 escenarios × 2 modalidades).
    - d = 1, 2, ..., 10
    - 7 configuraciones ARMA
    - 5 distribuciones
    - 4 varianzas
    - 2 modalidades (SIN_DIFF + CON_DIFF)
    """
    start_time = time.time()
    
    print("="*80)
    print("INICIANDO SIMULACIÓN COMPLETA: 2,800 FILAS")
    print("="*80)
    
    from pipeline import PipelineARIMA_MultiD_DobleModalidad
    
    pipeline = PipelineARIMA_MultiD_DobleModalidad(
        n_boot=1000,
        seed=42,
        verbose=False
    )
    
    df_final = pipeline.run_all(
        excel_filename="resultados_ARIMA_d1_a_d10_DOBLE_MODALIDAD_COMPLETO.xlsx",
        batch_size=20
    )
    
    # Análisis exhaustivo
    analisis_completo_doble_modalidad(df_final)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s ({elapsed/3600:.2f} horas)")
    
    return df_final


def main_test_reducido_doble():
    """
    Test reducido: 2 valores de d, 2 ARMA, 1 distribución, 1 varianza, 2 modalidades.
    Total: 2 × 2 × 1 × 1 × 2 = 8 filas base
    Con 12 pasos + 1 promedio = 104 filas totales
    """
    start_time = time.time()
    
    print("="*80)
    print("TEST REDUCIDO: DOBLE MODALIDAD (d=1,2)")
    print("="*80)
    
    from pipeline import PipelineARIMA_MultiD_DobleModalidad
    
    pipeline = PipelineARIMA_MultiD_DobleModalidad(
        n_boot=1000, seed=42, verbose=True
    )
    
    # Configuración reducida
    pipeline.D_VALUES = [1, 2]
    pipeline.ARMA_CONFIGS = [
        {'nombre': 'RW', 'phi': [], 'theta': []},
        {'nombre': 'AR(1)', 'phi': [0.6], 'theta': []}
    ]
    pipeline.DISTRIBUTIONS = ['normal']
    pipeline.VARIANCES = [1.0]
    
    df_final = pipeline.run_all(
        excel_filename="resultados_TEST_DOBLE_MODALIDAD.xlsx",
        batch_size=4
    )
    
    analisis_completo_doble_modalidad(df_final)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s")
    
    return df_final


def main_rango_d_doble_modalidad(d_min=1, d_max=5):
    """
    Rango personalizado de d con ambas modalidades.
    
    Args:
        d_min: Valor mínimo de d (default: 1)
        d_max: Valor máximo de d (default: 5)
    
    Ejemplo: main_rango_d_doble_modalidad(d_min=1, d_max=5)
    """
    start_time = time.time()
    
    print("="*80)
    print(f"SIMULACIÓN ARIMA d={d_min},...,{d_max} - DOBLE MODALIDAD")
    print("="*80)
    
    from pipeline import PipelineARIMA_MultiD_DobleModalidad
    
    pipeline = PipelineARIMA_MultiD_DobleModalidad(
        n_boot=1000, seed=42, verbose=False
    )
    
    # Configurar rango de d
    pipeline.D_VALUES = list(range(d_min, d_max + 1))
    
    total_base_scenarios = (len(pipeline.D_VALUES) * len(pipeline.ARMA_CONFIGS) * 
                           len(pipeline.DISTRIBUTIONS) * len(pipeline.VARIANCES))
    
    print(f"📊 Escenarios base: {total_base_scenarios}")
    print(f"   • Valores de d: {pipeline.D_VALUES}")
    print(f"   • Modalidades: 2 (SIN_DIFF + CON_DIFF)")
    print(f"   • Filas esperadas: ~{total_base_scenarios * 2 * 13}")
    print("="*80 + "\n")
    
    df_final = pipeline.run_all(
        excel_filename=f"resultados_ARIMA_d{d_min}_a_d{d_max}_DOBLE_MOD.xlsx",
        batch_size=20
    )
    
    analisis_completo_doble_modalidad(df_final)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s ({elapsed/3600:.2f} horas)")
    
    return df_final


def comparar_d_especificos_doble(d_lista=[1, 3, 5, 10]):
    """
    Compara valores específicos de d en ambas modalidades.
    
    Args:
        d_lista: Lista de valores de d a comparar (default: [1, 3, 5, 10])
    
    Ejemplo: comparar_d_especificos_doble([1, 5, 10])
    """
    start_time = time.time()
    
    print("="*80)
    print(f"COMPARACIÓN d={d_lista} - DOBLE MODALIDAD")
    print("="*80)
    
    from pipeline import PipelineARIMA_MultiD_DobleModalidad
    
    pipeline = PipelineARIMA_MultiD_DobleModalidad(
        n_boot=1000, seed=42, verbose=False
    )
    
    # Configurar valores específicos de d
    pipeline.D_VALUES = d_lista
    
    total_base_scenarios = (len(pipeline.D_VALUES) * len(pipeline.ARMA_CONFIGS) * 
                           len(pipeline.DISTRIBUTIONS) * len(pipeline.VARIANCES))
    
    print(f"📊 Escenarios base: {total_base_scenarios}")
    print(f"   • Modalidades: 2 (SIN_DIFF + CON_DIFF)")
    print("="*80 + "\n")
    
    filename = f"resultados_ARIMA_d_{'_'.join(map(str, d_lista))}_DOBLE_MOD.xlsx"
    df_final = pipeline.run_all(excel_filename=filename, batch_size=20)
    
    analisis_completo_doble_modalidad(df_final)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s ({elapsed/3600:.2f} horas)")
    
    return df_final

# ============================================================================
# ============================================================================
#  SIMULACION Tamaños crecientes
# ============================================================================
# ============================================================================

def main_comparacion_todos_procesos():
    """
    Ejecuta el estudio completo de tamaños para ARMA, ARIMA y SETAR
    """
    import time
    start_time = time.time()
    
    print("\n" + "="*80)
    print("🚀 INICIANDO PIPELINE UNIFICADO - TAMAÑOS CRECIENTES")
    print("="*80)
    print("\n📋 Configuración:")
    print("  • 3 tipos de procesos (ARMA, ARIMA, SETAR)")
    print("  • 7 configuraciones por proceso = 21 total")
    print("  • 5 tamaños diferentes (120, 240, 360, 600, 1200)")
    print("  • Proporción fija: 83% train / 17% calib")
    print("  • 5 distribuciones de ruido")
    print("  • 4 niveles de varianza")
    print("  • 12 pasos de predicción por escenario")
    print(f"  • TOTAL ESCENARIOS: 21 × 5 × 5 × 4 = 2,100")
    print(f"  • TOTAL FILAS: 2,100 × 12 = 25,200\n")
    
    all_results = []
    
    for proceso_tipo in ['ARMA', 'ARIMA', 'SETAR']:
        print(f"\n{'='*80}")
        print(f"📊 PROCESANDO: {proceso_tipo}")
        print(f"{'='*80}\n")
        
        pipeline = Pipeline140_TamanosCrecientes(
            n_boot=1000,
            seed=42,
            verbose=False,
            proceso_tipo=proceso_tipo
        )
        
        df_proceso = pipeline.run_all(
            excel_filename=f"RESULTADOS_TAMANOS_{proceso_tipo}.xlsx",
            batch_size=20,
            max_workers=12,
            save_frequency=3
        )
        
        all_results.append(df_proceso)
        print(f"✅ {proceso_tipo} completado: {len(df_proceso)} filas\n")
    
    # Consolidar
    print(f"\n{'='*80}")
    print("🔄 CONSOLIDANDO RESULTADOS")
    print(f"{'='*80}\n")
    
    df_final = pd.concat(all_results, ignore_index=True)
    output_file = "RESULTADOS_TAMANOS_TODOS.xlsx"
    df_final.to_excel(output_file, index=False)
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*80}")
    print("✅ PIPELINE COMPLETADO")
    print(f"{'='*80}")
    print(f"⏱  Tiempo total: {elapsed:.1f}s ({elapsed/3600:.2f} horas)")
    print(f"📊 Filas generadas: {len(df_final)}")
    print(f"💾 Archivo consolidado: {output_file}")
    print(f"📁 Archivos individuales:")
    for tipo in ['ARMA', 'ARIMA', 'SETAR']:
        print(f"   • RESULTADOS_TAMANOS_{tipo}.xlsx")
    print(f"{'='*80}\n")
    
    return df_final


def main_test_tamanos_reducido():
    """Test rápido con subset reducido"""
    print("\n" + "="*80)
    print("🧪 TEST REDUCIDO - TAMAÑOS CRECIENTES")
    print("="*80)
    
    pipeline = Pipeline140_TamanosCrecientes(
        n_boot=100,
        seed=42,
        verbose=True,
        proceso_tipo='ARMA'
    )
    
    # Reduce para test
    pipeline.ARMA_CONFIGS = pipeline.ARMA_CONFIGS[:2]
    pipeline.SIZE_COMBINATIONS = [
        {'tag': 'N=120', 'n_total': 120, 'n_train': 100, 'n_calib': 20},
        {'tag': 'N=240', 'n_total': 240, 'n_train': 199, 'n_calib': 41}
    ]
    pipeline.DISTRIBUTIONS = ['normal', 't-student']
    pipeline.VARIANCES = [0.5, 1.0]
    
    df_test = pipeline.run_all(
        excel_filename="TEST_TAMANOS.xlsx",
        batch_size=4,
        max_workers=2,
        save_frequency=2
    )
    
    print(f"\n✅ Test completado: {len(df_test)} filas")
    print(f"📁 Archivo: TEST_TAMANOS.xlsx")
    
    # Verifica NaN
    model_cols = [c for c in df_test.columns if c not in 
                  ['Paso', 'Tipo_Proceso', 'Proceso', 'Distribución', 'Varianza', 
                   'N_Train', 'N_Calib', 'N_Total', 'Size']]
    
    print("\n📊 Verificación de NaN:")
    nan_counts = df_test.isna().sum()
    for col in model_cols:
        if col in nan_counts and nan_counts[col] > 0:
            print(f"   ⚠️ {col}: {nan_counts[col]} NaN")
        elif col in df_test.columns:
            print(f"   ✅ {col}: Sin NaN")
    
    return df_test


# ============================================================================
# ============================================================================
#  SIMULACION proporciones
# ============================================================================
# ============================================================================
import pandas as pd
import numpy as np
import time


def main_proporciones_240_completo():
    """
    Ejecuta el estudio completo de proporciones para ARMA, ARIMA y SETAR
    """
    start_time = time.time()
    
    print("\n" + "="*80)
    print("🚀 INICIANDO PIPELINE UNIFICADO - PROPORCIONES 240")
    print("="*80)
    print("\n📋 Configuración:")
    print("  • 3 tipos de procesos (ARMA, ARIMA, SETAR)")
    print("  • 7 configuraciones por proceso = 21 total")
    print("  • 5 proporciones de calibración (10% a 50%)")
    print("  • 5 distribuciones de ruido")
    print("  • 4 niveles de varianza")
    print("  • 12 pasos de predicción por escenario")
    print(f"  • TOTAL ESCENARIOS: 21 × 5 × 5 × 4 = 2,100")
    print(f"  • TOTAL FILAS: 2,100 × 12 = 25,200\n")
    
    all_results = []
    
    for proceso_tipo in ['ARMA', 'ARIMA', 'SETAR']:
        print(f"\n{'='*80}")
        print(f"📊 PROCESANDO: {proceso_tipo}")
        print(f"{'='*80}\n")
        
        # Crear pipeline para este proceso
        pipeline = Pipeline240_ProporcionesVariables(
            n_boot=1000,
            seed=42,
            verbose=False,
            proceso_tipo=proceso_tipo
        )
        
        # Ejecutar
        df_proceso = pipeline.run_all(
            excel_filename=f"RESULTADOS_PROPORCIONES_240_{proceso_tipo}.xlsx",
            batch_size=20,
            max_workers=None,  # Auto-detecta
            save_frequency=3
        )
        
        all_results.append(df_proceso)
        print(f"✅ {proceso_tipo} completado: {len(df_proceso)} filas\n")
    
    # Consolidar resultados
    print(f"\n{'='*80}")
    print("🔄 CONSOLIDANDO RESULTADOS")
    print(f"{'='*80}\n")
    
    df_final = pd.concat(all_results, ignore_index=True)
    output_file = "RESULTADOS_PROPORCIONES_240_TODOS.xlsx"
    df_final.to_excel(output_file, index=False)
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*80}")
    print("✅ PIPELINE COMPLETADO")
    print(f"{'='*80}")
    print(f"⏱  Tiempo total: {elapsed:.1f}s ({elapsed/3600:.2f} horas)")
    print(f"📊 Filas generadas: {len(df_final)}")
    print(f"💾 Archivo consolidado: {output_file}")
    print(f"📁 Archivos individuales:")
    for tipo in ['ARMA', 'ARIMA', 'SETAR']:
        print(f"   • RESULTADOS_PROPORCIONES_240_{tipo}.xlsx")
    print(f"{'='*80}\n")
    
    return df_final

def test_proporciones_240_rapido():
    """
    Test rápido con subset reducido para validación
    """
    print("\n" + "="*80)
    print("🧪 TEST REDUCIDO - PROPORCIONES 240")
    print("="*80)
    
    # Solo ARMA, 2 configs, 2 proporciones, 2 dist, 2 var
    pipeline = Pipeline240_ProporcionesVariables(
        n_boot=100,  # Menos bootstraps
        seed=42,
        verbose=True,
        proceso_tipo='ARMA'
    )
    
    # Reduce configuraciones para test
    pipeline.CONFIGS['ARMA'] = [
        {'nombre': 'AR(1)', 'phi': [0.9], 'theta': []},
        {'nombre': 'ARMA(1,1)', 'phi': [0.6], 'theta': [0.3]}
    ]
    pipeline.SIZE_COMBINATIONS = [
        {'prop_tag': '10%', 'n_train': 216, 'n_calib': 24, 'prop_val': 0.10},
        {'prop_tag': '50%', 'n_train': 120, 'n_calib': 120, 'prop_val': 0.50}
    ]
    pipeline.DISTRIBUTIONS = ['normal', 't-student']
    pipeline.VARIANCES = [0.5, 1.0]
    
    # Ejecuta test
    df_test = pipeline.run_all(
        excel_filename="TEST_PROPORCIONES_240.xlsx",
        batch_size=4,
        max_workers=None,
        save_frequency=2
    )
    
    print(f"\n✅ Test completado: {len(df_test)} filas")
    print(f"📁 Archivo: TEST_PROPORCIONES_240.xlsx")
    
    # Verifica que no haya NaN
    nan_counts = df_test.isna().sum()
    model_cols = [c for c in df_test.columns if c not in ['Paso', 'Proceso', 'Distribución', 'Varianza', 'N_Train', 'N_Calib', 'Prop_Calib']]
    
    print("\n📊 Verificación de NaN:")
    for col in model_cols:
        if col in nan_counts and nan_counts[col] > 0:
            print(f"   ⚠️ {col}: {nan_counts[col]} NaN")
        elif col in df_test.columns:
            print(f"   ✅ {col}: Sin NaN")
    
    return df_test