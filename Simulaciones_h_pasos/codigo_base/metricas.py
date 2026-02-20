# metricas.py (VERSIÓN ULTRA-OPTIMIZADA)

import numpy as np
from numba import jit, prange

@jit(nopython=True, fastmath=True, cache=True)
def crps(F_samples: np.ndarray, x: float) -> float:
    """
    CRPS optimizado con Numba JIT.
    CRPS(F, x) = E_F|X - x| - (1/2) * E_F|X - X'|
    """
    n = len(F_samples)
    if n == 0:
        return np.nan
    
    # Término 1: E_F|X - x|
    term1 = 0.0
    for i in range(n):
        term1 += abs(F_samples[i] - x)
    term1 /= n
    
    # Término 2: (1/2) * E_F|X - X'| - usando fórmula eficiente
    # Para muestras ordenadas: E|X-X'| = (2/n²) * Σᵢ (2i - n - 1) * X_{(i)}
    sorted_samples = np.sort(F_samples)
    term2 = 0.0
    for i in range(n):
        term2 += (2.0 * i - n + 1.0) * sorted_samples[i]
    term2 = abs(term2) / (n * n)
    
    return term1 - term2


@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def ecrps_fast(forecast_samples: np.ndarray, ground_truth_samples: np.ndarray) -> float:
    """
    ECRPS ultra-optimizado con Numba paralelo.
    Pre-calcula término constante y paraleliza sobre ground_truth.
    """
    n_f = len(forecast_samples)
    n_g = len(ground_truth_samples)
    
    if n_f == 0 or n_g == 0:
        return np.nan
    
    # Pre-ordenar forecast para cálculo eficiente del término 2
    sorted_forecast = np.sort(forecast_samples)
    
    # Pre-calcular término 2 (constante para todos los g_i)
    term2_base = 0.0
    for i in range(n_f):
        term2_base += (2.0 * i - n_f + 1.0) * sorted_forecast[i]
    term2_base = abs(term2_base) / (n_f * n_f)
    
    # Calcular CRPS para cada muestra de ground truth en paralelo
    total_crps = 0.0
    for k in prange(n_g):
        g_k = ground_truth_samples[k]
        
        # Término 1 para este g_k
        term1 = 0.0
        for i in range(n_f):
            term1 += abs(forecast_samples[i] - g_k)
        term1 /= n_f
        
        total_crps += (term1 - term2_base)
    
    return total_crps / n_g


def ecrps(samples_F: np.ndarray, samples_G: np.ndarray) -> float:
    """Wrapper para compatibilidad."""
    F = np.asarray(samples_F, dtype=np.float64).flatten()
    G = np.asarray(samples_G, dtype=np.float64).flatten()
    
    if len(F) == 0 or len(G) == 0:
        return np.nan
    
    return ecrps_fast(F, G)