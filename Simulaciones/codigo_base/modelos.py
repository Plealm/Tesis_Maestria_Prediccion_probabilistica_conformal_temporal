# MODELOS CORREGIDOS - Sin re-entrenamiento en rolling window
# Solo incluyo las clases que necesitan corrección

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Tuple
from dataclasses import dataclass
import gc
import os
import warnings

# Configurar TensorFlow ANTES de importar
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
warnings.filterwarnings("ignore", category=UserWarning)

# Imports condicionales
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from bayes_opt import BayesianOptimization
    BAYESOPT_AVAILABLE = True
except ImportError:
    BAYESOPT_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.utils import check_random_state
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    try:
        tf.config.set_visible_devices([], 'GPU')
    except:
        pass
    from tensorflow.keras import layers, optimizers, Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

@dataclass
class CVResult:
    param_value: Union[int, float]
    mean_crps: float
    std_crps: float
    fold_scores: List[float]


class CircularBlockBootstrapModel:
    """CBB CORREGIDO: Congela block_length después de optimización."""

    def __init__(self, block_length: Union[int, str] = 'auto', n_boot: int = 1000,
                 random_state: int = 42, verbose: bool = False,
                 hyperparam_ranges: Dict = None, optimize: bool = True):
        self.block_length = block_length
        self.n_boot = n_boot
        self.random_state = random_state
        self.verbose = verbose
        self.hyperparam_ranges = hyperparam_ranges or {'block_length': [2, 50]}
        self.optimize = optimize
        self.rng = np.random.default_rng(random_state)
        self.best_params = {}
        self._frozen_block_length = None  # Valor congelado

    def _determine_block_length_optimal(self, n: int) -> int:
        """Heurística Politis-White (2004): l ≈ 1.5 × n^(1/3)"""
        if isinstance(self.block_length, (int, np.integer)):
            return max(2, int(self.block_length))
        
        l_opt = max(2, int(round(1.5 * (n ** (1/3)))))
        min_l, max_l = self.hyperparam_ranges.get('block_length', [2, min(50, n//2)])
        return min(max(l_opt, min_l), max_l)

    def freeze_hyperparameters(self, train_data: np.ndarray):
        """CRÍTICO: Congela block_length basado en datos de entrenamiento+calibración."""
        optimal_l = self._determine_block_length_optimal(len(train_data))
        self._frozen_block_length = optimal_l
        self.optimize = False
        if self.verbose:
            print(f"  CBB congelado: block_length={optimal_l}")

    def fit_predict(self, history: Union['pd.DataFrame', np.ndarray]) -> np.ndarray:
        """
        CORREGIDO: Usa block_length congelado si existe.
        NO re-optimiza en ventana rolling.
        """
        import pandas as pd
        series = history['valor'].values if isinstance(history, pd.DataFrame) else np.asarray(history).flatten()
        n = len(series)
        
        if n < 10:
            return np.full(self.n_boot, np.mean(series[-min(8, n):]))
        
        # FIX: Usar valor congelado si existe
        if self._frozen_block_length is not None:
            l = self._frozen_block_length
        elif isinstance(self.block_length, (int, np.integer)):
            l = max(2, int(self.block_length))
        else:
            # Solo calcular si no está congelado (primer paso)
            l = self._determine_block_length_optimal(n)
            if self.optimize:
                self.best_params = {'block_length': l}
        
        # CBB estándar
        within_block_pos = n % l
        starts = self.rng.integers(0, n, size=self.n_boot)
        positions = (starts + within_block_pos) % n
        return series[positions]


class SieveBootstrapModel:
    """Sieve Bootstrap CORREGIDO: Congela parámetros AR después de optimización."""

    def __init__(self, order: Union[int, str] = 'auto', n_boot: int = 1000,
                 random_state: int = 42, verbose: bool = False,
                 hyperparam_ranges: Dict = None, optimize: bool = True):
        self.order = order
        self.n_boot = n_boot
        self.random_state = random_state
        self.verbose = verbose
        self.hyperparam_ranges = hyperparam_ranges or {'order': [1, 20]}
        self.optimize = optimize
        self.rng = np.random.default_rng(random_state)
        self.best_params = {}
        
        # Cache para modelo congelado
        self._frozen_order = None
        self._frozen_params = None      # Parámetros AR congelados
        self._frozen_residuals = None   # Residuos congelados
        self._is_frozen = False

    def _determine_order_aic(self, series: np.ndarray) -> int:
        """AIC con 5 candidatos estratégicos."""
        from statsmodels.tsa.ar_model import AutoReg
        
        if isinstance(self.order, (int, np.integer)):
            return max(1, int(self.order))
        
        n = len(series)
        min_p, max_p = self.hyperparam_ranges.get('order', [1, min(20, n//3)])
        
        candidates = sorted(set([
            1, 2, max(3, min_p), (min_p + max_p) // 2, min(max_p, n//4)
        ]))
        candidates = [c for c in candidates if min_p <= c <= min(max_p, n//3)]
        
        best_aic, best_p = np.inf, 1
        for p in candidates:
            try:
                aic = AutoReg(series, lags=p).fit().aic
                if aic < best_aic:
                    best_aic, best_p = aic, p
            except:
                continue
        return best_p

    def freeze_hyperparameters(self, train_data: np.ndarray):
        """
        CRÍTICO: Congela order Y ajusta el modelo AR UNA SOLA VEZ.
        Los parámetros y residuos se calculan aquí y se reutilizan.
        """
        from statsmodels.tsa.ar_model import AutoReg
        
        series = train_data.flatten() if hasattr(train_data, 'flatten') else np.asarray(train_data)
        
        # Determinar orden óptimo
        optimal_p = self._determine_order_aic(series)
        self._frozen_order = optimal_p
        
        # Ajustar modelo AR con los datos de entrenamiento+calibración
        try:
            model = AutoReg(series, lags=optimal_p).fit()
            self._frozen_params = model.params.copy()
            self._frozen_residuals = model.resid - np.mean(model.resid)
            self._is_frozen = True
            
            if self.verbose:
                print(f"  Sieve Bootstrap congelado: order={optimal_p}, "
                      f"n_residuos={len(self._frozen_residuals)}")
        except Exception as e:
            if self.verbose:
                print(f"  Error congelando Sieve: {e}")
            self._is_frozen = False
        
        self.optimize = False

    def fit_predict(self, history: Union['pd.DataFrame', np.ndarray]) -> np.ndarray:
        """
        CORREGIDO: Usa parámetros congelados si existen.
        Solo los últimos p valores de la serie se usan para predecir.
        NO re-ajusta el modelo AR.
        """
        from statsmodels.tsa.ar_model import AutoReg
        import pandas as pd
        
        series = history['valor'].values if isinstance(history, pd.DataFrame) else np.asarray(history).flatten()
        n = len(series)
        
        if n < 10:
            return np.full(self.n_boot, np.mean(series[-min(8, n):]))
        
        # Determinar orden a usar
        if self._frozen_order is not None:
            p = self._frozen_order
        elif isinstance(self.order, (int, np.integer)):
            p = max(1, int(self.order))
        else:
            # Solo calcular si no está congelado (primer paso)
            p = self._determine_order_aic(series)
            if self.optimize:
                self.best_params = {'order': p}
        
        if p >= n - 1:
            p = max(1, n // 2)
        
        try:
            # FIX: Usar parámetros congelados si están disponibles
            if self._is_frozen and self._frozen_params is not None:
                # Usar parámetros y residuos congelados
                params = self._frozen_params
                residuals = self._frozen_residuals
                
                # Verificar que p coincida con los parámetros congelados
                expected_p = len(params) - 1  # params incluye intercepto
                if p != expected_p:
                    p = expected_p
            else:
                # NO CONGELADO: ajustar modelo nuevo (solo primer paso)
                model = AutoReg(series, lags=p).fit()
                params = model.params
                residuals = model.resid - np.mean(model.resid)
            
            # Predicción usando los últimos p valores de la serie ACTUAL
            last_p = series[-p:][::-1]
            ar_pred = params[0] + np.dot(params[1:], last_p)
            
            # Bootstrap de residuos (siempre de los residuos guardados/calculados)
            boot_resid = self.rng.choice(residuals, size=self.n_boot, replace=True)
            
            return ar_pred + boot_resid
            
        except Exception as e:
            if self.verbose:
                print(f"  Sieve error: {e}")
            return np.full(self.n_boot, np.mean(series[-min(8, n):]))


class LSPM:
    """
    LSPM - Least Squares Prediction Machine
    
    Retorna residuos conformales sin ponderar.
    Congela n_lags después de optimización.
    """
    
    def __init__(self, random_state=42, verbose=False):
        self.version = 'studentized'
        self.random_state = random_state
        self.verbose = verbose
        self.rng = np.random.default_rng(random_state)
        self.n_lags = None
        self.best_params = {}
        
        # Parámetros congelados
        self._frozen_n_lags = None
        self._is_frozen = False

    def optimize_hyperparameters(self, df, reference_noise):
        """Optimización trivial para LSPM (solo estructura)."""
        self.best_params = {'version': self.version}
        return None, -1.0

    def freeze_hyperparameters(self, train_data: np.ndarray):
        """
        CRÍTICO: Congela n_lags basado en datos de entrenamiento+calibración.
        """
        n = len(train_data)
        self._frozen_n_lags = max(1, int(n**(1/3)))
        self._is_frozen = True
        
        if self.verbose:
            print(f"  LSPM congelado: n_lags={self._frozen_n_lags}")

    def _calculate_critical_values(self, values: np.ndarray) -> np.ndarray:
        """
        Calcula residuos conformales studentizados.
        Usa n_lags congelado si está disponible.
        """
        # Determinar p (número de lags)
        if self._is_frozen and self._frozen_n_lags is not None:
            p = self._frozen_n_lags
        else:
            p = self.n_lags if self.n_lags and self.n_lags > 0 else max(1, int(len(values)**(1/3)))
        
        n = len(values)
        
        if n < 2 * p + 2:
            return np.array([])
        
        # Construir matrices de diseño
        y_full = values[p:]
        n_obs = len(y_full)
        
        # X_full: cada fila son los p lags
        X_full = np.column_stack([values[p-i-1:n-i-1] for i in range(p)])
        
        X_train = X_full[:-1]
        y_train = y_full[:-1]
        x_test = X_full[-1]
        
        # Agregar intercepto
        X_train_b = np.column_stack([np.ones(len(X_train)), X_train])
        x_test_b = np.concatenate([[1], x_test])
        X_bar = np.vstack([X_train_b, x_test_b])
        
        try:
            # Matriz hat: H = X(X'X)^{-1}X'
            XtX_inv = np.linalg.pinv(X_bar.T @ X_bar)
            H_bar = X_bar @ XtX_inv @ X_bar.T
        except np.linalg.LinAlgError:
            return np.array([])
        
        n_train = len(y_train)
        h_ii = np.diag(H_bar)[:n_train]
        h_n = H_bar[-1, -1]
        h_in = H_bar[:n_train, -1]
        h_ni = H_bar[-1, :n_train]
        
        # Predicciones leave-one-out
        y_hat_full = H_bar[:n_train, :n_train] @ y_train
        
        # Filtrar casos válidos
        valid = (np.abs(1 - h_ii) > 1e-10) & (np.abs(1 - h_n) > 1e-10)
        
        if not np.any(valid):
            return np.array([])
        
        # Calcular B_i para casos válidos
        sqrt_1_minus_h_n = np.sqrt(np.maximum(1 - h_n, 1e-10))
        sqrt_1_minus_h_ii = np.sqrt(np.maximum(1 - h_ii[valid], 1e-10))
        
        B_i = sqrt_1_minus_h_n + h_in[valid] / sqrt_1_minus_h_ii
        
        # Término 1: suma ponderada de y_train
        term1 = np.dot(h_ni, y_train) / sqrt_1_minus_h_n
        
        # Término 2: residuos studentizados
        resid_i = y_train[valid] - y_hat_full[valid]
        term2 = resid_i / sqrt_1_minus_h_ii
        
        # Filtrar B_i válidos
        valid_B = np.abs(B_i) > 1e-10
        
        if not np.any(valid_B):
            return np.array([])
        
        critical_values = (term1 + term2[valid_B]) / B_i[valid_B]
        
        return critical_values

    def fit_predict(self, df) -> np.ndarray:
        """
        Retorna residuos conformales directamente (distribución empírica sin ponderar).
        """
        values = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df)
        critical = self._calculate_critical_values(values.astype(np.float64))
        
        if len(critical) == 0:
            return np.full(1000, np.mean(values[-min(8, len(values)):]))
        
        return critical


class LSPMW(LSPM):
    """
    LSPMW - Least Squares Prediction Machine with Weighted Residuals
    
    Implementación según Barber et al. (2023) "Conformal Prediction Beyond Exchangeability".
    
    DIFERENCIA CON LSPM:
    -------------------
    - LSPM: Retorna residuos R_i (distribución empírica sin ponderar)
    - LSPMW: Retorna residuos R_i (para usar con pesos w_i = ρ^(n-i))
    
    Los pesos NO se aplican en fit_predict, sino externamente:
    - En cuantiles: Q_{1-α}(Σ w̃_i δ_{R_i})
    - En CRPS ponderado durante optimización
    
    Congela: n_lags y rho (NO congela residuos por adaptabilidad a drift)
    """
    
    def __init__(self, rho: float = 0.95, random_state: int = 42, 
                 verbose: bool = False):
        super().__init__(random_state=random_state, verbose=verbose)
        if not (0 < rho <= 1):
            raise ValueError("rho debe estar entre 0 y 1")
        self.rho = rho
        self.best_params = {'rho': rho}
        
        # Estado congelado: SOLO hiperparámetros
        self._frozen_rho = None
        # NO congelamos critical_values ni weights
    
    def freeze_hyperparameters(self, train_data: np.ndarray):
        """
        Congela SOLO hiperparámetros: n_lags y rho.
        
        NO congela critical_values porque:
        - LSPMW está diseñado para drift continuo
        - Los residuos deben recalcularse con datos actuales
        - w_i = ρ^(n-i) asume que "n" es el tiempo ACTUAL
        """
        # 1. Congelar n_lags (heredado de LSPM)
        super().freeze_hyperparameters(train_data)
        
        # 2. Congelar rho (usar el mejor encontrado por optimizador)
        if self.best_params and 'rho' in self.best_params:
            self._frozen_rho = self.best_params['rho']
        else:
            self._frozen_rho = self.rho
        
        if self.verbose:
            print(f"  ✅ LSPMW congelado: ρ={self._frozen_rho:.3f}, "
                  f"n_lags={self._frozen_n_lags}")
            print(f"     (residuos se recalculan dinámicamente)")
    
    def fit_predict(self, df: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Genera distribución predictiva PONDERADA mediante muestreo estratificado.
        
        DIFERENCIA CLAVE CON LSPM:
        --------------------------
        LSPM: Retorna residuos {R_1, ..., R_n} directamente (todos con igual probabilidad)
        LSPMW: Muestrea residuos con probabilidades proporcionales a w_i = ρ^(n-i)
        
        Esto NO es "expansión por replicación" sino MUESTREO PONDERADO:
        - Cada residuo R_i tiene probabilidad w̃_i de ser seleccionado
        - Residuos recientes tienen mayor probabilidad
        - El resultado es una distribución empírica ponderada
        
        Teoría (Barber et al. 2023):
        ----------------------------
        La distribución predictiva conformal es:
        F(y) = Σ w̃_i · 1{R_i ≤ y} + w̃_{n+1} · 1{+∞ ≤ y}
        
        Muestrear de esta distribución es equivalente a:
        1. Seleccionar índice i con probabilidad w̃_i
        2. Retornar R_i
        
        Esto es EXACTAMENTE lo que hace np.random.choice con p=weights.
        """
        values = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df)
        
        # Recalcular residuos usando n_lags congelado
        critical_vals = self._calculate_critical_values(values.astype(np.float64))
        
        # Fallback si no hay residuos
        if len(critical_vals) == 0:
            return np.full(1000, np.mean(values[-min(8, len(values)):]))
        
        # ═══ APLICAR PESOS PARA GENERAR DISTRIBUCIÓN PONDERADA ═══
        
        # Calcular pesos actuales (adaptativos al tamaño de ventana)
        rho = self._frozen_rho if self._is_frozen else self.rho
        n = len(critical_vals)
        
        # w_i = ρ^(n-1-i) para i=0,...,n-1
        exponents = np.arange(n - 1, -1, -1)
        weights = rho ** exponents
        
        # Normalizar para que sumen 1 (probabilidades de muestreo)
        # Nota: NO dividimos por (sum + 1) porque NO incluimos δ_{+∞}
        # Estamos muestreando SOLO de los residuos finitos
        weights = weights / weights.sum()
        
        # Muestrear 1000 valores de la distribución empírica ponderada
        # Esto implementa: F(y) = Σ w̃_i · 1{R_i ≤ y}
        n_samples = 1000
        sampled_residuals = self.rng.choice(
            critical_vals, 
            size=n_samples, 
            p=weights,
            replace=True
        )
        
        return sampled_residuals
    
    def get_weights(self, n: int) -> np.ndarray:
        """
        Calcula pesos w̃_i para n residuos según ecuación (10) del paper.
        
        Este método es AUXILIAR para uso externo (evaluación, conformal prediction).
        
        Args:
            n: Número de residuos
            
        Returns:
            Array de pesos normalizados w̃_i de tamaño n
            
        Formula:
            w_i = ρ^(n-1-i) para i=0,...,n-1 (más reciente = índice mayor)
            w̃_i = w_i / (Σw_j + 1) según ecuación (10) del paper
        """
        rho = self._frozen_rho if self._is_frozen else self.rho
        
        # w_i = ρ^(n-1-i) para i=0,...,n-1
        exponents = np.arange(n - 1, -1, -1)  # [n-1, n-2, ..., 1, 0]
        raw_weights = rho ** exponents
        
        # w̃_i = w_i / (Σw_j + 1) según ecuación (10)
        normalized_weights = raw_weights / (raw_weights.sum() + 1)
        
        return normalized_weights
    
    def compute_weighted_quantile(self, values: np.ndarray, alpha: float = 0.1) -> float:
        """
        Calcula cuantil ponderado Q_{1-α}(Σ w̃_i δ_{R_i} + w̃_{n+1} δ_{+∞})
        según ecuación (11) del paper.
        
        Este método es AUXILIAR para referencia teórica.
        
        Args:
            values: Residuos R_i ordenados cronológicamente
            alpha: Nivel de significancia
            
        Returns:
            Cuantil 1-α ponderado
        """
        if len(values) == 0:
            return np.inf
        
        # Calcular pesos para estos residuos
        weights = self.get_weights(len(values))
        
        # Ordenar valores y sus pesos
        sorted_idx = np.argsort(values)
        sorted_values = values[sorted_idx]
        sorted_weights = weights[sorted_idx]
        
        # Masa acumulada (sin incluir w̃_{n+1} que está en +∞)
        cumsum_weights = np.cumsum(sorted_weights)
        
        # Encontrar primer índice donde masa ≥ 1-α
        threshold = 1 - alpha
        idx = np.searchsorted(cumsum_weights, threshold, side='left')
        
        if idx >= len(sorted_values):
            # Si todos los pesos no llegan a 1-α, 
            # el peso restante w̃_{n+1} está en +∞
            return sorted_values[-1]
        
        return sorted_values[idx]


class AREPD:
    """AREPD CORREGIDO: Una vez congelado, NUNCA re-entrena."""
    
    def __init__(self, n_lags=5, rho=0.95, alpha=0.1, poly_degree=2,
                 random_state=42, verbose=False, optimize=True):
        from sklearn.utils import check_random_state
        
        self.n_lags = n_lags
        self.rho = rho
        self.alpha = alpha
        self.poly_degree = poly_degree
        self.random_state = random_state
        self.verbose = verbose
        self.optimize = optimize
        
        self.mean_val = None
        self.std_val = None
        self.rng = check_random_state(random_state)
        self.best_params = {}
        self._is_optimized = False
        
        # FIX CRÍTICO
        self._frozen_model = None
        self._frozen_mean = None
        self._frozen_std = None
        self._is_frozen = False  # NUEVO: flag explícito
        
        np.random.seed(random_state)
    
    def _create_lag_matrix(self, values: np.ndarray, n_lags: int, degree: int = 2):
        n = len(values) - n_lags
        if n <= 0:
            return np.array([]), np.array([])
        
        y = values[n_lags:]
        X_list = [np.ones((n, 1))]
        
        for lag in range(n_lags):
            lagged = values[lag:lag + n].reshape(-1, 1)
            for d in range(1, degree + 1):
                X_list.append(np.power(lagged, d))
        
        return np.hstack(X_list), y
    
    def freeze_hyperparameters(self, train_data: np.ndarray):
        """CRÍTICO: Entrena Ridge UNA VEZ y congela TODO."""
        from sklearn.linear_model import Ridge
        
        values = train_data.flatten() if hasattr(train_data, 'flatten') else np.asarray(train_data)
        
        if len(values) < self.n_lags * 2:
            self._is_frozen = False
            return
        
        if self.best_params:
            self.n_lags = self.best_params.get('n_lags', self.n_lags)
            self.rho = self.best_params.get('rho', self.rho)
            self.poly_degree = self.best_params.get('poly_degree', self.poly_degree)
        
        try:
            # Congelar scaler
            self._frozen_mean = np.nanmean(values)
            self._frozen_std = np.nanstd(values) + 1e-8
            normalized = (values - self._frozen_mean) / self._frozen_std
            
            X, y = self._create_lag_matrix(normalized, self.n_lags, self.poly_degree)
            if X.shape[0] == 0:
                self._is_frozen = False
                return
            
            # Pesos exponenciales
            weights = self.rho ** np.arange(len(y))[::-1]
            weights = weights / (weights.sum() + 1e-8)
            
            # ENTRENAR MODELO UNA SOLA VEZ
            model = Ridge(alpha=self.alpha, fit_intercept=False)
            model.fit(X, y, sample_weight=weights)
            
            self._frozen_model = model
            self._is_frozen = True
            
            if self.verbose:
                print(f"  AREPD congelado: n_lags={self.n_lags}, rho={self.rho}, "
                      f"mean={self._frozen_mean:.4f}, std={self._frozen_std:.4f}")
                
        except Exception as e:
            if self.verbose:
                print(f"  Error congelando AREPD: {e}")
            self._is_frozen = False
    
    def fit_predict(self, df) -> np.ndarray:
        """CORREGIDO: Si está congelado, SOLO predice."""
        try:
            values = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df)
            
            if len(values) < self.n_lags * 2:
                mean_val = self._frozen_mean if self._is_frozen else np.mean(values)
                return np.full(1000, mean_val)
            
            # VERIFICACIÓN CRÍTICA: Si está congelado, usar modelo existente
            if self._is_frozen:
                if self._frozen_model is None or self._frozen_mean is None:
                    return np.full(1000, np.mean(values))
                
                # Solo normalizar y predecir
                normalized = (values - self._frozen_mean) / self._frozen_std
                X, y = self._create_lag_matrix(normalized, self.n_lags, self.poly_degree)
                
                if X.shape[0] == 0:
                    return np.full(1000, self._frozen_mean)
                
                # Predicción con modelo congelado
                predictions = self._frozen_model.predict(X)
                samples = (predictions * self._frozen_std) + self._frozen_mean
                
                return samples
            
            # CÓDIGO ANTIGUO: Solo para primer paso
            if self.best_params:
                self.n_lags = self.best_params.get('n_lags', self.n_lags)
                self.rho = self.best_params.get('rho', self.rho)
                self.poly_degree = self.best_params.get('poly_degree', self.poly_degree)
            
            self.mean_val = np.nanmean(values)
            self.std_val = np.nanstd(values) + 1e-8
            normalized = (values - self.mean_val) / self.std_val
            
            X, y = self._create_lag_matrix(normalized, self.n_lags, self.poly_degree)
            if X.shape[0] == 0:
                return np.full(1000, self.mean_val)
            
            from sklearn.linear_model import Ridge
            weights = self.rho ** np.arange(len(y))[::-1]
            weights = weights / (weights.sum() + 1e-8)
            
            model = Ridge(alpha=self.alpha, fit_intercept=False)
            model.fit(X, y, sample_weight=weights)
            predictions = model.predict(X)
            
            samples = (predictions * self.std_val) + self.mean_val
            
            return samples
            
        except Exception as e:
            if self.verbose:
                print(f"    AREPD error: {e}")
            mean_val = self._frozen_mean if self._is_frozen else np.nanmean(df)
            return np.full(1000, mean_val if not np.isnan(mean_val) else 0)


class DeepARModel:
    """DeepAR CORREGIDO: Una vez congelado, NUNCA re-entrena."""
    
    def __init__(self, hidden_size=20, n_lags=5, num_layers=1, dropout=0.1, 
                 lr=0.01, batch_size=32, epochs=30, num_samples=1000,
                 random_state=42, verbose=False, optimize=True,
                 early_stopping_patience=5):
        
        import torch
        
        self.hidden_size = hidden_size
        self.n_lags = n_lags
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_samples = num_samples
        self.random_state = random_state
        self.verbose = verbose
        self.optimize = optimize
        self.early_stopping_patience = early_stopping_patience
        
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        self.best_params = {}
        self._is_optimized = False
        
        # FIX CRÍTICO: Modelo y scaler completamente congelados
        self._trained_model = None
        self._frozen_mean = None
        self._frozen_std = None
        self._training_history = []
        self._is_frozen = False  # NUEVO: flag explícito
        
        np.random.seed(random_state)
        torch.manual_seed(random_state)
    
    class _DeepARNN(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout):
            super().__init__()
            import torch.nn as nn
            drop = dropout if num_layers > 1 else 0
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                               batch_first=True, dropout=drop)
            self.fc_mu = nn.Linear(hidden_size, 1)
            self.fc_sigma = nn.Linear(hidden_size, 1)
        
        def forward(self, x):
            import torch
            out, _ = self.lstm(x)
            mu = self.fc_mu(out[:, -1, :])
            sigma = torch.exp(self.fc_sigma(out[:, -1, :])).clamp(min=1e-6, max=10)
            return mu, sigma
    
    def _train_with_early_stopping(self, X_t, y_t):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        n_train = int(0.8 * len(X_t))
        X_train, X_val = X_t[:n_train], X_t[n_train:]
        y_train, y_val = y_t[:n_train], y_t[n_train:]
        
        if len(X_val) < 5:
            X_val, y_val = X_train[-10:], y_train[-10:]
        
        self.model = self._DeepARNN(1, self.hidden_size, self.num_layers, self.dropout)
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        crit = nn.GaussianNLLLoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        self._training_history = []
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            perm = torch.randperm(len(X_train))
            
            for i in range(0, len(X_train), self.batch_size):
                idx = perm[i:i+self.batch_size]
                if len(idx) < 2:
                    continue
                mu, sig = self.model(X_train[idx])
                loss = crit(mu, y_train[idx], sig**2)
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss += loss.item()
            
            self.model.eval()
            with torch.no_grad():
                mu_val, sig_val = self.model(X_val)
                val_loss = crit(mu_val, y_val, sig_val**2).item()
            
            self._training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss / max(1, len(X_train) // self.batch_size),
                'val_loss': val_loss
            })
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= self.early_stopping_patience:
                if self.verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return self.model
    
    def freeze_hyperparameters(self, train_data: np.ndarray):
        """
        CRÍTICO: Entrena modelo UNA VEZ y congela TODO.
        Marca _is_frozen = True.
        """
        import torch
        
        values = train_data.flatten() if hasattr(train_data, 'flatten') else np.asarray(train_data)
        
        # Congelar scaler
        self._frozen_mean = np.nanmean(values)
        self._frozen_std = np.nanstd(values) + 1e-8
        
        # Aplicar hiperparámetros optimizados
        if self.best_params:
            self.n_lags = self.best_params.get('n_lags', self.n_lags)
            self.hidden_size = self.best_params.get('hidden_size', self.hidden_size)
            self.num_layers = self.best_params.get('num_layers', self.num_layers)
            self.dropout = self.best_params.get('dropout', self.dropout)
            self.lr = self.best_params.get('lr', self.lr)
        
        # ENTRENAR MODELO UNA SOLA VEZ
        try:
            norm_series = (values - self._frozen_mean) / self._frozen_std
            
            if len(norm_series) <= self.n_lags:
                self._is_frozen = False
                return
            
            X, y = [], []
            for i in range(len(norm_series) - self.n_lags):
                X.append(norm_series[i:i + self.n_lags])
                y.append(norm_series[i + self.n_lags])
            X, y = np.array(X), np.array(y)
            
            if len(X) < self.batch_size:
                self._is_frozen = False
                return
            
            X_t = torch.FloatTensor(X.reshape(-1, self.n_lags, 1))
            y_t = torch.FloatTensor(y.reshape(-1, 1))
            
            self._train_with_early_stopping(X_t, y_t)
            self._trained_model = self.model
            self._is_frozen = True
            
            if self.verbose:
                print(f"  DeepAR congelado: mean={self._frozen_mean:.4f}, "
                      f"std={self._frozen_std:.4f}, model_trained=True")
                
        except Exception as e:
            if self.verbose:
                print(f"  Error congelando DeepAR: {e}")
            self._is_frozen = False
    
    def fit_predict(self, df) -> np.ndarray:
        """
        CORREGIDO: Si está congelado, SOLO predice con modelo existente.
        NO re-entrena NUNCA después de freeze.
        """
        import torch
        import pandas as pd
        
        try:
            series = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df)
            
            # VERIFICACIÓN CRÍTICA: Si está congelado, usar modelo existente
            if self._is_frozen:
                if self._trained_model is None or self._frozen_mean is None:
                    return np.full(self.num_samples, np.mean(series))
                
                # Solo normalizar y predecir
                norm_series = (series - self._frozen_mean) / self._frozen_std
                
                if len(norm_series) < self.n_lags:
                    return np.full(self.num_samples, self._frozen_mean)
                
                # Predicción con modelo congelado
                self._trained_model.eval()
                with torch.no_grad():
                    last_seq = torch.FloatTensor(
                        norm_series[-self.n_lags:].reshape(1, self.n_lags, 1)
                    )
                    mu, sig = self._trained_model(last_seq)
                
                samples = np.random.normal(mu.item(), sig.item(), self.num_samples)
                samples = samples * self._frozen_std + self._frozen_mean
                
                return np.nan_to_num(samples, nan=self._frozen_mean)
            
            # CÓDIGO ANTIGUO: Solo para primer paso (antes de freeze)
            if self.best_params:
                self.n_lags = self.best_params.get('n_lags', self.n_lags)
                self.hidden_size = self.best_params.get('hidden_size', self.hidden_size)
            
            self.scaler_mean = np.nanmean(series)
            self.scaler_std = np.nanstd(series) + 1e-8
            norm_series = (series - self.scaler_mean) / self.scaler_std
            
            if len(norm_series) <= self.n_lags:
                return np.full(self.num_samples, self.scaler_mean)
            
            X, y = [], []
            for i in range(len(norm_series) - self.n_lags):
                X.append(norm_series[i:i + self.n_lags])
                y.append(norm_series[i + self.n_lags])
            X, y = np.array(X), np.array(y)
            
            if len(X) < self.batch_size:
                return np.full(self.num_samples, self.scaler_mean)
            
            X_t = torch.FloatTensor(X.reshape(-1, self.n_lags, 1))
            y_t = torch.FloatTensor(y.reshape(-1, 1))
            
            self._train_with_early_stopping(X_t, y_t)
            
            self.model.eval()
            with torch.no_grad():
                last_seq = torch.FloatTensor(norm_series[-self.n_lags:].reshape(1, self.n_lags, 1))
                mu, sig = self.model(last_seq)
            
            samples = np.random.normal(mu.item(), sig.item(), self.num_samples)
            samples = samples * self.scaler_std + self.scaler_mean
            
            return np.nan_to_num(samples, nan=self.scaler_mean)
            
        except Exception as e:
            if self.verbose:
                print(f"    DeepAR error: {e}")
            mean_val = self._frozen_mean if self._is_frozen else np.nanmean(df)
            return np.full(self.num_samples, mean_val if not np.isnan(mean_val) else 0)


# =============================================================================
# MondrianCPS - Mondrian Conformal Predictive System
# =============================================================================

class MondrianCPSModel:
    """Mondrian CPS CORREGIDO: Devuelve calibration scores directamente como LSPM."""
    
    def __init__(self, n_lags: int = 10, n_bins: int = 10, test_size: float = 0.25,
                 random_state: int = 42, verbose: bool = False, optimize: bool = True):
        try:
            import xgboost as xgb
            self.xgb = xgb
        except ImportError:
            raise ImportError("XGBoost requerido: pip install xgboost")
        
        self.n_lags = n_lags
        self.n_bins = n_bins
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose
        self.optimize = optimize
        self.rng = np.random.default_rng(random_state)
        
        self.base_model = self.xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=50,
            learning_rate=0.1,
            max_depth=3,
            random_state=self.random_state,
            n_jobs=1,
            verbosity=0
        )
        self.best_params = {}
        self._is_optimized = False
        self._fitted_artifacts = None
        self._is_frozen = False
    
    def _create_lag_matrix(self, series: np.ndarray):
        X, y = [], []
        for i in range(len(series) - self.n_lags):
            X.append(series[i:(i + self.n_lags)])
            y.append(series[i + self.n_lags])
        return np.array(X), np.array(y)
    
    def freeze_hyperparameters(self, train_data: np.ndarray):
        """CRÍTICO: Entrena XGBoost UNA VEZ y congela TODO."""
        values = train_data.flatten() if hasattr(train_data, 'flatten') else np.asarray(train_data)
        
        if self.best_params:
            self.n_lags = self.best_params.get('n_lags', self.n_lags)
            self.n_bins = self.best_params.get('n_bins', self.n_bins)
        
        try:
            if len(values) < self.n_lags * 2:
                self._is_frozen = False
                return
            
            X, y = self._create_lag_matrix(values)
            n_calib = max(10, int(len(X) * self.test_size))
            
            if n_calib >= len(X):
                self._fitted_artifacts = {
                    'fallback_mean': np.mean(values),
                    'is_fallback': True
                }
                self._is_frozen = True
                return
            
            X_train, X_calib = X[:-n_calib], X[-n_calib:]
            y_train, y_calib = y[:-n_calib], y[-n_calib:]
            
            # ENTRENAR MODELO UNA SOLA VEZ
            self.base_model.fit(X_train, y_train)
            calib_preds = self.base_model.predict(X_calib)
            
            bin_edges = None
            try:
                _, bin_edges = pd.qcut(calib_preds, self.n_bins, retbins=True, duplicates='drop')
            except:
                pass
            
            self._fitted_artifacts = {
                'calib_preds': calib_preds,
                'y_calib': y_calib,
                'bin_edges': bin_edges,
                'is_fallback': False
            }
            self._is_frozen = True
            
            if self.verbose:
                print(f"  MCPS congelado: n_lags={self.n_lags}, n_bins={self.n_bins}")
                
        except Exception as e:
            if self.verbose:
                print(f"  Error congelando MCPS: {e}")
            self._is_frozen = False
    
    def fit_predict(self, df) -> np.ndarray:
        """
        CORREGIDO: Devuelve calibration scores directamente (como LSPM).
        NO hace bootstrap ni agrega ruido artificial.
        """
        try:
            series = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df)
            
            # VERIFICACIÓN CRÍTICA: Si está congelado, usar modelo pre-entrenado
            if self._is_frozen:
                if self._fitted_artifacts is None:
                    return np.full(100, np.mean(series[-min(8, len(series)):]))
                
                if self._fitted_artifacts.get('is_fallback', False):
                    fallback = self._fitted_artifacts['fallback_mean']
                    return np.full(100, fallback)
                
                # ===== PREDICCIÓN CON MODELO CONGELADO =====
                x_test = series[-self.n_lags:].reshape(1, -1)
                point_pred = self.base_model.predict(x_test)[0]
                
                calib_preds = self._fitted_artifacts['calib_preds']
                y_calib = self._fitted_artifacts['y_calib']
                bin_edges = self._fitted_artifacts['bin_edges']
                
                # ===== MONDRIAN BINNING (LOCAL CALIBRATION) =====
                try:
                    if bin_edges is not None and len(bin_edges) > 1:
                        # Asignar predicciones de calibración a bins
                        bin_idx = np.digitize(calib_preds, bins=bin_edges) - 1
                        bin_idx = np.clip(bin_idx, 0, len(bin_edges) - 2)
                        
                        # Determinar bin del test point
                        test_bin = np.clip(np.digitize(point_pred, bins=bin_edges) - 1, 
                                          0, len(bin_edges) - 2)
                        
                        # Filtrar solo scores del mismo bin (MONDRIAN)
                        local_mask = (bin_idx == test_bin)
                        
                        # Si el bin está muy vacío, usar todos los datos
                        if np.sum(local_mask) < 5:
                            local_mask = np.ones(len(calib_preds), dtype=bool)
                    else:
                        # Sin binning válido, usar todos los datos (SCPS regular)
                        local_mask = np.ones(len(calib_preds), dtype=bool)
                except Exception as e:
                    if self.verbose:
                        print(f"    Binning falló: {e}, usando SCPS global")
                    local_mask = np.ones(len(calib_preds), dtype=bool)
                
                # ===== CALIBRATION SCORES (CONFORMAL PREDICTION) =====
                local_y = y_calib[local_mask]
                local_preds = calib_preds[local_mask]
                
                # Formula del paper: C_j = h(x_test) + (y_j - h(x_j))
                calibration_scores = point_pred + (local_y - local_preds)
                
                # FIX CRÍTICO: Devolver scores directamente (como LSPM)
                if len(calibration_scores) == 0:
                    return np.full(100, point_pred)
                
                return calibration_scores  # ← SIN BOOTSTRAP, SIN RUIDO
            
            # ===== CÓDIGO PARA PRIMER PASO (no congelado) =====
            if self.best_params:
                self.n_lags = self.best_params.get('n_lags', self.n_lags)
                self.n_bins = self.best_params.get('n_bins', self.n_bins)
            
            if len(series) < self.n_lags * 2:
                return np.full(100, np.mean(series[-min(8, len(series)):]))
            
            X, y = self._create_lag_matrix(series)
            n_calib = max(10, int(len(X) * self.test_size))
            
            if n_calib >= len(X):
                return np.full(100, np.mean(series[-min(8, len(series)):]))
            
            X_train, X_calib = X[:-n_calib], X[-n_calib:]
            y_train, y_calib = y[:-n_calib], y[-n_calib:]
            
            self.base_model.fit(X_train, y_train)
            calib_preds = self.base_model.predict(X_calib)
            
            x_test = series[-self.n_lags:].reshape(1, -1)
            point_pred = self.base_model.predict(x_test)[0]
            
            # Formula conformal: scores = predicción_test + errores_calibración
            calibration_scores = point_pred + (y_calib - calib_preds)
            
            if len(calibration_scores) == 0:
                return np.full(100, point_pred)
            
            return calibration_scores  # ← Devolver directamente
            
        except Exception as e:
            if self.verbose:
                print(f"    MCPS error: {e}")
            fallback = np.nanmean(series[-min(8, len(series)):]) if len(series) > 0 else 0
            return np.full(100, fallback)


# =============================================================================
# AV-MCPS - Adaptive Volatility Mondrian CPS
# =============================================================================

class AdaptiveVolatilityMondrianCPS:
    """
    AV-MCPS CORREGIDO: Devuelve calibration scores directamente (como LSPM).
    Congela modelo, bins Y volatilidad de referencia.
    """
    
    def __init__(self, n_lags: int = 15, n_pred_bins: int = 8, n_vol_bins: int = 4,
                 volatility_window: int = 20, test_size: float = 0.25,
                 random_state: int = 42, verbose: bool = False, optimize: bool = True):
        try:
            import xgboost as xgb
            self.xgb = xgb
        except ImportError:
            raise ImportError("XGBoost requerido: pip install xgboost")
        
        self.n_lags = n_lags
        self.n_pred_bins = n_pred_bins
        self.n_vol_bins = n_vol_bins
        self.volatility_window = volatility_window
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose
        self.optimize = optimize
        self.rng = np.random.default_rng(random_state)
        
        self.base_model = self.xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=50,
            learning_rate=0.1,
            max_depth=3,
            random_state=self.random_state,
            n_jobs=1,
            verbosity=0
        )
        self.best_params = {}
        self._is_optimized = False
        self._fitted_artifacts = None
        self._is_frozen = False  # ← AGREGAR
    
    def _create_lag_matrix(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(series) - self.n_lags):
            X.append(series[i:(i + self.n_lags)])
            y.append(series[i + self.n_lags])
        return np.array(X), np.array(y)
    
    def _calculate_volatility(self, series: np.ndarray) -> np.ndarray:
        """Calcula volatilidad rolling para calibración o predicción."""
        volatility = pd.Series(series).rolling(
            window=self.volatility_window, min_periods=1
        ).std().bfill().values
        return volatility[self.n_lags - 1: -1] if len(volatility) > self.n_lags else volatility
    
    def freeze_hyperparameters(self, train_data: np.ndarray):
        """CRÍTICO: Entrena XGBoost UNA VEZ y congela bins de predicción Y volatilidad."""
        values = train_data.flatten() if hasattr(train_data, 'flatten') else np.asarray(train_data)
        
        if self.best_params:
            self.n_lags = self.best_params.get('n_lags', self.n_lags)
            self.n_pred_bins = self.best_params.get('n_pred_bins', self.n_pred_bins)
            self.n_vol_bins = self.best_params.get('n_vol_bins', self.n_vol_bins)
            self.volatility_window = self.best_params.get('volatility_window', self.volatility_window)
        
        try:
            if len(values) < max(self.n_lags * 2, self.volatility_window):
                self._is_frozen = False
                return
            
            X, y = self._create_lag_matrix(values)
            vol_features = self._calculate_volatility(values)
            
            n_calib = max(10, int(len(X) * self.test_size))
            
            if n_calib >= len(X):
                self._fitted_artifacts = {
                    'fallback_mean': np.mean(values),
                    'is_fallback': True
                }
                self._is_frozen = True
                return
            
            X_train, X_calib = X[:-n_calib], X[-n_calib:]
            y_train, y_calib = y[:-n_calib], y[-n_calib:]
            
            # Calcular volatilidad de calibración
            if len(vol_features) >= n_calib:
                vol_calib = vol_features[-n_calib:]
            else:
                vol_calib = np.full(n_calib, np.std(values))
            
            # ENTRENAR MODELO UNA SOLA VEZ
            self.base_model.fit(X_train, y_train)
            calib_preds = self.base_model.predict(X_calib)
            
            # Crear bins para predicción y volatilidad
            pred_edges, vol_edges = None, None
            try:
                _, pred_edges = pd.qcut(calib_preds, self.n_pred_bins, retbins=True, duplicates='drop')
                _, vol_edges = pd.qcut(vol_calib, self.n_vol_bins, retbins=True, duplicates='drop')
            except Exception as e:
                if self.verbose:
                    print(f"    Warning: No se pudieron crear bins: {e}")
            
            self._fitted_artifacts = {
                'calib_preds': calib_preds,
                'y_calib': y_calib,
                'vol_calib': vol_calib,
                'pred_edges': pred_edges,
                'vol_edges': vol_edges,
                'is_fallback': False
            }
            self._is_frozen = True
            
            if self.verbose:
                print(f"  AV-MCPS congelado: n_lags={self.n_lags}, "
                      f"pred_bins={self.n_pred_bins}, vol_bins={self.n_vol_bins}")
                
        except Exception as e:
            if self.verbose:
                print(f"  Error congelando AV-MCPS: {e}")
            self._is_frozen = False
    
    def fit_predict(self, df) -> np.ndarray:
        """
        CORREGIDO: Devuelve calibration scores directamente (como LSPM).
        NO hace bootstrap ni agrega ruido artificial.
        """
        try:
            series = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df)
            
            # ===== VERIFICACIÓN CRÍTICA: Usar modelo congelado =====
            if self._is_frozen:
                if self._fitted_artifacts is None:
                    return np.full(100, np.mean(series[-min(8, len(series)):]))
                
                if self._fitted_artifacts.get('is_fallback', False):
                    fallback = self._fitted_artifacts['fallback_mean']
                    return np.full(100, fallback)
                
                # ===== PREDICCIÓN CON MODELO CONGELADO =====
                x_test = series[-self.n_lags:].reshape(1, -1)
                test_vol = np.std(series[-self.volatility_window:])
                
                point_pred = self.base_model.predict(x_test)[0]
                
                calib_preds = self._fitted_artifacts['calib_preds']
                y_calib = self._fitted_artifacts['y_calib']
                vol_calib = self._fitted_artifacts['vol_calib']
                pred_edges = self._fitted_artifacts['pred_edges']
                vol_edges = self._fitted_artifacts['vol_edges']
                
                # ===== ADAPTIVE VOLATILITY MONDRIAN BINNING =====
                try:
                    if pred_edges is not None and vol_edges is not None:
                        # Asignar calibration data a bins 2D (predicción × volatilidad)
                        pred_idx = np.clip(
                            np.digitize(calib_preds, pred_edges[:-1]) - 1, 
                            0, len(pred_edges) - 2
                        )
                        vol_idx = np.clip(
                            np.digitize(vol_calib, vol_edges[:-1]) - 1, 
                            0, len(vol_edges) - 2
                        )
                        
                        # Determinar bin del test point
                        test_pred_bin = np.clip(
                            np.digitize(point_pred, pred_edges[:-1]) - 1, 
                            0, len(pred_edges) - 2
                        )
                        test_vol_bin = np.clip(
                            np.digitize(test_vol, vol_edges[:-1]) - 1, 
                            0, len(vol_edges) - 2
                        )
                        
                        # Filtro 1: Mismo bin de predicción Y volatilidad (más restrictivo)
                        local_mask = (pred_idx == test_pred_bin) & (vol_idx == test_vol_bin)
                        
                        # Fallback 1: Si muy pocos datos, solo usar bin de predicción
                        if np.sum(local_mask) < 5:
                            local_mask = (pred_idx == test_pred_bin)
                            if self.verbose:
                                print(f"    Fallback: usando solo pred_bin={test_pred_bin}")
                            
                            # Fallback 2: Si aún muy pocos, usar todos los datos
                            if np.sum(local_mask) < 5:
                                local_mask = np.ones(len(calib_preds), dtype=bool)
                                if self.verbose:
                                    print(f"    Fallback: usando todos los datos")
                    else:
                        # Sin bins válidos, usar SCPS regular
                        local_mask = np.ones(len(calib_preds), dtype=bool)
                        if self.verbose:
                            print(f"    Sin bins válidos, usando SCPS global")
                            
                except Exception as e:
                    if self.verbose:
                        print(f"    Binning falló: {e}, usando SCPS global")
                    local_mask = np.ones(len(calib_preds), dtype=bool)
                
                # ===== CALIBRATION SCORES (CONFORMAL PREDICTION) =====
                local_y = y_calib[local_mask]
                local_preds = calib_preds[local_mask]
                
                # Formula del paper: C_j = h(x_test) + (y_j - h(x_j))
                calibration_scores = point_pred + (local_y - local_preds)
                
                # FIX CRÍTICO: Devolver scores directamente (como LSPM)
                if len(calibration_scores) == 0:
                    return np.full(100, point_pred)
                
                return calibration_scores  # ← SIN BOOTSTRAP, SIN RUIDO
            
            # ===== CÓDIGO PARA PRIMER PASO (no congelado) =====
            if self.best_params:
                self.n_lags = self.best_params.get('n_lags', self.n_lags)
                self.n_pred_bins = self.best_params.get('n_pred_bins', self.n_pred_bins)
                self.n_vol_bins = self.best_params.get('n_vol_bins', self.n_vol_bins)
                self.volatility_window = self.best_params.get('volatility_window', self.volatility_window)
            
            if len(series) < max(self.n_lags * 2, self.volatility_window):
                return np.full(100, np.mean(series[-min(8, len(series)):]))
            
            X, y = self._create_lag_matrix(series)
            vol_features = self._calculate_volatility(series)
            
            n_calib = max(10, int(len(X) * self.test_size))
            
            if n_calib >= len(X):
                return np.full(100, np.mean(series[-min(8, len(series)):]))
            
            X_train, X_calib = X[:-n_calib], X[-n_calib:]
            y_train, y_calib = y[:-n_calib], y[-n_calib:]
            
            if len(vol_features) >= n_calib:
                vol_calib = vol_features[-n_calib:]
            else:
                vol_calib = np.full(n_calib, np.std(series))
            
            self.base_model.fit(X_train, y_train)
            calib_preds = self.base_model.predict(X_calib)
            
            x_test = series[-self.n_lags:].reshape(1, -1)
            point_pred = self.base_model.predict(x_test)[0]
            
            # Formula conformal: scores = predicción_test + errores_calibración
            calibration_scores = point_pred + (y_calib - calib_preds)
            
            if len(calibration_scores) == 0:
                return np.full(100, point_pred)
            
            return calibration_scores  # ← Devolver directamente
            
        except Exception as e:
            if self.verbose:
                print(f"    AV-MCPS error: {e}")
            fallback = np.nanmean(series[-min(8, len(series)):]) if len(series) > 0 else 0
            return np.full(100, fallback)

# =============================================================================
# EnCQR-LSTM - Ensemble Conformalized Quantile Regression with LSTM
# =============================================================================

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

class EnCQR_LSTM_Model:
    """
    EnCQR-LSTM: Implementación corregida siguiendo el paper 
    'Ensemble Conformalized Quantile Regression for Probabilistic Time Series Forecasting'.
    
    CORRECCIÓN: En lugar de interpolar cuantiles (que causa bimodalidad), 
    ajustamos una distribución paramétrica a los cuantiles conformalizados.
    """
   
    def __init__(self, n_lags: int = 20, B: int = 3, units: int = 32, n_layers: int = 2,
                 lr: float = 0.005, batch_size: int = 16, epochs: int = 20,
                 num_samples: int = 1000, random_state: int = 42, verbose: bool = False,
                 optimize: bool = True, alpha: float = 0.05):
       
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, Model, optimizers
            from sklearn.preprocessing import MinMaxScaler
            self.tf = tf
            self.layers = layers
            self.Model = Model
            self.optimizers = optimizers
            self.MinMaxScaler = MinMaxScaler
        except ImportError:
            raise ImportError("TensorFlow y scikit-learn requeridos")
       
        self.n_lags = n_lags
        self.B = B
        self.units = units
        self.n_layers = n_layers
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_samples = num_samples
        self.random_state = random_state
        self.verbose = verbose
        self.optimize = optimize
        self.alpha = alpha
       
        self.scaler = self.MinMaxScaler()
        self.rng = np.random.default_rng(random_state)
       
        # Cuantiles clave para capturar la forma de la distribución
        self.quantiles = np.array([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
        
        self._trained_ensemble = None
        self._is_frozen = False
       
        self.tf.random.set_seed(random_state)
        np.random.seed(random_state)
   
    def _pin_loss(self, y_true, y_pred):
        """Pinball loss para múltiples cuantiles."""
        error = y_true - y_pred
        return self.tf.reduce_mean(
            self.tf.maximum(self.quantiles * error, (self.quantiles - 1) * error),
            axis=-1
        )
   
    def _build_lstm(self):
        x_in = self.layers.Input(shape=(self.n_lags, 1))
        x = x_in
        for i in range(self.n_layers - 1):
            x = self.layers.LSTM(self.units, return_sequences=True)(x)
            x = self.layers.Dropout(0.1)(x)
        x = self.layers.LSTM(self.units, return_sequences=False)(x)
        x = self.layers.Dense(len(self.quantiles))(x)
        
        model = self.Model(inputs=x_in, outputs=x)
        model.compile(optimizer=self.optimizers.Adam(learning_rate=self.lr), loss=self._pin_loss)
        return model
   
    def _prepare_data(self, series: np.ndarray, scaler=None):
        """Divide los datos en B subconjuntos DISJUNTOS (Lógica EnCQR)."""
        if scaler is not None:
            series_scaled = scaler.transform(series.reshape(-1, 1))
        else:
            series_scaled = self.scaler.fit_transform(series.reshape(-1, 1))
        
        X, y = [], []
        for i in range(len(series_scaled) - self.n_lags):
            X.append(series_scaled[i:(i + self.n_lags)])
            y.append(series_scaled[i + self.n_lags])
        X, y = np.array(X), np.array(y)
        
        n_samples = X.shape[0]
        batch_size_loo = n_samples // self.B
        batches = []
        for b in range(self.B):
            start = b * batch_size_loo
            end = (b + 1) * batch_size_loo if b < self.B - 1 else n_samples
            batches.append({'X': X[start:end], 'y': y[start:end]})
        return batches

    def _fit_skew_normal(self, quantile_values: np.ndarray) -> dict:
        """
        CORRECCIÓN PRINCIPAL: Ajusta una distribución Skew-Normal a los cuantiles.
        
        La Skew-Normal es ideal porque:
        1. Puede ser simétrica (cuando skew=0) o asimétrica
        2. Tiene 3 parámetros: location (μ), scale (σ), skew (α)
        3. Se reduce a Normal cuando skew=0
        4. Garantiza unimodalidad
        
        Teoría: Minimizamos la distancia entre los cuantiles empíricos 
        conformalizados y los cuantiles teóricos de la Skew-Normal.
        """
        def objective(params):
            loc, scale, skew = params
            if scale <= 0:
                return 1e10
            
            # Cuantiles teóricos de la distribución
            theoretical_quantiles = stats.skewnorm.ppf(self.quantiles, skew, loc=loc, scale=scale)
            
            # Minimizar error cuadrático medio
            return np.sum((quantile_values - theoretical_quantiles) ** 2)
        
        # Inicialización robusta
        q_median = quantile_values[len(quantile_values) // 2]
        q_iqr = quantile_values[-2] - quantile_values[1]  # Rango intercuartil aproximado
        
        initial_params = [
            q_median,           # location
            q_iqr / 1.35,       # scale (IQR ≈ 1.35σ para Normal)
            0.0                 # skew (empezar simétrico)
        ]
        
        # Optimización con restricciones
        bounds = [
            (quantile_values.min() - q_iqr, quantile_values.max() + q_iqr),  # loc
            (q_iqr / 10, q_iqr * 3),                                          # scale > 0
            (-5, 5)                                                            # skew
        ]
        
        result = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds)
        
        if not result.success:
            # Fallback a distribución Normal
            return {
                'loc': q_median,
                'scale': q_iqr / 1.35,
                'skew': 0.0
            }
        
        return {
            'loc': result.x[0],
            'scale': result.x[1],
            'skew': result.x[2]
        }

    def _quantiles_to_distribution(self, conf_q: np.ndarray) -> np.ndarray:
        """
        Genera muestras de la distribución ajustada a los cuantiles conformalizados.
        
        Este método respeta la teoría del paper:
        - Los cuantiles conformalizados (Eq. 12) definen los límites del intervalo
        - Ajustamos una distribución paramétrica que respeta estos límites
        - Muestreamos de esa distribución para obtener la distribución predictiva
        """
        # Asegurar monotonicidad
        conf_q_sorted = np.sort(conf_q)
        
        # Ajustar distribución Skew-Normal
        params = self._fit_skew_normal(conf_q_sorted)
        
        # Generar muestras de la distribución ajustada
        samples = stats.skewnorm.rvs(
            params['skew'],
            loc=params['loc'],
            scale=params['scale'],
            size=self.num_samples,
            random_state=self.random_state
        )
        
        return samples

    def _get_ensemble_loo_scores(self, batches):
        """Calcula scores de conformidad asimétricos usando Leave-One-Out (Eq. 11)."""
        ensemble_models = []
        loo_preds = [[] for _ in range(self.B)]
        
        for b in range(self.B):
            model = self._build_lstm()
            model.fit(batches[b]['X'], batches[b]['y'], epochs=self.epochs, 
                      batch_size=self.batch_size, verbose=0, shuffle=False)
            ensemble_models.append(model)
            
            # Predicciones LOO
            for i in range(self.B):
                if i != b:
                    loo_preds[i].append(model.predict(batches[i]['X'], verbose=0))
        
        # Calcular scores asimétricos (Eq. 11)
        scores_list = [[] for _ in range(len(self.quantiles))]
        for i in range(self.B):
            if loo_preds[i]:
                avg_pred = np.mean(loo_preds[i], axis=0)
                y_true = batches[i]['y'].reshape(-1, 1)
                
                for q_idx, tau in enumerate(self.quantiles):
                    q_val = avg_pred[:, q_idx].reshape(-1, 1)
                    if tau <= 0.5:
                        score = q_val - y_true
                    else:
                        score = y_true - q_val
                    scores_list[q_idx].extend(score.flatten())
        
        return ensemble_models, [np.array(s) for s in scores_list]

    def fit_predict(self, df) -> np.ndarray:
        series = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df)
        
        if not self._is_frozen:
            batches = self._prepare_data(series)
            models, scores = self._get_ensemble_loo_scores(batches)
            self._trained_ensemble = {'models': models, 'scores': scores, 'scaler': self.scaler}
            self._is_frozen = True

        # Predicción
        scaler = self._trained_ensemble['scaler']
        last_window = scaler.transform(series[-self.n_lags:].reshape(-1, 1)).reshape(1, self.n_lags, 1)
        
        # 1. Agregación de modelos
        preds = np.array([m.predict(last_window, verbose=0)[0] for m in self._trained_ensemble['models']])
        agg_q = np.mean(preds, axis=0)
        
        # 2. Conformalización (Eq. 12)
        conf_q = np.zeros_like(agg_q)
        for q_idx, tau in enumerate(self.quantiles):
            omega = np.quantile(self._trained_ensemble['scores'][q_idx], 1 - self.alpha)
            if tau <= 0.5:
                conf_q[q_idx] = agg_q[q_idx] - omega
            else:
                conf_q[q_idx] = agg_q[q_idx] + omega
        
        # 3. Desescalar
        conf_q_final = scaler.inverse_transform(conf_q.reshape(-1, 1)).flatten()
        
        # 4. Generar distribución suave y unimodal
        return self._quantiles_to_distribution(conf_q_final)

    def _cleanup(self):
        self.tf.keras.backend.clear_session()

class TimeBalancedOptimizer:
    """
    Optimizador Eficiente sin Restricción de Tiempo Rígida.
    
    Filosofía:
    - NO corta la evaluación por tiempo (evita sesgo contra Deep Learning).
    - Usa 'Smart Grids': Pocas configuraciones pero bien elegidas.
    - Usa 'Statistical Early Stopping': Detiene la validación si el CRPS converge.
    """
    
    def __init__(self, random_state: int = 42, verbose: bool = False):
        self.random_state = random_state
        self.verbose = verbose
        self.rng = np.random.default_rng(random_state)

    def _get_efficient_grid(self, model_name: str, n_train: int) -> List[Dict]:
        """
        Retorna un Grid de búsqueda pequeño pero efectivo (3-5 configuraciones).
        """
        # --- Modelos Bootstrap (Rápidos) ---
        if model_name == 'Block Bootstrapping':
            # Probar bloques pequeños, medios y grandes
            return [{'block_length': 5}, {'block_length': 9}, {'block_length': int(np.sqrt(n_train))}, {'block_length': int(n_train/5)}]
        
        elif model_name == 'Sieve Bootstrap':
            # Probar órdenes AR bajos y medios
            return [{'order': 5}, {'order': 10}, {'order': 20}]
        
        # --- Modelos Basados en Regresión (Medios) ---
        elif model_name == 'LSPMW':
            return [{'rho': 0.90}, {'rho': 0.95}, {'rho': 0.99}]
        
        elif model_name == 'AREPD':
            return [
                {'n_lags': 5, 'rho': 0.95, 'poly_degree': 2},
                {'n_lags': 10, 'rho': 0.90, 'poly_degree': 2},
                {'n_lags': 5, 'rho': 0.98, 'poly_degree': 3}
            ]
        
        elif model_name == 'MCPS':
            return [
                {'n_lags': 10, 'n_bins': 8},
                {'n_lags': 15, 'n_bins': 15}
            ]
        
        elif model_name == 'AV-MCPS':
            return [
                {'n_lags': 10, 'n_pred_bins': 8, 'n_vol_bins': 3},
                {'n_lags': 15, 'n_pred_bins': 10, 'n_vol_bins': 5}
            ]
        
        # --- Modelos Deep Learning (Lentos - Grid Mínimo) ---
        elif model_name == 'DeepAR':
            # Una config ligera y una más profunda
            return [
                {'hidden_size': 20, 'n_lags': 10, 'num_layers': 1, 'epochs': 25, 'lr': 0.01},
                {'hidden_size': 32, 'n_lags': 15, 'num_layers': 2, 'epochs': 30, 'lr': 0.005}
            ]
        
        elif model_name == 'EnCQR-LSTM':
            # EnCQR es muy costoso (Ensemble), probamos solo 2 variantes clave
            return [
                {'n_lags': 10, 'units': 24, 'epochs': 20},
                {'n_lags': 20, 'units': 32, 'epochs': 25}
            ]
        
        # Default
        return [{}]

    def _evaluate_with_convergence(self, model, train_data: np.ndarray,
                                   val_data: np.ndarray, min_steps: int = 15) -> float:
        """
        Evalúa el modelo en el set de validación usando predicción rolling.
        Usa convergencia estadística para parar antes si el error es estable.
        """
        from metricas import crps
        scores = []
        
        # Ventana para chequear estabilidad
        window = 10 
        
        for i in range(len(val_data)):
            # Construir historia creciente
            history = np.concatenate([train_data, val_data[:i]]) if i > 0 else train_data
            true_val = val_data[i]
            
            try:
                # Predicción
                if hasattr(model, 'fit_predict'):
                    # Manejo de tipos de entrada según el modelo
                    if "Bootstrap" in str(type(model)):
                        pred_samples = model.fit_predict(history)
                    else:
                        pred_samples = model.fit_predict(pd.DataFrame({'valor': history}))
                else:
                    continue
                
                # Calcular métrica
                pred_samples = np.asarray(pred_samples).flatten()
                score = crps(pred_samples, true_val)
                
                if not np.isnan(score):
                    scores.append(score)
            
            except Exception:
                continue
            
            # --- CRITERIO DE CONVERGENCIA ---
            # Si hemos evaluado suficientes pasos (min_steps)
            # y el promedio de los últimos X scores no cambia mucho respecto a los anteriores
            if len(scores) > min_steps + window:
                recent_mean = np.mean(scores[-window:])
                prev_mean = np.mean(scores[-2*window:-window])
                
                # Si el error se estabilizó (cambio < 1%), paramos para ahorrar tiempo
                # Esto asume que el modelo ya "aprendió" o falló consistentemente
                if prev_mean > 0 and abs(recent_mean - prev_mean) / prev_mean < 0.01:
                    break
        
        return np.mean(scores) if scores else np.inf

    def optimize_all_models(self, models: Dict, train_data: np.ndarray,
                           val_data: np.ndarray) -> Dict:
        """
        Optimiza todos los modelos buscando el mejor CRPS en validación.
        """
        import time
        import copy
        
        optimized_params = {}
        
        if self.verbose:
            print(f"\n⚡ OPTIMIZACIÓN EFICIENTE (Sin límite rígido de tiempo)")
        
        for name, model in models.items():
            model_start = time.time()
            
            # Obtener grid eficiente
            param_grid = self._get_efficient_grid(name, len(train_data))
            
            # Si no hay nada que optimizar
            if len(param_grid) <= 1 and not param_grid[0]:
                optimized_params[name] = {}
                continue
            
            best_score = np.inf
            best_params = {}
            
            for idx, params in enumerate(param_grid):
                try:
                    # Crear copia limpia del modelo
                    model_copy = copy.deepcopy(model)
                    
                    # Aplicar parámetros
                    for key, val in params.items():
                        if hasattr(model_copy, key):
                            setattr(model_copy, key, val)
                    
                    # Evaluar
                    score = self._evaluate_with_convergence(
                        model_copy, train_data, val_data
                    )
                    
                    if score < best_score:
                        best_score = score
                        best_params = params
                    
                    # Limpieza memoria
                    del model_copy
                    
                except Exception as e:
                    if self.verbose:
                        print(f"    Error en config {params}: {e}")
            
            # Guardar mejores parámetros
            optimized_params[name] = best_params
            
            # Aplicar al modelo original para referencia
            for key, val in best_params.items():
                if hasattr(model, key):
                    setattr(model, key, val)
            if hasattr(model, 'best_params'):
                model.best_params = best_params

            if self.verbose:
                elapsed = time.time() - model_start
                print(f"  ✓ {name}: Mejor CRPS={best_score:.4f} [{elapsed:.1f}s]")

        return optimized_params