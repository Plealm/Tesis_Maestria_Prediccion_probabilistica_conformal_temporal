# simulacion.py
import pandas as pd
import numpy as np
from scipy.stats import t
from typing import List, Dict, Tuple
import pandas as pd


class ARMASimulation:
    """
    Genera series temporales ARMA con diferentes tipos de ruido y puede proporcionar
    la distribución teórica del siguiente paso, dado el historial completo.
    """
    def __init__(self, model_type: str = 'AR(1)', phi: List[float] = [], theta: List[float] = [], 
                 noise_dist: str = 'normal', sigma: float = 1.0, seed: int = None, verbose: bool = False):
        """
        Inicializa el simulador ARMA.
        """
        self.model_type = model_type
        self.phi = np.array(phi)
        self.theta = np.array(theta)
        self.noise_dist = noise_dist
        self.sigma = sigma
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self.series = None
        self.errors = None

    def model_params(self) -> Dict:
        """Devuelve los parámetros del modelo en un diccionario."""
        return {
            'model_type': self.model_type, 
            'phi': self.phi.tolist(), 
            'theta': self.theta.tolist(), 
            'sigma': self.sigma
        }

    def simulate(self, n: int = 250, burn_in: int = 50, return_just_series: bool = False) -> Tuple[np.ndarray, np.ndarray]:
            """
            Simula una serie temporal ARMA.
            """
            total_length = n + burn_in
            p, q = len(self.phi), len(self.theta)
            errors = self._generate_errors(total_length + q) # Generar suficientes errores para el MA
            series = np.zeros(total_length)
            
            initial_values = self.rng.normal(0, self.sigma, max(p, q))
            if len(initial_values) > 0:
                series[:len(initial_values)] = initial_values

            for t in range(max(p, q), total_length):
                ar_part = np.dot(self.phi, series[t-p:t][::-1]) if p > 0 else 0
                ma_part = np.dot(self.theta, errors[t-q:t][::-1]) if q > 0 else 0
                series[t] = ar_part + ma_part + errors[t]
                
            if return_just_series:
                # Opción para devolver la serie completa (con burn-in) y None para los errores
                return series, None

            self.series = series[burn_in:]
            self.errors = errors[burn_in:burn_in + n] # Asegurarse de que los errores coincidan con la serie
            return self.series, self.errors
    
    def get_true_next_step_samples(self, series_history: np.ndarray, errors_history: np.ndarray, 
                                        n_samples: int = 10000) -> np.ndarray:
        """
        Calcula una muestra grande de la distribución real para el siguiente paso (X_{n+1}).
        
        Args:
            series_history (np.ndarray): El historial de valores de la serie (X_1, ..., X_n).
            errors_history (np.ndarray): El historial de errores que generaron la serie (e_1, ..., e_n).
            n_samples (int): El número de muestras a generar para estimar la densidad.

        Returns:
            Un array de numpy con las muestras de la distribución del siguiente paso.
        """
        p, q = len(self.phi), len(self.theta)

        if len(series_history) < p:
            raise ValueError(f"El historial de la serie es insuficiente para el orden AR(p={p}).")
        if len(errors_history) < q:
            raise ValueError(f"El historial de errores es insuficiente para el orden MA(q={q}).")

        deterministic_part = np.dot(self.phi, series_history[-p:][::-1]) if p > 0 else 0
        ma_part = np.dot(self.theta, errors_history[-q:][::-1]) if q > 0 else 0
        conditional_mean = deterministic_part + ma_part

        future_errors = self._generate_errors(n_samples)
        
        # Devolvemos las muestras directamente para que puedan ser usadas en un gráfico KDE
        return conditional_mean + future_errors

    def _generate_errors(self, n: int) -> np.ndarray:
        """Genera el término de error según la distribución especificada."""
        if self.noise_dist == 'normal':
            return self.rng.normal(0, self.sigma, n)
        if self.noise_dist == 'uniform':
            limit = np.sqrt(3) * self.sigma
            return self.rng.uniform(-limit, limit, size=n)
        if self.noise_dist == 'exponential':
            return self.rng.exponential(scale=self.sigma, size=n) - self.sigma
        if self.noise_dist == 't-student':
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
    
    def generate_series(self, n_total: int = 505, seed: int = None) -> np.ndarray:
        """
        MÉTODO NUEVO - 100% compatible con el pipeline honesto
        Genera una serie ARMA limpia de n_total puntos útiles (ya sin burn-in)
        Este es el que usa PipelineOptimizado
        """
        if seed is not None:
            old_rng_state = self.rng.bit_generator.state  # guardamos estado actual
            self.rng = np.random.default_rng(seed)
        
        # Usamos tu método simulate existente (¡genial!)
        series, _ = self.simulate(n=n_total, burn_in=200, return_just_series=False)
        
        # Restauramos el estado del RNG si se pasó seed
        if seed is not None:
            self.rng.bit_generator.state = old_rng_state
        
        return series
    

class ARIMASimulation:

    """
    Genera series temporales ARIMA(p,1,q) con diferentes tipos de ruido y puede
    proporcionar la distribución teórica del siguiente paso, dado el historial completo.
    El componente de integración (d=1) se maneja explícitamente.
    """
    def __init__(self, model_type: str = 'ARIMA(1,1,0)', phi: List[float] = [], theta: List[float] = [],
                 noise_dist: str = 'normal', sigma: float = 1.0, seed: int = None, verbose: bool = False):
        """
        Inicializa el simulador ARIMA.
        Los parámetros phi y theta corresponden al proceso ARMA subyacente de la serie diferenciada.
        """
        self.model_type = model_type
        self.phi = np.array(phi)
        self.theta = np.array(theta)
        self.noise_dist = noise_dist
        self.sigma = sigma
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self.series = None
        self.errors = None

    def model_params(self) -> Dict:
        """Devuelve los parámetros del modelo en un diccionario."""
        return {
            'model_type': self.model_type,
            'phi': self.phi.tolist(),
            'theta': self.theta.tolist(),
            'sigma': self.sigma
        }

    def simulate(self, n: int = 250, burn_in: int = 50, return_just_series: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simula una serie temporal ARIMA(p,1,q).
        1. Simula un proceso ARMA(p,q) para la serie diferenciada.
        2. Integra (suma acumulada) la serie ARMA para obtener la serie ARIMA final.
        """
        total_length = n + burn_in
        p, q = len(self.phi), len(self.theta)
        
        # 1. Simular el proceso ARMA(p,q) para la serie diferenciada
        errors = self._generate_errors(total_length + q)
        diff_series = np.zeros(total_length)
        
        initial_values = self.rng.normal(0, self.sigma, max(p, q))
        if len(initial_values) > 0:
            diff_series[:len(initial_values)] = initial_values

        for t in range(max(p, q), total_length):
            ar_part = np.dot(self.phi, diff_series[t-p:t][::-1]) if p > 0 else 0
            ma_part = np.dot(self.theta, errors[t-q:t][::-1]) if q > 0 else 0
            diff_series[t] = ar_part + ma_part + errors[t]

        # 2. Integrar la serie diferenciada para obtener la serie ARIMA
        # Se inicia desde 0 para la serie integrada
        integrated_series = np.cumsum(np.insert(diff_series, 0, 0))

        if return_just_series:
            return integrated_series, None

        # Descartar el período de burn-in
        self.series = integrated_series[burn_in+1:] # +1 por el 0 insertado
        self.errors = errors[burn_in:burn_in + n]
        return self.series, self.errors

    def get_true_next_step_samples(self, series_history: np.ndarray, errors_history: np.ndarray,
                                         n_samples: int = 10000) -> np.ndarray:
        """
        Calcula una muestra grande de la distribución real para el siguiente paso (Y_{n+1}).
        Para un ARIMA(p,1,q), Y_{n+1} = Y_n + X_{n+1}, donde X es el proceso ARMA(p,q) subyacente.
        La distribución de Y_{n+1} es la distribución de X_{n+1} desplazada por el último valor Y_n.
        """
        p, q = len(self.phi), len(self.theta)

        if len(series_history) < p + 1:
            raise ValueError(f"El historial de la serie es insuficiente para el orden AR(p={p}) en la serie diferenciada.")
        if len(errors_history) < q:
            raise ValueError(f"El historial de errores es insuficiente para el orden MA(q={q}).")
            
        # El último valor observado de la serie original es el componente de "localización"
        last_observed_value = series_history[-1]
        
        # Necesitamos el historial de la serie diferenciada para predecir el siguiente incremento
        diff_history = np.diff(series_history)

        # Predecir el siguiente valor del proceso ARMA subyacente (X_{n+1})
        deterministic_part = np.dot(self.phi, diff_history[-p:][::-1]) if p > 0 else 0
        ma_part = np.dot(self.theta, errors_history[-q:][::-1]) if q > 0 else 0
        conditional_mean_of_diff = deterministic_part + ma_part

        # Generar los futuros errores para el componente estocástico
        future_errors = self._generate_errors(n_samples)
        
        # Muestras del siguiente incremento (X_{n+1})
        next_step_diff_samples = conditional_mean_of_diff + future_errors
        
        # Muestras del siguiente valor de la serie ARIMA (Y_{n+1} = Y_n + X_{n+1})
        return last_observed_value + next_step_diff_samples

    def _generate_errors(self, n: int) -> np.ndarray:
        """Genera el término de error según la distribución especificada."""
        if self.noise_dist == 'normal':
            return self.rng.normal(0, self.sigma, n)
        if self.noise_dist == 'uniform':
            limit = np.sqrt(3) * self.sigma
            return self.rng.uniform(-limit, limit, size=n)
        if self.noise_dist == 'exponential':
            # Centrado en cero
            return self.rng.exponential(scale=self.sigma, size=n) - self.sigma
        if self.noise_dist == 't-student':
            df = 5 # Grados de libertad fijos
            # Escalar para que la varianza sea sigma^2
            scale_factor = self.sigma * np.sqrt((df - 2) / df)
            return t.rvs(df, scale=scale_factor, size=n, random_state=self.rng)
        elif self.noise_dist == 'mixture':
            # Mezcla de dos normales, centrada en cero
            n1 = int(n * 0.75)
            n2 = n - n1
            # Asegurar que la varianza total de la mezcla sea sigma^2
            variance_of_means = 0.75 * (-0.25 * self.sigma * 2)**2 + 0.25 * (0.75 * self.sigma * 2)**2
            if self.sigma**2 < variance_of_means:
                raise ValueError("La varianza de la mezcla no puede ser la sigma deseada.")
            component_std = np.sqrt(self.sigma**2 - variance_of_means)
            
            comp1 = self.rng.normal(-0.25 * self.sigma * 2, component_std, n1)
            comp2 = self.rng.normal(0.75 * self.sigma * 2, component_std, n2)
            mixture = np.concatenate([comp1, comp2])
            self.rng.shuffle(mixture)
            return mixture - np.mean(mixture) # Forzar media cero
        else:
            raise ValueError(f"Distribución de ruido no soportada: {self.noise_dist}")
        

class SETARSimulation:
    """
    Genera series temporales SETAR (Self-Exciting Threshold AutoRegressive) con diferentes 
    tipos de ruido y puede proporcionar la distribución teórica del siguiente paso.
    
    SETAR(k; p1, p2, ..., pk) con k regímenes y órdenes autorregresivos p1, p2, ..., pk.
    Esta implementación se enfoca en SETAR(2; p1, p2) con dos regímenes.
    """
    def __init__(self, model_type: str = 'SETAR(2;1,1)', 
                 phi_regime1: List[float] = [0.6], 
                 phi_regime2: List[float] = [-0.5],
                 threshold: float = 0.0,
                 delay: int = 1,
                 noise_dist: str = 'normal', 
                 sigma: float = 1.0, 
                 seed: int = None, 
                 verbose: bool = False):
        """
        Inicializa el simulador SETAR.
        
        Args:
            model_type: Nombre del modelo (e.g., 'SETAR(2;1,1)')
            phi_regime1: Coeficientes AR para el régimen 1 (y_{t-d} <= threshold)
            phi_regime2: Coeficientes AR para el régimen 2 (y_{t-d} > threshold)
            threshold: Umbral que separa los regímenes (r)
            delay: Retardo para la variable de umbral (d)
            noise_dist: Distribución del ruido ('normal', 'uniform', 'exponential', 't-student', 'mixture')
            sigma: Desviación estándar del ruido
            seed: Semilla para reproducibilidad
            verbose: Mostrar información adicional
        """
        self.model_type = model_type
        self.phi_regime1 = np.array(phi_regime1)
        self.phi_regime2 = np.array(phi_regime2)
        self.threshold = threshold
        self.delay = delay
        self.noise_dist = noise_dist
        self.sigma = sigma
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self.series = None
        self.errors = None
        self.regime_history = None  # Para tracking de regímenes

    def model_params(self) -> Dict:
        """Devuelve los parámetros del modelo en un diccionario."""
        return {
            'model_type': self.model_type,
            'phi_regime1': self.phi_regime1.tolist(),
            'phi_regime2': self.phi_regime2.tolist(),
            'threshold': self.threshold,
            'delay': self.delay,
            'sigma': self.sigma
        }

    def _determine_regime(self, series_history: np.ndarray) -> int:
        """
        Determina el régimen basado en el valor retardado de la serie.
        
        Args:
            series_history: Historial de la serie temporal
            
        Returns:
            1 si y_{t-d} <= threshold, 2 si y_{t-d} > threshold
        """
        if len(series_history) < self.delay:
            # Si no hay suficiente historia, usar régimen 1 por defecto
            return 1
        
        threshold_value = series_history[-self.delay]
        return 1 if threshold_value <= self.threshold else 2

    def simulate(self, n: int = 250, burn_in: int = 50, 
                 return_just_series: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simula una serie temporal SETAR.
        
        Args:
            n: Número de observaciones a generar (sin burn-in)
            burn_in: Período de calentamiento
            return_just_series: Si True, devuelve (serie_completa, None)
            
        Returns:
            Tupla (serie, errores) o (serie_completa, None)
        """
        total_length = n + burn_in
        p1, p2 = len(self.phi_regime1), len(self.phi_regime2)
        max_p = max(p1, p2, self.delay)
        
        # Generar errores
        errors = self._generate_errors(total_length)
        series = np.zeros(total_length)
        regime_history = np.zeros(total_length, dtype=int)
        
        # Inicialización con valores pequeños
        initial_values = self.rng.normal(0, self.sigma * 0.5, max_p)
        series[:max_p] = initial_values
        
        # Simular la serie
        for t in range(max_p, total_length):
            # Determinar el régimen basado en y_{t-d}
            regime = self._determine_regime(series[:t])
            regime_history[t] = regime
            
            if regime == 1:
                # Régimen 1: y_{t-d} <= threshold
                phi = self.phi_regime1
                p = p1
            else:
                # Régimen 2: y_{t-d} > threshold
                phi = self.phi_regime2
                p = p2
            
            # Calcular la parte AR
            if p > 0:
                ar_part = np.dot(phi, series[t-p:t][::-1])
            else:
                ar_part = 0
            
            series[t] = ar_part + errors[t]
        
        if return_just_series:
            return series, None

        # Guardar resultados sin burn-in
        self.series = series[burn_in:]
        self.errors = errors[burn_in:]
        self.regime_history = regime_history[burn_in:]
        
        return self.series, self.errors

    def get_true_next_step_samples(self, series_history: np.ndarray, 
                                   errors_history: np.ndarray,
                                   n_samples: int = 10000) -> np.ndarray:
        """
        Calcula una muestra grande de la distribución real para el siguiente paso (X_{n+1}).
        Para SETAR, esto depende del régimen determinado por y_{n-d+1}.
        
        Args:
            series_history: El historial de valores de la serie (X_1, ..., X_n)
            errors_history: El historial de errores (no se usa en SETAR puro, pero se mantiene por compatibilidad)
            n_samples: El número de muestras a generar
            
        Returns:
            Array de numpy con las muestras de la distribución del siguiente paso
        """
        p1, p2 = len(self.phi_regime1), len(self.phi_regime2)
        max_p = max(p1, p2)
        
        if len(series_history) < max(max_p, self.delay):
            raise ValueError(f"El historial de la serie es insuficiente. "
                           f"Se necesitan al menos {max(max_p, self.delay)} observaciones.")
        
        # Determinar el régimen para la predicción
        regime = self._determine_regime(series_history)
        
        if regime == 1:
            phi = self.phi_regime1
            p = p1
        else:
            phi = self.phi_regime2
            p = p2
        
        # Calcular la parte determinística
        if p > 0:
            deterministic_part = np.dot(phi, series_history[-p:][::-1])
        else:
            deterministic_part = 0
        
        # Generar errores futuros
        future_errors = self._generate_errors(n_samples)
        
        # Retornar las muestras
        return deterministic_part + future_errors

    def _generate_errors(self, n: int) -> np.ndarray:
        """Genera el término de error según la distribución especificada."""
        if self.noise_dist == 'normal':
            return self.rng.normal(0, self.sigma, n)
        if self.noise_dist == 'uniform':
            limit = np.sqrt(3) * self.sigma
            return self.rng.uniform(-limit, limit, size=n)
        if self.noise_dist == 'exponential':
            return self.rng.exponential(scale=self.sigma, size=n) - self.sigma
        if self.noise_dist == 't-student':
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

    def generate_series(self, n_total: int = 505, seed: int = None) -> np.ndarray:
        """
        MÉTODO NUEVO - 100% compatible con el pipeline honesto.
        Genera una serie SETAR limpia de n_total puntos útiles (ya sin burn-in).
        """
        if seed is not None:
            old_rng_state = self.rng.bit_generator.state
            self.rng = np.random.default_rng(seed)
        
        series, _ = self.simulate(n=n_total, burn_in=200, return_just_series=False)
        
        if seed is not None:
            self.rng.bit_generator.state = old_rng_state
        
        return series
    
    def get_regime_proportions(self) -> Dict[int, float]:
        """
        Calcula la proporción de observaciones en cada régimen.
        Solo funciona después de llamar a simulate().
        """
        if self.regime_history is None:
            raise ValueError("Primero debe llamar a simulate()")
        
        unique, counts = np.unique(self.regime_history, return_counts=True)
        total = len(self.regime_history)
        
        return {int(regime): count/total for regime, count in zip(unique, counts)}