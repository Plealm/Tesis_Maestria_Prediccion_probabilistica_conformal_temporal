
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np
import seaborn as sns
import pandas as pd
import os
from scipy import stats
from itertools import combinations
from metricas import calculate_pit, calculate_reliability, crps

GLOBAL_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

def get_model_colors(model_names):
    """Asigna un color fijo a cada modelo basado en su orden alfabético."""
    sorted_names = sorted(list(model_names))
    return {model: GLOBAL_COLORS[i % len(GLOBAL_COLORS)] for i, model in enumerate(sorted_names)}

# ==============================================================================
# 1. DIAGNÓSTICO BÁSICO (PIT / RELIABILITY)
# ==============================================================================
def generate_validation_report(predictions_dict, output_folder, output_prefix, model_colors):
    print(" 📊 Generando Reporte Estadístico Agregado (Diagnóstico)...")

    timestamps = sorted(predictions_dict.keys())
    y_true = np.array([predictions_dict[t]['true_value'] for t in timestamps])
    first_key = timestamps[0]
    models = list(predictions_dict[first_key]['predictions'].keys())

    pit_values = {m: [] for m in models}
    samples_matrix = {m: [] for m in models}
    
    TARGET_SAMPLES = 1000
    rng = np.random.default_rng(42)

    for t in timestamps:
        obs = predictions_dict[t]['true_value']
        for m in models:
            raw_preds = np.array(predictions_dict[t]['predictions'][m])
            raw_preds = raw_preds[np.isfinite(raw_preds)]
            
            if len(raw_preds) == 0:
                matrix_preds = np.full(TARGET_SAMPLES, np.nan)
            elif len(raw_preds) < TARGET_SAMPLES:
                matrix_preds = rng.choice(raw_preds, size=TARGET_SAMPLES, replace=True)
            elif len(raw_preds) > TARGET_SAMPLES:
                matrix_preds = rng.choice(raw_preds, size=TARGET_SAMPLES, replace=False)
            else:
                matrix_preds = raw_preds
            
            samples_matrix[m].append(matrix_preds)
            
            if len(raw_preds) > 0:
                pit_values[m].append(calculate_pit(obs, raw_preds))
            else:
                pit_values[m].append(np.nan)

    for m in models:
        samples_matrix[m] = np.array(samples_matrix[m])

    # Histograma PIT
    n_models = len(models)
    cols = 3
    rows = (n_models + cols - 1) // cols
    fig_pit, axes_pit = plt.subplots(rows, cols, figsize=(15, 4 * rows), constrained_layout=True)
    axes_pit = axes_pit.flatten()

    for i, m in enumerate(models):
        if i < len(axes_pit):
            vals = [v for v in pit_values[m] if not np.isnan(v)]
            if len(vals) > 0:
                sns.histplot(vals, bins=10, stat="density", ax=axes_pit[i], color=model_colors[m], alpha=0.6, kde=False)
            axes_pit[i].axhline(1.0, color='k', linestyle='--', linewidth=1)
            axes_pit[i].set_title(f"PIT: {m}")
            axes_pit[i].set_xlim(0, 1)

    for j in range(i + 1, len(axes_pit)):
        axes_pit[j].axis('off')
        
    plt.savefig(os.path.join(output_folder, f"{output_prefix}_stat_PIT.png"))
    plt.close()

    # Reliability Curves
    quantiles = np.linspace(0.01, 0.99, 99)
    fig_rel, ax_rel = plt.subplots(figsize=(8, 8))
    ax_rel.plot([0, 1], [0, 1], 'k--', label="Ideal")

    reliability_data = []
    for m in models:
        valid_mask = ~np.isnan(pit_values[m])
        if np.sum(valid_mask) > 0:
            y_t = y_true[valid_mask]
            y_s = samples_matrix[m][valid_mask]
            emp_coverage = calculate_reliability(y_t, y_s, quantiles)
            ax_rel.plot(quantiles, emp_coverage, label=m, color=model_colors[m], linewidth=2)
            for q, cov in zip(quantiles, emp_coverage):
                reliability_data.append({'Modelo': m, 'Nominal': q, 'Empirical': cov})

    ax_rel.set_title("Reliability Curves")
    # CAMBIO: Leyenda arriba a la izquierda
    ax_rel.legend(loc='upper left')
    plt.savefig(os.path.join(output_folder, f"{output_prefix}_stat_Reliability.png"))
    plt.close()

    return pd.DataFrame(reliability_data)

# ==============================================================================
# 2. GRAFICACIÓN TIPO A: 24 SUBPLOTS VERTICALES
# ==============================================================================
def plot_type_a_vertical_stack(group_name, models_to_plot, predictions_dict, df_results, 
                               model_colors, output_folder, output_prefix):
    """
    Genera una imagen ultra-alta con 24 filas x 1 columna.
    """
    steps = sorted(predictions_dict.keys())[:24] 
    n_steps = len(steps)
    
    # Altura dinámica: 2.5 pulgadas por paso -> 24 * 2.5 = 60 pulgadas de alto
    fig, axes = plt.subplots(nrows=n_steps, ncols=1, figsize=(10, 2.5 * n_steps), constrained_layout=True)
    
    if n_steps == 1: axes = [axes]
    
    fig.suptitle(f"Evolución Paso a Paso - {group_name}", fontsize=20, fontweight='bold', y=1.002)

    for i, t in enumerate(steps):
        ax = axes[i]
        step_num = t + 1
        data = predictions_dict[t]
        true_val = data['true_value']
        preds_dict = data['predictions']
        
        crps_row = df_results.loc[df_results['Paso'] == step_num].iloc[0]
        legend_labels = []

        # Determinar límites X comunes para este paso y estos modelos
        vals_step = [true_val]
        for m in models_to_plot:
            if m in preds_dict:
                p = np.array(preds_dict[m])
                p = p[np.isfinite(p)]
                vals_step.extend(p)
        
        if len(vals_step) > 1:
            mn, mx = np.min(vals_step), np.max(vals_step)
            margin = (mx - mn) * 0.2
            ax.set_xlim(mn - margin, mx + margin)

        for m in models_to_plot:
            if m not in preds_dict: continue
            
            p = np.array(preds_dict[m])
            p = p[np.isfinite(p)]
            color = model_colors.get(m, 'gray')
            
            crps_val = crps_row[m]
            crps_txt = f"{crps_val:.3f}" if pd.notna(crps_val) else "NaN"

            if len(p) > 1 and np.var(p) > 0:
                try:
                    sns.kdeplot(p, fill=True, color=color, alpha=0.2, ax=ax, linewidth=2)
                except:
                    ax.hist(p, bins=30, density=True, color=color, alpha=0.2)
            
            legend_labels.append(f"{m}: {crps_txt}")

        ax.axvline(true_val, color='black', linestyle='--', linewidth=2, label='Real')
        
        # Título a la derecha para no chocar con la leyenda izquierda
        ax.set_title(f"Step {step_num} | Real: {true_val:.2f}", fontsize=12, fontweight='bold', loc='right')
        ax.set_ylabel("Densidad")
        ax.grid(True, alpha=0.3)
        
        # CAMBIO: Leyenda (colores) movida a 'upper left'
        full_txt = "CRPS:\n" + "\n".join(legend_labels)
        at = AnchoredText(full_txt, prop=dict(size=9), frameon=True, loc='upper left')
        at.patch.set_alpha(0.8)
        ax.add_artist(at)

    safe_name = group_name.replace(" ", "_").replace("+", "_plus_")
    filename = os.path.join(output_folder, f"{output_prefix}_TypeA_{safe_name}.png")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

# ==============================================================================
# 3. GRAFICACIÓN TIPO B: CORTE TRANSVERSAL POR PASO (TODOS LOS MODELOS)
# ==============================================================================
def plot_type_b_step_comparison(target_step_idx, predictions_dict, df_results, 
                                all_models, model_colors, output_folder, output_prefix):
    """
    Pasos 1, 8, 16, 24. Todos los modelos uno bajo el otro.
    """
    t_idx = target_step_idx - 1 
    if t_idx not in predictions_dict: return

    data = predictions_dict[t_idx]
    true_val = data['true_value']
    preds_dict = data['predictions']
    timestamp = data['timestamp']
    
    sorted_models = sorted(all_models)
    n_models = len(sorted_models)

    fig, axes = plt.subplots(n_models, 1, figsize=(10, 2.0 * n_models), constrained_layout=True, sharex=True)
    if n_models == 1: axes = [axes]
    
    timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M')
    fig.suptitle(f"Análisis Detallado Paso {target_step_idx} | {timestamp_str}\nValor Real: {true_val:.2f}", 
                 fontsize=16, fontweight='bold')

    all_vals_step = [true_val]
    for m in sorted_models:
        p = preds_dict.get(m, [])
        p = np.array(p)
        p = p[np.isfinite(p)]
        if len(p) > 0: all_vals_step.extend(p)
    
    if len(all_vals_step) > 1:
        min_x, max_x = np.min(all_vals_step), np.max(all_vals_step)
        margin = (max_x - min_x) * 0.15
        xlims = (min_x - margin, max_x + margin)
    else:
        xlims = (true_val - 1, true_val + 1)
    
    crps_row = df_results.loc[df_results['Paso'] == target_step_idx].iloc[0]

    for i, m in enumerate(sorted_models):
        ax = axes[i]
        color = model_colors.get(m, 'blue')
        preds = np.array(preds_dict.get(m, []))
        preds = preds[np.isfinite(preds)]
        val_crps = crps_row[m]
        crps_txt = f"CRPS: {val_crps:.4f}" if pd.notna(val_crps) else "NaN"

        if len(preds) > 1 and np.var(preds) > 0:
            try:
                sns.kdeplot(preds, fill=True, color=color, alpha=0.5, ax=ax, linewidth=2)
            except:
                ax.hist(preds, bins=30, density=True, color=color, alpha=0.5)
        
        ax.axvline(true_val, color='black', linestyle='--', linewidth=2)
        
        # Título (Nombre Modelo) movido a la derecha para balancear con el texto izquierdo
        ax.set_title(m, loc='right', fontsize=11, fontweight='bold', color=color)
        
        # CAMBIO: Texto CRPS movido a la izquierda (x=0.01)
        ax.text(0.01, 0.8, crps_txt, transform=ax.transAxes, ha='left',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='#ccc'))
        
        ax.set_xlim(xlims)
        ax.grid(True, alpha=0.3)
        ax.set_yticks([]) # Quitar eje Y para limpieza

    filename = os.path.join(output_folder, f"{output_prefix}_TypeB_Step{target_step_idx:02d}_AllModels.png")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

# ==============================================================================
# 4. MÓDULO DE ANÁLISIS ESTADÍSTICO (DIEBOLD-MARIANO & RANKING)
# ==============================================================================
def diebold_mariano_test(errors1, errors2):
    """Test DM básico para vectores de errores."""
    d = np.array(errors1) - np.array(errors2)
    n = len(d)
    if n < 2: return np.nan, np.nan
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1) / n
    if var_d <= 0: return 0.0, 1.0
    dm_stat = mean_d / np.sqrt(var_d)
    # P-value two-sided
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value

def run_advanced_statistics(df_results, output_folder):
    """
    Implementa el análisis estadístico solicitado:
    1. Heatmap de comparaciones DM.
    2. Ranking basado en victorias significativas.
    3. Boxplot de distribución de CRPS.
    """
    analysis_dir = os.path.join(output_folder, "Analisis")
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)

    print(f"\n🔬 Ejecutando Análisis Estadístico en '{analysis_dir}'...")

    # Identificar modelos (excluyendo columnas no numéricas)
    cols_meta = ['Paso', 'Valor_Observado', 'timestamp']
    modelos = [c for c in df_results.columns if c not in cols_meta]
    
    # ---------------------------------------------------------
    # A. Matriz Diebold-Mariano y Ranking
    # ---------------------------------------------------------
    n_models = len(modelos)
    matriz_dm = np.zeros((n_models, n_models))
    matriz_pvals = np.ones((n_models, n_models))
    
    # Ajuste Bonferroni simple
    n_comparisons = (n_models * (n_models - 1)) / 2
    alpha = 0.05
    alpha_bonf = alpha / n_comparisons if n_comparisons > 0 else alpha

    ranking_data = []

    for i, m1 in enumerate(modelos):
        wins = 0
        losses = 0
        ties = 0
        for j, m2 in enumerate(modelos):
            if i == j: 
                continue
            
            # Usamos CRPS como métrica de error (se quiere minimizar)
            # DM test: H0: error_m1 = error_m2
            e1 = df_results[m1].values
            e2 = df_results[m2].values
            
            # Diebold-Mariano sobre CRPS (Loss Function = CRPS)
            dm_stat, p_val = diebold_mariano_test(e1, e2) # d = e1 - e2
            
            matriz_pvals[i, j] = p_val
            
            if p_val < alpha_bonf:
                if dm_stat < 0: 
                    # Mean(d) < 0 => Mean(e1) < Mean(e2) => m1 es mejor
                    matriz_dm[i, j] = 1 # Victoria fila
                    wins += 1
                else:
                    # m1 es peor
                    matriz_dm[i, j] = -1 # Derrota fila
                    losses += 1
            else:
                ties += 1
        
        score = wins - losses
        ranking_data.append({
            'Modelo': m1,
            'Victorias': wins,
            'Derrotas': losses,
            'Empates': ties,
            'Score_Neto': score,
            'Win_Rate': wins / (n_models - 1)
        })

    # Guardar Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matriz_dm, annot=True, fmt='.0f', cmap='RdYlGn', center=0,
                xticklabels=modelos, yticklabels=modelos, 
                cbar_kws={'label': '1=Fila Gana, -1=Fila Pierde'}, ax=ax)
    ax.set_title(f"Matriz de Superioridad Significativa (DM Test, N={len(df_results)})", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "DM_Superiority_Matrix.png"))
    plt.close()

    # Guardar Excel Ranking
    df_rank = pd.DataFrame(ranking_data).sort_values('Score_Neto', ascending=False)
    df_rank.insert(0, 'Rank', range(1, len(df_rank) + 1))
    
    excel_path = os.path.join(analysis_dir, "Ranking_Estadistico_DM.xlsx")
    df_rank.to_excel(excel_path, index=False)
    print(f"   ✓ Ranking DM guardado: {excel_path}")

    # ---------------------------------------------------------
    # B. Boxplot de Distribución de CRPS (Variabilidad)
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    
    # Ordenar modelos por mediana de CRPS
    median_crps = df_results[modelos].median().sort_values()
    sorted_models_by_med = median_crps.index.tolist()
    
    sns.boxplot(data=df_results[sorted_models_by_med], orient='h', palette='viridis')
    plt.title("Distribución de CRPS por Modelo (Ordenado por Mediana)")
    plt.xlabel("CRPS Value")
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "CRPS_Distribution_Boxplot.png"))
    plt.close()

    # ---------------------------------------------------------
    # C. Evolución Acumulada del CRPS Promedio
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    for m in modelos:
        cum_mean = df_results[m].expanding().mean()
        plt.plot(df_results['Paso'], cum_mean, label=m)
    
    plt.title("Evolución del CRPS Promedio Acumulado")
    plt.xlabel("Pasos de Predicción")
    plt.ylabel("CRPS Promedio Acumulado")
    # CAMBIO: Leyenda arriba a la izquierda (dentro del gráfico)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "CRPS_Cumulative_Evolution.png"))
    plt.close()


# ==============================================================================
# 5. ORQUESTADOR PRINCIPAL
# ==============================================================================
def visualize_predictions(pipeline, series_index=0, output_prefix="dataset", output_folder="Resultados"):
    print(f"\n🎨 Iniciando visualización avanzada para serie {series_index}...")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 1. Ejecutar Pipeline
    df_results, predictions_dict = pipeline.run_evaluation(
        series_index=series_index,
        save_predictions=True
    )

    if predictions_dict is None or df_results.empty:
        print("❌ Error: No se obtuvieron resultados.")
        return

    model_names = [c for c in df_results.columns if c not in ['Paso', 'Valor_Observado', 'timestamp']]
    color_map = get_model_colors(model_names)

    # 2. Guardar Resultados Brutos y Diagnósticos
    excel_name = os.path.join(output_folder, f"{output_prefix}_resultados_full.xlsx")
    df_rel = generate_validation_report(predictions_dict, output_folder, output_prefix, color_map)

    with pd.ExcelWriter(excel_name, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='Detalle_Pasos', index=False)
        if not df_rel.empty:
            df_rel.to_excel(writer, sheet_name='Reliability_Data', index=False)
    print(f"  💾 Excel guardado: {excel_name}")

    # 3. Definir Grupos para Gráficas
    mean_scores = df_results[model_names].mean().sort_values()
    sorted_models = mean_scores.index.tolist()
    
    top_3_best = sorted_models[:3]
    top_3_worst = sorted_models[-3:]
    top_2_best_plus_worst = sorted_models[:2] + [sorted_models[-1]]

    print("\n  📸 Generando Imágenes Tipo A (Vertical Stack 24 Pasos)...")
    
    # A1. Individuales (9 imágenes)
    for model in model_names:
        plot_type_a_vertical_stack(
            group_name=f"Model_{model}",
            models_to_plot=[model],
            predictions_dict=predictions_dict,
            df_results=df_results,
            model_colors=color_map,
            output_folder=output_folder,
            output_prefix=output_prefix
        )
    
    # A2. Comparativas (3 imágenes)
    plot_type_a_vertical_stack("Top3_Mejores", top_3_best, predictions_dict, df_results, 
                               color_map, output_folder, output_prefix)
    
    plot_type_a_vertical_stack("Top3_Peores", top_3_worst, predictions_dict, df_results, 
                               color_map, output_folder, output_prefix)
    
    plot_type_a_vertical_stack("VS_Top2Mejor_1Peor", top_2_best_plus_worst, predictions_dict, df_results, 
                               color_map, output_folder, output_prefix)

    print("\n  📸 Generando Imágenes Tipo B (Corte Transversal Pasos 1, 8, 16, 24)...")
    target_steps = [1, 8, 16, 24]
    for step in target_steps:
        plot_type_b_step_comparison(
            target_step_idx=step,
            predictions_dict=predictions_dict,
            df_results=df_results,
            all_models=model_names,
            model_colors=color_map,
            output_folder=output_folder,
            output_prefix=output_prefix
        )

    # 4. Análisis Estadístico Avanzado (Carpeta 'Analisis')
    run_advanced_statistics(df_results, output_folder)

    print(f"\n✨ Visualización completada.")
    print(f"   - Tipo A: 12 imágenes (24 subplots verticales).")
    print(f"   - Tipo B: 4 imágenes.")
    print(f"   - Diagnóstico: PIT/Reliability.")
    print(f"   - Análisis: Carpeta 'Analisis' con Rankings y Heatmaps.")