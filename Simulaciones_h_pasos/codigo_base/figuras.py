import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
from matplotlib.offsetbox import AnchoredText

class PlotManager:
    _STYLE = {
        'figsize_step': (9, 3.5),
        'colors': {
            'Teórica': '#9467bd', 'Block Bootstrapping': '#1f77b4',
            'Sieve Bootstrap': '#2ca02c', 'LSPM': '#d62728',
            'LSPMW': '#ff7f0e', 'AREPD': '#8c564b',
            'MCPS': '#e377c2', 'AV-MCPS': '#7f7f7f',
            'DeepAR': '#bcbd22', 'EnCQR-LSTM': '#17becf'
        }
    }

    @classmethod
    def plot_individual_model_evolution(cls, scenario_name: str, model_name: str, 
                                       predictions_dict: dict, df_results: pd.DataFrame, 
                                       save_path: str):
        steps = sorted(predictions_dict.keys())
        n_steps = len(steps)
        fig, axes = plt.subplots(nrows=n_steps, ncols=1, 
                                figsize=(cls._STYLE['figsize_step'][0], 
                                         cls._STYLE['figsize_step'][1] * n_steps))
        
        if n_steps == 1: axes = [axes]
        model_color = cls._STYLE['colors'].get(model_name, '#333333')
        true_color = cls._STYLE['colors']['Teórica']

        fig.suptitle(f"Comparación de Densidades: {model_name}\nEscenario: {scenario_name}", 
                    fontsize=16, fontweight='bold', y=0.99)

        for i, step_idx in enumerate(steps):
            ax = axes[i]
            step_num = step_idx + 1
            data = predictions_dict[step_idx]
            true_dist = data['true_distribution']
            model_preds = data['model_predictions'].get(model_name, np.array([]))
            
            # Limpiar datos para el gráfico
            true_dist = true_dist[np.isfinite(true_dist)]
            model_preds = model_preds[np.isfinite(model_preds)]

            if len(true_dist) > 0:
                sns.kdeplot(true_dist, fill=True, color=true_color, alpha=0.3, 
                            linewidth=2.5, ax=ax, label='Densidad Teórica (Ground Truth)')
            
            if len(model_preds) > 0:
                sns.kdeplot(model_preds, fill=True, color=model_color, alpha=0.2, 
                            linewidth=2.5, ax=ax, label=f'Predicción: {model_name}')
                ax.axvline(np.mean(model_preds), color=model_color, linestyle=':', alpha=0.8)

            ax.axvline(np.mean(true_dist), color='black', linestyle='--', alpha=0.6, label='Media Teórica')

            # --- ECRPS RECUADRO ---
            try:
                score = df_results[df_results['Paso'] == step_num][model_name].values[0]
                text_score = f"ECRPS: {score:.5f}"
            except:
                text_score = "ECRPS: Error/NaN"

            at = AnchoredText(text_score, prop=dict(size=12, fontweight='bold'), 
                             frameon=True, loc='upper right')
            at.patch.set_facecolor('white')
            at.patch.set_alpha(0.9)
            ax.add_artist(at)
            
            ax.set_title(f"Paso {step_num}", loc='left', fontsize=12, fontweight='bold')
            ax.set_ylabel("Densidad")
            ax.legend(loc='upper left', fontsize=9) # Forzar leyenda en cada paso

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)