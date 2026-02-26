# -*- coding: utf-8 -*-
"""
扩展对比分析脚本：Human vs LLM Rewrite vs LLM Polish (Multi-Group Support)
支持多组文本分析，生成 CSV 数据，进行统计对比和可视化。
"""
import json
import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.interpolate import make_interp_spline
import calc
from viz import smooth_for_plot
import seaborn as sns
import pandas as pd

# 配置
WINDOW_SIZE = 200
STEP_SIZE = 50

def load_corpus(path="./corpus/corpus.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def analyze_text(text, label, group_id):
    """
    分析单个文本，返回所有指标
    """
    # 基础计算
    drifts, entropies = calc.compute_drift_and_entropy(text, WINDOW_SIZE, STEP_SIZE)
    acc = calc.compute_acceleration(drifts)
    acf = calc.compute_autocorrelation(drifts)
    freqs, mag = calc.compute_fft(drifts)
    hurst, _, _ = calc.compute_hurst(drifts)
    dfa, _ = calc.compute_dfa(drifts)
    ar1_phi, ar1_sigma = calc.fit_ar1(drifts)
    
    # 统计标量
    mean_drift = np.mean(drifts)
    std_drift = np.std(drifts)
    mean_entropy = np.mean(entropies)
    std_entropy = np.std(entropies)
    mean_acc = np.mean(acc)
    std_acc = np.std(acc)
    acf_lag1 = acf[1] if len(acf) > 1 else 0.0
    
    return {
        'group_id': group_id,
        'label': label,
        'length': len(text),
        'drifts': drifts,       # Array data (not for CSV)
        'entropies': entropies, # Array data
        'acc': acc,             # Array data
        'acf': acf,             # Array data
        'freqs': freqs,         # Array data
        'mag': mag,             # Array data
        # Scalar Metrics for CSV/Stats
        'mean_drift': mean_drift,
        'std_drift': std_drift,
        'mean_entropy': mean_entropy,
        'std_entropy': std_entropy,
        'mean_acc': mean_acc,
        'std_acc': std_acc,
        'acf_lag1': acf_lag1,
        'hurst': hurst,
        'dfa': dfa,
        'ar1_phi': ar1_phi,
        'ar1_sigma': ar1_sigma
    }

def save_results_to_csv(results, filepath):
    """
    将统计结果保存为 CSV
    """
    headers = [
        'group_id', 'label', 'length',
        'mean_drift', 'std_drift', 
        'mean_entropy', 'std_entropy',
        'mean_acc', 'std_acc',
        'acf_lag1', 'hurst', 'dfa',
        'ar1_phi', 'ar1_sigma'
    ]
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for res in results:
            row = [
                res['group_id'], res['label'], res['length'],
                res['mean_drift'], res['std_drift'],
                res['mean_entropy'], res['std_entropy'],
                res['mean_acc'], res['std_acc'],
                res['acf_lag1'], res['hurst'], res['dfa'],
                res['ar1_phi'], res['ar1_sigma']
            ]
            writer.writerow(row)
    print(f"Results saved to {filepath}")

def print_statistical_summary(results):
    """
    打印分组统计摘要
    """
    try:
        df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (np.ndarray, list))} for r in results])
        print("\n" + "="*80)
        print("Statistical Summary (Mean ± Std)")
        print("="*80)
        
        metrics = [
            'mean_drift', 'std_drift', 'mean_entropy', 'std_entropy', 
            'acf_lag1', 'hurst', 'dfa', 'ar1_phi', 'ar1_sigma'
        ]
        
        summary = df.groupby('label')[metrics].agg(['mean', 'std'])
        print(summary.T)
        print("="*80 + "\n")
    except ImportError:
        pass

# =============================================================================
# Matrix Plotting Functions (N Groups x 4 Columns)
# Columns: Human, LLM Rewrite, LLM Polish, Overlay
# =============================================================================

COLORS = {'Human': '#66fFA8', 'LLM Rewrite': '#FF646D', 'LLM Polish': '#6488FF'}
LABELS_ORDER = ['Human', 'LLM Rewrite', 'LLM Polish']

def _get_group_data(results):
    """Helper to organize results by group_id"""
    groups = {}
    for r in results:
        gid = r['group_id']
        if gid not in groups:
            groups[gid] = {}
        groups[gid][r['label']] = r
    # Sort by group id (assuming numeric or comparable)
    try:
        sorted_keys = sorted(groups.keys(), key=lambda x: int(x))
    except:
        sorted_keys = sorted(groups.keys())
    return groups, sorted_keys

def _setup_matrix_figure(n_rows, title_suffix):
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Set Column Titles
    cols = ['Human', 'LLM Rewrite', 'LLM Polish', 'Overlay']
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=14, fontweight='bold')
        
    return fig, axes

def plot_phase_space_matrix(results, output_dir):
    """
    绘制 N行 x 4列 的相空间矩阵图
    使用 3阶样条插值 + 彩虹渐变 (Time)
    """
    groups, group_ids = _get_group_data(results)
    n_rows = len(group_ids)
    fig, axes = _setup_matrix_figure(n_rows, "Phase Space")

    for i, gid in enumerate(group_ids):
        row_data = groups[gid]
        
        # 1-3 Columns: Individual
        for j, label in enumerate(LABELS_ORDER):
            ax = axes[i, j]
            if i == 0: ax.set_title(label, fontsize=14, fontweight='bold')
            if j == 0: ax.set_ylabel(f"Group {gid}\nDrift", fontsize=12)
            
            if label in row_data:
                _plot_single_phase_space(ax, row_data[label], color_time=True)
            else:
                ax.text(0.5, 0.5, "N/A", ha='center', va='center')

        # 4th Column: Overlay
        ax_overlay = axes[i, 3]
        if i == 0: ax_overlay.set_title("Overlay", fontsize=14, fontweight='bold')
        for label in LABELS_ORDER:
            if label in row_data:
                _plot_single_phase_space(ax_overlay, row_data[label], color_time=False, label=label)
        ax_overlay.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "phase_space_matrix.png"))
    plt.close()

def _plot_single_phase_space(ax, res, color_time=True, label=None):
    drifts = res['drifts']
    entropies = res['entropies']
    n = min(len(entropies), len(drifts))
    x_raw = np.array(entropies[:n])
    y_raw = np.array(drifts[:n])

    # Spline Interpolation (Same logic as compare.py)
    k_order = min(3, n - 1)
    if n > k_order:
        try:
            t_orig = np.arange(n)
            t_smooth = np.linspace(0, n - 1, 500)
            spline_y = make_interp_spline(t_orig, y_raw, k=k_order)
            y_smooth = spline_y(t_smooth)
            spline_x = make_interp_spline(t_orig, x_raw, k=k_order)
            x_smooth = spline_x(t_smooth)
        except:
            x_smooth, y_smooth = x_raw, y_raw
    else:
        x_smooth, y_smooth = x_raw, y_raw

    # 始终使用对应类别的颜色
    c = COLORS.get(res['label'], 'gray')

    if color_time:
        # 彩虹渐变散点 + 对应类别颜色的轨迹线
        ax.plot(x_smooth, y_smooth, color=c, alpha=0.6, linewidth=1.5)
        scatter = ax.scatter(x_raw, y_raw, c=np.arange(n), cmap='gist_rainbow', s=20, alpha=0.8, edgecolor='none')
    else:
        # Overlay 模式：实线
        ax.plot(x_smooth, y_smooth, color=c, alpha=0.6, linewidth=1.5, label=label)

    ax.grid(alpha=0.2)

def plot_kde_matrix(results, output_dir):
    """
    N x 4 KDE Plot
    """
    groups, group_ids = _get_group_data(results)
    n_rows = len(group_ids)
    fig, axes = _setup_matrix_figure(n_rows, "Drift KDE")
    
    # Calculate global range for consistent x-axis
    all_vals = np.concatenate([r['drifts'] for r in results])
    x_min, x_max = min(all_vals), max(all_vals)
    x_grid = np.linspace(x_min, x_max, 100)

    for i, gid in enumerate(group_ids):
        row_data = groups[gid]
        for j, label in enumerate(LABELS_ORDER):
            ax = axes[i, j]
            if i == 0: ax.set_title(label, fontsize=14, fontweight='bold')
            if j == 0: ax.set_ylabel(f"Group {gid}", fontsize=12)
            
            if label in row_data:
                _plot_kde(ax, row_data[label]['drifts'], x_grid, COLORS[label])
        
        # Overlay
        ax_over = axes[i, 3]
        if i == 0: ax_over.set_title("Overlay", fontsize=14, fontweight='bold')
        for label in LABELS_ORDER:
            if label in row_data:
                _plot_kde(ax_over, row_data[label]['drifts'], x_grid, COLORS[label], label=label)
        ax_over.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "drift_kde_matrix.png"))
    plt.close()

def _plot_kde(ax, data, x_grid, color, label=None):
    try:
        kde = gaussian_kde(data)
        ax.plot(x_grid, kde(x_grid), color=color, label=label, linewidth=1.5)
        ax.fill_between(x_grid, kde(x_grid), color=color, alpha=0.1)
    except:
        pass
    ax.grid(alpha=0.2)

def plot_trend_matrix(results, output_dir):
    """
    N x 4 Trend Plot
    """
    groups, group_ids = _get_group_data(results)
    n_rows = len(group_ids)
    fig, axes = _setup_matrix_figure(n_rows, "Drift Trend")

    for i, gid in enumerate(group_ids):
        row_data = groups[gid]
        for j, label in enumerate(LABELS_ORDER):
            ax = axes[i, j]
            if i == 0: ax.set_title(label, fontsize=14, fontweight='bold')
            if j == 0: ax.set_ylabel(f"Group {gid}", fontsize=12)
            
            if label in row_data:
                _plot_trend(ax, row_data[label]['drifts'], COLORS[label])
        
        # Overlay
        ax_over = axes[i, 3]
        if i == 0: ax_over.set_title("Overlay", fontsize=14, fontweight='bold')
        for label in LABELS_ORDER:
            if label in row_data:
                _plot_trend(ax_over, row_data[label]['drifts'], COLORS[label], label=label)
        ax_over.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "drift_trend_matrix.png"))
    plt.close()

def _plot_trend(ax, data, color, label=None):
    _, _, x_smooth, y_smooth = smooth_for_plot(data)
    x_norm = np.linspace(0, 1, len(x_smooth))
    ax.plot(x_norm, y_smooth, color=color, label=label, linewidth=1.5)
    ax.grid(alpha=0.2)

def plot_acf_matrix(results, output_dir):
    """
    N x 4 ACF Plot
    """
    groups, group_ids = _get_group_data(results)
    n_rows = len(group_ids)
    fig, axes = _setup_matrix_figure(n_rows, "ACF")

    for i, gid in enumerate(group_ids):
        row_data = groups[gid]
        for j, label in enumerate(LABELS_ORDER):
            ax = axes[i, j]
            if i == 0: ax.set_title(label, fontsize=14, fontweight='bold')
            if j == 0: ax.set_ylabel(f"Group {gid}", fontsize=12)
            
            if label in row_data:
                _plot_acf(ax, row_data[label]['acf'], COLORS[label])
        
        # Overlay
        ax_over = axes[i, 3]
        if i == 0: ax_over.set_title("Overlay", fontsize=14, fontweight='bold')
        for label in LABELS_ORDER:
            if label in row_data:
                _plot_acf(ax_over, row_data[label]['acf'], COLORS[label], label=label)
        ax_over.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "acf_matrix.png"))
    plt.close()

def _plot_acf(ax, acf, color, label=None):
    lags = np.arange(min(20, len(acf)))
    ax.plot(lags, acf[:len(lags)], color=color, label=label, marker='.', linewidth=1.5)
    ax.grid(alpha=0.2)
    ax.set_ylim(-0.5, 1.0)

def plot_fft_matrix(results, output_dir):
    """
    N x 4 FFT Plot
    """
    groups, group_ids = _get_group_data(results)
    n_rows = len(group_ids)
    fig, axes = _setup_matrix_figure(n_rows, "FFT")

    for i, gid in enumerate(group_ids):
        row_data = groups[gid]
        for j, label in enumerate(LABELS_ORDER):
            ax = axes[i, j]
            if i == 0: ax.set_title(label, fontsize=14, fontweight='bold')
            if j == 0: ax.set_ylabel(f"Group {gid}", fontsize=12)
            
            if label in row_data:
                _plot_fft(ax, row_data[label]['freqs'], row_data[label]['mag'], COLORS[label])
        
        # Overlay
        ax_over = axes[i, 3]
        if i == 0: ax_over.set_title("Overlay", fontsize=14, fontweight='bold')
        for label in LABELS_ORDER:
            if label in row_data:
                _plot_fft(ax_over, row_data[label]['freqs'], row_data[label]['mag'], COLORS[label], label=label)
        ax_over.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fft_matrix.png"))
    plt.close()

def _plot_fft(ax, freqs, mag, color, label=None):
    ax.semilogy(freqs, mag, color=color, label=label, alpha=0.8, linewidth=1.5)
    ax.grid(alpha=0.2)

def plot_distinction_heatmap(results, output_dir):
    """
    绘制指标区分度热力图 (Effect Size Heatmap)
    展示不同指标在区分 [Human vs Rewrite], [Human vs Polish], [Rewrite vs Polish] 时的效力 (Cohen's d)
    """
    df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (np.ndarray, list))} for r in results])
    
    metrics = [
        'mean_drift', 'std_drift', 'mean_entropy', 'std_entropy', 
        'acf_lag1', 'hurst', 'dfa', 'ar1_phi', 'ar1_sigma'
    ]
    
    pairs = [
        ('Human', 'LLM Rewrite'),
        ('Human', 'LLM Polish'),
        ('LLM Rewrite', 'LLM Polish')
    ]
    
    distinction_data = []
    
    for metric in metrics:
        row = {'Metric': metric}
        for label_a, label_b in pairs:
            # Get values
            vals_a = df[df['label'] == label_a][metric].values
            vals_b = df[df['label'] == label_b][metric].values
            
            if len(vals_a) < 2 or len(vals_b) < 2:
                d = 0
            else:
                # Calculate Cohen's d
                mean_a, mean_b = np.mean(vals_a), np.mean(vals_b)
                std_a, std_b = np.std(vals_a, ddof=1), np.std(vals_b, ddof=1)
                n_a, n_b = len(vals_a), len(vals_b)
                
                s_pooled = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
                if s_pooled == 0:
                    d = 0
                else:
                    d = abs(mean_a - mean_b) / s_pooled # Use absolute value to show magnitude of distinction
            
            row[f"{label_a} vs {label_b}"] = d
        distinction_data.append(row)
        
    dist_df = pd.DataFrame(distinction_data).set_index('Metric')
    
    # Sort by average distinction
    dist_df['avg_dist'] = dist_df.mean(axis=1)
    dist_df = dist_df.sort_values('avg_dist', ascending=False).drop(columns=['avg_dist'])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(dist_df, annot=True, cmap='Reds', fmt=".2f", linewidths=0.5)
    plt.title("Metric Distinction Power (Cohen's d)\nHigher value = Better Separation")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distinction_heatmap.png"))
    plt.close()

def process_mode(data, mode_name, output_base, remove_newlines=False):
    """
    执行一种模式的完整分析流程
    """
    print(f"\nStarting analysis: {mode_name}...")
    output_dir = os.path.join(output_base, mode_name)
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    for group in data:
        gid = group.get('group_id', 'unknown')
        print(f"  Processing Group {gid}...")
        
        for key, label in [('human', 'Human'), ('llm_rewrite', 'LLM Rewrite'), ('llm_polish', 'LLM Polish')]:
            if key in group and group[key]:
                content = group[key]
                if remove_newlines:
                    content = content.replace("\n\n", "").replace("\n", "") 
                    pass 
                
                # Check if content is string
                if isinstance(content, str):
                    if remove_newlines:
                         content = content.replace("\n\n", "")
                    
                    res = analyze_text(content, label, gid)
                    all_results.append(res)
    
    # Save CSV
    save_results_to_csv(all_results, os.path.join(output_dir, "metrics.csv"))
    
    # Statistics
    print_statistical_summary(all_results)
    
    # Plots
    print("  Generating plots...")
    plot_kde_matrix(all_results, output_dir)
    plot_trend_matrix(all_results, output_dir)
    plot_acf_matrix(all_results, output_dir)
    plot_fft_matrix(all_results, output_dir)
    plot_phase_space_matrix(all_results, output_dir)
    plot_distinction_heatmap(all_results, output_dir)
    
    print(f"Finished {mode_name}.")

def main():
    data = load_corpus()
    base_dir = "results/comparison_plus"
    
    # Mode 1: With Paragraph Cut (Original text)
    process_mode(data, "paragraph_cut", base_dir, remove_newlines=False)
    
    # Mode 2: Without Paragraph Cut (Remove \n\n)
    process_mode(data, "without_paragraph_cut", base_dir, remove_newlines=True)

if __name__ == "__main__":
    main()
