# -*- coding: utf-8 -*-
"""
绘图模块（使用样条插值平滑，保留所有原始数据点）
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# 配置 matplotlib
plt.rcParams['font.sans-serif'] = ['Maple Mono NF CN', 'Consolas', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

def _save_plot(filepath):
    """保存并关闭当前图表"""
    # 如果路径只是文件名，默认存到 results 目录（保持兼容性）
    if os.path.dirname(filepath) == "":
        filepath = os.path.join("results", filepath)
        
    os.makedirs(os.path.dirname(filepath), exist_ok=True)  # 确保目录存在
    print(f"[Plot] Generating {filepath}...")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def smooth_for_plot(y, enabled=True, num_points=500):
    """
    使用三次样条插值生成平滑曲线（仅用于绘图）
    返回：(x_original, y_original, x_smooth, y_smooth)
    """
    if not enabled:
        x_orig = np.arange(len(y))
        return x_orig, y, x_orig, y

    n = len(y)
    if n < 3:
        x_orig = np.arange(n)
        return x_orig, y, x_orig, y

    x_orig = np.arange(n)
    # 三次样条插值（k=3），要求至少4个点；若只有3个点，则用 k=2
    k = min(3, n - 1)
    try:
        spline = make_interp_spline(x_orig, y, k=k)
    except ValueError:
        # 回退到线性插值（极少数情况）
        return x_orig, y, x_orig, y

    x_smooth = np.linspace(0, n - 1, num_points)
    y_smooth = spline(x_smooth)

    return x_orig, y, x_smooth, y_smooth

def plot_drift_curve(drifts, filename="1_drift_curve.png"):
    """1. 绘制漂移曲线"""
    x_raw, y_raw, x_smooth, y_smooth = smooth_for_plot(drifts)

    plt.figure(figsize=(10, 4))
    plt.plot(x_raw, y_raw, color='#bdc3c7', alpha=0.5, label='Raw Drift', linewidth=0.5, marker='o', markersize=2)
    plt.plot(x_smooth, y_smooth, color='#2c3e50', label='Smoothed Trend', linewidth=1.5)

    plt.xlabel("Window Index")
    plt.ylabel("JS Divergence")
    plt.title("1. Local Distribution Drift (Velocity)")
    plt.legend()
    plt.grid(alpha=0.3)
    _save_plot(filename)

def plot_acceleration(acc, filename="2_drift_acceleration.png"):
    """2. 绘制漂移加速度（二阶导数）"""
    x_raw, y_raw, x_smooth, y_smooth = smooth_for_plot(acc)

    plt.figure(figsize=(10, 4))
    plt.plot(x_raw, y_raw, color='#95a5a6', alpha=0.5, label='Raw Acc', linewidth=0.5, marker='o', markersize=2)
    plt.plot(x_smooth, y_smooth, color='#e74c3c', label='Smoothed Acc', linewidth=1.5)

    plt.xlabel("Window Index")
    plt.ylabel("d(Drift)/dt")
    plt.title("2. Drift Acceleration (Rate of Change)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    _save_plot(filename)

def plot_autocorrelation(acf, filename="3_drift_acf.png"):
    """3. 绘制自相关函数（通常不需要平滑，但保留接口一致性）"""
    lags = np.arange(len(acf))

    plt.figure(figsize=(10, 4))
    plt.plot(lags, acf, color='#3498db', linewidth=1.5, marker='o', markersize=2)
    plt.fill_between(lags, 0, acf, alpha=0.2, color='#3498db')

    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title("3. Drift Autocorrelation Function (ACF)")
    plt.grid(alpha=0.3)
    tau = np.where(acf < 1/np.e)[0]
    if len(tau) > 0:
        t = tau[0]
        plt.axvline(t, color='red', linestyle='--', alpha=0.7, label=f'τ = {t}')
        plt.legend()
    _save_plot(filename)

def plot_fft_spectrum(freqs, magnitude, filename="4_drift_fft.png"):
    """4. 绘制频域分析（通常也不平滑）"""
    plt.figure(figsize=(10, 4))
    plt.semilogy(freqs, magnitude, color='#8e44ad', linewidth=1, marker='o', markersize=2)

    plt.xlabel("Frequency")
    plt.ylabel("Magnitude (Log Scale)")
    plt.title("4. Drift Frequency Spectrum (FFT)")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    _save_plot(filename)

def plot_phase_space(entropies, drifts, filename="5_phase_space.png"):
    """5. 绘制熵-漂移相图（对 drift 使用平滑以使轨迹更清晰）"""
    n = min(len(entropies), len(drifts))
    x = np.array(entropies[:n])
    y_raw = np.array(drifts[:n])

    # 对 y 做平滑用于连线（但散点仍用原始值）
    _, _, t_smooth, y_smooth = smooth_for_plot(y_raw)
    # 为了在相图中正确对应 x，我们需对 x 也插值（因为 y_smooth 是基于时间索引的）
    t_orig = np.arange(n)
    x_spline = make_interp_spline(t_orig, x, k=min(3, n-1)) if n >= 2 else lambda t: x[0]
    x_smooth = x_spline(t_smooth)

    plt.figure(figsize=(12, 8))
    plt.plot(x_smooth, y_smooth, color='#2c3e50', alpha=0.3, linewidth=0.8)
    scatter = plt.scatter(x, y_raw, c=np.arange(n), cmap='viridis', s=15, alpha=0.8, edgecolor='none')

    plt.colorbar(scatter, label='Time (Window Index)')
    plt.xlabel("Shannon Entropy (Complexity)")
    plt.ylabel("Drift (Instability)")
    plt.title("5. Phase Space: Entropy vs Drift")
    plt.grid(alpha=0.3)
    _save_plot(filename)

def plot_summary(drifts, acc, acf, freqs, magnitude, entropies, filename="summary_plot.png"):
    """绘制汇总图 (5合1)"""
    fig = plt.figure(figsize=(18, 18))
    gs = fig.add_gridspec(4, 2)
    
    # 1. Drift Curve
    ax1 = fig.add_subplot(gs[0, 0])
    x_raw, y_raw, x_smooth, y_smooth = smooth_for_plot(drifts)
    ax1.plot(x_raw, y_raw, color='#bdc3c7', alpha=0.5, label='Raw Drift', linewidth=0.5)
    ax1.plot(x_smooth, y_smooth, color='#2c3e50', label='Smoothed Trend', linewidth=1.5)
    ax1.set_title("1. Local Distribution Drift")
    ax1.set_ylabel("JS Divergence")
    ax1.grid(alpha=0.3)
    ax1.legend()

    # 2. Acceleration
    ax2 = fig.add_subplot(gs[0, 1])
    x_raw, y_raw, x_smooth, y_smooth = smooth_for_plot(acc)
    ax2.plot(x_raw, y_raw, color='#95a5a6', alpha=0.5, label='Raw Acc', linewidth=0.5)
    ax2.plot(x_smooth, y_smooth, color='#e74c3c', label='Smoothed Acc', linewidth=1.5)
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.set_title("2. Drift Acceleration")
    ax2.set_ylabel("d(Drift)/dt")
    ax2.grid(alpha=0.3)
    ax2.legend()

    # 3. ACF
    ax3 = fig.add_subplot(gs[1, 0])
    lags = np.arange(len(acf))
    ax3.plot(lags, acf, color='#3498db', linewidth=1.5)
    ax3.fill_between(lags, 0, acf, alpha=0.2, color='#3498db')
    ax3.set_title("3. Autocorrelation (ACF)")
    ax3.set_ylabel("Correlation")
    ax3.grid(alpha=0.3)

    # 4. FFT
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.semilogy(freqs, magnitude, color='#8e44ad', linewidth=1)
    ax4.set_title("4. Frequency Spectrum")
    ax4.set_ylabel("Magnitude (Log)")
    ax4.grid(True, which="both", ls="-", alpha=0.2)

    # 5. Phase Space
    ax5 = fig.add_subplot(gs[2:, :]) # 跨两行、两列
    n = min(len(entropies), len(drifts))
    x = np.array(entropies[:n])
    y_raw = np.array(drifts[:n])
    
    # 重新计算平滑（因为 summary 是独立的）
    _, _, t_smooth, y_smooth = smooth_for_plot(y_raw)
    t_orig = np.arange(n)
    x_spline = make_interp_spline(t_orig, x, k=min(3, n-1)) if n >= 2 else lambda t: x[0]
    x_smooth = x_spline(t_smooth)

    ax5.plot(x_smooth, y_smooth, color='#2c3e50', alpha=0.3, linewidth=0.8)
    scatter = ax5.scatter(x, y_raw, c=np.arange(n), cmap='viridis', s=15, alpha=0.8, edgecolor='none')
    plt.colorbar(scatter, ax=ax5, label='Time')
    ax5.set_title("5. Phase Space: Entropy vs Drift")
    ax5.set_xlabel("Entropy")
    ax5.set_ylabel("Drift")
    ax5.grid(alpha=0.3)

    plt.tight_layout()
    _save_plot(filename)