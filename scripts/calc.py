# -*- coding: utf-8 -*-
"""
核心计算模块
包含：分布计算、JS散度、香农熵、Drift曲线、加速度、自相关、FFT
"""
import numpy as np
from collections import Counter
from scipy.signal import savgol_filter

EPS = 1e-12

def char_distribution(segment):
    """计算字符分布"""
    cnt = Counter(segment)
    total = sum(cnt.values())
    return {k: v / total for k, v in cnt.items()}

def shannon_entropy(dist):
    """计算香农熵 (bits)"""
    entropy = 0.0
    for p in dist.values():
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy

def js_divergence(p, q):
    """计算 JS 散度"""
    keys = set(p) | set(q)
    p_vec = np.array([p.get(k, 0.0) for k in keys])
    q_vec = np.array([q.get(k, 0.0) for k in keys])
    m = 0.5 * (p_vec + q_vec)

    def kl(a, b):
        return np.sum(a * np.log((a + EPS) / (b + EPS)))

    return 0.5 * kl(p_vec, m) + 0.5 * kl(q_vec, m)

def smooth_signal(y, smooth_enabled=True):
    """使用 Savitzky-Golay 滤波器平滑曲线"""
    n = len(y)
    if not smooth_enabled or n < 11: return y
    
    # 动态调整窗口大小：数据长度的 5% 或至少 25
    raw_win = max(25, int(n * 0.05))
    if raw_win % 2 == 0:
        raw_win += 1
        
    win = min(n - 1 if n % 2 == 0 else n, raw_win)
    if win < 3: return y
    
    # 使用 2 阶多项式拟合
    smoothed = savgol_filter(y, window_length=win, polyorder=2)
    return smoothed  # 注意：这里不再强制 clip 到 0-1，因为加速度可能有负值，但在 JS 散度场景下调用者应自行处理

def compute_drift_and_entropy(text, window_size=500, step_size=100):
    """
    计算 Drift 序列和 Entropy 序列
    返回: (drifts, entropies)
    """
    windows = []
    entropies = []
    
    # 1. 生成窗口并计算分布和熵
    for i in range(0, len(text) - window_size, step_size):
        seg = text[i:i + window_size]
        dist = char_distribution(seg)
        windows.append(dist)
        entropies.append(shannon_entropy(dist))
        
    # 2. 计算相邻窗口的 JS 散度 (Drift)
    drifts = []
    for i in range(len(windows) - 1):
        d = js_divergence(windows[i], windows[i + 1])
        drifts.append(d)
        
    # 对齐长度：Drift 比 Entropy 少一个（因为是差分），我们在 Drift 头部补 0 或者截断 Entropy
    # 为了方便绘图对齐，通常丢弃最后一个 Entropy 或者第一个 Entropy
    # 这里我们选择返回 len(drifts) 长度的数据，丢弃最后一个 entropy 以匹配 drift[i] (window[i] -> window[i+1])
    # 实际上 drift[i] 对应的是 window[i] 和 window[i+1] 之间的变化
    # 我们保留 entropies[:-1] 对应 drift[i] 的起始状态
    
    return np.array(drifts), np.array(entropies[:-1])

def compute_acceleration(drifts):
    """计算 Drift 的变化率（加速度）"""
    # 使用 np.gradient 计算中心差分
    return np.gradient(drifts)

def compute_autocorrelation(series):
    """计算自相关函数 (ACF)"""
    n = len(series)
    # 标准化
    series = series - np.mean(series)
    # 自相关
    r = np.correlate(series, series, mode='full')[-n:]
    # 归一化，使得 lag=0 时为 1
    r = r / (r[0] + EPS)
    return r

def detect_period(acf, min_lag=1):
    """
    通过 ACF 检测主周期
    返回: (period_lag, period_strength)
    """
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(acf, height=0)
    
    valid_peaks = [p for p in peaks if p >= min_lag]
    
    if not valid_peaks:
        return None, 0.0
        
    best_peak = max(valid_peaks, key=lambda p: acf[p])
    return best_peak, acf[best_peak]

def compute_hurst(series):
    """
    计算 Hurst 指数 (R/S 分析法)
    
    参数:
        series: 一维数值序列
        
    返回:
        H: Hurst 指数 (0.5=随机, >0.5=持久性, <0.5=反持久性)
        c: 常数项
        data: (log_rs, log_n) 用于绘图的元组
    """
    series = np.array(series)
    N = len(series)
    if N < 20: return 0.5, 0, ([], [])
    
    # 划分不同的子区间长度 n
    min_n = 10
    max_n = N // 2
    # 生成对数分布的区间长度
    n_vals = np.unique(np.logspace(np.log10(min_n), np.log10(max_n), num=20).astype(int))
    n_vals = n_vals[n_vals > 5] # 过滤过小的区间
    
    rs_vals = []
    
    for n in n_vals:
        # 将序列分割成长度为 n 的块
        num_chunks = N // n
        rs_chunk_vals = []
        
        for i in range(num_chunks):
            chunk = series[i*n : (i+1)*n]
            mean = np.mean(chunk)
            # 计算累积离差
            y = np.cumsum(chunk - mean)
            # 极差 R
            R = np.max(y) - np.min(y)
            # 标准差 S
            S = np.std(chunk, ddof=1)
            
            if S == 0:
                continue
                
            rs_chunk_vals.append(R / S)
            
        if rs_chunk_vals:
            rs_vals.append(np.mean(rs_chunk_vals))
        else:
            rs_vals.append(np.nan)
            
    # 移除无效值
    mask = ~np.isnan(rs_vals)
    n_vals = n_vals[mask]
    rs_vals = np.array(rs_vals)[mask]
    
    if len(n_vals) < 3:
        return 0.5, 0, (np.array([]), np.array([]))
        
    # 双对数回归: log(R/S) = H * log(n) + c
    log_n = np.log10(n_vals)
    log_rs = np.log10(rs_vals)
    
    H, c = np.polyfit(log_n, log_rs, 1)
    
    return H, c, (log_n, log_rs)

def compute_dfa(series, order=1):
    """
    计算 DFA (去趋势波动分析) 指数
    
    参数:
        series: 一维数值序列
        order: 去趋势的多项式阶数 (默认1，即线性去趋势)
        
    返回:
        alpha: DFA 指数 (类似 Hurst, ~0.5=白噪声, ~1.0=1/f噪声, ~1.5=布朗运动)
        data: (log_scales, log_flucts) 用于绘图的元组
    """
    series = np.array(series)
    N = len(series)
    if N < 20: return 0.5, (np.array([]), np.array([]))
    
    # 1. 积分 (累积和，并去均值)
    y = np.cumsum(series - np.mean(series))
    
    # 2. 设定尺度 scale
    min_scale = 4  # 降低最小尺度以适应短序列
    max_scale = N // 4
    if max_scale < min_scale:
         max_scale = N // 2 # 再次尝试放宽
         
    if max_scale < min_scale:
        return 0.5, (np.array([]), np.array([]))

    scales = np.unique(np.logspace(np.log10(min_scale), np.log10(max_scale), num=20).astype(int))
    scales = scales[scales > order + 2] # 保证有足够点进行多项式拟合
    
    if len(scales) < 3:
         # 尝试直接使用所有整数尺度
         scales = np.arange(min_scale, max_scale + 1)
         scales = scales[scales > order + 2]
    
    fluctuations = []
    
    for s in scales:
        # 将序列分割成长度为 s 的窗口
        # 为了利用更多数据，可以处理剩余部分（此处简单起见直接截断或重叠）
        # 这里使用非重叠窗口
        num_segments = N // s
        if num_segments < 1: continue
        
        rms = []
        
        for i in range(num_segments):
            segment = y[i*s : (i+1)*s]
            x = np.arange(s)
            
            # 3. 拟合趋势 (多项式)
            coeffs = np.polyfit(x, segment, order)
            trend = np.polyval(coeffs, x)
            
            # 4. 计算去趋势后的均方根波动
            rms.append(np.sqrt(np.mean((segment - trend)**2)))
            
        if rms:
            fluctuations.append(np.mean(rms))
        
    # 5. 双对数回归: log(F(s)) = alpha * log(s) + c
    if len(fluctuations) < 3:
        return 0.5, (np.array([]), np.array([]))
        
    log_scales = np.log10(scales[:len(fluctuations)])
    log_flucts = np.log10(fluctuations)
    
    alpha, c = np.polyfit(log_scales, log_flucts, 1)
    
    return alpha, (log_scales, log_flucts)


def fit_ar1(series):
    """
    拟合 AR(1) 模型: X_t = c + phi * X_{t-1} + epsilon_t
    
    参数:
        series: 一维数值序列
        
    返回:
        phi: 自回归系数 (衡量连续性/惯性)
        sigma: 残差标准差 (衡量随机波动强度)
    """
    series = np.array(series)
    N = len(series)
    if N < 3:
        return 0.0, 0.0
        
    # 构造 y = X_t 和 x = X_{t-1}
    y = series[1:]
    x = series[:-1]
    
    # 线性回归
    slope, intercept = np.polyfit(x, y, 1)
    phi = slope
    
    # 计算残差
    residuals = y - (intercept + phi * x)
    sigma = np.std(residuals, ddof=1)
    
    return phi, sigma


def compute_fft(series):
    """计算 FFT 频谱"""
    # 去直流分量
    series_centered = series - np.mean(series)
    # 加窗 (Hanning) 减少频谱泄漏
    window = np.hanning(len(series_centered))
    fft_vals = np.fft.rfft(series_centered * window)
    freqs = np.fft.rfftfreq(len(series_centered))
    magnitude = np.abs(fft_vals)
    return freqs, magnitude
