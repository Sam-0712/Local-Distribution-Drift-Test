# Local Distribution Drift (LDDT)

This repository contains the implementation for **Local Distribution Drift (LDDT)**, a statistical mechanics approach to distinguishing human writing from LLM-generated text. 

Unlike typical detection methods that rely on neural network embeddings or semantic analysis, LDDT focuses on the "dynamics" of text generation by analyzing character distribution changes in a sliding window. The core hypothesis is that human writing is a non-stationary stochastic process (high noise, low memory), while LLM generation is a highly auto-regressive process (low noise, high memory).

## Methodology

### 1. Drift Definition
We treat text as a sequence of characters $T$. By sliding a window of size $w$ with step $s$, we calculate the character distribution $P_i$ for each window. The **Drift** is defined as the Jensen-Shannon (JS) divergence between adjacent windows:

$$ d_i = \text{JS}(P_i \| P_{i+1}) $$

This sequence of drifts captures the local statistical instability of the text.

### 2. Metrics
From the drift sequence, we compute several time-series metrics:
- **Mean Drift & Std Drift**: Basic statistical moments.
- **Acceleration**: The rate of change of the drift (2nd derivative).
- **Autocorrelation (ACF)**: Measures the "memory" of the process. $\rho(1)$ is critical for distinguishing the "inertia" of the generation process.
- **FFT Spectrum**: Analyzes energy distribution across frequencies.
- **Hurst Exponent ($H$)**: Measures long-range dependence (persistence vs. anti-persistence).
- **DFA Exponent ($\alpha$)**: Detrended Fluctuation Analysis, a robust measure of self-similarity.
- **AR(1) Coefficient ($\phi$)**: Fits an Auto-Regressive model $d_t = \phi d_{t-1} + \epsilon$. This is the most discriminative feature, representing the "inertia" of the text generation mechanism.

## Key Findings

Analysis reveals a distinct separation between **Human**, **LLM Rewrite**, and **LLM Polish** texts:

1.  **Macro vs. Micro Statistics**:
    -   **Macro statistics** (Mean Drift, Entropy Std) show huge effect sizes ($d > 4.0$) between Human and LLM Rewrite.
    -   However, **LLM Polish** texts can mimic Human texts perfectly in these macro statistics (the "Trojan Horse" effect), showing almost no difference ($d < 0.35$).

2.  **The Power of Dynamics**:
    -   The **AR(1) coefficient ($\phi$)** and **Autocorrelation ($\rho(1)$)** successfully expose the "disguise" of LLM Polish.
    -   Even when macro statistics are indistinguishable, LLM Polish texts exhibit dynamics closer to LLM Rewrite (higher memory/inertia) than to Human texts (high noise).

3.  **Response to Noise**:
    -   Removing paragraph breaks (flattening the text structure) affects Human and LLM Polish texts significantly but leaves LLM Rewrite largely unaffected. This confirms that LLM generation is inherently smoother and less dependent on structural "noise."

## Results Visualization

### Phase Space Analysis
Plotting Entropy against Drift reveals the trajectory of the text's state.
- **Human**: High curvature, chaotic trajectory.
- **LLM**: Low curvature, smooth trajectory.

![Phase Space Analysis](/pic/95fcd883-d550-4ed9-83f2-c7919c6a3552.png)

### Drift Distribution
LLM texts consistently show lower drift and lower variance, indicating a more "conservative" generation process.

![Drift Distribution](/pic/4aafc52f-3bf9-4afb-a99d-6cf5a2a5fde0.png)

### Autocorrelation & Dynamics
The ACF decay and AR(1) coefficients provide the strongest discriminative power.

![Autocorrelation](/pic/86f2dc95-32c9-41e9-b124-941a4a75ceb0.png)
![FFT Spectrum](/pic/592025bb-e7ea-4fb7-8615-de59e1fc5c73.png)

## Code Structure

-   `src/calc.py`: Core calculation functions (Drift, Entropy, ACF, FFT, Hurst, DFA, AR fitting).
-   `src/viz.py`: Visualization utilities using Matplotlib (smoothing, plotting helpers).
-   `compare_plus.py`: Main execution script. Handles multi-group comparison (Human vs. Rewrite vs. Polish) and generates statistical summaries and matrix plots.

## Usage

1.  Prepare your corpus in `corpus.json` with the following structure:
    ```json
    [
      {
        "group_id": "1",
        "human": "Original text...",
        "llm_rewrite": "LLM rewritten text...",
        "llm_polish": "LLM polished text..."
      }
    ]
    ```

2.  Run the analysis:
    ```bash
    python compare_plus.py
    ```

3.  Results will be saved in the `results/comparison_plus/` directory, including:
    -   `metrics.csv`: Statistical scalars for each text.
    -   Matrix plots comparing Phase Space, KDE, Trends, ACF, and FFT.
    -   `distinction_heatmap.png`: Cohen's d effect size heatmap for metric distinction power.

## Dependencies

-   Python 3.x
-   NumPy
-   SciPy
-   Matplotlib
-   Seaborn
-   Pandas
