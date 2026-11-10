# Literature Review: Reliability of Probabilistic Forecasts

**Paper:** Gneiting, T., Balabdaoui, F., & Raftery, A. E. (2007). *Probabilistic forecasts, calibration and sharpness*. Journal of the Royal Statistical Society: Series B, 69(2), 243–268. https://academic.oup.com/jrsssb/article/69/2/243/7109375

**Research context:** Health monitoring and early warning for LEAR v3 electricity-price forecasts.

## Main question
**What makes a probabilistic forecast reliable, and which metrics should we use to evaluate it?**

A reliable probabilistic forecast should be **calibrated** and **as sharp as possible subject to calibration** (Gneiting et al., 2007).

- **Calibration** means statistical consistency between the predicted distributions and realized observations. For example, a nominal 90% prediction interval should contain approximately 90% of observed prices over many forecasts.
- **Sharpness** means concentration of the predictive distributions: narrower prediction intervals are more informative, *provided that calibration is maintained*.
- **PIT (probability integral transform)** evaluates probabilistic calibration. For a predictive cumulative distribution function (CDF) \(F_t\) and observed price \(y_t\), the PIT is \(p_t=F_t(y_t)\). For ideal continuous forecasts, PIT values should be uniformly distributed. U-shaped PIT histograms often suggest forecasts that are too narrow; hump-shaped histograms often suggest forecasts that are too wide; sloped histograms can suggest bias. **Uniform PIT is not sufficient to prove forecast reliability**: the paper demonstrates that several non-ideal forecasters have approximately uniform PIT histograms.
- **Proper scoring rules** evaluate complete predictive distributions with a numerical penalty and encourage truthful probability forecasts. The paper discusses **CRPS** (continuous ranked probability score), which evaluates the CDF against observations and is more robust than the **logarithmic score**. For both, lower average scores indicate better forecasting performance. CRPS is expressed in the same units as the forecast variable (€/MWh for electricity prices).

The authors also distinguish *probabilistic*, *exceedance* and *marginal* calibration. These are theoretically distinct. Marginal calibration compares the average forecast CDF with the empirical CDF of observed outcomes. The paper does not provide a straightforward operational sample diagnostic for its formal definition of exceedance calibration.

## Which metrics should monitor the absolute performance of LEAR v3?

| Priority | Metric | What it tells us |
|---|---|---|
| **Core** | **Mean CRPS** | Overall quality of probabilistic forecasts; a sustained increase signals poorer performance, though greater market difficulty may also raise CRPS. |
| **Core** | **Empirical prediction-interval coverage** | Whether observed prices fall inside nominal intervals at the expected frequency (e.g., 80%, 85%, 90%). The corresponding expected noncoverage rates are 20%, 15%, 10%. |
| **Core** | **PIT histogram** | Whether probability forecasts appear calibrated; helps diagnose underdispersion, overdispersion or bias. |
| **Core** | **Prediction-interval width / sharpness** | Whether forecasts are informative; interpret narrower intervals together with coverage, not alone. |
| **Diagnostic** | **Marginal calibration plot** | Whether the predicted overall price distribution matches the observed price distribution. |
| **Diagnostic** | **PIT autocorrelation** | Whether PIT values display unexplained temporal patterns; interpretation must account for forecast horizons and dependence. |
| **Optional** | **Logarithmic score** | Density-based performance; highly sensitive to observations assigned very low density. |
| **Optional, event-specific** | **Brier score** | Quality of predicted probabilities for selected price events, such as negative prices. |

**Recommended minimal LEAR v3 monitoring set:** **CRPS + empirical coverage + PIT histogram + prediction-interval width.** Use marginal calibration and PIT autocorrelation to investigate emerging problems.

These assess **absolute forecast performance**, rather than only whether LEAR v3 outperforms a competing model. For an early-warning system, evaluate their evolution over time and, where useful, separately by delivery hour or market regime. A rise in CRPS alone does **not** establish model failure because market conditions can become harder to forecast. Monitoring windows, alert thresholds and statistical change-detection rules must be chosen and validated separately; **the paper does not specify these for LEAR v3**.

## Short paper summary

Gneiting et al. (2007) propose evaluating probabilistic forecasts by **maximizing sharpness subject to calibration**. They show that calibration cannot be judged from PIT uniformity alone: non-ideal forecasts may have apparently uniform PIT histograms. Their diagnostic framework combines PIT histograms, marginal calibration plots, sharpness diagrams and proper scoring rules such as CRPS and the logarithmic score. In a wind-speed forecasting case study, these complementary diagnostics identify differences in forecast quality that a single coverage measure would miss.

**Takeaway for LEAR v3:** Monitor **calibration** (PIT and interval coverage), **sharpness** (interval width) and **overall predictive quality** (CRPS) together. This is a foundation for probabilistic forecast health monitoring, not a complete early-warning algorithm.
