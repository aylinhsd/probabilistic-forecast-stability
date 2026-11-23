# Experiment 15c — Reliable detector selection under a common notification budget

**Input:** Actual saved probabilistic forecast outputs: ENTSOE_Probabilistic_LEAR_v3_predictions.csv  
**Fit period:** 2025-07-01 to 2025-12-07 (160 days)  
**Reference remainder (not assumed verified null):** 2025-12-08 to 2026-02-28  
**Historical evaluation:** 2026-03-01 to 2026-10-13 (227 days)  
**Target:** <=1 grouped notification per 30 synthetic null days, with 7 day cooldown.

## Controlled test method

1. Fit only on the first chronological reference days. Bootstrap contiguous 7-day baseline blocks to construct no-change series.
2. For each detector, select parameters on calibration synthetic null trials, not test trials. Match the **combined** notification budget across four signals.
3. Evaluate held-out synthetic no-change sequences and inject signal-level sustained, gradual, and temporary changes with known onset.
4. Save unverified historical notifications separately; no historical alert is called a false alarm.

## Method comparison

```
      method                 settings  cal_budget_met  cal_null_notifications_per_100_days  heldout_null_notifications_per_100_days  sustained_detected_30d_pct  weak_sustained_detected_30d_pct  moderate_sustained_detected_30d_pct  gradual_detected_30d_pct  temporary_detected_30d_pct  sustained_median_delay_detected_days  incremental_sustained_detection_pct  paired_null_notifications_in_event_window_pct  historical_evaluation_notifications_unlabeled  historical_notif_per_100_days
Page-Hinkley {"h": 18, "delta": 0.15}            True                                 2.75                                     2.80                        98.7                             96.0                                100.0                      91.7                        79.3                                   7.0                                 26.7                                           72.0                                             23                          10.13
        EWMA   {"L": 4, "alpha": 0.2}            True                                 3.12                                     3.35                        94.7                             85.0                                 99.0                      87.3                        77.0                                   6.0                                 26.7                                           68.0                                             25                          11.01
       CUSUM      {"h": 12, "k": 0.5}            True                                 2.42                                     2.64                        93.7                             81.0                                100.0                      84.0                        64.7                                   7.0                                 37.7                                           56.0                                             25                          11.01
   Fixed P95              {"q": 0.99}            True                                 2.79                                     3.08                        82.3                             77.0                                 82.0                      78.7                        71.3                                   7.0                                 14.3                                           68.0                                             26                          11.45
       ADWIN           {"delta": 0.1}            True                                 1.10                                     1.02                        81.3                             55.0                                 93.0                      66.0                        19.3                                  30.0                                 62.3                                           20.0                                              4                           1.76
```

## Interpretation and limitations

- `heldout_null_notifications_per_100_days` describes **synthetic no-change** notifications, not confirmed real-world false alarms.
- `sustained_detected_30d_pct` is the fraction of **injected signal shifts** detected within 30 days, irrespective of whether a paired null series would also have alerted.
- `incremental_sustained_detection_pct` is stricter: records an injected-series notification only when the same paired no-change series had none in that window.
- Historical notifications may be justified or unjustified; no ground-truth failure labels were supplied.
- Simulation changes only ONE monitoring signal at a time; it does not modify raw forecasts consistently. For coherent forecast-level validation, use Experiment 15b.
- The original reference period was selected after volatility inspection. The fit window may include seasonal or behavioral patterns not representative of a stationary null.
- The selected detector settings are research candidates, not deployment-ready settings; evaluate in several chronological folds and on external models.
- Upward risks of coverage below/above nominal are separate signals; high interval width can also reflect appropriate uncertainty.

## Next steps

1. Audit signal definitions against Exp 14/15; investigate intervals and regime-aware thresholds.
2. Validate and integrate forecast-level perturbations from Exp 15b, including unchanged forecast nulls.
3. Add documented real model deterioration labels or external expert adjudication.
4. Choose a detector based on held-out detection, delayed/missed events, and allowed notifications, not synthetic detection alone.
5. Add a separate action-policy study before recommending recalibration/retraining.
