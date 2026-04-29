# Offline SARIMA Experiment Summaries

This document provides a high-level summary of the main offline experiments conducted using `offline_sarima.py` to better understand Chocolatine's forecasting behavior, model lifecycle, and anomaly detection logic.

---

# Experiment 1: Forecast Horizon Drift

## Purpose

Determine whether live IODA SARIMA prediction intervals remained consistent with the saved model dump (`arma_model_dump_gtr.xlsx`) or changed over time.

## Method

- Used the same ARMA parameters and prediction intervals from the model dump.
- Queried live `gtr-sarima` API outputs at selected timestamps.
- Compared:

  - Offline prediction interval widths  
  - Live API threshold differences (`predicted - threshold`)


## Metrics Reviewed

Statistics were calculated by comparing offline predicted and threshold values against corresponding API predicted and threshold values.

Reported metrics:

- Mean / median / max relative error
- Count of mismatches above 5%


## Key Findings

- Prediction intervals eventually diverged from the saved model dump.
- Drift appeared consistent with a model refresh / retraining cycle of ~60 days.
- Intervals later converged back to values matching the exported model dump.
- Suggests models may be periodically regenerated and redeployed.
- Relative error between API and offline forecasts improved substantially once prediction intervals aligned with the model dump.


---

# Experiment 2: Priming Sensitivity (History + Bootstrap Windows)

## Purpose

Evaluate how initialization settings impact short-term forecast accuracy.

This combined experiment tested:

1. Total historical data loaded before forecasting  
2. Recent history used during `bootstrapHistory()`

## Method

### History Windows Tested

- 6 weeks  
- 8 weeks  
- 10 weeks  
- 12 weeks

### Bootstrap Windows Tested

- 2 weeks  
- 4 weeks  
- 6 weeks  
- 8 weeks  
- 10 weeks  
- 12 weeks

## Metrics Reviewed

Statistics were calculated using observed signal values versus offline forecast values after removing alertable outage periods.

Reported metrics:

- Mean / median absolute error
- Mean / median / max relative error
- Share of forecasts with error > 10%

## Key Findings

- Forecast quality changed meaningfully depending on initialization settings.
- Moderate history windows (6 weeks) often performed better than excessively long windows.
- Smaller to medium bootstrap windows (2 or 4 weeks) often adapted better to recent behavior.
- Very large windows could reduce responsiveness.

---

# Experiment 3: Seasonal Slot Sensitivity

## Purpose

Test how changing seasonal repetition assumptions impacts forecasts.

## Method

Modified slot grouping periods used in seasonal historical buckets.

Compared:

- 1 week  
- 2 weeks  
- 4 weeks

## Metrics Reviewed

Statistics were calculated using observed signal values versus offline forecast values after removing alertable outage periods.

Reported metrics:

- Mean / median absolute error
- Mean / median / max relative error
- Share of forecasts with error > 10%

## Key Findings

- Forecast quality was sensitive to seasonal slot assumptions.
- Weekly repetition patterns appear meaningful for GTR signals.


---

# Experiment 4: Model Failure Analysis

## Purpose

Investigate extreme API forecast values observed during certain outages (approximately `1.844674e+19`).

## Method

- Filtered timestamps where the live API returned extremely large predicted values (~`1.844674e+19`).
- Noted these values were close to `2^64`, suggesting possible unsigned overflow of negative forecasts.
- Added `2^64` to negative offline forecasts and subtracted `2^64` from API values to reconstruct signed equivalents.
- Compared reconstructed values with offline forecasts using relative error metrics to test the overflow hypothesis.

## Metrics Reviewed

Statistics were calculated using timestamps where API predictions returned abnormal extreme values and compared against reconstructed offline forecasts.

Reported metrics:

- Mean / median relative error

## Key Findings

- Extremely large API values were consistent with possible numeric overflow.
- Failures often coincided with negative offline prediction states.

---

# Overall Conclusions

- Offline reconstruction successfully approximated live Chocolatine forecasts.
- Initialization choices materially impact prediction quality.
- Production models likely refresh periodically.
- Extreme outages may expose numerical edge cases.

---