# Chocolatine Offline Testing

**Authors**: Georgia Tech VIP for Internet Outage Detection & Analysis (IODA) - Michael Kolby, Pranav Anumandla, Elizabeth Iaryguine

Offline SARIMA experimentation framework for reproducing, evaluating, and supporting implementation of the Chocolatine forecasting pipeline for IODA anomaly detection.

Chocolatine is an experimental forecasting system developed for IODA to model internet activity signals, generate expected baselines, and detect abnormal behavior such as outages or sudden traffic disruptions.

Original source code: [InetIntel/chocolatine](https://github.com/InetIntel/chocolatine)

This repository rebuilds core Chocolatine prediction behavior in an offline environment using exported model parameters, historical IODA signal data, and adapted source-code logic from:

- `libchocolatine.py`
- `arimapredictor.py`
- `asyncfetcher.py`
- `arimabuilder.py`
- `modeller.py`
- `arima.pyx`

This project was developed to evaluate how the IODA Google Transparency Report (GTR) SARIMA pipeline behaves under different parameter settings, forecast horizons, and outage scenarios to inform future deployment decisions.


--- 

## Objective

Reconstruct the behavior of a `ChocGtrTimeSeries` object offline and review how forecasting outputs change under controlled modifications to:

- historical fetch duration
- bootstrap history length
- seasonal slot grouping (`slotmod`)
- forward forecast duration
- outage / collapse scenarios

---

## Repository Structure
- `offline_sarima.py`: main offline testing framework

- `arma_model_dump_gtr.xlsx`: exported IODA parameters used for forecasting

- `annotated_source/` – commented and explained versions of core Chocolatine source files

- `source/` - original Chocolatine source files

- `docs/` – Chocolatine source code notes and experiment summaries

- `sample_outputs/` - example Excel exports and sample files generated during testing

- `test_results/` - consolidated outputs and summarized results from completed experiments


## Reconstructed Chocolatine Workflow

```python
s = choc.ChocGtrTimeSeries(args.fqid)
s.assignDatabaseModel(dbmodel)
```

Uses parameters exported from `arma_model_dump_gtr.xlsx`:
- `ar_param`
- `ma_param`
- `model_type`
- `pred_intervals`


## Recreated Logic

### processHistoryData()

`offline_process_history_data()` rebuilds the following:
- `s.history`
- `s.histslots`
- `s.smallesthist`
- `s.baseline`

Then initializes:

```python
s.predictor = ArimaPredictor(s.arma, s.datafreq)
s.predictor.bootstrapHistory(...)
```

### processLiveData()

`offline_process_live_data()` recreates rolling prediction logic:

```python
s.predicted = s.predictor.forecast(12)
s.predictor.appendHistory(...)
```

## Forecast Structure

- `datafreq = 1800` seconds
- 30-minute cadence
- 12 forecast steps = 6-hour block
- Weekly seasonality via `s.ppw`

---
## Setup & Running Offline Tests

### Required Files

The offline framework expects the following files in the repository:

- `offline_sarima.py`
- `arma_model_dump_gtr.xlsx`
- Chocolatine source files in `/source`

### Python Dependencies

```bash
pip install pandas numpy regex requests openpyxl scipy statsmodels confluent_kafka pycountry
```

---

## Model Experiments

### Forecast Horizon Experiment
- Tracked live prediction interval changes to estimate how long deployed parameters remained active before refresh or rebuild

- **Key Finding:** Threshold comparisons indicated a finite deployed model runtime, with prediction interval regeneration observed after `~66 days`

### Model Failure Analysis
- Investigated giant API values near `1.8446744e19`

- **Key Finding:** Evidence suggests negative forecasts may wrap into unsigned 64-bit values

### Priming Experiment
- Tested fetch-history and bootstrap durations to analyze forecast accuracy.

- **Key Finding:** `6 / 4` minimized error between predicted and observed values, while 10/4 most closely matched live outputs

### Seasonal Slot Experiment
- Tested seasonal grouping sizes `slotmod` for weekly historical medians

- Experiments conducted with `6 / 4` configuration from Priming Experiment

- **Key Finding:** `slotmod = 4` produced the strongest forecasting accuracy across tested signals

---

## Limitations

- Offline environment does not include full production schedulers or cache refresh logic
- Hidden live state may differ from exported parameters
- Some failure conclusions are inference-based

## Future Work

1. Automate offline testing for faster parameter sweeps and repeatable experiments.
2. Expand validation across more countries, services, and time windows.
3. Study more outage failures to better understand collapse behavior.
4. Centralize results tracking for easier comparison and ranking.
5. Test top settings in staging/live environments to support deployment decisions.

## Development Notes

AI-assisted tools were used as a supplemental resource for debugging, documentation, and Chocolatine source code navigation. All experiment methodology, implementation, result interpretation, and final conclusions were directed and validated by the repository author.
