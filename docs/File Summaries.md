# File Summaries
This document summarizes the Chocolatine source files used to understand offline SARIMA prediction, model training, and anomaly detection.

---

## Core Prediction Files

### `libchocolatine.py`

- Main orchestration layer for Chocolatine.
- Defines time series classes for different IODA signal types.
- Connects historical data fetching, model loading, forecasting, and anomaly detection into one pipeline.

#### Responsibilities

- Creates per-series objects when a new live stream appears.
- Searches for an existing saved model in the database.
- Requests a new model through Kafka if needed.
- Requests historical data to begin predictions.
- Processes historical data.
- Reviews each live point to determine whether to generate an alert.

---

### `asyncfetcher.py`

- Handles retrieval of historical IODA data from the API.
- Builds API requests depending on the signal type.
- Supplies history used for model initialization and training.

#### Responsibilities

- Parses different time series types (BGP, IBR, Active Probing, GTR).
- Builds API query paths.
- Fetches historical data.
- Pushes results through queues for the Chocolatine detector.

---

### `arimapredictor.py`

- Used after a model has already been trained.
- Performs online forecasting using the trained ARIMA model.

#### Responsibilities

- Maintains recent signal history.
- Fills missing values using seasonal medians.
- Applies seasonal differencing.
- Generates short-term forecasts.
- Updates history when outages occur so the baseline does not shift too aggressively.

---

## Model Training Files

### `modeller.py`

- Manages offline generation of forecasting models used by Chocolatine.
- Receives model requests, gets historical IODA data, and determines whether enough data exists to train a model.
- If sufficient data exists, it submits ARIMA training jobs.
- Once complete, it returns ARMA parameters and MAD values used for anomaly detection.

#### Responsibilities

- Runs a separate service to build or refresh models.
- Listens on Kafka for model requests.
- Fetches sufficient historical data for modelling.
- Determines whether a time series is appropriate for modelling.
- Chooses a `ZERO` model for mostly-zero data.
- Submits jobs to the ARIMA worker pool.
- Publishes generated models to Kafka.

---

### `arimabuilder.py`

- Manages parallel training of ARIMA models.

#### Responsibilities

- Creates worker processes.
- Sends training jobs to workers.
- Each worker trains a model using `arima.pyx`.
- Returns results to the main process.
- Worker pool can run many jobs simultaneously.

---

### `arima.pyx`

- Main ARIMA statistical engine used by Chocolatine.

#### Responsibilities

- Performs seasonal differencing.
- Selects fitted ARMA parameters.
- Computes error statistics and MAD intervals.

---

# File Interactions

---

## Main Relationships

1. **libchocolatine.py ↔ asyncfetcher.py**  
   Uses asyncfetcher to collect historical IODA data required before forecasting.

2. **libchocolatine.py ↔ modeller.py**  
   Requests new models when saved models are missing or stale.

3. **modeller.py ↔ asyncfetcher.py**  
   Uses historical data to determine whether and how to train models.

4. **modeller.py ↔ arimabuilder.py**  
   Sends model-building jobs to worker processes.

5. **arimabuilder.py ↔ arima.pyx**  
   Workers call ARIMA engine to fit models and compute MAD intervals.

6. **libchocolatine.py ↔ arimapredictor.py**  
   Uses trained models to generate live forecasts and thresholds.

---

# Full System Sequence

---

1. `libchocolatine.py` creates a new series object and checks PostgreSQL for an existing model.

2. If missing or stale, it sends a Kafka request for a new model.

3. It requests historical signal data through `asyncfetcher.py`.

4. `asyncfetcher.py` queries the IODA API and returns the data.

5. `libchocolatine.py` processes the history and initializes an `ArimaPredictor`.

6. `modeller.py` receives the Kafka request and fetches sufficient training data.

7. If suitable, `modeller.py` creates a job for `arimabuilder.py`.

8. Worker processes in `arimabuilder.py` call `arima.pyx` to fit a model.

9. Trained ARMA parameters and MAD intervals are returned to `modeller.py`.

10. `modeller.py` publishes the finished model through Kafka.

11. `libchocolatine.py` receives the updated model and rebuilds forecasting objects.

12. Live observations continue to be tested against thresholds for anomaly detection.
