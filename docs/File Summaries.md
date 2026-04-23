##### *File Summaries*



###### **libchocolatine.py**

* Main detector of outages
* Owns 'ChocolatineDetector' object (also per-series objects)
* Methods:

 	- Series object when a new live stream appears

 	- Searches for an existing model in the database

 	- Requests a new model through Kafka if needed

 	- Requests historical data to begin predictions

 	- Processes historical data

 	- Reviews each live point to decide whether to generate an alert



###### **asynfecther.py**

* Turns Chocolatine series key into an IODA API request
* Methods:

 	- Parses different time series (BGP, IBR, Active Probing, gtr)

 	- Builds API query path

 	- Fetches history

 	- Pushes results through queues for the Chocolatine detector



###### **arimapredictor.py**

* File implemented after model identified
* Methods:

 	- Review recent history

 	- Smooth missing values using seasonal medians

 	- Maintain differenced history (seasonal differencing to eliminate weekly seasonality)

 	- Generates short-term forecasts (forecast method in file)

 	- Updates history when outages/anomalies occur to avoid shifting the baseline of model too aggressively



###### **arimabuilder.py**

* Methods:

 	- Creates multiprocessing worker process

 	- Sends jobs to workers for model construction ('ChocArimaJob' object estimates model for specific time series,'ChocArimaTrainer' object is background process running one jobs)

 	- Worker pool ('ChocArimaPool') manages collections of workers to run multiple jobs at once

 	- Calls code from arima.pyx to fit a model

 	- Returns model parameters and interval data for modeler



###### **modeller.py**

* Runs a separate service to build or refresh models when requested
* Methods:

 	- Listens on Kafka for model requests

 	- Fetches sufficient historical data for modelling

 	- Checks where time series is appropriate for modelling

 	- Decides whether to use a 'ZERO' model for mostly-zero data

 	- Submits model-building job to ARIMA pool

 	- Publishes generated model to Kafka



###### **arima.pyx**

* Responsible for statistical ARIMA/ARMA analysis
* Math engine that performs expensive model analysis used by the worker pool





##### *File Interactions \& Sequence*



**Interactions:**

1. **libchocolatine.py <--> asynfecther.py**: when libchocolatine.py takes in a live time series, it leverages classes/methods from asyncfetcher.py (AsyncHistoryFetcher, runAsyncFetcher()) to collect the appropriate data from the IODA API to begin prediction in libchocolatine.py
2. **libchocolatine.py <--> modeller.p**y: when a new series is created, libchocolatine.py checks the database for an existing saved model or sends a Kafka model request. modeller.py performs the Kafka requests, fetches model history, and decides whether the series should use a 'ZERO' model or a fitted ARMA model.
3. **modeller.py <--> asyncfetcher.py**: asyncfetcher.py provides the raw historical signal data (fetchIodaMeta(),fetchhIodaHistoricBlocking()) for modeller.py to decide whether and how to fit data to a model
4. **modeller.py <--> arimabuilder.py**: modeller.py identifies suitable time series for modelling and creates a ChocArimaJob object and submits it to the worker pool. arimabuilder.py accepts jobs and assigns them to workers based on the decisions of modeler.py
5. **arimabuilder.py <--> arima.pyx**: for each worker, they computer weekly differencing order from step size, creates Arima object from arima.pyx calls prepare\_analysis() from arima.pyx, collects model and MAD-based intervals, and packages results for modeller.py
6. **libchocolatine.py <--> arimapredictor.py**: when libchocolatine.py has history and a usable model, it creates an ArimaPredictor object from arimapredictor.py. Calls bootstrapHistory() from arimapredictor.py to compare observed value to predicted and threshold.



**Sequence:**

1. In libchocolatine.py, createNewSeries() creates the correct per-series object (ChocBgpTimeSeries, ChocGtrTimeSeries, ChocTelescopeTimeSeries, or ChocActiveTimeSeries) and then calls lookupModelInDatabase() from libchocolatine.py to see whether a fresh saved model already exists in PostgreSQL. If the model is missing or stale, createNewSeries() sends a Kafka model request using \_sendModelRequest(), also from libchocolatine.py. If no database model exists, it temporarily assigns a default model.
2. Still in libchocolatine.py, processLiveData() calls sendHistoryRequest() from libchocolatine.py, which creates a history job tuple (serieskey, timestamp, 10 weeks, s.datafreq) and pushes it onto the detector’s histRequest queue.
3. In libchocolatine.py, runTestInstance(detector, kafkaconf) function starts an AsyncHistoryFetcher object imported from asyncfetcher.py. It passes detector.histRequest and detector.histReply into that fetcher, so the fetcher can read requested history jobs and return completed history results.
4. In asyncfetcher.py, the AsyncHistoryFetcher object reads jobs from the queue and its async fetch() function calls formHistoryQuery() from asyncfetcher.py to translate the series key into the proper IODA API URL. fetch() then performs the HTTP request to the IODA API and pushes the returned history onto the detector’s histReply queue.
5. Back in libchocolatine.py, the detector’s run() method reads from histReply. When history arrives, run() calls processHistoryData() from libchocolatine.py. That method stores the history, computes time-slot medians, and then creates an ArimaPredictor object from arimapredictor.py. It calls ArimaPredictor.bootstrapHistory() and then ArimaPredictor.forecast(12) to initialize live forecasting.
6. If a better model is needed, modeller.py receives the Kafka model request. Its ChocModeller class uses \_fetchData() and deriveBestModel() from modeller.py. That file imports fetchIodaMeta and fetchIodaHistoricBlocking from asyncfetcher.py, so it reuses the same IODA query logic, but through the blocking fetch path rather than the detector’s async queue path. deriveBestModel() either decides the series should use a "ZERO" model or creates a ChocArimaJob object imported from arimabuilder.py.
7. In arimabuilder.py, the ChocArimaPool object receives that ChocArimaJob. One of its workers runs runArimaTrainer() from arimabuilder.py, which creates an arima.Arima() object from arima.pyx and calls prepare\_analysis(). That returns the fitted ARMA model and the MAD-based interval widths.
8. Back in modeller.py, the run() method collects completed worker results from the ChocArimaPool, builds a reply message, and publishes that model over Kafka using the producer configured in setupKafkaProducer(). If database storage is enabled, insertDatabaseRow() from modeller.py also writes the fitted model into PostgreSQL.
9. Back in libchocolatine.py, the detector’s run() method polls Kafka for model replies. When it receives one, it calls updateSeriesWithNewModel() from libchocolatine.py. That method updates the series’ AR/MA parameters and interval widths, then rebuilds the ArimaPredictor from arimapredictor.py by calling bootstrapHistory() and forecast(12) again if history is already available.
10. libchocolatine.py continues running processLiveData(), which compares each observed value against the prediction threshold and updates the predictor state. The predictor update itself is done through ArimaPredictor.appendHistory() from arimapredictor.py, while the decision logic remains in libchocolatine.py.
