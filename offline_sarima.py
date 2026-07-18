# This source code is Copyright (c) 2026 Georgia Tech Research Corporation. All
# Rights Reserved. Permission to copy, modify, and distribute this software and
# its documentation for academic research and education purposes, without fee,
# and without a written agreement is hereby granted, provided that the above
# copyright notice, this paragraph and the following three paragraphs appear in
# all copies. Permission to make use of this software for other than academic
# research and education purposes may be obtained by contacting:
#
#  Office of Technology Licensing
#  Georgia Institute of Technology
#  926 Dalney Street, NW
#  Atlanta, GA 30318
#  404.385.8066
#  techlicensing@gtrc.gatech.edu
#
# This software program and documentation are copyrighted by Georgia Tech
# Research Corporation (GTRC). The software program and documentation are 
# supplied "as is", without any accompanying services from GTRC. GTRC does
# not warrant that the operation of the program will be uninterrupted or
# error-free. The end-user understands that the program was developed for
# research purposes and is advised not to rely exclusively on the program for
# any reason.
#
# IN NO EVENT SHALL GEORGIA TECH RESEARCH CORPORATION BE LIABLE TO ANY PARTY FOR
# DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING
# LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION,
# EVEN IF GEORGIA TECH RESEARCH CORPORATION HAS BEEN ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE. GEORGIA TECH RESEARCH CORPORATION SPECIFICALLY DISCLAIMS ANY
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED
# HEREUNDER IS ON AN "AS IS" BASIS, AND  GEORGIA TECH RESEARCH CORPORATION HAS
# NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
# MODIFICATIONS.
#
# This source code is part of the chocolatine software. The original
# chocolatine software is Copyright (c) 2021 The Regents of the University of
# California. All rights reserved. Permission to copy, modify, and distribute
# this software for academic research and education purposes is subject to the
# conditions and copyright notices in the source code files and in the
# included LICENSE file.

"""
offline_sarima.py

Offline reconstruction of Chocolatine's GTR SARIMA model for
validation, experimentation, and outage detection analysis.
"""

# ==========================================================
# SETUP REQUIREMENTS
# Before running this file, ensure the following resources
# are located in the same working directory:
#
# Required model parameter file:
#   - arma_model_dump_gtr.xlsx
#
# Required Chocolatine source files:
#   - libchocolatine.py
#   - arimapredictor.py
#   - asyncfetcher.py
#   - arimabuilder.py
#   - modeller.py
#   - arima.pyx
#
# Required Python packages:
#   pip install pandas numpy regex requests openpyxl scipy
#   statsmodels confluent_kafka pycountry
# ==========================================================
from arimapredictor import ArimaPredictor
from asyncfetcher import fetchIodaHistoricBlocking
import libchocolatine as choc
import pandas as pd 
import numpy as np  
import regex as re 
import statistics
import requests
import argparse, sys
import pycountry

IODA_API="https://api.ioda.inetintel.cc.gatech.edu/v2/signals/raw"

# Set argument parser to ensure all arguments are valid for assignDatabaseModel() method on a ChocGtrTimeSeries object
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fqid", type=str, help="The FQID of the signal you want to run SARIMA against")
parser.add_argument("-md", "--model", type=str, help="The model type for the predictions must be either ARMA or ZERO")
parser.add_argument("-t", "--time", type=int, help="The Unix timestamp where you want the predictions to begin")
parser.add_argument("-d", "--duration", type=int, help="The length of time that you want predictions to be generated for, in seconds")
parser.add_argument("-a", "--ar", type=int, help="The AR parameter for the SARIMA model")
parser.add_argument("-m", "--ma", type=int, help="The MA parameter for the SARIMA model")
parser.add_argument("-p", "--predintervals", type=str, help="The prediction intervals from the SARIMA model, expressed in the format {A,B,C,..N}")

args = parser.parse_args()

# ==========================================================
# USER CONFIGURATION
# - Choose country, service, start time, duration, etc.
# - Choose parameters to manipulate for testing
# ==========================================================
desired_country = "Iran"
desired_service = "WEB_SEARCH"
args.time = 1770877800 # Thursday, February 12, 2026 at 6:30:00 AM UTC
test_durations = [6*60*60, 1*24*60*60, 3*24*60*60, 5*24*60*60, 6*24*60*60, 7*24*60*60, 8*24*60*60, 9*24*60*60, 10*24*60*60, 25*24*60*60, 30*24*60*60, 60*24*60*60]
args.duration = test_durations[1]
DEBUG_ERROR_METRICS = False

# Change the following for experimentation. Otherwise, assign respective baseline values
test_num_weeks = [6*7*24*60*60, 8*7*24*60*60, 10*7*24*60*60, 12*7*24*60*60]
history_duration = test_num_weeks[2] # baseline value is 10 weeks
bootstrap_durations = [2, 4, 6, 8, 10, 12] 
bd = bootstrap_durations[1] # baseline value is 4 weeks
slotmod_durations = [1, 2, 4] 
sd = slotmod_durations[1] # baseline value is 2 weeks

# Set following metrics to True to generate metrics for experiments
DEBUG_INTERVAL_METRICS = False
DEBUG_PRIMING_METRICS = False
DEBUG_FAILURE_METRICS = False
DEBUG_THRESHOLD_TESTS = False # reserved for future analysis

# ==========================================================
# LOAD MODEL PARAMETER DUMP
# Read saved ARMA parameters and intervals
# ==========================================================
model_dump = pd.read_excel("arma_model_dump_gtr.xlsx").copy()

# Extract prediction intervals from data dump and create a list of lists, 
# where each list contains 12 intervals corresponding to
# the forecast methodology of 12 steps per prediction
pdlist = model_dump["pred_intervals"].fillna("").apply(lambda s: [0.0]*12 if s.strip("{}").strip() == "" 
else [float(x) for x in s.strip("{}").split(",") if x.strip()])

# Create a dictionary with country names mapped to country codes from the pycountry python library
country_codes = {c.name.lower(): c.alpha_2 for c in pycountry.countries}

# We identify and store the country's matching code from the country_codes dictionary 
for country in country_codes:
    if desired_country.lower() in country:
        code = country_codes[country]
        break

google_service_types = []

# Filter the data dump to only contain entries for the desired country
country_rows = model_dump[model_dump["entitycode"] == code]

# Extract service type (e.g., WEB_SEARCH) from fqid
for idx, row in country_rows.iterrows():
    google_service_types.append(row["fqid"].split(".")[3])

# Review the available google service types for desired_country
#print(len(google_service_types))
#print(google_service_types)

# Identify row from data dump that matches desired_service
i = 0
for idx, service in enumerate(google_service_types):
    if desired_service == service:
        i = idx
        break

# Filter to the row that represents desired_country and desired_service
target_row = country_rows.iloc[i]
idx = target_row.name

args.predintervals = pdlist.iloc[idx]
args.ar = int(target_row["ar_param"])
args.ma = int(target_row["ma_param"])
args.fqid = target_row["fqid"]
args.md = target_row["model_type"]

# ==========================================================
# ARGUMENT DEFAULTS AND VALIDATION
# Use configured defaults when command-line values are missing
# ==========================================================

# Adjust this starting time depending on desired starting time for offline predictions 
# Select times at increments of exactly 30 minutes due to behavior of GTR
# Use this link to convert dates into Unix Timestamps: https://www.epochconverter.com/
# Starting times for predictions must begin at 00:30, 06:30, 12:30, 18:30 UTC
if args.time is None:
    # Baseline Validation Date
    args.time = 1770877800 # Thursday, February 12, 2026 at 6:30:00 AM UTC

    # Recent Outage Start Times
    #args.time = 1767895200 # Thursday, January 8, 2026 at 6:00:00 PM UTC
    #args.time = 1772263800 # Saturday, February 28, 2026 at 7:30:00 AM UTC

# Adjust this duration depending on desired time frame for analysis
# test_durations is a list containing the total seconds in 1, 3, 5, 6, and 7+ days 
if args.duration is None:
    args.duration = test_durations[1]

if args.fqid is None:
    print("Error: must provide a signal FQID")
    parser.print_help()
    sys.exit(1)

if args.md is None:
    print("Error: must provide a model")
    parser.print_help()
    sys.exit(1)

if args.time is None:
    print("Error: must provide a prediction start time")
    parser.print_help()
    sys.exit(1)

if args.duration is None or args.duration <= 0:
    print("Error: must provide a valid duration")
    parser.print_help()
    sys.exit(1)

if args.predintervals is None:
    print("Error: must provide the prediction intervals for the model")
    parser.print_help()
    sys.exit(1)

if args.ar is None or args.ar < 0 or args.ar >= 3:
    print("Warning: did not provide a valid AR value, defaulting to 1")
    args.ar = 1

if args.ma is None or args.ma < 0 or args.ma >= 3:
    print("Warning: did not provide a valid MA value, defaulting to 1")
    args.ma = 1

# ==========================================================
# BUILD OFFLINE TEST SERIES OBJECT
# Mimics model loading from libchocolatine.py
# ==========================================================
# Arguments are set to run the assignDatabaseModel() method on a ChocGtrTimeSeries object
dbmodel = {
    'ar_param': args.ar,
    'ma_param': args.ma,
    'model_type': args.md,
    'generated_at': 0,
    'pred_intervals': args.predintervals
    }

# Identify the type of time series used for offline testing 
# For these tests, we will only work with GTR time series
seriestype = args.fqid.split('.')[0]
if seriestype == "google_tr":
    s = choc.ChocGtrTimeSeries(args.fqid)
else:
    print("Unsupported series type '%s'" % (seriestype))
    sys.exit(1)

s.assignDatabaseModel(dbmodel)

# ==========================================================
# OFFLINE RECONSTRUCTION OF CHOCOLATINE PIPELINE
# - Adapted from libchocolatine.py for offline environment
# - These methods were adapted from Chocolatine's live 
# detector so forecasts could be reproduced and tested 
# offline using historical IODA signals
# ==========================================================

# ----------------------------------------------------------
# Rebuild processHistoryData()
# Initializes model state using historical observations
# ----------------------------------------------------------
def offline_process_history_data(s, hist, bd, sd):
    s.history = None
    s.histslots = {}
    s.smallesthist = []
    s.predictor = None
    s.predicted = None
    s.pred_intervals = []
    s.predict_source = ""
    s.baseline = -1

    ts = hist["from"]
    step = hist["step"]
    native = hist["nativeStep"]

    assert step == native, f"step ({step}) != nativeStep ({native})"

    res = []

    for v in hist["values"]:
        # Build history in the format expected by ArimaPredictor
        res.append({
            "timestamp": pd.Timestamp(ts, unit="s"),
            "signalValue": v
        })

        ts += step

        if v is not None:
            slot = ts % (60 * 60 * 24 * 7 * 2)

            if slot not in s.histslots:
                s.histslots[slot] = [int(v)]
            else:
                s.histslots[slot].append(int(v))

            if len(s.smallesthist) < 40:
                s.smallesthist.append(v)
                s.smallesthist = sorted(s.smallesthist)
            elif v < s.smallesthist[-1]:
                s.smallesthist[-1] = v
                s.smallesthist = sorted(s.smallesthist)

    s.history = res

    # Build predictor only if not ZERO model
    if s.modeltype != choc.CHOC_MODEL_TYPE_ZERO:
        medians = {}
        for k, v in s.histslots.items():
            medians[k] = statistics.median(sorted(v))

        if s.arma_source == "default":
            s.setDefaultPredIntervals(12)
        s.predictor = ArimaPredictor(s.arma, s.datafreq)
        s.predictor.bootstrapHistory(
            s.history[-bd * s.ppw:],
            medians,
            60 * 60 * 24 * 7 * sd
        )

        #print("datafreq:", s.datafreq) # should be 1800, matching the step size of GTR
        #print("ppw:", s.ppw) # assigned value for weekly seasonality which is calculated using s.datafreq

        s.predicted = s.predictor.forecast(12)
        s.pred_intervals = s.arma_mads_scores.copy()
        s.predict_source = s.arma_source

    s.baseline = max(1, statistics.median(s.smallesthist))

# ----------------------------------------------------------
# Rebuild processLiveData()
# Simulates live streaming predictions & outage detection
# ----------------------------------------------------------
def offline_process_live_data(s, timestamp, value):
    event = None

    if s.modeltype == choc.CHOC_MODEL_TYPE_ZERO:
        event = {
            "timestamp": timestamp,
            "observed": value,
            "predicted": 0,
            "threshold": 0,
            "norm_threshold": 0,
            "alertable": False,
            "baseline": s.baseline,
        }
        return event

    if s.history is None or s.predictor is None:
        return None

    while s.predicted is None or len(s.predicted) == 0 or timestamp > s.predicted[-1]["timestamp"]:
        if s.predicted and len(s.predicted) > 0:
            for p in s.predicted:
                s.predictor.appendHistory(
                    p["forecast"],
                    p["timestamp"],
                    p["forecast"],
                    False
                )

        s.predicted = s.predictor.forecast(12)
        s.pred_intervals = s.arma_mads_scores.copy()
        s.predict_source = s.arma_source

        if s.predicted is None:
            return None

    # Walk through the forecast list until we find the forecast
    # corresponding to this timestamp.
    while len(s.predicted) > 0:
        p = s.predicted[0]

        if p["timestamp"] < timestamp and p["forecast"] is not None:
            s.predictor.appendHistory(
                p["forecast"],
                p["timestamp"],
                p["forecast"],
                False
            )
            s.predicted = s.predicted[1:]
            continue

        if p["timestamp"] > timestamp:
            break

        # Matching timestamp found
        if p["forecast"] is None:
            s.predictor.appendHistory(value, timestamp, None, False)
            break

        if s.pred_intervals is None or p["index"] >= len(s.pred_intervals):
            break

        s.predicted = s.predicted[1:]

        event = {
            "timestamp": timestamp,
            "observed": value,
            "predicted": int(p["forecast"]),
            "threshold": int(p["forecast"] - s.pred_intervals[p["index"]]),
            "norm_threshold": int(p["forecast"] - (s.pred_intervals[p["index"]] / 4)),
            "alertable": False,
            "baseline": int(s.baseline),
        }

        if event["threshold"] <= 0:
            event["threshold"] = 0

        if event["predicted"] - event["threshold"] < 1.0:
            event["threshold"] = event["predicted"] - 1.0
            event["norm_threshold"] = event["predicted"]

        if value < event["threshold"]:
            event["alertable"] = True
            s.predictor.appendHistory(value, timestamp, p["forecast"], True)

        elif value > p["forecast"] + s.pred_intervals[p["index"]]:
            event["alertable"] = False
            s.predictor.appendHistory(value, timestamp, p["forecast"], True)

        else:
            event["alertable"] = False
            s.predictor.appendHistory(value, timestamp, p["forecast"], False)

        break

    return event

# ==========================================================
# FETCH HISTORICAL DATA AND PRIME MODEL
# - Load prior observations needed before forecasting
# ==========================================================
# Collect initial history to prime the prediction engine
# fetchIodaHistoricBlocking() fetches time series data from a certain history_duration 
# before an args.time (i.e., 10 weeks before Mon Jan 12 2026)
jsondata, meta = fetchIodaHistoricBlocking(IODA_API, args.fqid, args.time, history_duration)
#print(jsondata, meta) 

# Initialize model state using historical data (required before forecasting)
offline_process_history_data(s, jsondata, bd, sd)
# Debugging tests
#print("pred_intervals:", len(s.pred_intervals)) # should be 12
#for item in s.predicted: # should generate 12 forward predictions (ensures args.time is properly set for prediction interval alignment)
    #print(item)

# Grab the data you want to test the prediction engine against
# This data will reflect the observed time series data at a certain args.duration 
# after a starting args.time (i.e., 1 day after Mon Jan 12 2026)
preddata, meta = fetchIodaHistoricBlocking(IODA_API, args.fqid, args.time + args.duration, args.duration)
#print(preddata)

# ==========================================================
# RUN OFFLINE FORECAST SIMULATION
# - Process future observations sequentially as live data
# ==========================================================
results = []
timestamp = preddata["from"]
step = preddata["step"]
for val in preddata["values"]:
    if val is None:
        timestamp += step
        continue

    event = offline_process_live_data(s, timestamp, val)

    if event is not None:
        results.append(event)
    
    timestamp += step

# Store results as a dataframe to conveniently review testing results
results_df = pd.DataFrame(results).reset_index(drop = True)
results_df["model_relative_error"] = abs((results_df["observed"] - results_df["predicted"] ) / results_df["observed"])
#print(results_df)

# ==========================================================
# COMPARE AGAINST LIVE IODA API OUTPUT
# - Evaluate offline predictions vs gtr-sarima endpoint
# ==========================================================
# Create API request given the parameters for offline testing to compare IODA predictions to offline predictions
url_API = f"https://api.ioda.inetintel.cc.gatech.edu/v2/signals/raw/country/{code}?from={args.time}&until={args.time + args.duration}&datasource=gtr-sarima&sourceParams={desired_service}&maxPoints=3000"
response = requests.get(url_API)
data = response.json()
sarima_data = data["data"][0][0]["values"]
#print(sarima_data)

import_data = []
for event in sarima_data:
    observed_val = event[0]["agg_values"]["observed"]
    predicted_val = event[0]["agg_values"]["predicted"]
    threshold_val = event[0]["agg_values"]["threshold"]
    import_data.append({"api_observed": observed_val, "api_predicted": predicted_val, "api_threshold": threshold_val})
import_df = pd.DataFrame(import_data).reset_index(drop = True) # store IODA API SARIMA predictions in a dataframe to compare against offline predictions
import_df["api_alertable"] = np.where(import_df["api_observed"] < import_df["api_threshold"], True, False)
#print(import_df)
#print("API alert count:", len(import_df[import_df["api_alertable"] == True]))

# Calculate relative error of offline predicted and threshold values against the API values
combined_df = pd.concat([results_df, import_df], axis = 1)
combined_df["predicted_error"] = np.where(combined_df["api_predicted"] != 0,
    np.round(abs((combined_df["api_predicted"] - combined_df["predicted"]) / combined_df["api_predicted"]), 5), np.nan)
combined_df["threshold_error"] = np.where(combined_df["api_threshold"] != 0,
    np.round(abs((combined_df["api_threshold"] - combined_df["threshold"]) / combined_df["api_threshold"]), 5), np.nan)  

combined_df["threshold_difference"] = combined_df["predicted"] - combined_df["threshold"]
combined_df["api_threshold_difference"] = combined_df["api_predicted"] - combined_df["api_threshold"]

relative_error_df = combined_df[["timestamp", "alertable","api_alertable", "observed", "api_observed", "predicted", "api_predicted", "threshold",   
"api_threshold", "predicted_error", "threshold_error", "threshold_difference", "api_threshold_difference"]]
#print(relative_error_df)

# Testing metrics, set DEBUG_ERROR_METRICS = True
if DEBUG_ERROR_METRICS:
    print("mean predicted relative error:", round(relative_error_df["predicted_error"].mean(), 3))
    print("median predicted relative error:", round(relative_error_df["predicted_error"].median(), 3))
    print("max predicted relative error:", round(max(relative_error_df["predicted_error"]), 3))
    print("mean threshold relative error:", round(relative_error_df["threshold_error"].mean(), 3))
    print("median threshold relative error:", round(relative_error_df["threshold_error"].median(), 3))
    print("max threshold relative error:", round(max(relative_error_df["threshold_error"]), 3))
    print("instances of predicted error > 5%:", len(relative_error_df[relative_error_df["predicted_error"] > 0.05]))
    print("instances of threshold error > 5%:", len(relative_error_df[relative_error_df["threshold_error"] > 0.05]))

# ==========================================================
# OFFLINE EXPERIMENTS
# ==========================================================

# ----------------------------------------------------------
# Experiment 1: Forecast Horizon Drift
#   - Determine when live intervals diverge from saved model 
#   dump (should expect adjustment after ~60 day period)
#   - Calculate difference between API predicted values & 
#   threshold values, representing the API prediction intervals
#   - Compute metrics such as mean absolute error, mean 
#   relative error, etc. for offline vs. API predictions/thresholds
#
# INSTRUCTIONS:
#   - Starting times for predictions must begin at 00:30, 06:30, 
#   12:30, 18:30 UTC
#   - Select any valid args.time but keep args.duration at 
#   1 day
#   - Report API prediction intervals every 2 weeks. Once 
#   intervals began to shift, report intervals every day or 6 hours
#   to identify when model refresh occurs 
#   - Set DEBUG_INTERVAL_METRICS = True and report relevant metrics
# ----------------------------------------------------------
#print(relative_error_df[relative_error_df["api_predicted"].isna()]) # review whether model was offline during specified duration 

# Testing metrics, set DEBUG_INTERVAL_METRICS = True
if DEBUG_INTERVAL_METRICS:
    print(list(relative_error_df["api_threshold_difference"])[:12]) # results for live prediction intervals from IODA at args.time
    print(args.predintervals) # prediction intervals collected from data dump used for offline testing
    
    print("mean predicted relative error:", round(relative_error_df["predicted_error"].mean(), 3))
    print("median predicted relative error:", round(relative_error_df["predicted_error"].median(), 3))
    print("max predicted relative error:", round(max(relative_error_df["predicted_error"]), 3))
    print("mean threshold relative error:", round(relative_error_df["threshold_error"].mean(), 3))
    print("median threshold relative error:", round(relative_error_df["threshold_error"].median(), 3))
    print("max threshold relative error:", round(max(relative_error_df["threshold_error"]), 3))
    print("instances of predicted error > 5%:", len(relative_error_df[relative_error_df["predicted_error"] > 0.05]))
    print("instances of threshold error > 5%:", len(relative_error_df[relative_error_df["threshold_error"] > 0.05]))

# ----------------------------------------------------------
# Experiment 2: Priming & Seasonal Slot Sensitivity
#   - Evaluate Chocolatine accuracy by comparing predicted
#  values to observations
#
# Priming Experiment
#   - Compare configurations of 6, 8, 10, 12 weeks 
#   of history and 2, 4, 6, 8, 10, 12 weeks bootstrapped
#   - Compute metrics such as mean absolute error, mean 
#   relative error, etc. for predicted vs. observed values
#
# Seasonal Slot Experiment
#   - Compare ideal configuration from Priming Experiment 
#   (often 6 / 4) with slotmod input for the bootstrapHistory(),
#   which tells the model how often time patterns repeat when 
#   grouping old timestamps into seasonal buckets
#   - Compute metrics such as mean absolute error, mean 
#   relative error, etc. for predicted vs. observed values
#
# INSTRUCTIONS:
#   - Starting times for predictions must begin at 00:30, 06:30, 
#   12:30, 18:30 UTC
#   - Test different values for history_duration, bd, and sd
#   under USER CONFIGURATION
#   - Set DEBUG_PRIMING_METRICS = True and report relevant metrics
# ----------------------------------------------------------
test_results_df = results_df.copy()
test_results_df = test_results_df[test_results_df["alertable"] != True].reset_index(drop = True)
#print("alertable events in sample:", len(results_df) - len(test_results_df))

# Testing metrics, set DEBUG_PRIMING_METRICS = True
if DEBUG_PRIMING_METRICS:
    print(f"{desired_country} Test")
    print("mean absolute error:", round((abs(test_results_df["observed"] - test_results_df["predicted"]).mean())))
    print("median absolute error:", round((abs(test_results_df["observed"] - test_results_df["predicted"]).median())))
    print("mean relative error:", round((test_results_df["model_relative_error"]).mean(), 3))
    print("median relative error:", round((test_results_df["model_relative_error"]).median(), 3))
    print("max relative error:", round(max(test_results_df["model_relative_error"]), 3))
    print("proportion of predictions w/ relative error > 10%:", round(len(test_results_df[test_results_df["model_relative_error"] > 0.10])/len(test_results_df), 3))

# ----------------------------------------------------------
# Experiment 3: Model Failure Analysis
#   - Analyze overflow / extreme prediction values
#   - Compute metrics such as mean relative error and
#   median relative error between offline negative predictions
#   and reconstructed API predictions
#
# INSTRUCTIONS:
#   - args.time must be set to date when outage occurred
#   based on IODA signals for desired_country
#   - args.duration should be set depending on the duration of 
#   outage under consideration 
#   - Set DEBUG_FAILURE_METRICS = True and report relevant metrics
# ----------------------------------------------------------
model_failure_df = relative_error_df.copy()
model_failure_df = model_failure_df[model_failure_df["api_predicted"] >= (10**19)].copy() #  filter for events where the live model defaulted it prediction values to ~1.844674e+19

model_failure_df["test_predicted"] = np.array([2**64], dtype=object)[0] + model_failure_df["predicted"].astype(object)
model_failure_df["api_test_predicted"] = -(np.array([2**64], dtype=object)[0] - model_failure_df["api_predicted"].astype(object))
model_failure_df["api_predicted"] = model_failure_df["api_predicted"].apply(lambda x: int(round(x)))
model_failure_df["test_relative_error"] = abs((model_failure_df["api_test_predicted"] - (model_failure_df["predicted"]))/model_failure_df["api_test_predicted"])
model_failure_df = model_failure_df[["timestamp", "predicted", "api_predicted", "test_predicted", "api_test_predicted", "test_relative_error"]]

# Testing metrics, set DEBUG_FAILURE_METRICS = True
if DEBUG_FAILURE_METRICS:
    print(model_failure_df)
    #print(model_failure_df[(model_failure_df["predicted"] < 0) & (model_failure_df["api_predicted"] > (10**19))]) # check to see if negative prediction values correlate with model failures (i.e., prediction values of ~1.844674e+19)
    print("negative offline predictions count:", len(results_df[results_df["predicted"] < 0])) # count of offline timestamps where predictions are negative 
    print("NaN API predictions count:", len(import_df[import_df["api_predicted"].isna()])) # count of timestamps from IODA API where SARIMA was likely offline 
    print("mean relative error:", round(model_failure_df["test_relative_error"].mean(), 3))
    print("median relative error", round(model_failure_df["test_relative_error"].median(), 3))

# ----------------------------------------------------------
# Experiment 4: Threshold Potential Error Cases
# - Identify zero or negative threshold behavior for future research
# ----------------------------------------------------------
potential_errors_overview = []
for idx, event in enumerate(results):
    if event["threshold"] == 0:
        potential_errors_overview.append({"Potential Error": "Zero value threshold", "Event Index": idx})
    if event["norm_threshold"] < 0:
        potential_errors_overview.append({"Potential Error": "Negative normal threshold value", "Event Index": idx})
#print(potential_errors_overview)

# From potential_errors_overview, store error types and the corresponding entry to aid data review process
potential_errors = []
for idx, event in enumerate(results):
    for error in potential_errors_overview:
        error_type = error["Potential Error"]
        if error["Event Index"] == idx:
            potential_errors.append({error_type: event})
potential_errors_df = pd.DataFrame(potential_errors).reset_index(drop = True)

if DEBUG_THRESHOLD_TESTS:
    print(potential_errors_df)
    print("potential error count:", len(potential_errors_df))

# ==========================================================
# EXPORTS
# - Excel exports used to review experiment outputs 
# ==========================================================
#results_df.to_excel(f"{desired_country}_results_{args.time}_{int(args.duration/(24*60*60))}_days.xlsx", index = False)
#import_df.to_excel(f"{desired_country}_API_predictions_{args.time}_{int(args.duration/(24*60*60))}_days.xlsx")
#model_failure_df.to_excel(f"{desired_country}_outage_results_{args.time}_{int(args.duration/(24*60*60))}_days.xlsx", index = False)
