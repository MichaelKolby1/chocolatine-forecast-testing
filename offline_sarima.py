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

# -----------------------------------------------------------------------------------------------------------
# IMPORTANT!!! - Before running file, ensure you have downloaded the csv dump with 
# model parameters from Shane (arma_model_dump_gtr.csv). Also, ensure you've downloaded 
# the following files from the Chocolatine GitHub repository under the same folder:
#   - libchocolatine.py
#   - arimapredictor.py
#   - asyncfetcher.py
#   - arimabuilder.py
#   - modeller.py
#   - arima.pyx
#
# Additionally, ensure you have the following modules installed by running this code in your command prompt:
#   - pip install pandas numpy regex requests openpyxl scipy statsmodels confluent_kafka pycountry
# -----------------------------------------------------------------------------------------------------------


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
parser.add_argument("-t", "--time", type=int, help="The unix timestamp where you want the predictions to begin")
parser.add_argument("-d", "--duration", type=int, help="The length of time that you want predictions to be generated for, in seconds")
parser.add_argument("-a", "--ar", type=int, help="The AR parameter for the SARIMA model")
parser.add_argument("-m", "--ma", type=int, help="The MA parameter for the SARIMA model")
parser.add_argument("-p", "--predintervals", type=str, help="The prediction intervals from the SARIMA model, expressed in the format {A,B,C,..N}")

args = parser.parse_args()

# Load ARMA model parameters for each country/service
model_dump = pd.read_excel("arma_model_dump_gtr.xlsx").copy()

# Extract prediction intervals from data dump and create a list of lists, 
# where each list contains 12 intervals corresponding to
# the forecast methodology of 12 steps per prediction
pdlist = model_dump["pred_intervals"].fillna("").apply(lambda s: [0.0]*12 if s.strip("{}").strip() == "" 
else [float(x) for x in s.strip("{}").split(",") if x.strip()])

# Create a dictionary with country names mapped to country codes from the pycountry python library
country_codes = {c.name.lower(): c.alpha_2 for c in pycountry.countries}

# Indicate a desired country for testing
# We identify and store the country's matching code from the country_codes dictionary 
test_countries = ["Iran", "Germany", "Nigeria", "South Africa", "France"]
desired_country = test_countries[0]
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

# Uncomment to review the available google service types
#print(len(google_service_types))
#print(google_service_types)

# Indicate a desired service type for testing
desired_service = "WEB_SEARCH"
i = 0
for idx, service in enumerate(google_service_types):
    if desired_service == service:
        i = idx
        break

# Filter to the row that represents the desired country and desired service type
target_row = country_rows.iloc[i]
idx = target_row.name

args.predintervals = pdlist.iloc[idx]
args.ar = int(target_row["ar_param"])
args.ma = int(target_row["ma_param"])
args.fqid = target_row["fqid"]
args.md = target_row["model_type"]

# Adjust this starting time depending on desired starting time for offline predictions 
# Select times at increments of exactly 30 minutes due to behavior of GTR
# Use this link to convert dates into Unix Timestamps: https://www.epochconverter.com/
# Only select times which lie on these intervals 
#   - 00:30
#   - 06:30
#   - 12:30
#   - 18:30
if args.time is None:
    #args.time = 1768631400 # Saturday, January 17, 2026 at 6:30:00 AM GMT+0000
    args.time = 1770877800 # Thursday, February 12, 2026 at 6:30:00 AM GMT+0000

    # Recent outage start times
    #args.time = 1767895200 # Thursday, January 8, 2026 at 6:00:00 PM GMT+0000
    #args.time = 1772263800 # Saturday, February 28, 2026 at 7:30:00 AM GMT+0000

    # Forecast Horizon Experiment (should expect prediction intervals to adjust after ~60 day period)
    # First observable prediction intervals different than model dump populate on 1762907400
    # Note: sarima may have been offline prior to 1762907400, no API requests can be made for
    # period of at least 10 days prior to this date (recommend further tests to verify)
    #args.time = 1762907400 # Wednesday, November 12, 2025 at 12:30:00 AM GMT+0000
    #args.time = 1763166600 # Saturday, November 15, 2025 at 12:30:00 AM GMT+0000
    #args.time = 1764549000 # Monday, December 1, 2025 at 12:30:00 AM GMT+0000
    #args.time = 1765758600 # Monday, December 15, 2025 at 12:30:00 AM GMT+0000
    #args.time = 1767227400 # Thursday, January 1, 2026 at 12:30:00 AM GMT+0000
    #args.time = 1768437000 # Thursday, January 15, 2026 at 12:30:00 AM

    #args.time = 1768113000 # Sunday, January 11, 2026 at 6:30:00 AM GMT+0000
    #args.time = 1768177800 # Monday, January 12, 2026 at 12:30:00 AM GMT+0000
    #args.time = 1768264200 # Tuesday, January 13, 2026 at 12:30:00 AM GMT+0000
    #args.time = 1768350600 # Wednesday, January 14, 2026 at 12:30:00 AM GMT+0000
    #args.time = 1768372200 # Wednesday, January 14, 2026 at 6:30:00 AM GMT+0000
    #args.time = 1768393800 # Wednesday, January 14, 2026 at 12:30:00 PM GMT+0000
    #args.time = 1768415400 # Wednesday, January 14, 2026 at 6:30:00 PM GMT+0000
    #args.time = 1768523400 # Thursday, January 15, 2026 at 12:30:00 AM GMT+0000
    #args.time = 1768458600 # Thursday, January 15, 2026 at 6:30:00 AM GMT+0000
    #args.time = 1768523400 # Friday, January 16, 2026 at 12:30:00 AM GMT+0000
    #args.time = 1768545000 # Friday, January 16, 2026 at 6:30:00 AM GMT+0000
    #args.time = 1768588200 # Friday, January 16, 2026 at 6:30:00 PM GMT+0000

    # 1st instances when model prediction intervals are near identical to data dump 
    #args.time = 1768609800 # Saturday, January 17, 2026 at 12:30:00 AM GMT+0000
    #args.time = 1768869000 # Tuesday, January 20, 2026 at 12:30:00 AM GMT+0000

    # 1st instance when model prediction intervals exactly match data dump
    #args.time = 1770877800 # Thursday, February 12, 2026 at 6:30:00 AM GMT+0000

# Adjust this duration depending on desired time frame for analysis
# test_durations is a list containing the total seconds in 1, 3, 5, 6, and 7+ days 
test_durations = [6*60*60, 1*24*60*60, 3*24*60*60, 5*24*60*60, 6*24*60*60, 7*24*60*60, 8*24*60*60, 9*24*60*60, 10*24*60*60, 25*24*60*60, 30*24*60*60, 60*24*60*60]
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

# Arguments are set to run the assignDatabaseModel() method on a ChocGtrTimeSeries object
dbmodel = {
    'ar_param': args.ar,
    'ma_param': args.ma,
    'model_type': args.md,
    'generated_at': 0,
    'pred_intervals': args.predintervals
    }

# Example FQID for a GTR signal
# google_tr.AF.BI.GMAIL.traffic
# format is datasource.continent.countrycode.service.metric

# Identify the type of time series used for offline testing 
# For these tests, we will only work with GTR time series
seriestype = args.fqid.split('.')[0]
if seriestype == "google_tr":
    s = choc.ChocGtrTimeSeries(args.fqid)
else:
    print("Unsupported series type '%s'" % (seriestype))
    sys.exit(1)

s.assignDatabaseModel(dbmodel)

# Adapted the processHistoryData() method from libchocolatine.py since we cannot create 
# a ChocolatineDetector object in our offline testing environment
# See "March 12th - Directed Group Work" on Notion for detailed explanations about Chocolatine source code
def offline_process_history_data(s, hist):
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

        bootstrap_duration = [2, 4, 6, 8, 10, 12] # original value is 4
        slotmod_duration = [1, 2, 4] # original value is 2
        s.predictor = ArimaPredictor(s.arma, s.datafreq)
        s.predictor.bootstrapHistory(
            s.history[-bootstrap_duration[1] * s.ppw:],
            medians,
            60 * 60 * 24 * 7 * slotmod_duration[1]
        )

        #print("datafreq:", s.datafreq) # should be 1800, matching the step size of GTR
        #print("ppw:", s.ppw) # assigned value for weekly seasonality which is calculated using s.datafreq

        s.predicted = s.predictor.forecast(12)
        s.pred_intervals = s.arma_mads_scores.copy()
        s.predict_source = s.arma_source

    s.baseline = max(1, statistics.median(s.smallesthist))

# Adapted the processLiveData() method from libchocolatine.py since we cannot create 
# a ChocolatineDetector object in our offline testing environment
# See "March 12th - Directed Group Work" on Notion for detailed explanations about Chocolatine source code
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

# Collect initial history to prime the prediction engine
# fetchIodaHistoricBlocking() fetches time series data from a certain history_duration 
# before an args.time (i.e., 10 weeks before Mon Jan 12 2026)
test_num_weeks = [6, 8, 10, 12]
history_duration = test_num_weeks[2]*7*24*60*60
jsondata, meta = fetchIodaHistoricBlocking(IODA_API, args.fqid, args.time, history_duration)
#print(jsondata, meta) 

# Initialize model state using historical data (required before forecasting)
offline_process_history_data(s, jsondata)
# Debugging tests
#print("history length:", len(s.history)) # should equal 3360
#print("smallesthist length:", len(s.smallesthist)) # should be <= 40
#print("baseline:", s.baseline) # should be a positive number 
#print("pred_intervals:", len(s.pred_intervals)) # should be 12
#for item in s.predicted: # should generate 12 forward predictions (ensures args.time is properly set for prediction interval alignment)
    #print(item)


# Grab the data you want to test the prediction engine against
# This data will reflect the observed time series data from a certain args.duration passed an args.time,
# where args.duration is the time frame used for testing
preddata, meta = fetchIodaHistoricBlocking(IODA_API, args.fqid, args.time + args.duration, args.duration)
#print(preddata)

# Process each data point in turn as though they were streamed live
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

# Priming Experiment / Seasonal Slot Experiment tests
"""
test_results_df = results_df.copy()
test_results_df = test_results_df[test_results_df["alertable"] != True].reset_index(drop = True)

#print("alertable events in sample:", len(results_df) - len(test_results_df))
print(f"{desired_country} Test")
print("mean absolute error:", round((abs(test_results_df["observed"] - test_results_df["predicted"]).mean())))
print("median absolute error:", round((abs(test_results_df["observed"] - test_results_df["predicted"]).median())))
print("mean relative error:", round((test_results_df["model_relative_error"]).mean(), 3))
print("median relative error:", round((test_results_df["model_relative_error"]).median(), 3))
print("max relative error:", round(max(test_results_df["model_relative_error"]), 3))
print("proportion of predictions w/ relative error > 10%:", round(len(test_results_df[test_results_df["model_relative_error"] > 0.10])/len(test_results_df), 3))
"""

# Create API request given the parameters for offline testing to compare IODA predictions to offline predictions
url_API = f"https://api.ioda.inetintel.cc.gatech.edu/v2/signals/raw/country/{code}?from={args.time}&until={args.time + args.duration}&datasource=gtr-sarima&sourceParams={desired_service}&maxPoints=3000"
response = requests.get(url_API)
data = response.json()
#print(data)
sarima_data = data["data"][0][0]["values"]
#print("imported data length:", len(sarima_data))
#print(sarima_data)

import_data = []
for event in sarima_data:
    observed_val = event[0]["agg_values"]["observed"]
    predicted_val = event[0]["agg_values"]["predicted"]
    threshold_val = event[0]["agg_values"]["threshold"]
    import_data.append({"api_observed": observed_val, "api_predicted": predicted_val, "api_threshold": threshold_val})
import_df = pd.DataFrame(import_data).reset_index(drop = True)
import_df["api_alertable"] = np.where(import_df["api_observed"] < import_df["api_threshold"], True, False)
#print(import_df)
#import_df.to_excel("sample_API_predictions_2.xlsx")

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
#print("mean predicted relative error:", round(relative_error_df["predicted_error"].mean(), 3))
#print("median predicted relative error:", round(relative_error_df["predicted_error"].median(), 3))
#print("max predicted relative error:", round(max(relative_error_df["predicted_error"]), 3))
#print("mean threshold relative error:", round(relative_error_df["threshold_error"].mean(), 3))
#print("median threshold relative error:", round(relative_error_df["threshold_error"].median(), 3))
#print("max threshold relative error:", round(max(relative_error_df["threshold_error"]), 3))
#print("instances of predicted error > 5%:", len(relative_error_df[relative_error_df["predicted_error"] > 0.05]))
#print("instances of threshold error > 5%:", len(relative_error_df[relative_error_df["threshold_error"] > 0.05]))


# Model Failure Analysis (1 week post-outages)
test_relative_error_df = relative_error_df.copy()
test_relative_error_df = test_relative_error_df[test_relative_error_df["api_predicted"] >= (10**19)].copy() #  filter for events where the live model defaulted it prediction values to ~1.844674e+19

test_relative_error_df["test_predicted"] = np.array([2**64], dtype=object)[0] + test_relative_error_df["predicted"].astype(object)
test_relative_error_df["api_test_predicted"] = -(np.array([2**64], dtype=object)[0] - test_relative_error_df["api_predicted"].astype(object))
test_relative_error_df["api_predicted"] = test_relative_error_df["api_predicted"].apply(lambda x: int(round(x)))
test_relative_error_df["test_relative_error"] = abs((test_relative_error_df["api_test_predicted"] - (test_relative_error_df["predicted"]))/test_relative_error_df["api_test_predicted"])
test_relative_error_df = test_relative_error_df[["timestamp", "predicted", "api_predicted", "test_predicted", "api_test_predicted", "test_relative_error"]]
#print(test_relative_error_df)
#print(test_relative_error_df[(test_relative_error_df["predicted"] < 0) & (test_relative_error_df["api_predicted"] > (10**19))]) # check to see if negative prediction values correlate with model failures (i.e., prediction values of ~1.844674e+19)
#print("negative offline predictions count:", len(results_df[results_df["predicted"] < 0]))
#print("NaN API predictions count:", len(import_df[import_df["api_predicted"].isna()]))
#test_relative_error_df.to_excel(f"{desired_country}_outage_sample_results.xlsx", index = False)


# Forecast Horizon Experiment tests
#print(list(relative_error_df["api_threshold_difference"])[:12]) # results for live prediction intervals from IODA at args.time
#print(args.predintervals) # prediction intervals collected from data dump used for offline testing

#print(relative_error_df[relative_error_df["api_predicted"].isna()]) # review whether model was offline during specified duration 
#print((1768609800 - 1762907400)/(60*60*24)) # 66, which is the number of days between prediction interval generation times detected by Iran tests


# Flag events for errors that may relate to model failures
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
#print(potential_errors_df)
#print("potential error count:", len(potential_errors_df))


# Export sample results for review
#results_df.to_excel(f"{desired_country}_sample_results_{int(args.duration/(24*60*60))}_days.xlsx", index = False)

