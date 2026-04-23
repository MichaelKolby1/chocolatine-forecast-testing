# This source code is Copyright (c) 2023 Georgia Tech Research Corporation. All
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

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller  # Check stationarity
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.tools.sm_exceptions
import warnings

# ARIMA error
from numpy.linalg import LinAlgError

# Error estimation
from sklearn.metrics import mean_squared_error
from math import sqrt

import sys
from math import ceil, floor

class ArimaPredictor(object):
    def __init__(self, armaparams, datafreq):
        self.datafreq = datafreq
        self.arma = (armaparams[0], 0, armaparams[1])
        self.ppw = int((7 * 24 * 60 * 60) / self.datafreq)

        self.history = None
        self.diff_history = None

    def updateArmaParams(self, armaparams):
        self.arma = (armaparams[0], 0, armaparams[1])

    def test_signal(self, df):
        dftest = adfuller(df)
        dfout = pd.Series(dftest[0:4],
                    index=['TestStatistic', 'p-value', '# Lags', 'Observations Used'])

        for k,v in dftest[4].items():
            dfout['Critical Value (%s)' % k] = v

        if dfout['TestStatistic'] < dfout['Critical Value (1%)']:
            res = 1
        elif dfout['TestStatistic'] < dfout['Critical Value (5%)']:
            res = 5
        elif dfout['TestStatistic'] < dfout['Critical Value (10%)']:
            res = 10
        else:
            res = 0

        if not res:
            print("Time series %s is not stationary!" % (df.name))
        else:
            print("Time series %s has a %u%% chance to be stationary" % \
                    (df.name, 100 - res))


    def bootstrapHistory(self, hist, medians, slotmod):

        # prune None values from start and end of history
        while len(hist) > 0:
            h = hist[0]
            if h['signalValue'] != None:
                break
            hist = hist[1:]

        while len(hist) > 0:
            h = hist[-1]
            if h['signalValue'] != None:
                break
            hist = hist[:-1]

        self.history = hist
        self.diff_history = []

        # replace any "None" history values with a pre-calculated
        # default (i.e. the median of all observed values for that
        # particular time of the week)
        for i in range(0, len(self.history)):
            slot = self.history[i]['timestamp'].timestamp() % slotmod

            if slot not in medians:
                continue
            self.history[i]['signalValue'] = medians[slot]

            if self.history[i]['signalValue'] is None or \
                    self.history[i - self.ppw]['signalValue'] is None:

                self.diff_history.append({
                    "timestamp": self.history[i]['timestamp'],
                    "signalDiff": 0
                })
            else:
                self.diff_history.append({
                    "timestamp": self.history[i]['timestamp'],
                    "signalDiff": self.history[i]['signalValue'] - self.history[i - self.ppw]['signalValue']
                })

                

        #try:
        #    self.test_signal(self.history)
        #    self.test_signal(self.diff_history)
        #except LinAlgError:
        #    raise

        ind = -(3 * self.ppw)
        # We should only need to keep the last two weeks of historic data
        self.history = self.history[ind:]
        self.diff_history = self.diff_history[ind:]

    def appendHistory(self, value, timestamp, forecast, useForecastDiff):
        pdts = pd.Timestamp(timestamp, unit='s')
        origval = value

        if self.history[-self.ppw]['signalValue'] is None:
            diff = 0
            value = forecast        # just use the forecast as substitute
        elif useForecastDiff:
            # save a "history" value that is the forecast but shifted
            # slightly towards the observed value -- this will allow us to
            # eventually adjust in the event of a permanent change in the
            # level of the signal, without letting outages or outliers have
            # too much of an effect on future predictions
            if (forecast > value):
                value = forecast - min(0.2 * (forecast - value), 0.2 * forecast)
            elif forecast >= 0:
                value = forecast + min(0.2 * (value - forecast), 0.2 * forecast)
            else:
                # Observed value is potentially anomalous.
                # Forecast is negative, so is also dodgy.
                # Our best bet is probably just to pretend we saw the same
                # value as last time and hope things come right in the long
                # run?
                value = self.history[-self.ppw]['signalValue']

            # We're supposed to be using our forecast, but if the forecast is
            # further away from the previous value than our observed current
            # value, it makes more sense to trust the observation than the
            # forecast. This prevents the future forecasts from
            # overcompensating when they have to adjust to a major change in
            # the series.
            fc_diff = value - self.history[-self.ppw]['signalValue']
            act_diff = origval - self.history[-self.ppw]['signalValue']

            if abs(fc_diff) < abs(act_diff):
                diff = fc_diff
            else:
                diff = act_diff
        else:
            diff = value - self.history[-self.ppw]['signalValue']

        self.diff_history.append({'timestamp': pdts, 'signalDiff': diff})

        self.history.append({'timestamp': pdts, 'signalValue': value})
        self.history = self.history[-(3 * self.ppw):]
        self.diff_history = self.diff_history[-(3 * self.ppw):]

    def forecast(self, maxahead=12):
        ts = self.diff_history[-1]['timestamp'].timestamp()
        endts = ts + (maxahead * self.datafreq)
        endts -= (endts % (maxahead * self.datafreq))

        n = int((endts - ts) / self.datafreq)

        if n <= 0:
            return None

        #for i in range(-20, 0):
        #    print(self.history[i]['timestamp'], self.history[i]['signalValue'],
        #            self.diff_history[i]['signalDiff'],
        #            self.history[i - self.ppw]['signalValue'],
        #            self.history[i - self.ppw]['timestamp'])

        df = pd.DataFrame.from_records(self.diff_history, index="timestamp")

        order = self.arma
        model = ARIMA(df.values, order=order, trend='t')

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                fitted = model.fit()
            except ValueError:
                return None

        predictions = []

        forecast = fitted.get_forecast(steps=n)
        p_ts = ts + self.datafreq

        for i in range(n):
            p = forecast.predicted_mean[i]
            if self.history[-self.ppw + 1] is None:
                inv = None
            else:
                inv = p + self.history[-self.ppw + i]['signalValue']
            sd = forecast.se_mean[i]

            predictions.append({
                "timestamp" : p_ts,
                "diff": p, "forecast": inv, "sdev": sd, "index": i,
            })

            p_ts += self.datafreq

        return predictions
