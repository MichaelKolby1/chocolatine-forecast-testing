# This file is part of chocolatine
#
# Copyright (C) 2021 The Regents of the University of California
# All Rights Reserved
#
# Permission to copy, modify, and distribute this software and its
# documentation for academic research and education purposes, without fee, and
# without a written agreement is hereby granted, provided that
# the above copyright notice, this paragraph and the following paragraphs
# appear in all copies.
#
# Permission to make use of this software for other than academic research and
# education purposes may be obtained by contacting:
#
# Office of Innovation and Commercialization
# 9500 Gilman Drive, Mail Code 0910
# University of California
# La Jolla, CA 92093-0910
# (858) 534-5815
# invent@ucsd.edu
#
# This software program and documentation are copyrighted by The Regents of the
# University of California. The software program and documentation are supplied
# "as is", without any accompanying services from The Regents. The Regents does
# not warrant that the operation of the program will be uninterrupted or
# error-free. The end-user understands that the program was developed for
# research purposes and is advised not to rely exclusively on the program for
# any reason.
#
# IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY FOR
# DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING
# LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION,
# EVEN IF THE UNIVERSITY OF CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE. THE UNIVERSITY OF CALIFORNIA SPECIFICALLY DISCLAIMS ANY
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED
# HEREUNDER IS ON AN "AS IS" BASIS, AND THE UNIVERSITY OF CALIFORNIA HAS NO
# OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
# MODIFICATIONS.
#
####
#
# Portions of this source code are Copyright (c) 2023 Georgia Tech Research
# Corporation. All Rights Reserved. Permission to copy, modify, and distribute
# this software and its documentation for academic research and education
# purposes, without fee, and without a written agreement is hereby granted,
# provided that the above copyright notice, this paragraph and the following
# three paragraphs appear in all copies. Permission to make use of this
# software for other than academic research and education purposes may be
# obtained by contacting:
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

import multiprocessing, queue, time, signal
import pandas as pd
from threadpoolctl import threadpool_limits

import pyximport
pyximport.install()
from . import arima

class ChocArimaTrainer(object):
    def __init__(self, trainerid):
        self.trainerid = trainerid
        self.inq = multiprocessing.Queue()
        self.outq = multiprocessing.Queue()
        self.oob = multiprocessing.Queue()
        self.process = None

    def setProcess(self, p):
        self.process = p

    def getProcess(self):
        return self.process

    def addJob(self, job):
        self.inq.put(job)

    def addResult(self, model):
        self.outq.put(model)

    def halt(self):
        self.oob.put(ChocArimaJob(None, None, None, None, 0))

    def getResult(self):
        try:
            res = self.outq.get(False)
        except queue.Empty:
            return None
        return res

    def getNextJob(self):
        try:
            job = self.oob.get(False)
            return job
        except queue.Empty:
            pass

        try:
            job = self.inq.get(False)
        except queue.Empty:
            return None
        return job

class ChocArimaPool(object):
    def __init__(self, workers):
        self.numworkers = workers
        self.workers = []
        self.nextassign = 0

    def startWorkers(self):
        if self.numworkers <= 0:
            return -1

        for i in range(0, self.numworkers):
            self.workers.append(ChocArimaTrainer(i))
            p = multiprocessing.Process(target=runArimaTrainer, daemon=True,
                    args=(self.workers[i],),
                    name="ChocolatineArimaTrainer-%d" % (i))
            self.workers[i].setProcess(p)
            p.start()

        return self.numworkers

    def haltWorkers(self):
        for i in range(0, self.numworkers):
            self.workers[i].halt()
            self.workers[i].getProcess().join()


    def getCompleted(self):
        done = []

        for i in range(0, self.numworkers):
            while True:
                res = self.workers[i].getResult()
                if res is None:
                    break
                done.append(res)

        return done

    def addJob(self, job):
        self.workers[self.nextassign].addJob(job)
        self.nextassign += 1

        if self.nextassign >= self.numworkers:
            self.nextassign = self.nextassign % self.numworkers

class ChocArimaJob(object):
    def __init__(self, fqid, requestor, arma_limit, history, stepsize):
        self.fqid = fqid
        self.requestor = requestor
        self.arma_limit = arma_limit
        self.history = history
        self.stepsize = stepsize


def runArimaTrainer(trainer):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    while True:
        job = trainer.getNextJob()
        if job is None:
            time.sleep(1)
            continue
        if job.fqid is None or job.requestor is None or job.history is None:
            break
        if job.stepsize <= 0:
            break

        difforder = int((24 * 7 * 60 * 60) / job.stepsize)
        if difforder < 1:
            break

        print("Generating ARIMA model for '%s'" % (job.fqid))
        model = arima.Arima(
                [job.history.last_valid_index() - pd.DateOffset(weeks=1),
                        job.history.last_valid_index()],
                12,
                [difforder],
                job.arma_limit,
                False,
                False,
                0,
                50)
        start = time.time()
        with threadpool_limits(limits=4, user_api='blas'):
            try:
                success, restuple = model.prepare_analysis(job.history, difforder, None, False)
            except:
                print(job)
                raise

        if success == 0:
            res = {'fqid': job.fqid, 'arma': None, 'mads': [],
                   'modeltime': 0, 'requestor': job.requestor,
                   'eststart': start}
        else:
            end = time.time()
            res = {'fqid': job.fqid, 'arma': restuple[2], 'mads': restuple[3],
                   'modeltime': end - start, 'requestor': job.requestor,
                   'eststart': start}

        trainer.addResult(res)

