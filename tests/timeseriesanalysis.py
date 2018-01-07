import pyflux as pf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import scipy.stats as stat


class TimeSeries(object):
    def __init__(self, full_data, ar, ma, integ, end=False):
        self.full_data = full_data
        self.ar = ar
        self.ma = ma
        self.integ = integ
        self.end = end #if end is False, zero is the isolated point

    def _get_gap_indices(self):
        lowval = sys.maxsize
        highval = -1e99
        for i in range(0, len(self.full_data)):
            if self.full_data[i] > highval:
                highval = self.full_data[i]
            if self.full_data[i] < lowval:
                lowval = self.full_data[i]

        gap_indices = []
        for j in range(0, len(self.full_data)):
            if self.full_data[j] == lowval:
                if ((j == 0) or (j == len(self.full_data) - 1)) and (self.full_data[j] == lowval):
                    gap_indices.append(j)
                elif (self.full_data[j - 1] != lowval) or (self.full_data[j + 1] != lowval):  # just get gap start and end points
                    gap_indices.append(j)

        gap_ranges = np.empty((int((len(gap_indices) / 2)), 2))
        for k in range(0, len(gap_ranges)):
            l = k * 2
            gap_ranges[k][0] = gap_indices[l]
            gap_ranges[k][1] = gap_indices[l + 1]

        return gap_ranges  # inclusive

    def _get_gap_number(self):
        gaps = self._get_gap_indices()
        return len(gaps)

    def _generate_data_segments(self): #type (string) = dataframe or array
        gaps = self._get_gap_indices()
        data_segments = []

        if self.end is False:
            data_segments.append(('0', [self.full_data[0]]))
        else:
            key = '0' + '-' + str(gaps[0][1])
            data_segments.append((key, []))
            for i in range(0, gaps[0][1]):
                data_segments[0][1].append(self.full_data[i])

        for i in range(1, len(gaps)):
            key = str(int(gaps[i-1][1]+1)) + '-' + str(int(gaps[i][0]))
            data_segments.append((key, []))
            for j in range(int(gaps[i-1][1]+1), int(gaps[i][0])):
                data_segments[i][1].append(self.full_data[j])

        if self.end is False:
            key = str(int(gaps[len(gaps)-1][1])) + '-' + str(len(self.full_data))
            data_segments.append((key, []))
            for i in range(int(gaps[len(gaps)-1][1]+1), len(self.full_data)):
                data_segments[len(data_segments)-1][1].append(self.full_data[i])
        else:
            data_segments.append((str(len(self.full_data)), [self.full_data[len(self.full_data)-1]]))

        return data_segments

    def _latent_variable_distribution(self, set_a, set_b):
        a = np.asarray(set_a)
        b = np.asarray(set_b)
        a_model = pf.ARIMA(a, self.ar, self.ma, self.integ)
        b_model = pf.ARIMA(b, self.ar, self.ma, self.integ)
        a_modelfit = a_model.fit()
        b_modelfit = b_model.fit()

        a_vars = a_model.latent_variables
        a_latent_vars = a_vars.dahlia()
        a_vals = a_latent_vars[0]
        a_indicators = a_latent_vars[1]
        a_factors = a_latent_vars[2]
        a_means = np.empty(len(a_vals))
        a_sdevs = np.empty(len(a_vals))

        b_vars = b_model.latent_variables
        b_latent_vars = b_vars.dahlia()
        b_vals = b_latent_vars[0]
        b_indicators = b_latent_vars[1]
        b_factors = b_latent_vars[2]
        b_means = np.empty(len(b_vals))
        b_sdevs = np.empty(len(b_vals))

        for i in range(len(a_means)):
            a_means[i] = a_vals[i].mean()
            a_sdevs[i] = a_vals[i].std()
            b_means[i] = b_vals[i].mean()
            b_sdevs[i] = b_vals[i].std()

        values = np.empty((len(a_vals), len(a_vals[0])))
        for y in range(0, len(a_vals)):
            #datasets with the 'b' indicator are scaled differently, use z-scores to average equivalent values in a and b distribution
            if ( b_indicators== 'b' and a_indicators[y] != 'b') or (a_indicators[y] == b_indicators[y] != 'b'):
                for z in range(0, len(a_vals[0])):
                    pA = stat.norm(a_means[y],a_sdevs[y]).cdf(a_vals[y][z])
                    bVal = stat.norm(b_means[y], b_sdevs[y]).ppf(pA)
                    avg = (a_vals[y][z] + bVal) / 2
                    values[y][z] = avg
            elif a_indicators[y] != 'b' and b_indicators[y] != 'b':
                for z in range(0, len(a_vals[0])):
                    pB = stat.norm(b_means[y],b_sdevs[y]).cdf(b_vals[y][z])
                    aVal = stat.norm(a_means[y], a_sdevs[y]).ppf(pB)
                    avg = (b_vals[y][z] + aVal) / 2
                    values[y][z] = avg
            else:
                for z in range(0, len(a_vals[0])):
                    pA = stat.norm(a_means[y], a_sdevs[y]).cdf(a_vals[y][z])
                    bVal = stat.norm(b_means[y], b_sdevs[y]).ppf(pA)
                    avg = (a_vals[y][z] + bVal) / 2
                    values[y][z] = avg

        return values

    def _summarize_latent_variables(self, vals):
        m = np.empty(len(vals))
        s = np.empty(len(vals))
        for a in range(0, len(vals)):
            m[a] = vals[a].mean()
            s[a] = vals[a].std()
        lv = np.empty((len(m), 2))
        for a in range(0, len(m)):
            lv[a][0] = m[a]
            lv[a][1] = s[a]
        return lv

    def _linear_regression(self, set_a, set_b, ax, bx): #ax= 120, bx = 180
        set_a = np.asarray(set_a)
        set_b = np.asarray(set_b)
        ay = set_a[set_a.size-1]
        by = set_b[0]
        m = (by-ay)/(bx-ax)
        b = (-m*ax) + (ay)
        linregvals = np.empty((bx-ax))
        for i in range(0, (bx-ax)):
            linregvals[i] = (m*(i+ax)) + b
        return linregvals

    def _noise_generator(self, set_a, set_b):
        set_a = np.asarray(set_a)
        set_b = np.asarray(set_b)
        model_a = pf.LocalLevel(set_a, family = pf.Normal())
        model_b = pf.LocalLevel(set_b, family = pf.Normal())
        model_a_fit = model_a.fit()
        model_b_fit = model_b.fit()
        noise_a = model_a.return_noise()
        noise_b = model_b.return_noise()
        return noise_a, noise_b

    def _single_noise_generator(self, set):
        set = np.asarray(set)
        model = pf.LocalLevel(set, family = pf.Normal())
        model_fit = model.fit()
        noise = model.return_noise()
        return noise

    def _single_randomization(self, linregvals, noise):
        randomized_series = np.empty(len(linregvals))
        for i in range(0, len(linregvals)):
            mean_noise = noise.mean()
            sdev_noise = noise.std()
            noise_value = np.random.normal(mean_noise, sdev_noise, 1)
            randomized_series[i] = linregvals[i] + noise_value
        return randomized_series

    def _randomization(self, linregvals, noise_a, noise_b):
        randomized_series = np.empty(len(linregvals))
        for i in range(0, len(linregvals)):
            j = i+1
            ratio = j / 60 #noise_b/noise_a
            b_factor = int(ratio * len(noise_b))
            a_factor = int((1-ratio)*len(noise_a))
            sample_a = np.random.choice(noise_a, a_factor)
            sample_b = np.random.choice(noise_b, b_factor)
            length = len(sample_a) + len(sample_b)
            noise = np.empty(length)
            for j in range(0, length):
                if j < len(sample_a):
                    noise[j] = sample_a[j]
                else:
                    noise[j] = sample_b[j-len(sample_a)]
            mean_noise = noise.mean()
            sdev_noise = noise.std()
            noise_value = np.random.normal(mean_noise, sdev_noise, 1)
            randomized_series[i] = linregvals[i] + noise_value
        return randomized_series

    def create(self):
        data_segments = self._generate_data_segments()
        gaps = self._get_gap_number()
        gapsets = gaps - 1
        latent_variables = []
        linregvals = []
        noise = []
        randomized_series = []
        if self.end is False:
            x1 = data_segments[0][0]
            x2 = data_segments[1][0].split('-')
            linregvals0 = self._linear_regression(data_segments[0][1], data_segments[1][1], int(x1), int(x2[0]))
            noise0 = self._single_noise_generator(data_segments[1][1])
            randomized_series.append(self._single_randomization(linregvals0, noise0))
            for i in range(0, gapsets):
                raw_latent_variables = self._latent_variable_distribution(data_segments[i+1][1], data_segments[i+2][1])
                latent_variables.append(self._summarize_latent_variables(raw_latent_variables))
                x1 = data_segments[i + 1][0].split('-')
                x2 = data_segments[i + 2][0].split('-')
                linregvals.append(self._linear_regression(data_segments[i + 1][1], data_segments[i + 2][1],
                                                          int(x1[1]), int(x2[0])))
                noise.append((self._noise_generator(data_segments[i + 1][1], data_segments[i + 2][1])))
                randomized_series.append(self._randomization(linregvals[i], noise[i][0], noise[i][1]))

        else:
            for i in range(0, gapsets):
                raw_latent_variables = self._latent_variable_distribution(data_segments[i][1], data_segments[i+1][1])
                latent_variables.append(self._summarize_latent_variables(raw_latent_variables))
                x1 = data_segments[i][0].split('-')
                x2 = data_segments[i + 1][0].split('-')
                linregvals.append(self._linear_regression(data_segments[i][1], data_segments[i + 1][1],
                                                          int(x1[1]), int(x2[0])))
                noise.append((self._noise_generator(data_segments[i][1], data_segments[i + 1][1])))
                randomized_series.append(self._randomization(linregvals[i], noise[i][0], noise[i][1]))
            x1 = data_segments[len(data_segments)-2][0].split('-')
            x2 = data_segments[len(data_segments)-1][0]
            linregvals0 = self._linear_regression(data_segments[len(data_segments)-2][1],
                                                  data_segments[len(data_segments)-1][1], int(x1[1]), int(x2))
            noise0 = self._single_noise_generator(data_segments[len(data_segments)-2][1])
            randomized_series.append(self._single_randomization(linregvals0, noise0))

        return randomized_series

    def plot(self, series, lv=True):
        a_array, b_array, c_array = self._generate_data_segments()
        raw_latent_variables = self._latent_variable_distribution(a_array, b_array)
        latent_variables = self._summarize_latent_variables(raw_latent_variables)
        if lv is True:
            model = pf.ARIMA(series, self.ar, self.ma, self.integ, latent_variables)
        else:
            model = pf.ARIMA(series, self.ar, self.ma, self.integ)
        model_fit = model.fit()
        model.plot_fit()

    def plot_z(self, latent_variables):
        for i in range(0, len(latent_variables)):
            plt.plot(latent_variables[i])
        plt.show()

    def stitch(self, gapdata, plot=False):
        data_segments = self._generate_data_segments()
        #length = len(a_array) + len(b_array) + len(gapdata) - 2
        data = []
        """for i in range(0, length):
                if i < len(a_array):
                    data[i] = a_array[i]
                elif i >= len(a_array) and i < (len(gapdata) - 1):
                    data[i] = gapdata[i - len(a_array) + 1]
                else:
                    data[i] = b_array[i - len(a_array) - (len(gapdata) - 2)]"""


        for i in range(0, len(data_segments)):
            data.extend(data_segments[i][1])
            if i < len(gapdata):
                data.extend(gapdata[i])

        data = np.asarray(data)

        if plot is False:
            return data
        else:
            plt.plot(data)
            plt.show()



loadpath = 'data/gapData/0gapraw14.txt'
load = np.loadtxt(loadpath)
plt.plot(load)
plt.show()

timeseries = TimeSeries(load, 3,3,1)
gapdata = timeseries.create()
timeseries.stitch(gapdata, plot=True)



#testmodel = pf.ARIMA(linVals, 2,2,1, latentvars)
#testmodel_fit = testmodel.fit()
#testmodel.plot_fit(figsize = (15,10))
