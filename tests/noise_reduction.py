# Author Dahlia Dry
# Last Modified 12/8/2017
# This program contains various methods for reducing noise in light curve data
# These methods are judged qualitatively through observing graphs
# These methods are also judged indirectly by their effect on the overall accuracy of the neural net

import numpy as np
import matplotlib.pyplot as plt
import pyflux as pf
import pandas as pd
from scipy.stats import gaussian_kde
from matplotlib import colors
import matplotlib.cm as cmx


class Stats(object):
    def __init__(self, full_data):
        self.full_data = full_data

    def _get_noise(self):
        model = pf.LocalLevel(self.full_data, family=pf.Normal())
        x = model.fit()
        noise = model.return_noise()
        return noise

    def _get_margin(self):
        model = pf.LocalLevel(self.full_data, family=pf.Normal())
        x = model.fit()
        data = model.return_local_level()
        margin = data['margin']
        return margin

    def _get_ma(self):
        model = pf.LocalLevel(self.full_data, family = pf.Normal())
        x = model.fit()
        data = model.return_local_level()
        ma = data['data']
        return ma

    def durbin_koopman_simulation(self, n = 10, plot = True):
        df = pd.DataFrame(self.full_data)
        model = pf.LLEV(df)
        x = model.fit()
        if plot is True:
            plt.figure(figsize = (15,5))
            for i in range(n):
                print(model.latent_variables.get_z_values)
                plt.plot(model.index, model.simulation_smoother(
                    model.latent_variables.get_z_values())[0][0:model.index.shape[0]])
            plt.show()
        else:
            data = []
            for i in range(10):
                data.append(model.simulation_smoother(
                    model.latent_variables.get_z_values())[0][0:model.index.shape[0]])
            return model.index, data

    def plot_gaussian_density_sim(self, n = 10):
        data = self.durbin_koopman_simulation(n, plot=False)
        x_a = data[0]
        x = []  #np.empty(n*len(x_a))
        for i in range(n):
            x.extend(x_a)
        x = np.asarray(x)
        y = data[1]
        y = np.reshape(y, n*len(y[0]))

        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        fig, ax = plt.subplots()
        norm = colors.Normalize(vmin=z[0], vmax=z[len(z) - 1])
        colormap = plt.get_cmap('jet')
        scalarMap = cmx.ScalarMappable(norm=norm, cmap=colormap)
        print(scalarMap.get_clim)
        color = np.empty((len(z), 4))
        for idx in range(len(z)):
            color[idx] = scalarMap.to_rgba(z[idx])
            ax.scatter(x[idx], y[idx], c=color[idx], s=30, edgecolor='')
        plt.show()

    



def test_colormap():
    NCURVES = 10
    np.random.seed(101)
    curves = [np.random.random(20) for i in range(NCURVES)]
    values = range(NCURVES)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # replace the next line
    # jet = colors.Colormap('jet')
    # with
    jet = cm = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    print scalarMap.get_clim()

    lines = []
    for idx in range(len(curves)):
        line = curves[idx]
        colorVal = scalarMap.to_rgba(values[idx])
        colorText = (
            'color: (%4.2f,%4.2f,%4.2f)' % (colorVal[0], colorVal[1], colorVal[2])
        )
        retLine, = ax.plot(line,
                           color=colorVal,
                           label=colorText)
        lines.append(retLine)
    # added this to get the legend to work
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right')
    ax.grid()
    plt.show()


# test_colormap()
loadpath = 'data/rawData/0raw21.txt'
load = np.loadtxt(loadpath)
stats = Stats(load)
#stats.durbin_koopman_simulation()
stats.plot_gaussian_density_sim()