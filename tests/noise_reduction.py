import numpy as np
import matplotlib.pyplot as plt
import pyflux as pf
import pandas as pd

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

    def durbin_koopman_simulation(self, plot = True):
        model = pf.LLEV(self.full_data)
        x = model.fit()
        if plot is True:
            plt.figure(figsize = (15,5))
            for i in range(10):
                print(model.latent_variables.get_z_values)
                plt.plot(model.index, model.simulation_smoother(
                    model.latent_variables.get_z_values())[0][0:model.index.shape[0]])
            plt.show()

def test():
    nile = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/datasets/Nile.csv')
    nile.index = pd.to_datetime(nile['time'].values, format='%Y')
    plt.figure(figsize=(15, 5))
    plt.plot(nile.index, nile['Nile'])
    plt.ylabel('Discharge Volume')
    plt.title('Nile River Discharge')
    plt.show()
    model = pf.LLEV(data=nile, target='Nile')
    x = model.fit()
    plt.figure(figsize=(15, 5))
    for i in range(10):
        plt.plot(model.index, model.simulation_smoother(
            model.latent_variables.get_z_values())[0][0:model.index.shape[0]])
    plt.show()


test()
loadpath = 'data/rawData/0raw14.txt'
load = np.loadtxt(loadpath)
stats = Stats(load)
stats.durbin_koopman_simulation()