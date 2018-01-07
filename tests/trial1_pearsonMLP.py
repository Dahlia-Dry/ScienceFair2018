import pickle
import numpy as np
from ELCA import transit
import multiprocessing as mp
from itertools import product
from sklearn import preprocessing
from example_data import dataGenerator
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import glob




def make_data(inputData):
    fullData = pd.DataFrame(columns = ['light_curve', 'status'])
    for i in range(0, (int(len(inputData.results)/10))):
        for j in range(1, 3):
            if j == 1:
                placeTrue = pd.DataFrame({'light_curve': inputData.results[i][j], 'status': 0}, columns = ['light_curve', 'status'])
                print(placeTrue[0:])
                fullData.append(placeTrue)
            elif j == 2:
                placeFalse = pd.DataFrame({'light_curve':inputData.results[i][j], 'status': 1}, columns = ['light_curve', 'status'])
                fullData.append(placeFalse)
    return fullData

def make_figs(inputData):
    xlist = np.arange(0, len(inputData.results[0][1]))
    for i in range(0, len(inputData.results)):
        for j in range(1,3):
            if j==1:
                string0 = '0' + 'fig'+ str(i) + '.png'
                plt.plot(inputData.results[i][j])
                plt.savefig(string0)
                plt.close()
            elif j ==2:
                string1 = '1' + 'fig' + str(i) + '.png'
                plt.plot(inputData.results[i][j])
                plt.savefig(string1)
                plt.close()

def crop_figs():
    files = glob.glob("~/ScienceFair2018/lightCurvePlots/trainData/*.png")
    for myFile in files:
        img = cv2.imread(myFile,0)
        print(img)
        (lx,ly) = img.shape
        crop = img[55:lx-53, 80:ly-62]
        cv2.imwrite(myFile,crop)
        print(myFile)



if __name__ == "__main__":
    # Generate time data
    settings = {'ws': 360, 'dt': 2}
    # window size (ws/dt = num pts) (MINUTES)
    # time step (observation cadence) (MINUTES)
    npts = settings['ws'] / settings['dt']
    hw = (0.5 * npts * settings['dt']) / 60. / 24.  # half width in days
    t = np.linspace(1 - hw, 1 + hw, npts)
    dt = t.max() - t.min()

    # default transit parameters
    init = {'rp': 0.1, 'ar': 12.0,  # Rp/Rs, a/Rs
            'per': 2, 'inc': 90,  # Period (days), Inclination
            'u1': 0.5, 'u2': 0,  # limb darkening (linear, quadratic)
            'ecc': 0, 'ome': 0,  # Eccentricity, Arg of periastron
            'a0': 1, 'a1': 0,  # Airmass extinction terms
            'a2': 0, 'tm': 1}  # tm = Mid Transit time (Days)

    # training data
    pgrid = {
        'rp': (np.array([200, 500, 1000, 2500, 5000, 10000]) / 1e6) ** 0.5,  # transit depth (ppm) -> Rp/Rs

        'per': np.linspace(*[2, 4, 5]),
        'inc': np.array([86, 87, 90]),
        'sig_tol': np.linspace(*[1.5, 4.5, 4]),  # generate noise based on X times the tdepth

        # stellar variability systematics
        'phi': np.linspace(*[0, np.pi, 4]),
        'A': np.array([250, 500, 1000, 2000]) / 1e6,
        'w': np.array([6, 12, 24]) / 24.,  # periods in days
        'PA': [-4 * dt, 4 * dt, 100],  # doubles amp, zeros amp between min time and max time, 1000=no amplitude change
        'Pw': [-12 * dt, 4 * dt, 100],  # -12dt=period halfs, 4dt=period doubles, 1000=no change
    }

    pgrid_test = {
        # TEST data
        'rp': (np.array([200, 500, 1000, 2500, 5000, 10000]) / 1e6) ** 0.5,  # transit depth (ppm) -> Rp/Rs
        'per': np.linspace(*[2, 4, 5]),
        'inc': np.array([86, 87, 90]),
        'sig_tol': np.linspace(*[0.25, 3, 12]),  # generate noise based on X times the tdepth

        # stellar variability systematics
        'phi': np.linspace(*[0, np.pi, 4]),
        'A': np.array([250, 500, 1000, 2000]) / 1e6,
        'w': np.array([6, 12, 24]) / 24.,  # periods in days
        'PA': [-4 * dt, 4 * dt, 100],  # doubles amp, zeros amp between min time and max time, 1000=no amplitude change
        'Pw': [-12 * dt, 4 * dt, 100],  # -12dt=period halfs, 4dt=period doubles, 1000=no change
    }

    #olddata = dataGenerator(**{'pgrid': pgrid, 'settings': settings, 'init': init})
    #olddata.generate()
    #make_figs(olddata)
    crop_figs()




