from ELCA import lc_fitter, transit
import numpy as np

if __name__ == "__main__":

    t = np.linspace(0.85,1.05,200)

    init = { 'rp':0.06, 'ar':14.07,       # Rp/Rs, a/Rs
             'per':3.336817, 'inc':88.75, # Period (days), Inclination
             'u1': 0.3, 'u2': 0,          # limb darkening (linear, quadratic)
             'ecc':0, 'ome':0,            # Eccentricity, Arg of periastron
             'a0':1, 'a1':0,              # Airmass extinction terms
             'a2':0, 'tm':0.95 }          # tm = Mid Transit time (Days)

    # only report params with bounds, all others will be fixed to initial value
    mybounds = {
              'rp':[0,1],
              'tm':[min(t),max(t)],
              'a0':[-np.inf,np.inf],
              'a1':[-np.inf,np.inf]
              }


    # GENERATE NOISY DATA
    data = transit(time=t, values=init) + np.random.normal(0, 2e-4, len(t))
    dataerr = np.random.normal(300e-6, 50e-6, len(t))

    myfit = lc_fitter(t,data,
                        dataerr=dataerr,
                        init= init,
                        bounds= mybounds,
                        )

    for k in myfit.data['LS']['freekeys']:
        print( '{}: {:.6f} +- {:.6f}'.format(k,myfit.data['LS']['parameters'][k],myfit.data['LS']['errors'][k]) )

    myfit.plot_results(show=True,phase=True)
    print (data.size)
    # explore the output of the fitting
    print( myfit.data['LS'].keys() )