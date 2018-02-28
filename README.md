# Challenging Limitations: Using Deep Learning, Time Series Analysis, and Statistical Methods for Noise Reduction to Develop an Innovative Approach to Exoplanet Candidate Detection Using Earth-Based Telescopes
Dahlia Dry, 2018

## Abstract
The current limitations of our efforts toward exoplanet discovery can be attributed largely to the immense cost of the
observational tools needed to look for signs of exoplanetary transits of target stars. Seeking to increase the accessibility
of astronomy in a way that would heighten the rate at which new exoplanet candidates are identified, I attempted to
engineer a comprehensive system for the observation and identification of possible new exoplanet candidates in areas
not currently being explored by space telescopes. I proposed that this can be done by making observations using an
existing worldwide network of earth-based telescopes rather than the few space telescopes currently available to us. To
address the issues of atmospheric noise and periodic gaps in observation windows that arise when working with data
from earth-based telescopes, a three-tiered approach was adopted to develop the analytical software. First, a
convolutional neural net (CNN) was created to differentiate between transit and non-transit light curves generated using a
Gaussian distribution of stellar parameters obtained from analysis of Kepler light curves translated into SIFT image
descriptors. Then, 12-hour segments were deleted from the data to simulate daytime periods in which an Earth-based
telescope cannot make observations. The gaps were then filled in using Gaussian Local Level and ARIMA models. Noise
was artificially added into the Kepler observations to simulate observing conditions on earth, and software was developed
to remove that noise through a process involving Gaussian Local Level Models and Durbin-Koopman simulations. The
result of this process was software which classified transit light curves transformed under simulated earth-based
observation conditions with 88% accuracy.

