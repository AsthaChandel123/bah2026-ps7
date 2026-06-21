PROBLEM STATEMENT 7

AI-enabled Detection of Exoplanets from Noisy Astronomical Light Curves
Objective
Develop an AI-based data analysis pipeline capable of automatically detecting exoplanet transit signals from noisy astronomical light curve data.

Details
Exoplanet detection through transit photometry requires the identification of extremely small brightness variations in stars. For light curves of astronomical sources present in crowded fields, there can be significant contaminations arising from effects such as stellar blending by foreground or background sources in the aperture, and the intrinsic noise in the data due to the detector's response, to name a few.

Apart from contamination, the brightness variations in the light curves can be due to a transiting planet across the host star's disk, an eclipsing stellar companion in binary star systems, or even starspots. Different phenomena give rise to distinct features in light curves, which, however, become difficult to disentangle while dealing with noisy datasets in crowded fields.

The developed algorithm should, therefore, be able to achieve the following :

Identify datasets with periodic dips in the star's light curve, potentially mimicking astrophysical phenomena.
Develop a classification framework to categorize the light curve dips into transits, eclipses, blends, and other astrophysical categories.
Apply the classifier on the given science datasets and correctly categorize the type of signals present in the data.
Provide signal-to-noise ratio or significance levels of the identified events.
For transit signals, estimate the parameters associated with the phenomenon, such as the transit depth, period, and duration.
Data Required
TESS raw light curves should be downloaded and used from the publicly available repository at https://archive.stsci.edu/tess/tic_ctl.html. It is advised to download a sector's high-cadence data for this work, which would contain around 20-30k light curves of different stars.

A curated dataset will also be provided for different types of classifiers for already known exoplanets, false positives, and eclipsing binaries, etc., which would be useful to train the AI model.

Suggested Tools/Technologies
Publicly available Python tools and libraries can be used. No specialised software is needed to analyse the data.

Expected Outcomes
An AI-driven pipeline that robustly identifies and classifies dips (transit features) present in noisy light curves into potential astrophysical signals
Estimation of the following parameters by light curve fitting on scientific datasets :
Orbital period
Transit Duration
Transit depth
Visualization of the light curve along with the detected and classified astrophysical signal
Provide the confidence level of the detected signal
A report (max 3 pages) is required, including the methodology, assumptions made, tools and libraries used, and how uncertainties are estimated.
Evaluation Criteria
Accuracy of the pipeline/software in event detection and classification
Accuracy of the parameters estimated
Methods/Approach used
Visualization and clarity
