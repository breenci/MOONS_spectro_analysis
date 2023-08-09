# MOONS_spectro_analysis

This repo will provide python scripts for finding the focus of the MOONS spectrograph and analysing spectrograph performance.

It builds on scripts developed for cooldown 3 by William Taylor.

The tools developed here should allow quick and easy analysis of MOONS frames for future cooldowns.

## Goals

* Develop scripts to:
    1. Read a folder of containing images from a focus sweep.
    2. Select line features and analyse PSF using a range of metrics (e.g. FWHM, encircled energy).
    3. Find optimal DAM positions using the above metrics.
* Ensure that all analysis code is fully documented and easy to use. Users should be able to run analysis from the command line without changing the raw code. 
* Code should be fully tested prior to Cooldown 4 using data from previous cooldowns.