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

## Installation

To intsall this code you will need to clone this repo and install the required packages. Packages can be installed using the environment.yml file provided. To do this run the following commands:

```conda env create -f environment.yml```

```conda activate espectro```

Alternatively, you can install the required packages manually. The required packages are:

numpy, matplotlib, astropy, pandas, lmfit, scipy, mpl_point_clicker

## Usage
#### focus_finder.py
The code is split into two main scripts. focus_finder.py reads the input images from a sweep, allows the user to select lines for analysis, and then performs the analysis using the metrics FWHM and encircled energy. The output of this script is a csv file containing the results of the analysis. The script should be run from the command line using the following command: 

```python focus_finder.py <path> -p <point file> -d <dark> -b <box size> -v <cmap min> <cmap max>```

**path**: User should provide the path to the folder containing the images to be analysed. Uses unix style path with wild cards (e.g. *.fits returns all fits files in a folder)

**dark** (optional): User can provide a dark frame to be subtracted from the images. If no dark is provided then no dark subtraction is performed.

**box size** (optional): User can provide the size of the box used to extract the line features. If no box size is provided then the default value of 30 is used.

**cmap min and cmap max** (optional): User can provide the min and max values for the colour map used to display the images. If no values are provided then the default values of 0 and 1000 are used.

**point file** (optional): User can specify a txt file containing the x and y coordinates of the lines to be analysed. If no file is provided then the user should select lines in gui.

#### focus_plane_fitting.py

This script finds the optimal position of the DAM motors using the results from focus_finder.py. The script should be run from the command line using the following command:

```python focus_plane_fitting.py <output file> --metric <metrics> --weights <weights>```

**output file**: User should provide the path to the csv file containing the results of the analysis from focus_finder.py.

**metric**: User should provide a list of metrics to be used in the fitting. The metrics should be present as columns in the csv file. Currently the code supports the metrics FWHMx, FWHMy, and encircled energy.

**weights**: User should provide a list of weights to be used in the fitting. The weights will be used to create a combined score for each line. The weights should be provided in the same order as the metrics

## Example

The following example uses data from cooldown 3. The data is located in the folder ```copies_test_3A.01.13``` which has been provided. The data is a focus sweep of the MOONS spectrograph. The data is in the form of 2D images. The first step is to run focus_finder.py to analyse the data. The following command should be used:

```python focus_finder.py "data/raw/copies_test_3A.01.13/test_3A.01.13.YJ1.ARC*.fits" -d "data/raw/copies_test_3A.01.13/test_3A.01.13.YJ1.DARK.01.fits" -b 30 -v 0 1000 -p "points.txt"```

Then to find the optimal DAM positions the following command should be used:

```python focus_plane_fitting.py placeholder_test_name_output.csv --metric "FWHMx" "FWHMy" "EE_fixed"  --weights 1 1 1```

This should produce the following output:

```FWHMx: DAM1 = -0.4114456035616124 DAM2 = 0.08651857233725857 DAM3 = 0.22778065884760645```

```FWHMy: DAM1 = -0.3487468602556448 DAM2 = -0.15368252898053875 DAM3 = 0.35259705128570634```

```EE_fixed: DAM1 = -0.46485178410238626 DAM2 = 0.11728636086189323 DAM3 = 0.2550852634384641```

```Score: DAM1 = -0.37894175563177035 DAM2 = 0.047965336433471104 DAM3 = 0.22551447736637167```


