import numpy as np
import glob
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
import re
from focus_finder_gui import pointSelectGUI
from matplotlib.gridspec import GridSpec
from astropy.modeling.models import custom_model
from astropy.modeling.fitting import LevMarLSQFitter
import math


def extract_variables_and_export(filenames, pattern, column_names=None):
    # Initialize lists to store extracted variables
    variable_data = []

    # Loop through each filename and extract variables using the pattern
    for filename in filenames:
        match = re.search(pattern, filename)
        if match:
            variables = [match.group(i) for i in range(1, match.lastindex + 1)]
            variable_data.append(variables)

    # Get the number of variables
    num_variables = match.lastindex if match else 0

    # Create a Pandas DataFrame
    if column_names is None:
        column_names = [f"Variable_{i}" for i in range(1, num_variables + 1)]
    else:
        if len(column_names) != num_variables:
            raise ValueError(f"Number of column names ({len(column_names)}) must match the number of variables ({num_variables})")

    df = pd.DataFrame(variable_data, columns=column_names)
    df['filename'] = filenames

    return df


# Write a function which takes a 2D array, an X and Y coordinate, and a box size
# and returns a 2D array of the box centered on the X and Y coordinate. The
# function should also check that the box is within the bounds of the array.
def get_box(array, x, y, box_size=30):
    # Get the shape of the array
    ny, nx = array.shape

    # Check that the box is within the bounds of the array
    if x - box_size < 0 or x + box_size > nx or y - box_size < 0 or y + box_size > ny:
        raise ValueError("Box is not within the bounds of the array")

    x_discrete = int(x)
    y_discrete = int(y)
    
    # Get the box
    box = array[y_discrete - box_size:y_discrete + box_size, x_discrete - box_size:x_discrete + box_size]

    return box


# write a function which takes reads a fits file and returns the data as a 2D array
def read_fits(filename, ext=0):
    # Read the fits file
    with fits.open(filename) as hdul:
        data = hdul[ext].data
        header = hdul[ext].header
    
    return data, header


# write a function which takes a list of fits files, and a list of X, Y
# coordinates as input. The function should read the fits files using read_fits
# and extract the box around the X, Y coordinates using get_box for each file. It
# should then return a 3D array of the boxes.
def get_boxes_from_files(fits_files, X, Y, box_size=30):
    """
    Process a list of FITS files and extract boxes around given coordinates.
    
    :param fits_files: List of FITS file paths.
    :param coordinates_list: List of (x, y) coordinates.
    :param box_size: Size of the box around each coordinate.
    :return: A dictionary with filenames as keys and 3D arrays of boxes as values.
    """
    result = {}
    # Loop through each filename and coordinates
    for filename in fits_files:
        n = 0
        boxes = np.zeros((len(X), box_size*2, box_size*2))
        for x, y in zip(X, Y):
            # Read the fits file
            print(filename)
            data, header = read_fits(filename)

            # Get the box
            box = get_box(data, x, y, box_size=box_size)
            boxes[n, :, :] = box
            n += 1
        result[filename] = boxes
        
    return result


@custom_model
def gaussianTest2(x,y, height=1., center_x=1., center_y=1., width_x=1., width_y=1., theta=0., base=0.0):
    width_x = float(width_x)
    width_y = float(width_y)
    
    return (height*np.exp(-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2))+base


"""
Returns (height, x, y, width_x, width_y): the gaussian parameters of a 2D distribution by calculating its moments
"""
def moments(data):
	total = data.sum()
	#print("moments: ",total)
	X, Y = np.indices(data.shape)
	x = (X*data).sum()/total
	y = (Y*data).sum()/total
	col = data[:, int(y)]
	width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
	row = data[int(x), :]
	width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
#	print("moments:",width_x,width_y)
	height = data.max()


	if math.isnan(width_x):
		width_x = 4
	if math.isnan(width_y):
		width_y = 4
    
	base = np.median(data)

	return height, x, y, width_x, width_y, base


"""
Returns (height, x, y, width_x, width_y) the gaussian parameters of a 2D distribution found by a fit
"""

def fitgaussian(data,inputParams=None,weights=None):
    
    # Get input parameters
    if inputParams is None:
        params = moments(data)
    else:
        params = inputParams

    #print("PAR",params)

    # Mathematical model
    M = gaussianTest2(*params)
    
    #initiate fitting routines
    lmf = LevMarLSQFitter()
    
    # Define blank grid
    x,y = np.mgrid[:len(data),:len(data)]
    
    # Fit the function, as defined in M, to the actual data
    if weights is None:
        fit = lmf(M,x,y,data)
    else:
        fit = lmf(M,x,y,data,weights=weights)


    (height, y, x, width_y, width_x,theta,base) = (fit.height.value,fit.center_x.value,fit.center_y.value,fit.width_x.value,fit.width_y.value,fit.theta.value,fit.base.value)
    
    fwhmx = np.abs(2.0*np.sqrt(2.0*np.log(2.0))*width_x)
    fwhmy = np.abs(2.0*np.sqrt(2.0*np.log(2.0))*width_y)

    return height, x, y, fwhmx, fwhmy, theta, base


def main():
    # Sample list of filenames
    filenames = glob.glob("data/raw/test_3A.01.05/test*.fits")

    # Define the regular expression pattern
    pattern = r'(\w{3})\.(\d{3})'

    # Define custom column names (optional)
    custom_column_names = ["Lamp", "X"]
    
    box_size = 15
    # Call the function and get the extracted variables as a Pandas DataFrame
    extracted_data = extract_variables_and_export(filenames, pattern, 
                                                  column_names=custom_column_names)
    
    # # sort the DataFrame by the X column
    extracted_data = extracted_data.sort_values(by="X")
    fn_list = extracted_data['filename'].tolist()
    
    # open GUI
    gui = pointSelectGUI(fn_list)
    gui.run()
    box_centres = gui.selection['Selected Points']
    
    box_dict = get_boxes_from_files(fn_list, box_centres[:,0], box_centres[:,1])

    # Plot the boxes for a given file
    nplots = len(box_centres)
    ncols = 2 
    
    example_box = box_dict[fn_list[11]][0]

    params = fitgaussian(example_box)

    fig, ax1 = plt.subplots()
    ax1.imshow(example_box, origin='lower', vmin=0, vmax=1000)
    ax1.scatter(params[1], params[2], marker='x', color='red')
    
    
    
    # fig, ax = plt.subplots(nrows=1, ncols=nplots)
    # for i in range(nplots):
    #     ax[i].imshow(box_dict[fn_list[11]][i], origin='lower', vmin=0, vmax=1000)
        
    plt.show() 
    
    
    
if __name__ == "__main__":
    main()

