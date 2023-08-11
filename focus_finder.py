import numpy as np
import glob
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
import re
from focus_finder_gui import pointSelectGUI
import lmfit


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
        boxes = []
        for x, y in zip(X, Y):
            # Read the fits file
            data, header = read_fits(filename)

            # Get the box
            box = get_box(data, x, y, box_size=box_size)
            boxes.append(box)
            n += 1
        result[filename] = boxes
        
    return result

def main():
    # Sample list of filenames
    filenames = glob.glob("data/raw/test_3A.01.05/test*.fits")
    preload_selection = "selected_points.txt"

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
    gui = pointSelectGUI(fn_list, point_file=preload_selection)
    gui.run()
    box_centres = gui.selection['Selected Points']
    
    box_dict = get_boxes_from_files(fn_list, box_centres[:,0], box_centres[:,1], 
                                    box_size=box_size)

    
    # intialize the gaussian model
    g2d_model = lmfit.models.Gaussian2dModel()
    
    nDAMpos = len(fn_list)
    n_points = len(box_centres)
    
    col_names = ['DAM X', 'Xc', 'Yc', 'sigmax', 'sigmay']
    
    # create an empty dataframe with column names from col_names
    output_df = pd.DataFrame(columns=col_names, index=range(n_points*nDAMpos))
    
    # intialize the gaussian model
    g2d_model = lmfit.models.Gaussian2dModel()
    
    counter = 0
    # Loop through each box in each file and fit a 2D Gaussian
    for fn in fn_list:
        # get dam position from extracted_data
        DAMx = int(extracted_data.loc[extracted_data['filename'] == fn, 'X'].iloc[0])
        
        for box in box_dict[fn]:
            box_centre_counter = 0
            X, Y = np.meshgrid(np.arange(box.shape[0]), np.arange(box.shape[1]))
            # flatten X, Y and box to guess the parameters
            flatX = X.flatten()
            flatY = Y.flatten()
            flatbox = box.flatten()
            
            # guess the parameters
            params = g2d_model.guess(flatbox, flatX, flatY)
            
            # fit the model to the data
            fit_result = g2d_model.fit(box, params, x=X, y=Y)
            
            FWHMx = fit_result.params['fwhmx'].value
            FWHMy = fit_result.params['fwhmy'].value
            Xc = fit_result.params['centerx'].value
            Yc = fit_result.params['centery'].value
            
            output_df.loc[counter] = [DAMx, Xc, Yc, FWHMx, FWHMy]
            
            counter += 1
    
    
    # save the dataframe to a csv file
    output_df.to_csv('output.csv', index=False)
    # plot the boxes from the first file
    # find the number of boxes in the first file
    nboxes = len(box_dict[fn_list[0]])
        
    # extract the fitted parameters for the boxes at DAMx_0
    df_130 = output_df.loc[output_df['DAM X'] == 142]
    

    # make a figure with nboxes subplots
    fig, ax = plt.subplots(1, nboxes, figsize=(15, 5))
    for i in range(nboxes):
        ax[i].imshow(box_dict[fn_list[12]][i])
        ax[i].scatter(df_130['Xc'].iloc[i], df_130['Yc'].iloc[i], color='r')
        ax[i].set_title('Box {}'.format(i+1))
        
    plt.show()
    
    
if __name__ == "__main__":
    main()

