import numpy as np
import glob
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
import re
from focus_finder_gui import pointSelectGUI
import lmfit
import argparse
from matplotlib.backends.backend_pdf import PdfPages
import os



def extract_variables_and_export(filenames, pattern, column_names=None):
    # Initialize lists to store extracted variables
    variable_data = []

    # Loop through each filename and extract variables using the pattern
    for filename in filenames:
        match = re.search(pattern, filename)
        if match:
            variables = []
            for capture_group in match.groups():
                if capture_group[0] == 'p':
                    variables.append(int(capture_group[1:]))
                elif capture_group[0] == 'm':
                    variables.append(-int(capture_group[1:]))
                else:
                    variables.append(capture_group)
                    
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
    
    # also record the poistion of the box 0,0 in the original array
    box0 = [y_discrete - box_size, x_discrete - box_size]

    return box, box0


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
def get_boxes_from_files(fits_files, X, Y, box_size=30, dark=None):
    """
    Process a list of FITS files and extract boxes around given coordinates.
    
    :param fits_files: List of FITS file paths.
    :param coordinates_list: List of (x, y) coordinates.
    :param box_size: Size of the box around each coordinate.
    :return: A dictionary with filenames as keys and 3D arrays of boxes as values.
    """
    result = {}
    box0s = []
    # Loop through each filename and coordinates
    for filename in fits_files:
        n = 0
        boxes = []
        for x, y in zip(X, Y):
            # Read the fits file
            data, header = read_fits(filename)
            
            # Subtract the dark frame if provided
            if dark is not None:
                dark_data, _ = read_fits(dark)
                data -= dark_data

            # Get the box
            box, box0 = get_box(data, x, y, box_size=box_size)
            boxes.append(box)
            box0s.append(box0)
            n += 1
        result[filename] = boxes
        
    return result, np.array(box0s)

# TODO fix this function to calculate encircled energy
def ensquared(array, radius, center):
    """
    Calculate the ensquared energy of a 2D array.
    
    :param array: 2D array.
    :param radius: Radii of the aperture.
    :param center: Center of the aperture.
    :return: Ensquared energy.
    """
    outer_box,_ = get_box(array, center[0], center[1], box_size=radius[0])
    inner_box,_ = get_box(array, center[0], center[1], box_size=radius[1])
    
    print(outer_box.shape, inner_box.shape)
    
    return 1 - np.sum(inner_box)/np.sum(outer_box)
    


def main():
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Find the best focus position")
    # add an argument for folder containing the data
    parser.add_argument("folder", help="Folder containing the data")
    
    # add optional arguments for box size and preload selection
    parser.add_argument("-b", "--box_size", type=int, default=30, help="Size of the box around each point")
    parser.add_argument("-p", "--preload_selection", help="File containing preloaded selection")
    parser.add_argument("-d", "--dark", help="Dark frame to subtract from the data")
    
    # parse the arguments
    args = parser.parse_args()
    
    # Sample list of filenames
    filenames = glob.glob(args.folder)
    
    # TODO looks like this information will not be in filename. Will likely need
    # to extract from template obd file. WT to provide this?
    # ---------------------------------------------------------------------
    # Define the regular expression pattern
    pattern = r'\.S(\w{1}\d{3})\.X(\w{1}\d{3})\.Y(\w{1}\d{3})\.Z(\w{1}\d{3})'

    # Define custom column names (optional)
    custom_column_names = ["S", "X", "Y", "Z"]
    
    # Call the function and get the extracted variables as a Pandas DataFrame
    extracted_data = extract_variables_and_export(filenames, pattern, 
                                                  column_names=custom_column_names)
    # ---------------------------------------------------------------------
    
    # sort the DataFrame by the X column
    # implications for analysis later?
    extracted_data = extracted_data.sort_values(by="X")
    fn_list = extracted_data['filename'].tolist()
    
    # create a GUI to select points. Preload the selection if specified
    if args.preload_selection:
        preload_selection = args.preload_selection
        gui = pointSelectGUI(fn_list, point_file=preload_selection, 
                             DAM_positions=extracted_data['X'].tolist(), box_size=args.box_size)
    else:    
        gui = pointSelectGUI(fn_list, DAM_positions=extracted_data['X'].tolist(), 
                             box_size=args.box_size)

    # run the GUI
    print("Running GUI")
    gui.run()
    # get the selected points
    box_centres = gui.selection['Selected Points']
        
    # Do dark subtraction if specified
    # TODO can this be done in box loop? Potentially faster (?) but might make 
    # plotting more difficult
    print("Extracting boxes for analysis")
    if args.dark:
        # get the boxes around the selected points
        box_dict, box_origin = get_boxes_from_files(fn_list, box_centres[:,0], box_centres[:,1], 
                                        box_size=args.box_size, dark=args.dark)
        print("Dark frame subtracted. Filename: {}".format(args.dark))
    else:
        # get the boxes around the selected points
        box_dict, box_origin = get_boxes_from_files(fn_list, box_centres[:,0], box_centres[:,1], 
                                        box_size=args.box_size)
        print("No dark frame subtracted")

    # create an empty dataframe with column names from col_names
    # for each DAM poistion there are n points
    nDAMpos = len(fn_list)
    n_points = len(box_centres)
    col_names = ['File', 'Point ID', 'DAM X', 'DAM Y', 'DAM Z', 'Xc', 'Yc', 
                 'FWHMx', 'FWHMy', 'EE']
    output_df = pd.DataFrame(columns=col_names, index=range(n_points*nDAMpos))
    
    # intialize the gaussian model
    g2d_model = lmfit.models.Gaussian2dModel()
    
    counter = 0
    # Loop through each box in each file and fit a 2D Gaussian
    print("Running analysis")
    for fn in fn_list:
        # get dam position from extracted_data
        DAMx = int(extracted_data.loc[extracted_data['filename'] == fn, 'X'].iloc[0])
        DAMy = int(extracted_data.loc[extracted_data['filename'] == fn, 'Y'].iloc[0])
        DAMz = int(extracted_data.loc[extracted_data['filename'] == fn, 'Z'].iloc[0])

        box_counter = 0
        for box in box_dict[fn]:
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
            Xc = fit_result.params['centerx'].value + box_origin[counter,0]
            Yc = fit_result.params['centery'].value + box_origin[counter,1]
            
            # encircled energy calculation
            EE = ensquared(box, [5, 3], [fit_result.params['centerx'].value, fit_result.params['centery'].value])
            
            # save the results to the dataframe
            output_df.loc[counter] = [fn, box_counter, DAMx, DAMy, DAMz, Xc, Yc, FWHMx, FWHMy, EE]
            box_counter += 1
            counter += 1
    print("Analysis complete")
    
    # save the dataframe to a csv file
    test_name = 'placeholder_test_name'
    output_df.to_csv(test_name + 'output.csv', index=False)


    # ---------------------------------------------------------------------
    # Plotting
    
    # plot the each box with the fitted gaussian
    # TODO add a ellipse around the FWHM
    # TODO add pixel coordinates a titles for each subplot
    pdf_filename = test_name + 'boxes.pdf'
    num_cols = 2
    
    with PdfPages(pdf_filename) as pdf:
            for filename, arrays in box_dict.items():
                num_arrays = len(arrays)
                num_rows = int(np.ceil(num_arrays // num_cols))
                fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 3 * num_rows))
                plt.subplots_adjust(hspace=0.5)
                
                point_info = output_df[output_df['File'] == filename]
                for i, ax in enumerate(axes.flatten()):
                    if i < num_arrays:
                        ax.imshow(arrays[i])
                        ax.scatter(point_info['Xc'].iloc[i] - box_origin[i, 0], 
                                   point_info['Yc'].iloc[i] - box_origin[i, 1],
                                   color='r',s=10)
                    ax.axis('off')
                
                plt.suptitle(os.path.basename(filename))
                plt.tight_layout()
                pdf.savefig()
                plt.close()
        

if __name__ == "__main__":
    main()

