import numpy as np
import glob
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
import re
from focus_finder_gui import pointSelectGUI
from matplotlib.gridspec import GridSpec
from astropy.modeling import models, fitting

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


def main():
    # Sample list of filenames
    filenames = glob.glob("data/raw/test_3A.01.05/test*.fits")

    # Define the regular expression pattern
    pattern = r'(\w{3})\.(\d{3})'

    # Define custom column names (optional)
    custom_column_names = ["Lamp", "X"]
    
    # Call the function and get the extracted variables as a Pandas DataFrame
    extracted_data = extract_variables_and_export(filenames, pattern, 
                                                  column_names=custom_column_names)
    
    # # sort the DataFrame by the X column
    extracted_data = extracted_data.sort_values(by="X")
    fn_list = extracted_data['filename'].tolist()
    print(fn_list)
    
if __name__ == "__main__":
    main()

