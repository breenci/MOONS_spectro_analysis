import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import UnivariateSpline
import math


def find_plane(p1, p2, p3):
    # Convert points to numpy arrays for vector operations
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    # Calculate vectors between points
    v1 = p2 - p1
    v2 = p3 - p1

    # Calculate the normal vectors of the planes using cross product
    normal = np.cross(v1, v2)

    # Extract coefficients (A, B, C) from the normal vectors
    A, B, C = normal[:, 0], normal[:, 1], normal[:, 2]

    # Calculate the constant terms (D) in the plane equations
    D = -np.sum(normal * p1, axis=1)

    return A, B, C, D


# write a function to fit a plane to a set of points
def plane_fitter(point_coords):
    '''Fit a plane to a set of points and return the unit normal.'''

    # make sure points are in a numpy array
    point_coords = np.array(point_coords)
    
    # Subtract the centroid from the set of points
    centroid = np.mean(point_coords, axis=0)
    cntrd_pnts = point_coords - centroid
    
    # Perform singular value decomposition
    svd, _, _ = np.linalg.svd(cntrd_pnts.T)
    
    # Final column of the svd gives the unit normal
    norm = svd[:,2]
    
    # # Take the +ve norm
    # if norm[1]<0:
    #     norm = -1*norm
    
    # extract the coefficients of the plane
    (A, B, C) = norm
    D = -np.sum(norm * centroid)
    
    return A, B, C, D


def find_point_on_plane(A, B, C, D, known_coords, missing_coord='z'):
    """Find the value of the missing coordinate given the coefficients of the plane"""
    
    if missing_coord == 'z':
        x, y = known_coords
        z = (-A * x - B * y - D) / C
        missing_coord_val = z
        
    if missing_coord =='y':
        x, z = known_coords
        y = (-A * x - C * z - D) / B
        missing_coord_val = y
    
    if missing_coord =='x':
        y, z = known_coords
        x = (-B * y - C * z - D) / A
        missing_coord_val = x
        
    return missing_coord_val


# write a function which takes Z and a score as input and fits a spline to the
# score. Then find the Z value at the minimum of the spline over the range of Z
# if there are any outliers, remove them and refit the spline
def fit_spline(Z, score, k=4, outlier_f=1.5, minZ_step=0.01):
    
    # fit the spline
    spline_1D = UnivariateSpline(Z, score, k=k)
    
    # outlier removal using residual
    residual = score - spline_1D(Z)
    # find the outliers
    outliers_mask = np.abs(residual) > outlier_f * np.std(residual)
    # remove the outliers present and refit the spline
    if outliers_mask.sum() > 0:
        Z = Z[~outliers_mask]
        score = score[~outliers_mask]
        spline_1D = UnivariateSpline(Z, score, k=k)
        
    # find the minimum of the spline
    spline_points = np.arange(Z[0], Z[-1], minZ_step/2)
    spline_min_Z = spline_points[np.argmin(spline_1D(spline_points))]
    spline_min_score = spline_1D(spline_min_Z)
    
    return spline_1D, (spline_min_Z, spline_min_score)


def create_subplot_grid(num_plots):
    # Calculate the number of rows and columns for the subplot grid
    num_rows = int(math.sqrt(num_plots))
    num_cols = math.ceil(num_plots / num_rows)

    # Create the subplot grid
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

    # Flatten the axes array if needed
    if num_plots == 1:
        axes = [axes]

    return fig, axes


# write a function which finds the minimum of a score as a function of Z for
# each box id
def find_minima(coords, score, ID_arr, DAM_stoutlier_f=1.5, minZ_step=0.01, 
                score_label='Score', limit=None):
    # make sure the arrays are numpy arrays
    coords = np.array(coords)
    score = np.array(score)
    ID_arr = np.array(ID_arr)
    
    # separate the Z and ID values for each box
    unique_IDs = np.unique(ID_arr)
    
    # initialise an array to hold the minima
    minima = np.zeros((len(unique_IDs), 3))
    
    if limit != None:
        under_range = np.zeros((len(unique_IDs), 2))
    
    # create a square subplot grid
    fig, axes = create_subplot_grid(len(unique_IDs))
    axes = axes.flatten()
    
    # for each point ID, find the minimum of the score as a function of Z and
    # the X and Y coordinates of the minimum
    for n, ID in enumerate(unique_IDs):
        
        # find the coordinates of the point ID
        coords_id = coords[ID_arr == ID]
        
        # X and Y min is just the median of the X and Y coordinates
        # TODO: Check how much the points move in X and Y
        X_id = coords_id[:,0]
        Y_id = coords_id[:,1]
        X_min = np.median(X_id)
        Y_min = np.median(Y_id)
        
        # Z is found using spline fit
        Z_id = coords_id[:,2]
        score_id = score[ID_arr == ID]
        spline_fit, spline_min = fit_spline(Z_id, score_id, outlier_f=DAM_stoutlier_f, minZ_step=minZ_step)
        
        # store the result
        minima[n] = np.array([X_min, Y_min, spline_min[0]])
        
        spline_points = np.arange(Z_id[0], Z_id[-1], minZ_step/5)
        
        # if a limit is given find the range of values of spline points below 
        # the limit
        if limit != None:
            under_limit = spline_points[spline_fit(spline_points) < limit]
            Z_range_low = np.min(under_limit)
            Z_range_high = np.max(under_limit)
            under_range[n] = np.array([Z_range_low, Z_range_high])
            axes[n].plot(under_limit, spline_fit(under_limit), linewidth=10, color='g')
            
        
        # sanity check the spline minima with a plot
        axes[n].scatter(Z_id, score_id)
        axes[n].plot(spline_points, spline_fit(spline_points))
        axes[n].scatter(spline_min[0], spline_min[1], color='r')
        axes[n].set_xlabel('Z')
        axes[n].set_ylabel(score_label)
        axes[n].set_title(f'Point ID = {ID}')
    
    plt.show()
    
    if limit != None:
        return unique_IDs, minima, under_range
    
    else:  
        return unique_IDs, minima


def get_score(df, metric_names, weights):
    
    # initilise array to hold weighted metrics
    wmetric_arr = np.zeros((len(df), len(metric_names)))
    
    for n, name in enumerate(metric_names):
        metric = df[name]
        weight = weights[n]
        weighted_metric = metric * weight
        wmetric_arr[:,n] = weighted_metric
        
    # sum the weighted metrics
    score = np.sum(wmetric_arr, axis=1)
    
    return score


def main():
    # Define the DAM offsets
    # Coordinate system is X, Y are in the plane of the detector and Z is along
    # the direction of travel of the motors 
    DAM_offsets = [[0, -263.5, 0] ,[-228.2, 131.7, 0], [228.2, 131.7, 0]]

    
    # Convert pixel coordinates to mm
    pixel_size = 0.015 #15 micron pixels
    DAM_step_size = 0.01 # 10 micron steps
    
    # read in the output file
    output_df = pd.read_csv('output.csv')
    
    # convert the DAM positions to mm
    output_df['DAM X'] = output_df['DAM X'] * DAM_step_size
    output_df['DAM Y'] = output_df['DAM Y'] * DAM_step_size
    output_df['DAM Z'] = output_df['DAM Z'] * DAM_step_size
    
    # convert the Xc, Yc, FWHMx and FWHMy to mm
    output_df['Xc'] = output_df['Xc'] * pixel_size
    output_df['Yc'] = output_df['Yc'] * pixel_size
    # output_df['FWHMx'] = output_df['FWHMx'] * pixel_size
    # output_df['FWHMy'] = output_df['FWHMy'] * pixel_size
    
    # rename the columns to reflect unit change
    # output_df.rename(columns={'DAM X': 'DAM X (mm)', 'DAM Y': 'DAM Y (mm)', 'DAM Z': 'DAM Z (mm)'}, inplace=True)
    
    # confusingly the individual DAMS are labelled DAM1, DAM2 and DAM3
    # These are three DAMs with individually each have an XYZ position
    # I will refer to DAM1 as DAM1, DAM2 as DAM2 and DAM3 as DAM3 to avoid
    # variable name confusion
    
    # create variables for 3D locations of DAM1, DAM2 and DAM3 for plotting
    DAM1 = np.broadcast_to(np.array(DAM_offsets[0]), (len(output_df), 3)).copy()
    DAM1[:,2] = DAM1[:,2] + output_df['DAM X']
    
    DAM2 = np.broadcast_to(np.array(DAM_offsets[1]), (len(output_df), 3)).copy()
    DAM2[:,2] = DAM2[:,2] + output_df['DAM Y']
    
    DAM3 = np.broadcast_to(np.array(DAM_offsets[2]), (len(output_df), 3)).copy()
    DAM3[:,2] = DAM3[:,2] + output_df['DAM Z']
    
    mixed_score = get_score(output_df, ['FWHMx'], [1])
    # find the minimum of the FWHMx as a function of Z for each DAM
    IDs, FWHMx_minima, range = find_minima(output_df[['Xc', 'Yc', 'DAM X']], 
                                    mixed_score, output_df['Point ID'], DAM_stoutlier_f=1.5, 
                                    minZ_step=0.01, limit=3.1)
    
    
    N_points = 1000
    range_coords = np.zeros((len(IDs) * N_points, 3))
    for n, ID in enumerate(IDs):
        Xc = FWHMx_minima[n,0]
        Yc = FWHMx_minima[n,1]
        
        Z_range = range[n]
        print(Z_range)
        Z_full = np.linspace(Z_range[0], Z_range[1], N_points)
        X_full = np.repeat(Xc, len(Z_full))
        Y_full = np.repeat(Yc, len(Z_full))
        in_range_coords = np.vstack((X_full, Y_full, Z_full)).T
        
        range_coords[n * N_points:(n+1) * N_points,:] = in_range_coords
        
        
        
    
        
        
    # plot the DAM positions in 3D
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # Plot the DAM1 positions
    ax.scatter(DAM1[:,0], DAM1[:,1], DAM1[:,2], color='m', label='DAM1', alpha=0.5)
    # Plot the DAM2 positions
    ax.scatter(DAM2[:,0], DAM2[:,1], DAM2[:,2], color='k', label='DAM2', alpha=0.5)
    #plot the DAM3 positions
    ax.scatter(DAM3[:,0], DAM3[:,1], DAM3[:,2], color='g', label='DAM3', alpha=0.5)
    # plot the fit centres
    # ax.scatter(output_df['Xc'], output_df['Yc'], output_df['DAM X'], color='b', label='Fit Centres')
    ax.scatter(FWHMx_minima[:,0], FWHMx_minima[:,1], FWHMx_minima[:,2], color='r', label='Spline Minima')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()
    
    # fit a plane to the spline minima
    (A, B, C, D) = plane_fitter(FWHMx_minima)
    
    
    # plot the spline mimima and the plane
    x = np.linspace(np.min(FWHMx_minima[:,0]), np.max(FWHMx_minima[:,0]), 100)
    y = np.linspace(np.min(FWHMx_minima[:,1]), np.max(FWHMx_minima[:,1]), 100)
    X, Y = np.meshgrid(x, y)
    Z = (-A * X - B * Y - D) / C
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(FWHMx_minima[:,0], FWHMx_minima[:,1], FWHMx_minima[:,2], color='r', label='Spline Minima')
    ax.plot_surface(X, Y, Z, alpha=0.5)
    plt.show()
    
    
    # find the intersection of the plane with the DAM1, DAM2 and DAM3 axes
    DAM1_z = find_point_on_plane(A, B, C, D, DAM_offsets[0][:2], missing_coord='z')
    DAM2_z = find_point_on_plane(A, B, C, D, DAM_offsets[1][:2], missing_coord='z')
    DAM3_z = find_point_on_plane(A, B, C, D, DAM_offsets[2][:2], missing_coord='z')
    
    
    # new meshgrid for plotting
    x = np.linspace(-270, 270, 100)
    y = np.linspace(-270, 131, 100)
    X, Y = np.meshgrid(x, y)
    Z = (-A * X - B * Y - D) / C
    

    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(DAM1[:,0], DAM1[:,1], DAM1[:,2], color='m', label='DAM1', alpha=0.01)
    # Plot the DAM2 positions
    ax.scatter(DAM2[:,0], DAM2[:,1], DAM2[:,2], color='k', label='DAM2', alpha=0.01)
    #plot the DAM3 positions
    ax.scatter(DAM3[:,0], DAM3[:,1], DAM3[:,2], color='g', label='DAM3', alpha=0.01)
    ax.scatter(DAM1[0,0], DAM1[0,1], DAM1_z, color='b', label='DAM1_z')
    ax.scatter(DAM2[0,0], DAM2[0,1], DAM2_z, color='b', label='DAM2_z')
    ax.scatter(DAM3[0,0], DAM3[0,1], DAM3_z, color='b', label='DAM3_z')
    ax.scatter(FWHMx_minima[:,0], FWHMx_minima[:,1], FWHMx_minima[:,2], color='r', label='Spline Minima')
    #surface plot
    ax.plot_surface(X, Y, Z, alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    

    # fit a plane to the acceptable range of Z values
    (Ar, Br, Cr, Dr) = plane_fitter(range_coords)
    # fit a plane to the spline minima
    (A, B, C, D) = plane_fitter(FWHMx_minima) 
    
    x = np.linspace(np.min(FWHMx_minima[:,0]), np.max(FWHMx_minima[:,0]), 100)
    y = np.linspace(np.min(FWHMx_minima[:,1]), np.max(FWHMx_minima[:,1]), 100)
    X, Y = np.meshgrid(x, y)
    Zr = (-Ar * X - Br * Y - Dr) / Cr
    Z = (-A * X - B * Y - D) / C
    
    print(Ar, Br, Cr, Dr)
    print(A, B, C, D)

    # plot the acceptable range of Z values
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(range_coords[:,0], range_coords[:,1], range_coords[:,2], color='g', label='Acceptable Range')
    ax.plot_surface(X, Y, Zr, alpha=0.5)
    ax.plot_surface(X, Y, Z, alpha=0.5)

    plt.show()
    
    
    

    # plt.show()

if __name__ == "__main__":
    main()

     
     

