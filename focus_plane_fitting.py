import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import UnivariateSpline


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
    spline_min_Z= spline_points[np.argmin(spline_1D(spline_points))]
    spline_min_score = spline_1D(spline_min_Z)
    
    return spline_1D, (spline_min_Z, spline_min_score)
    



def main():
    DAM_offsets = [[0, 0,-263.5] ,[0, -228.2,131.7], [0, 228.2,131.7]]

    
    # Convert pixel coordinates to mm
    pixel_size = 0.015 #15 micron pixels
    DAM_step_size = 0.01 # 10 micron steps
    
    output_df = pd.read_csv('output.csv')
    
    # Get X values of DAMX
    DAMXx = output_df['DAM X'].values * DAM_step_size
    # get xyz values of DAMX
    DAMX = np.tile(DAM_offsets[0], (len(DAMXx), 1))
    DAMX[:,0] = DAMXx
    
    # repeat for DAMY and DAMZ
    DAMYx = output_df['DAM Y'].values * DAM_step_size
    DAMY = np.tile(DAM_offsets[1], (len(DAMYx), 1))
    DAMY[:,0] = DAMYx
    
    DAMZx = output_df['DAM Z'].values * DAM_step_size
    DAMZ = np.tile(DAM_offsets[2], (len(DAMZx), 1))
    DAMZ[:,0] = DAMZx
    
    Yc = output_df['Xc'].values * pixel_size
    Zc = output_df['Yc'].values * pixel_size
    Xc = DAMXx
    
    spline_mins = np.zeros((len(np.unique(output_df['Point ID'].values)), 3))
        # get the FWXMx values for points with point id == 0
    for point_id in np.unique(output_df['Point ID'].values):
        id0_df = output_df[output_df['Point ID'] == point_id]
        
        Zc_id0 = id0_df['DAM X'].values * DAM_step_size
        FWHMx = id0_df['FWHMx'].values
        FWHMx[5] = FWHMx[5] * 1.5

        spline_1D, spline_min = fit_spline(Zc_id0, FWHMx)
        
        spline_points = np.linspace(Zc_id0[0], Zc_id0[-1], 100)
        
        # get the mean Xc, Yc for each ID
        mean_Xc = np.mean(id0_df['Xc'].values * pixel_size)
        mean_Yc = np.mean(id0_df['Yc'].values * pixel_size)
        
        min_coord = np.array([mean_Xc, mean_Yc, spline_min[0]])
        spline_mins[point_id] = min_coord
        
        # fig, ax = plt.subplots()
        # ax.scatter(Zc_id0, FWHMx)
        # ax.plot(spline_points, spline_1D(spline_points))
        # ax.scatter(spline_min[0], spline_min[1], color='r')
        # ax.set_xlabel('Z')
        # ax.set_ylabel('FWHMx')
        # ax.set_title(f'Point ID = {point_id}')
    
    # print(spline_mins)
    # # plot the DAM positions in 3D
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # # Plot the DAMX positions
    # ax.scatter(DAMX[:,0], DAMX[:,1], DAMX[:,2], color='m', label='DAMX', alpha=0.5)
    # # Plot the DAMY positions
    # ax.scatter(DAMY[:,0], DAMY[:,1], DAMY[:,2], color='k', label='DAMY', alpha=0.5)
    # #plot the DAMZ positions
    # ax.scatter(DAMZ[:,0], DAMZ[:,1], DAMZ[:,2], color='g', label='DAMZ', alpha=0.5)
    # # plot the fit centres
    # # ax.scatter(Xc, Yc, Zc, color='b', label='Fit Centres')
    # ax.scatter(spline_mins[:,2], spline_mins[:,0], spline_mins[:,1], color='r', label='Spline Minima')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.legend()
    # plt.show()
    
    # fit a plane to the spline minima
    (A, B, C, D) = plane_fitter(spline_mins)
    
    
    # plot the spline mimima and the plane
    x = np.linspace(np.min(spline_mins[:,0]), np.max(spline_mins[:,0]), 100)
    y = np.linspace(np.min(spline_mins[:,1]), np.max(spline_mins[:,1]), 100)
    X, Y = np.meshgrid(x, y)
    Z = (-A * X - B * Y - D) / C
    
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # ax.scatter(spline_mins[:,0], spline_mins[:,1], spline_mins[:,2], color='r', label='Spline Minima')
    # ax.plot_surface(X, Y, Z, alpha=0.5)
    # plt.show()
    
    
    # find the intersection of the plane with the DAMX, DAMY and DAMZ axes
    DAMX_z = find_point_on_plane(A, B, C, D, DAM_offsets[0][1:], missing_coord='z')
    DAMY_z = find_point_on_plane(A, B, C, D, DAM_offsets[1][1:], missing_coord='z')
    DAMZ_z = find_point_on_plane(A, B, C, D, DAM_offsets[2][1:], missing_coord='z')
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(DAMX[:,0], DAMX[:,1], DAMX[:,2], color='m', label='DAMX')
    # Plot the DAMY positions
    ax.scatter(DAMY[:,0], DAMY[:,1], DAMY[:,2], color='k', label='DAMY')
    #plot the DAMZ positions
    ax.scatter(DAMZ[:,0], DAMZ[:,1], DAMZ[:,2], color='g', label='DAMZ')
    ax.scatter(DAMX_z, DAMX[:,1], DAMX[:,2], color='b', label='DAMX_z')
    ax.scatter(DAMY_z, DAMY[:,1], DAMY[:,2], color='b', label='DAMY_z')
    ax.scatter(DAMZ_z, DAMZ[:,1], DAMZ[:,2], color='b', label='DAMZ_z')
    ax.scatter(spline_mins[:,2], spline_mins[:,0], spline_mins[:,1], color='r', label='Spline Minima')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
    
    
    
    
    
    
    

    

if __name__ == "__main__":
    main()

     
     

