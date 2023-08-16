import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd


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
    
    print(Zc)
    
    
    # plot the DAM positions in 3D
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # Plot the DAMX positions
    ax.scatter(DAMX[:,0], DAMX[:,1], DAMX[:,2], color='r', label='DAMX')
    # Plot the DAMY positions
    ax.scatter(DAMY[:,0], DAMY[:,1], DAMY[:,2], color='b', label='DAMY')
    #plot the DAMZ positions
    ax.scatter(DAMZ[:,0], DAMZ[:,1], DAMZ[:,2], color='g', label='DAMZ')
    # plot the fit centres
    ax.scatter(Xc, Yc, Zc, color='k', label='Fit Centres')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()

     
     

