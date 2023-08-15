import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd


def find_plane(points):
    # Ensure we have exactly 3 points
    if len(points) != 3:
        raise ValueError("Exactly 3 points are required to define a plane.")

    # Convert points to numpy arrays
    p1 = np.array(points[0])
    p2 = np.array(points[1])
    p3 = np.array(points[2])

    # Calculate vectors between points
    v1 = p2 - p1
    v2 = p3 - p1

    # Calculate the cross product of the vectors to get the normal vector
    normal = np.cross(v1, v2)

    # Normalize the normal vector
    normal /= np.linalg.norm(normal)

    # Calculate the coefficients of the plane equation (Ax + By + Cz + D = 0)
    A, B, C = normal
    D = -np.dot(normal, p1)

    return A, B, C, D



def main():
    DAM_positions = np.array([[150, 0,-263.5] ,[130, -228.2,131.7], [130, 228.2,131.7]])
    A, B, C, D = find_plane(DAM_positions)
    
    # Create a grid of points in the x-y plane in the range of the DAM positions
    x = np.linspace(120, 160, 100)
    y = np.linspace(-300, 300, 100)
    
    X, Y = np.meshgrid(x, y)
    
    Z = (-A * X - B * Y - D) / C
    
    
    
    # plot the DAM positions in 3D
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(DAM_positions[:,0], DAM_positions[:,1], DAM_positions[:,2])
    ax.plot_surface(X, Y, Z, alpha=0.2)
    plt.show()
    
if __name__ == "__main__":
    main()

     
     

