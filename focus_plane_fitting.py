 

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy import optimize
import pickle
import copy
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import math

from matplotlib.backends.backend_pdf import PdfPages



def zFinder(motorXs, motorYs, motorZs, x, y):
    '''
    # The detector plane is defiend by the coordinates of the three DAM motors, which are:
    motor1: (motorXs[0],motorYs[0],motorZs[0])
    motor2: (motorXs[1],motorYs[1],motorZs[1])
    motor3: (motorXs[2],motorYs[2],motorZs[2])

    Function returns the z value of the point "x,y" in the plane.
    
    '''
    
    B = np.c_[motorXs, motorYs, np.ones(len(motorXs))]
    D,_,_,_ = scipy.linalg.lstsq(B, motorZs)
    
    z=D[0]*x + D[1]*y + D[2]
    
    return z



def bestEE(dam_zValues, xpos, ypos, EE, motorXs=0, motorYs=0, fix=0, DAMstart=[0,0,0], increment=0.01):
    numSpots = len(EE[:,0])
    numFiles = len(EE[0,:])

    # Note this function is very similar to that used for the FWHM fits. But there is no X and Y component of the fit to consider.
    bestFocusEE = np.zeros(numSpots)
    bestPeakEE = np.zeros(numSpots)
    
    bestxPos = np.zeros(numSpots)
    yfitX = np.zeros(shape = (numSpots,numFiles))
    zPoints = np.zeros(shape = (numSpots,numFiles))

    
    
    for i in range(numSpots):

        # Need to find the z location of each spot defined in DAM coordinate space. Do this for each spot at a time for all z positions
        zValues = np.zeros(numFiles)
        for j in range(numFiles):
            motorZs = np.array(DAMstart)+(j*increment)
            zValues[j] = zFinder(motorXs, motorYs, motorZs, xpos[i,j], ypos[i,j])

        
        zPoints[i,:] = zValues
        
        # Assign weights to the fit to ensure only inlcudes the point near the best focus. This is crudely done at the moment!!
        weights = EE[i,:]**4
 
        # Check for any Nans
        for f in range(numFiles):
            if math.isnan(EE[i,f]):
                weights[f] = -100
                EE[i,f] = 0.5
                
       
        # Fit peak flux
        peakFitcoeffs = np.polyfit(zValues,EE[i,:],deg=2,w=weights)
     
        # Plot the peaks, with the fit overlaid
        yfitX[i,:] = peakFitcoeffs[0]*zValues**2 + peakFitcoeffs[1]*zValues + peakFitcoeffs[2]

        # Store bestfocus location and value
        bestFocusEE[i] = -peakFitcoeffs[1]/(2*peakFitcoeffs[0])         
        bestPeakEE[i] = peakFitcoeffs[0]*bestFocusEE[i]**2 + peakFitcoeffs[1]*bestFocusEE[i] + peakFitcoeffs[2]
            

        # Find the best location of the spot for the best focus
        bestxPos[i] = EE[i,np.argmin(yfitX[i,:])]

        #print(i,bestFocusX[i],bestPeakX[i],bestFocusY[i],bestPeakY[i],bestxPos[i],bestyPos[i])        

    return bestFocusEE,bestPeakEE,bestxPos,yfitX,zPoints
        





def bestFocus(dam_zValues, xpos, ypos, originalFWHMx, originalFWHMy, motorXs=0, motorYs=0, fix=0, DAMstart = [0.40, 0.4, 0.4], increment=0.01):
    
    numSpots = len(originalFWHMx[:,0])
    numFiles = len(originalFWHMx[0,:])


    # Define some blank arrays for later populating
    # 
    bestFocusX = np.zeros(numSpots)         # Z location of the best focus position, when optimising for X FWHM
    bestPeakX = np.zeros(numSpots)          # Value of the best focus position, when optimising for X - i.e. the smallest FWHM along dispersion direction.
    bestFocusY = np.zeros(numSpots)
    bestPeakY = np.zeros(numSpots)

    bestxPos = np.zeros(numSpots)           # Pixel location of best position position
    bestyPos = np.zeros(numSpots)

    yfitX = np.zeros(shape = (numSpots,numFiles))   # Fit of the data - this is only really needed for plotting
    yfitY = np.zeros(shape = (numSpots,numFiles))

    zPoints = np.zeros(shape = (numSpots,numFiles)) # a blank array for storing the z position of all the spots


    # This is a really crude way of getting the data to ignore some frames/spots that are 'bad' for various reasons - normally clear outliers to the PSF fit, often caused by the assymetry as we go out of focus.
    if fix == 1:
        originalFWHMx[15,1] = 11.0
    if fix == 2:
        originalFWHMx[15,1:3] = 11.0
    if fix == 4:
        originalFWHMx[15,1:3] = 11.0
        originalFWHMx[20,0] = 15.0
    if fix == 5:
        originalFWHMx[15,1:3] = 11.0
        originalFWHMx[20,7] = 7.0
    if fix == 7:
        originalFWHMx[15,1:3] = 11.0
        originalFWHMx[20,0:2] = 11.0



    # Convert the position of spots from pixel space to mm.
    # Note this change the xpos and ypos values for rest of the code, which is fine, but should be noted!
    pixelScale = 0.015  # each pixel = 15 microns.

    xpos = (xpos-2046)*pixelScale
    ypos = (ypos-2046)*pixelScale
    

    
    for i in range(numSpots):

        # Need to find the z location of each spot defined in DAM coordinate space. Do this for each spot at a time for all z positions.
        zValues = np.zeros(numFiles)
        for j in range(numFiles):
            motorZs = np.array(DAMstart)+(j*increment)   # a three-element array with the DAM Z position for each motor.
            ## Note, code is not well tested for the DAMs having different starting values, i.e. not 'Piston' motion.
            zValues[j] = zFinder(motorXs, motorYs, motorZs, xpos[i,j], ypos[i,j])

        zPoints[i,:] = zValues
        
        # Assign weights to the fit to ensure only inlcudes the point near the best focus
        # This is a bit of a fudge and could be improved. 
        weightUpperLimit = 5
        weightLowerLimit = 2
        weightsX = np.zeros(len(dam_zValues))
        weightsX = np.where(originalFWHMx[i,:] < weightUpperLimit, 1, 0)
        weightsX = np.where(originalFWHMx[i,:] < weightLowerLimit, 0, 1)

        weightsY = np.zeros(len(dam_zValues))
        weightsY = np.where(originalFWHMy[i,:] < weightUpperLimit, 1, 0)
        weightsY = np.where(originalFWHMy[i,:] < weightLowerLimit, 0, 1)


        # Find coefficients of the fit minimum FWHM
        peakFitcoeffsX = np.polyfit(zValues,originalFWHMx[i,:],deg=2,w=weightsX)
        peakFitcoeffsY = np.polyfit(zValues,originalFWHMy[i,:],deg=2,w=weightsY)
    
        # Fit to the FWHM data 
        yfitX[i,:] = peakFitcoeffsX[0]*zValues**2 + peakFitcoeffsX[1]*zValues + peakFitcoeffsX[2]
        yfitY[i,:] = peakFitcoeffsY[0]*zValues**2 + peakFitcoeffsY[1]*zValues + peakFitcoeffsY[2]

        # Store bestfocus location and value
        bestFocusX[i] = -peakFitcoeffsX[1]/(2*peakFitcoeffsX[0])         
        bestFocusY[i] = -peakFitcoeffsY[1]/(2*peakFitcoeffsY[0])         
        bestPeakX[i] = peakFitcoeffsX[0]*bestFocusX[i]**2 + peakFitcoeffsX[1]*bestFocusX[i] + peakFitcoeffsX[2]
        bestPeakY[i] = peakFitcoeffsY[0]*bestFocusY[i]**2 + peakFitcoeffsY[1]*bestFocusY[i] + peakFitcoeffsY[2]


        # Find the best location of the spot for the best focus
        # Note the x and y pixel position change as you sweep through focus. The plane of best therefore needs to know the measured x/y location at the DAM position nearest to the best focus Z location.
        bestxPos[i] = xpos[i,np.argmin(yfitX[i,:])]
        bestyPos[i] = ypos[i,np.argmin(yfitY[i,:])]

        #print(i,bestFocusX[i],bestPeakX[i],bestFocusY[i],bestPeakY[i],bestxPos[i],bestyPos[i])        

    return bestFocusX,bestPeakX,bestFocusY,bestPeakY,bestxPos,bestyPos,yfitX,yfitY,zPoints
        

def findPlane(bestxPos,bestyPos,bestFocus,motorXs,motorYs,X,Y):
 
    # best-fit linear plane
    A = np.c_[bestxPos, bestyPos, np.ones(bestxPos.shape[0])]
    C,_,_,_ = scipy.linalg.lstsq(A, bestFocus)

    
    # evaluate it on grid
    Z = C[0]*X + C[1]*Y + C[2]

    # Define motor coords and find the position of the DAMS. Kinda the whole point!!
    fittedMotorZs = C[0]*motorXs + C[1]*motorYs + C[2]

    print('motorZs: ',fittedMotorZs)

    return Z,C,fittedMotorZs



def main():
    font = {'family' : 'normal','weight' : 'normal','size'   : 8}
    plt.rc('font', **font)

    # Need to specify the DAM starting values and the increment between frames.
    DAM_starts = [-0.1, -0.1, -0.1]
    increments = 0.01000

    focusData = True

    # Plotting options....
    savePlotsPDF = False
    plotBestFocus = True   
    gridSpots = True                    
    movements = True
    encircled = True
    fittedPlane = True


    # If two files are specified, then the first if the list of fits for a sweep, the second is a best fit frame. 
    if len(sys.argv) > 2:
        plotBest = True
    else:
        plotBest = False



    # Motor positions, we thought these might flip - they don't seem to, which I still don't fully understand.
    option = 4

    if option == 1:
       motor1 = [0,263.5]
       motor2 = [228.2,-131.7]
       motor3 = [-228.2,-131.7]

    if option == 2:
       motor1 = [0,263.5]
       motor3 = [228.2,-131.7]
       motor2 = [-228.2,-131.7]

    if option == 3:
       motor1 = [0,-263.5]
       motor2 = [228.2,131.7]
       motor3 = [-228.2,131.7]

    if option == 4:
       motor1 = [0,-263.5]
       motor2 = [-228.2,131.7]
       motor3 = [228.2,131.7]


    motorXs = np.array([motor1[0],motor2[0],motor3[0]])
    motorYs = np.array([motor1[1],motor2[1],motor3[1]])



    # Read in arrays saved at the end of the focus finder code.
    with open(sys.argv[1],'rb') as f:  
        xValues, fwhx1, fwhy1, xpos1, ypos1, spotLabels1,  originalFWHMx1, originalFWHMy1, EE = pickle.load(f)


    if plotBest:
        with open(sys.argv[2],'rb') as f:  
            xValuesBest, fwhxBest, fwhyBest, xposBest, yposBest, spotLabelsBest,  originalFWHMxBest, originalFWHMyBest, EEBest = pickle.load(f)


   
    
    # Current workign directory - needed later.
    cwd = os.getcwd()

    # Find the number of spots and number of files. Also need to define the 'name' of the spots which is simply the spot's XY coordinate. 
    numSpots = len(fwhx1[:,0])
    numFiles = len(fwhx1[0,:])
    print('No of frames: ', numFiles)
    
    xnames = np.zeros(numSpots)
    ynames = np.zeros(numSpots)
 
    for i in range(numSpots):
        xnames[i] = int(xpos1[i,1])
        ynames[i] = int(ypos1[i,1])


        
    #dam_zValues = np.arange(DAM_starts[0],DAM_starts[0]+(numFiles*increments),increments)
    dam_zValues = np.array(DAM_starts[0])
    for i in range(numFiles-1):
        if i == 0:
            dam_zValues = np.append(dam_zValues,dam_zValues+increments)
        else:
            dam_zValues = np.append(dam_zValues,dam_zValues[-1]+increments)

    # Check the dam_zValues are correct  
    print(dam_zValues)


    # Find the fits to the focus data - the main bit!
    bestFocusX,bestPeakX,bestFocusY,bestPeakY,bestxPos,bestyPos,yfitX,yfitY,zPoints = bestFocus(dam_zValues, xpos1, ypos1, originalFWHMx1, originalFWHMy1, motorXs, motorYs, fix=7, DAMstart=DAM_starts, increment=increments)

    # Find the fits to the EE data - the main bit!
    bestFocusEE,bestPeakEE,bestEEPos,yfitEE,zPointsEE = bestEE(dam_zValues, xpos1, ypos1, EE, motorXs, motorYs, fix=7, DAMstart=DAM_starts, increment=increments)


    nx=5  # Needed for plotting - crude again!!
    if savePlotsPDF:
        fileName = 'summaryPlots.pdf'
        pp = PdfPages(cwd+'/'+fileName)
   

    if plotBestFocus:
        plt.scatter(bestxPos,bestyPos,c = bestPeakY, s=80, vmin=1,vmax=4)
        plt.colorbar()
        plt.show()
        
  
    if gridSpots: 
        fig,grid = plt.subplots(nx, nx, figsize=(14,8))
        fig.subplots_adjust(top = 0.90, bottom = 0.05, right = 0.95, left = 0.05, hspace=.4, wspace=0.4)

        fig.suptitle('Best focus, X and Y', fontsize=12)
                  
        for i in range(numSpots):
            if i < nx:
                row,col = (nx-1)-i,0
            elif i>(nx-1) and i<(nx*2):
                row,col = (nx-1)-(i-nx),1
            elif i>((nx*2)-1) and i<(nx*3):
                row,col = (nx-1)-(i-(2*nx)),2
            elif i>((nx*3)-1) and i<(nx*4):
                row,col = (nx-1)-(i-(3*nx)),3
            elif i>((nx*4)-1) and i<(nx*5):
                row,col = (nx-1)-(i-(4*nx)),4


            # Sub-plot titles
            grid[row,col].set_title(str(int(xnames[i])) + ', ' + str(int(ynames[i]))) 
               

            # Plot the FWHM fitting data
            grid[row,col].plot(dam_zValues,originalFWHMx1[i,:],'blue')
            grid[row,col].plot(dam_zValues,originalFWHMy1[i,:],'red')

            # Overplot the fits to the data if wanted.
            #   grid[row,col].plot(dam_zValues,yfitX[i,:],'cyan')
            #   grid[row,col].plot(dam_zValues,yfitY[i,:],'pink')

            # If a best fit frame has been given, this will overlay the best fit, as dashed lines
            if plotBest:
                grid[row,col].plot([dam_zValues[0],dam_zValues[-1]],[fwhxBest[i],fwhxBest[i]],'blue',linestyle='--')
                grid[row,col].plot([dam_zValues[0],dam_zValues[-1]],[fwhyBest[i],fwhyBest[i]],'red',linestyle='--')
            
          
            # Plotting ranges
            grid[row,col].set_ylim([0,6])
            grid[row,col].set_xlim([dam_zValues[0],dam_zValues[-1]])

      
        plt.show()

        if savePlotsPDF:
            # close the figure to enable making the next page in the pdf
            pp.savefig(fig)
            plt.close()


    if movements:
        # Plot the motion of the spots.
        fig,grid = plt.subplots(nx, nx, figsize=(10,10))
        fig.subplots_adjust(top = 0.90, bottom = 0.05, right = 0.95, left = 0.05, hspace=.4, wspace=0.4)

        fig.suptitle('Spot movements', fontsize=12)
                  
        for i in range(numSpots):
            if i < nx:
                row,col = (nx-1)-i,0
            elif i>(nx-1) and i<(nx*2):
                row,col = (nx-1)-(i-nx),1
            elif i>((nx*2)-1) and i<(nx*3):
                row,col = (nx-1)-(i-(2*nx)),2
            elif i>((nx*3)-1) and i<(nx*4):
                row,col = (nx-1)-(i-(3*nx)),3
            elif i>((nx*4)-1) and i<(nx*5):
                row,col = (nx-1)-(i-(4*nx)),4



            grid[row,col].set_title(str(int(xnames[i])) + ', ' + str(int(ynames[i]))) 
               
           
            grid[row,col].scatter(xpos1[i,:], ypos1[i,:],c=np.arange(0,numFiles))

   
            bbb = 4   # Size of the box to plot around the data. Use same size for all.
            grid[row,col].set_xlim([np.median(xpos1[i,:])-bbb,np.median(xpos1[i,:])+bbb])
            grid[row,col].set_ylim([np.median(ypos1[i,:])-bbb,np.median(ypos1[i,:])+bbb])


      
        plt.show()

        if savePlotsPDF:
            # close the figure to enable making the next page in the pdf
            pp.savefig(fig)
            plt.close()





    if encircled:
        # Make plots for the encircled energy.
        fig,grid = plt.subplots(nx, nx, figsize=(14,8))
        fig.subplots_adjust(top = 0.90, bottom = 0.05, right = 0.95, left = 0.05, hspace=.4, wspace=0.4)


        fig.suptitle('Encircled energy: (3x3 box)/(7x7 box)', fontsize=12)
        
        for i in range(numSpots):
            if i < nx:
                row,col = (nx-1)-i,0
            elif i>(nx-1) and i<(nx*2):
                row,col = (nx-1)-(i-nx),1
            elif i>((nx*2)-1) and i<(nx*3):
                row,col = (nx-1)-(i-(2*nx)),2
            elif i>((nx*3)-1) and i<(nx*4):
                row,col = (nx-1)-(i-(3*nx)),3
            elif i>((nx*4)-1) and i<(nx*5):
                row,col = (nx-1)-(i-(4*nx)),4



            grid[row,col].set_title(str(int(xnames[i])) + ', ' + str(int(ynames[i]))) 
               
           
            grid[row,col].plot(dam_zValues,EE[i,:],'blue')
            grid[row,col].plot(dam_zValues,yfitEE[i,:],'cyan')


            if plotBest:
                grid[row,col].plot([dam_zValues[0],dam_zValues[-1]],[EEBest[i],EEBest[i]],'blue',linestyle='--')
          

            grid[row,col].set_ylim([0,1])
            grid[row,col].set_xlim([dam_zValues[0],dam_zValues[-1]])

            #grid[row,col].set_xlim([-0.3,dam_zValues[-1]])


      
        plt.show()


        if savePlotsPDF:
            # close the figure to enable making the next page in the pdf
            pp.savefig(fig)
            plt.close()


    if plotBest:
        fig,grid = plt.subplots(1, 1, figsize=(5,5))
        fig.subplots_adjust(top = 0.90, bottom = 0.05, right = 0.95, left = 0.05, hspace=.4, wspace=0.4)

        bestArray = np.reshape(fwhxBest, (5,5))
        print(bestArray)
        print(fwhxBest)
        #for i in range(5):
        #    print(np.transpose(fwhxBest[i*5:((i+1)*5)]))
        #    bestArray[i,:] = np.transpose(fwhxBest[i*5:(i+1*5)])
        grid.imshow(bestArray)
        plt.suptitle('Measured FWHM in x of best plane')
        #plt.colorbar()
        plt.show()

        if savePlotsPDF:
            # close the figure to enable making the next page in the pdf
            pp.savefig(fig)
            plt.close()



    ## Very simple way to remove any bad spots from the fit to the plane. 
    removeBad = [15,20]
      
    if len(removeBad) > 0:
        print('cleanup',bestFocusX[removeBad[0]])
        for b in range(len(removeBad)):
            bestFocusX[removeBad[b]] = np.median(bestFocusX)
            bestFocusY[removeBad[b]] = np.median(bestFocusY)




    # Create regular grid covering the domain of the data
    X,Y = np.meshgrid(np.arange(-300, 300, 10), np.arange(-300, 300, 10))
    XX = X.flatten()
    YY = Y.flatten()


    # Fit the data to the plane - the key bit!!
    print('FWHM X Best:')
    Z1,C1,motorZ1 = findPlane(bestxPos,bestyPos,bestFocusX,motorXs,motorYs,X,Y)
    print('FWHM Y Best:')
    Z1,C1,motorZ1 = findPlane(bestxPos,bestyPos,bestFocusY,motorXs,motorYs,X,Y)


    # Merge the X and Y fits to get an average frame for best FWHM in X and Y, if wanted. 
    bestXX = np.concatenate((bestxPos,bestxPos),axis=0)
    bestYY = np.concatenate((bestyPos,bestyPos),axis=0)
    bestFocusXY = np.concatenate((bestFocusX,bestFocusY),axis=0)

    print('FWHM averaged:')
    #Z8,C8,motorZ8 = findPlane(bestXX,bestYY,bestFocusXY,motorXs,motorYs,X,Y)


    # Find the best fit plane for the Encircled Energy plane
    print('Encircled')
    Z1,C1,motorZ1 = findPlane(bestxPos,bestyPos,bestFocusEE,motorXs,motorYs,X,Y)


    # As a test can create a sub array of just the corner points and fit against those.
    idx = [0,4,20,24]
    subX = np.array([bestxPos[idx[0]],bestxPos[idx[1]],bestxPos[idx[2]],bestxPos[idx[3]]])
    subY = np.array([bestyPos[idx[0]],bestyPos[idx[1]],bestyPos[idx[2]],bestyPos[idx[3]]])
    subEE = np.array([bestFocusEE[idx[0]],bestFocusEE[idx[1]],bestFocusEE[idx[2]],bestFocusEE[idx[3]]])

    print('Corners encircled')
    Z1,C1,motorZ1 = findPlane(subX,subY,subEE,motorXs,motorYs,X,Y)


    
    # Useful coordinates for plotting
    de = 31.0 # Detector size (hald length)
    dposX = np.array([-de,de,de,-de,-de])
    dposY = np.array([-de,-de,de,de,-de])
    dposZ = C1[0]*dposX + C1[1]*dposY + C1[2]

    ringX = 263.5*np.cos(np.arange(0,6.5,0.1))
    ringY = 263.5*np.sin(np.arange(0,6.5,0.1))
    ringZ = C1[0]*ringX + C1[1]*ringY + C1[2]

    # plot points and fitted surface
    if fittedPlane:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax = fig.gca()

        # Plot surface
        ax.plot_surface(X, Y, Z1, rstride=1, cstride=1, alpha=0.2)
        ax.scatter(bestxPos, bestyPos, bestFocusX, c=bestPeakX, s=20)
        #ax.scatter(subX,subY,subEE, c=subEE, s=20)


        # Plot detector edge and DAM ring outline
        ax.plot(dposX, dposY, dposZ, c='black',linewidth=0.5)
        ax.plot(ringX, ringY, ringZ, c='black',linewidth=0.5)

        # Plot motor locations
        ax.scatter(motorXs,motorYs,motorZ1,c='red',s=50)
        
        plt.xlabel('X')
        plt.ylabel('Y')
        ax.set_zlabel('Focus')
        ax.set_zlim3d(0.4,0.6)

        lims = -40,40
        lims = -250,250
        ax.set_xlim3d(lims)
        ax.set_ylim3d(lims)
        ax.set_zlim3d(lims)
        ax.axis('auto')
        #ax.axis('tight')
        plt.show()


    if savePlotsPDF:
        # Close the thumbnails file
        pp.close()




    


main()

