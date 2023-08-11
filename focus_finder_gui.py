from mpl_point_clicker import clicker
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.widgets import Cursor, Slider, Button
from matplotlib.gridspec import GridSpec
import numpy as np
import glob


class pointSelectGUI():
    """Matplotlib GUI for point selection"""
    
    def __init__(self, fn_list, point_file=None):
        self.fn_list = fn_list
        self.point_file = point_file
        self.selection = None
        
    def run(self):
        print("Running GUI: FN = ", self.fn_list[0])
        self.arr_list = read_files(self.fn_list)
        
        # Create a figure with grid
        self.fig = plt.figure(figsize=(8,8))
        grid = GridSpec(7, 4)
        imax = self.fig.add_subplot(grid[:4,:])
        self.im = imax.imshow(self.arr_list[0], vmin=0, vmax=1000)
        cursor = Cursor(imax, useblit=True, color='red', linewidth=2)
        imax.set_aspect('auto')
        
        self.klicker = clicker(imax, ['Selected Points'], markers=['x'])
        
        # Load preselected points
        if self.point_file != None:
            try:
                preselection = np.loadtxt(self.point_file)
                self.klicker.set_positions({"Selected Points":preselection})
                self.klicker._update_points()
            except Exception as e:
                print(f"Error: Unable to load the file '{self.point_file}'.")
                print(f"Error message: {e}")
            
        # add slider for file selection
        slideax = self.fig.add_subplot(grid[5, :])
        fn_slider = Slider(ax=slideax, valmin=0, valmax=len(self.fn_list), valstep=1,
                           label='Slide')
        fn_slider.on_changed(self._slider_update)
        
        # add button to confim selection
        bax = self.fig.add_subplot(grid[6, :])
        bsave = Button(bax, 'Save and Run')
        bsave.on_clicked(self._button_callback)
        
        plt.show()
    
    def _slider_update(self, val):
        print(self.fn_list[val])
        self.im.set_data(self.arr_list[val])
        self.fig.canvas.draw_idle()
        
    def _button_callback(self, event):
        self.selection = self.klicker.get_positions()
        np.savetxt('selected_points.txt', self.selection['Selected Points'])
        plt.close()
    
        
        


def read_files(fn_list):
    """Read MOONS frame fits file and output frame as array"""

    arr_list = [] # list to store arrays
    
    for fn in fn_list:
        with fits.open(fn) as hdul:
            frame_data = hdul[0].data
        arr_list.append(frame_data)
    
    return arr_list

        
            
if __name__ == '__main__':
    fn_list = glob.glob('data/raw/test_3A.01.05/test*.fits')
    print(fn_list)
    gui = pointSelectGUI(fn_list)
    gui.run()
    print(gui.selection)
