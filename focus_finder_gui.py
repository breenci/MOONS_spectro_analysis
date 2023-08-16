from mpl_point_clicker import clicker
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.widgets import Cursor, Slider, Button, CheckButtons
from matplotlib.gridspec import GridSpec
import numpy as np
import glob


class pointSelectGUI():
    """Matplotlib GUI for point selection"""
    
    def __init__(self, fn_list, point_file=None, DAM_positions=None, box_size=30):
        self.fn_list = fn_list
        self.point_file = point_file
        self.selection = None
        self.DAM_positions = DAM_positions
        self.box_size = box_size
        
                
    def run(self):
        print("Running GUI", self.fn_list[0])
        self.arr_list = read_files(self.fn_list)
        self.current_arr = self.arr_list[0]
        
        # Create a figure with grid
        self.fig = plt.figure(figsize=(12,8))
        grid = GridSpec(7, 4)
        imax = self.fig.add_subplot(grid[:5,:3])
        self.im = imax.imshow(self.arr_list[0], vmin=0, vmax=100)
        cursor = Cursor(imax, useblit=True, color='red', linewidth=.5)
        imax.set_aspect('auto')
        
        self.klicker = clicker(imax, ['Selected Points'], markers=['x'], 
                               colors=['red'], legend_loc='upper center')
        
        # Load preselected points
        if self.point_file != None:
            try:
                preselection = np.loadtxt(self.point_file)
                self.klicker.set_positions({"Selected Points":preselection})
                self.klicker._update_points()
            except Exception as e:
                print(f"Error: Unable to load the file '{self.point_file}'.")
                print(f"Error message: {e}")
        
        # add a subplot to show most recent point
        subax = self.fig.add_subplot(grid[:5, 3])
        self.subim = subax.imshow(np.array(np.zeros((self.box_size, self.box_size))), vmin=0, vmax=100)
        # turn off axis
        subax.axis('off')
        self.klicker.on_point_added(self._on_points_changed)      
        
        
        # add slider for file selection
        # If DAM positions are given, add slider for DAM position selection
        if self.DAM_positions != None:
            DAM_slider_ax = self.fig.add_subplot(grid[5, :])
            DAM_slider = Slider(ax=DAM_slider_ax, valmin=0, valmax=len(self.DAM_positions), valstep=1,
                                label='DAM')
            DAM_slider.on_changed(self._slider_update)
        
        else:
            slideax = self.fig.add_subplot(grid[5, :])
            fn_slider = Slider(ax=slideax, valmin=0, valmax=len(self.fn_list), valstep=1,
                            label='File Index')
            fn_slider.on_changed(self._slider_update)
        
        # add button to confim selection
        bax = self.fig.add_subplot(grid[6, :3])
        brun = Button(bax, 'Run Analysis and Close')
        brun.on_clicked(self._run_button_callback)
        
        # Add check box to save points
        bax = self.fig.add_subplot(grid[6, 3])
        bsave = Button(bax, 'Save Points')
        bsave.on_clicked(self._save_button_callback)
        plt.show()
    
    def _slider_update(self, val):
        print(self.fn_list[val])
        self.im.set_data(self.arr_list[val])
        self.fig.canvas.draw_idle()
        self.current_arr = self.arr_list[val]
            
    def _run_button_callback(self, event):
        self.selection = self.klicker.get_positions()
        plt.close()
        
    def _save_button_callback(self, event):
        pos = self.klicker.get_positions()
        np.savetxt('points.txt', pos["Selected Points"])
        
    def _on_points_changed(self, position, klass):
        print(f"Point added: {position}")
        box = self._get_box(position)
        self.subim.set_data(box)
        self.fig.canvas.draw_idle()

    
    def _get_box(self, point):
        """Return a box around the given point"""
        x, y = point
        x = int(x)
        y = int(y)
        box = self.current_arr[y-self.box_size:y+self.box_size, x-self.box_size:x+self.box_size]
        return box

        
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
