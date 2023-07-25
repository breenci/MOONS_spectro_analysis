import wx
import numpy as np
from astropy.io import fits
from PIL import Image
import sys
import os

class MyCanvas(wx.ScrolledWindow):
    def __init__(self, parent, id = -1, size = wx.DefaultSize, filepath = None):
        wx.ScrolledWindow.__init__(self, parent, id, (0, 0), size=size, style=wx.SUNKEN_BORDER)
        
        openFile = fits.open(filepath)
        dataArray = np.array(openFile[0].data)
#        dataArray = np.array(openFile[1].data)
        
        dataArray = np.where(dataArray < 0, 0.1, dataArray)
        dataArray = np.log(dataArray)
        dataArray = 256*dataArray/np.amax(dataArray)
#        dataArray = ((256*dataArray/np.amax(dataArray))-200)*10

        cwd = os.getcwd()
        
        holdName = cwd + "/your_file.bmp"

        im = Image.fromarray(dataArray)
        im = im.convert("L")
        im.save(holdName)

        self.image = wx.Image(holdName)
        
        self.w = self.image.GetWidth()
        self.h = self.image.GetHeight()
        self.bmp = wx.BitmapFromImage(self.image)
        
        #self.a = np.arange(10)
        
        self.SetVirtualSize((self.w, self.h))
        self.SetScrollRate(20,20)
        self.SetBackgroundColour(wx.Colour(0,0,0))
        
        self.buffer = wx.EmptyBitmap(self.w, self.h)
        dc = wx.BufferedDC(None, self.buffer)
        dc.SetBackground(wx.Brush(self.GetBackgroundColour()))
        dc.Clear()
        self.DoDrawing(dc)
        
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_LEFT_UP, self.OnClickL)
        self.Bind(wx.EVT_RIGHT_UP, self.OnClickR)

        self.xpos = []
        self.ypos = []
        self.directions = []
        self.increment = 0
    
    def OnClickL(self, event):
        pos = self.CalcUnscrolledPosition(event.GetPosition())
        if len(self.xpos) > 0:
            if np.sqrt((self.xpos[-1]-pos.x)**2 + (self.ypos[-1]-pos.y)**2) > 1:
                self.xpos.append(pos.x)
                self.ypos.append(pos.y)
                self.directions.append(1)
            else:
                self.directions[-1] = self.directions[-1] + 1
        else:
            self.xpos.append(pos.x)
            self.ypos.append(pos.y)
            self.directions.append(1)

        print('-------------------------')
        print("       xCentres = ",self.xpos)
        print("       yCentres = ",self.ypos)
        print("       orientations = ",self.directions)

    def OnClickR(self, event):
        pos = self.CalcUnscrolledPosition(event.GetPosition())
        if len(self.xpos) > 0:
            if np.sqrt((self.xpos[-1]-pos.x)**2 + (self.ypos[-1]-pos.y)**2) > 1:
                self.xpos.append(pos.x)
                self.ypos.append(pos.y)
                self.directions.append(3)
            else:
                self.directions[-1] = self.directions[-1] + 1
        else:
            self.xpos.append(pos.x)
            self.ypos.append(pos.y)
            self.directions.append(1)
        
        print('-------------------------')
        print("       xboxCentres = ",self.xpos)
        print("       yboxCentres = ",self.ypos)
        #print("       orientations = ",self.directions)

    def OnPaint(self, event):
        dc = wx.BufferedPaintDC(self, self.buffer, wx.BUFFER_VIRTUAL_AREA)
    
    def DoDrawing(self, dc):
        dc.DrawBitmap(self.bmp, 0, 0)

class MyFrame(wx.Frame):
    def __init__(self, parent=None, id=-1, filepath = None):
        wx.Frame.__init__(self, parent, id, title=filepath)
        self.canvas = MyCanvas(self, -1, filepath = filepath)
        
        self.canvas.SetMinSize((self.canvas.w, self.canvas.h))
        self.canvas.SetMaxSize((self.canvas.w, self.canvas.h))
        self.canvas.SetBackgroundColour(wx.Colour(0, 0, 0))
        vert = wx.BoxSizer(wx.VERTICAL)
        horz = wx.BoxSizer(wx.HORIZONTAL)
        vert.Add(horz,0, wx.EXPAND,0)
        vert.Add(self.canvas,1,wx.EXPAND,0)
        self.SetSizer(vert)
        vert.Fit(self)
        self.Layout()


if __name__ == '__main__':
    app = wx.App()
    app.SetOutputWindowAttributes(title='stdout')
    wx.InitAllImageHandlers()
    cwd = os.getcwd()
    
    #filepath = '/Users/wdt/ERIS/testing/distortionSpots/Dist Mask 3-1.bmp'
    #filepath = '/Users/wdt/ERIS/testing/coolDown6/edgeFocus/ERIS_NIXIMG_TEC_LABCHECKINTERNALFOCUS235_0025.fits'
    #filepath = '/Users/wdt/ERIS/testing/coolDown7/edgeTests/angles/' + sys.argv[1]
    #filepath = '/Users/wdt/ERIS/testing/coolDown7/pupils/' + sys.argv[1]
    filepath = cwd + '/' + sys.argv[1]
    if filepath:
        print(filepath)
        myframe = MyFrame(filepath=filepath)
        myframe.Center()
        myframe.Show()
        app.MainLoop()
        print('Finished')
