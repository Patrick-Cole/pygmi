# -----------------------------------------------------------------------------
# Name:        change.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2019 Council for Geoscience
# Licence:     GPL-3.0
#
# This file is part of PyGMI
#
# PyGMI is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyGMI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------
"""Change Detection."""

import datetime
from pathlib import Path
from xml.etree import ElementTree
from PyQt5 import QtWidgets, QtCore
import numpy as np
import pandas as pd
import geopandas as gpd
from osgeo import gdal
from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import colormaps
import matplotlib.animation as manimation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT

from pygmi.raster.datatypes import Data
from pygmi.misc import frm
from pygmi.raster.ginterp import histcomp, norm255
from pygmi.misc import BasicModule


class CreateSceneList(BasicModule):
    """
    Create Scene List.

    This class creates a list of scenes for use in change detection.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.indata = {'tmp': True}

        self.shapefile = QtWidgets.QLineEdit('')
        self.scenefile = QtWidgets.QLineEdit('')
        self.isrecursive = QtWidgets.QCheckBox('Recursive file search')
        self.useall = QtWidgets.QCheckBox('Use all scenes')

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gridlayout_main = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        # helpdocs = menu_default.HelpButton('pygmi.rsense.change.cscene')
        pb_shape = QtWidgets.QPushButton('Load shapefile or kml file')
        pb_scene = QtWidgets.QPushButton('Set scene directory')

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle(r'Create Scene List')

        gridlayout_main.addWidget(self.shapefile, 0, 0, 1, 1)
        gridlayout_main.addWidget(pb_shape, 0, 1, 1, 1)

        gridlayout_main.addWidget(self.scenefile, 1, 0, 1, 1)
        gridlayout_main.addWidget(pb_scene, 1, 1, 1, 1)

        gridlayout_main.addWidget(self.useall, 2, 0, 1, 2)
        gridlayout_main.addWidget(self.isrecursive, 3, 0, 1, 2)

        gridlayout_main.addWidget(buttonbox, 5, 1, 1, 3)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        pb_shape.pressed.connect(self.get_shape)
        pb_scene.pressed.connect(self.get_scene)

    def settings(self, nodialog=False):
        """
        Entry point into item.

        Parameters
        ----------
        nodialog : bool, optional
            Run settings without a dialog. The default is False.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        if not nodialog:
            tmp = self.exec_()

            if tmp != 1:
                return tmp

        idir = self.scenefile.text()
        sfile = self.shapefile.text()

        if idir == '' or sfile == '':
            return False

        if not self.useall.isChecked():
            if sfile[-3:] == 'shp':
                ddpoints = get_shape_coords(sfile)
            else:
                ddpoints = get_kml_coords(sfile)
            ddpoints2 = Polygon(ddpoints)

        if self.isrecursive.isChecked():
            subfiles = Path(idir).rglob('*.tif')
        else:
            subfiles = Path(idir).glob('*.tif')

        subfiles = list(subfiles)
        dtime = []
        flist = []
        nodates = False
        for ifile in self.piter(subfiles):
            dataset = gdal.Open(str(ifile), gdal.GA_ReadOnly)
            metadata = dataset.GetMetadata()

            if not self.useall.isChecked():
                gtr = dataset.GetGeoTransform()
                cols = dataset.RasterXSize
                rows = dataset.RasterYSize
                dxlim = (gtr[0], gtr[0]+gtr[1]*cols)
                dylim = (gtr[3]+gtr[5]*rows, gtr[3])

                coords = [[dxlim[0], dylim[0]],
                          [dxlim[0], dylim[1]],
                          [dxlim[1], dylim[1]],
                          [dxlim[1], dylim[0]],
                          [dxlim[0], dylim[0]]]

                coords2 = Polygon(coords)
                if not coords2.contains(ddpoints2):
                    continue

            if 'TIFFTAG_DATETIME' not in metadata:
                dt = datetime.datetime(1900, 1, 1)
                nodates = True
            else:
                dtimestr = metadata['TIFFTAG_DATETIME']
                dt = datetime.datetime.strptime(dtimestr, '%Y:%m:%d %H:%M:%S')

            dtime.append(dt)
            flist.append(ifile)

        if nodates is True:
            self.showprocesslog('Some of your scenes do not have dates. '
                                'Correct this in the output spreadsheet')

        if not flist:
            self.showprocesslog('No scenes could be found. Please make sure '
                                'that your shapefile or kml file is in the '
                                'area of your scenes and in the same '
                                'projection.')
            return False

        self.showprocesslog('Updating spreadsheet...')

        df = pd.DataFrame()
        df['Datetime'] = dtime
        df['Filename'] = flist
        df['Use'] = True
        df['Shapefile'] = sfile

        df.sort_values('Datetime', inplace=True)

        self.outdata['SceneList'] = df

        self.showprocesslog('Saving to disk...')

        ext = ('Scene List File (*.xlsx)')

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.parent, 'Save File', '.', ext)
        if filename == '':
            return False

        df.to_excel(filename, index=False)

        return True

    def loadproj(self, projdata):
        """
        Load project data into class.

        Parameters
        ----------
        projdata : dictionary
            Project data loaded from JSON project file.

        Returns
        -------
        chk : bool
            A check to see if settings was successfully run.

        """
        self.shapefile.setText(projdata['shapefile'])
        self.scenefile.setText(projdata['scenefile'])
        self.isrecursive.setChecked(projdata['recursive'])

        chk = self.settings(True)

        return chk

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        projdata : dictionary
            Project data to be saved to JSON project file.

        """
        projdata = {}

        projdata['shapefile'] = self.shapefile.text()
        projdata['scenefile'] = self.scenefile.text()
        projdata['recursive'] = self.isrecursive.isChecked()

        return projdata

    def get_shape(self, filename=''):
        """
        Get shape filename.

        Parameters
        ----------
        filename : str, optional
            Input filename. The default is ''.

        Returns
        -------
        None.

        """
        ext = ('shapefile or kml file (*.shp *.kml)')

        if filename == '':
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                self.parent, 'Open File', '.', ext)
            if filename == '':
                return

        self.shapefile.setText(filename)

    def get_scene(self, directory=''):
        """
        Get Scene Directory.

        Parameters
        ----------
        directory : str, optional
            Directory path as a string. The default is ''.

        Returns
        -------
        None.

        """
        if directory == '':
            directory = QtWidgets.QFileDialog.getExistingDirectory(
                self.parent, 'Select Directory')

            if directory == '':
                return

        self.scenefile.setText(directory)


class LoadSceneList(BasicModule):
    """Load scene list."""

    def __init__(self, parent=None):
        super().__init__(parent)

    def settings(self, nodialog=False):
        """
        Entry point into item.

        Parameters
        ----------
        nodialog : bool, optional
            Run settings without a dialog. The default is False.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        if not nodialog:
            ext = 'Scene List File (*.xlsx)'

            self.ifile, _ = QtWidgets.QFileDialog.getOpenFileName(
                self.parent, 'Open Scene List Spreadsheet', '.', ext)
            if self.ifile == '':
                return False

        df = pd.read_excel(self.ifile)

        self.outdata['SceneList'] = df
        return True

    def loadproj(self, projdata):
        """
        Load project data into class.

        Parameters
        ----------
        projdata : dictionary
            Project data loaded from JSON project file.

        Returns
        -------
        chk : bool
            A check to see if settings was successfully run.

        """
        self.ifile = projdata['ifile']

        chk = self.settings(True)

        return chk

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        projdata : dictionary
            Project data to be saved to JSON project file.

        """
        projdata = {}

        projdata['ifile'] = self.ifile

        return projdata


class MyMplCanvas(FigureCanvasQTAgg):
    """Simple canvas with a sine plot."""

    def __init__(self, parent=None, width=10, height=8, dpi=100,
                 bands=(0, 1, 2)):
        self.fig = Figure(figsize=(width, height), dpi=dpi)

        self.ax1 = self.fig.add_subplot(111)
        self.im1 = None
        self.bands = bands
        self.parent = parent
        self.rcid = None
        self.manip = 'RGB'
        self.cbar = None
        self.capture_active = False
        self.writer = None

        super().__init__(self.fig)

        self.setParent(parent)

        FigureCanvasQTAgg.setSizePolicy(self,
                                        QtWidgets.QSizePolicy.Expanding,
                                        QtWidgets.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)

        self.fig.canvas.mpl_connect('button_release_event', self.onClick)

    def capture(self):
        """
        Capture.

        Returns
        -------
        None.

        """
        self.capture_active = not self.capture_active

        if self.capture_active:
            ext = ('GIF (*.gif)')
            wfile, _ = QtWidgets.QFileDialog.getSaveFileName(
                self.parent, 'Save File', '.', ext)
            if wfile == '':
                self.capture_active = not self.capture_active
                return

            self.writer = manimation.PillowWriter(fps=4)
            self.writer.setup(self.fig, wfile)
        else:
            self.writer.finish()

    def compute_initial_figure(self, dat, dates, points):
        """
        Compute initial figure.

        Parameters
        ----------
        dat : PyGMI Data
            PyGMI dataset.
        dates : str
            Dates to show on title.
        points : numpy array
            Points to plot.

        Returns
        -------
        None.

        """
        extent = []

        rtmp1 = dat[self.bands[2]].data
        rtmp2 = dat[self.bands[1]].data
        rtmp3 = dat[self.bands[0]].data

        rtmp1 = (rtmp1-rtmp1.min())/rtmp1.ptp()
        rtmp2 = (rtmp2-rtmp2.min())/rtmp2.ptp()
        rtmp3 = (rtmp3-rtmp3.min())/rtmp3.ptp()

        alpha = np.logical_not(rtmp1 == 0.)

        dtmp = np.array([rtmp1, rtmp2, rtmp3, alpha])
        dtmp = np.moveaxis(dtmp, 0, 2)
        dtmp = dtmp*255
        dtmp = dtmp.astype(np.uint8)

        extent = dat[self.bands[0]].extent

        self.im1 = self.ax1.imshow(dtmp, extent=extent)
        self.ax1.plot(points[:, 0], points[:, 1])
        self.cbar = None

        self.fig.suptitle(dates)

    def update_plot(self, dat, dates):
        """
        Update plot.

        Parameters
        ----------
        dat : PyGMI Data
            PyGMI dataset.
        dates : str
            Dates to show on title.

        Returns
        -------
        None.

        """
        extent = dat[self.bands[0]].extent
        if self.manip == 'NDWI':
            if self.cbar is None:
                self.cbar = self.figure.colorbar(self.im1, format=frm)
            green = np.ma.masked_equal(dat[2].data, 0)
            nir = np.ma.masked_equal(dat[4].data, 0)
            green = green.astype(float)
            nir = nir.astype(float)
            dtmp = (green-nir)/(green+nir)
            self.im1.set_clim(-1, 1)
            self.im1.set_clim(-1, 1)
            self.im1.set_cmap(colormaps['PiYG_r'])
        elif self.manip == 'NDVI':
            if self.cbar is None:
                self.cbar = self.figure.colorbar(self.im1, format=frm)
            red = np.ma.masked_equal(dat[3].data, 0)
            nir = np.ma.masked_equal(dat[4].data, 0)
            red = red.astype(float)
            nir = nir.astype(float)
            dtmp = (nir-red)/(nir+red)
            self.im1.set_clim(-1, 1)
            self.im1.set_clim(-1, 1)
            self.im1.set_cmap(colormaps('PiYG'))
        else:
            if self.cbar is not None:
                self.cbar.remove()
                self.cbar = None

            mask = (dat[self.bands[2]].data == 0.)
            red = np.ma.array(dat[self.bands[2]].data, mask=mask)
            green = np.ma.array(dat[self.bands[1]].data, mask=mask)
            blue = np.ma.array(dat[self.bands[0]].data, mask=mask)

            red = histcomp(red, nbr_bins=10000, perc=2.)
            green = histcomp(green, nbr_bins=10000, perc=2.)
            blue = histcomp(blue, nbr_bins=10000, perc=2.)

            red = norm255(red)
            green = norm255(green)
            blue = norm255(blue)

            red[mask] = 0
            green[mask] = 0
            blue[mask] = 0

            alpha = ~mask
            alpha = alpha*255

            dtmp = np.array([red, green, blue, alpha])
            dtmp = np.moveaxis(dtmp, 0, 2)
            dtmp = dtmp.astype(np.uint8)
            self.im1.set_clim(0, 255)
            self.im1.set_clim(0, 255)

        self.im1.set_data(dtmp)
        self.im1.set_extent(extent)
        self.fig.suptitle(dates)
        self.ax1.xaxis.set_major_formatter(frm)
        self.ax1.yaxis.set_major_formatter(frm)

        self.fig.canvas.draw()

    def onClick(self, event):
        """
        On click event.

        Parameters
        ----------
        event : TYPE
            Unused.

        Returns
        -------
        None.

        """
        self.rcid = self.fig.canvas.mpl_connect('draw_event', self.redraw)

    def redraw(self, event):
        """
        Redraw event.

        Parameters
        ----------
        event : TYPE
            Unused.

        Returns
        -------
        None.

        """
        self.fig.canvas.mpl_disconnect(self.rcid)
        self.parent.newdata(self.parent.curimage)


class SceneViewer(BasicModule):
    """Scene Viewer."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.df = None
        self.pbar2 = QtWidgets.QProgressBar()

        self.setWindowTitle("View Change Data")

        self.file_menu = QtWidgets.QMenu('&File', self)
        self.help_menu = QtWidgets.QMenu('&Help', self)

        self.help_menu.addAction('&About', self.about)
        self.file_menu.addAction('&Quit', self.fileQuit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)

        vlayout = QtWidgets.QVBoxLayout(self)
        hlayout = QtWidgets.QHBoxLayout()
        hlayout2 = QtWidgets.QHBoxLayout()

        self.canvas = MyMplCanvas(self, width=5, height=4, dpi=100)

        mpl_toolbar = NavigationToolbar2QT(self.canvas, self)
        self.slider = QtWidgets.QScrollBar(QtCore.Qt.Horizontal)
        self.button1 = QtWidgets.QPushButton('Start Capture')
        self.button2 = QtWidgets.QPushButton('Update Scene List File')
        self.button3 = QtWidgets.QPushButton('Next Scene')
        self.cb_use = QtWidgets.QCheckBox('Use Scene')
        self.cb_display = QtWidgets.QCheckBox('Only Display Scenes Flagged '
                                              'for Use')
        self.cb_display.setChecked(True)
        self.manip = QtWidgets.QComboBox()

        actions = ['RGB', 'NDVI', 'NDWI']
        self.manip.addItems(actions)

        hlayout2.addWidget(QtWidgets.QLabel('Band Manipulation:'))
        hlayout2.addWidget(self.manip)
        hlayout.addWidget(self.button3)
        hlayout.addWidget(self.button2)
        hlayout.addWidget(self.button1)
        vlayout.addWidget(self.canvas)
        vlayout.addWidget(mpl_toolbar)
        vlayout.addWidget(self.slider)
        vlayout.addWidget(self.cb_display)
        vlayout.addWidget(self.cb_use)
        vlayout.addLayout(hlayout2)
        vlayout.addLayout(hlayout)
        vlayout.addWidget(self.pbar2)

        self.curimage = 0

        mpl_toolbar.actions()[0].triggered.connect(self.home_callback)
        self.slider.valueChanged.connect(self.newdata)
        self.cb_use.stateChanged.connect(self.flaguse)
        self.button2.clicked.connect(self.updateanim)
        self.button3.clicked.connect(self.nextscene)
        self.button1.clicked.connect(self.capture)
        self.manip.currentIndexChanged.connect(self.manip_change)

    def settings(self, nodialog=False):
        """
        Entry point into item.

        Parameters
        ----------
        nodialog : bool, optional
            Run settings without a dialog. The default is False.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        if 'SceneList' not in self.indata:
            self.showprocesslog('You need a scene list.')
            return False

        self.df = self.indata['SceneList']
        sfile = self.df['Shapefile'][0]

        dates = self.df.Datetime[self.curimage]
        dat = self.get_tiff(self.df.Filename[self.curimage], firstrun=True)
        points = get_shape_coords(sfile, False)
        self.slider.setMaximum(len(self.df)-1)
        self.cb_use.setChecked(bool(self.df.Use[self.curimage]))

        self.canvas.bands = list(dat.keys())

        self.canvas.compute_initial_figure(dat, dates, points)

        if not nodialog:
            tmp = self.exec_()

            if tmp != 1:
                return tmp
        return True

    def loadproj(self, projdata):
        """
        Load project data into class.

        Parameters
        ----------
        projdata : dictionary
            Project data loaded from JSON project file.

        Returns
        -------
        chk : bool
            A check to see if settings was successfully run.

        """
        return False

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        projdata : dictionary
            Project data to be saved to JSON project file.

        """
        projdata = {}

        return projdata

    def manip_change(self, event):
        """
        Change manipulation.

        Parameters
        ----------
        event : TYPE
            Unused.

        Returns
        -------
        None.

        """
        self.canvas.manip = self.manip.currentText()
        self.newdata(self.curimage)

    def updateanim(self, event):
        """
        Update animation file.

        Parameters
        ----------
        event : TYPE
            Unused.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        ext = ('Scene List File (*.xlsx)')

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.parent, 'Save File', '.', ext)
        if filename == '':
            return False

        self.df.to_excel(filename, index=False)

        return True

    def nextscene(self, event):
        """
        Get next scene.

        Parameters
        ----------
        event : TYPE
            Unused.

        Returns
        -------
        None.

        """
        self.slider.setValue(self.slider.value()+1)

    def flaguse(self, event):
        """
        Flag the scene for use.

        Parameters
        ----------
        event : TYPE
            Unused.

        Returns
        -------
        None.

        """
        self.df.loc[self.curimage, 'Use'] = self.cb_use.isChecked()

    def home_callback(self, event):
        """
        Home callback.

        Parameters
        ----------
        event : TYPE
            Unused.

        Returns
        -------
        None.

        """
        self.newdata(self.curimage)

    def newdata(self, indx, capture=False):
        """
        Get new dataset.

        Parameters
        ----------
        indx : int
            Current index.
        capture : bool, optional
            Option to capture the scene. The default is False.

        Returns
        -------
        None.

        """
        if not self.df.Use[indx] and self.cb_display.isChecked():
            if capture is False:
                return
        else:
            self.curimage = indx

        dates = self.df.Datetime[indx]
        dat = self.get_tiff(self.df.Filename[self.curimage])
        self.cb_use.setChecked(bool(self.df.Use[self.curimage]))

        self.canvas.update_plot(dat, dates)

    def capture(self):
        """Capture."""
        self.slider.valueChanged.disconnect()

        self.canvas.capture()
        for indx in self.df.index:
            self.slider.setValue(indx)
            self.newdata(indx, capture=True)
            self.canvas.writer.grab_frame()

        self.canvas.capture()
        self.slider.valueChanged.connect(self.newdata)
        self.slider.setValue(self.curimage)

    def fileQuit(self):
        """
        File quit.

        Returns
        -------
        None.

        """
        self.close()

    def closeEvent(self, cevent):
        """
        Close event.

        Parameters
        ----------
        cevent : TYPE
            Unused.

        Returns
        -------
        None.

        """
        self.fileQuit()

    def about(self):
        """
        About.

        Returns
        -------
        None.

        """
        QtWidgets.QMessageBox.about(self, "About",
                                    """Timeseries Plot""")

    def get_tiff(self, ifile, firstrun=False):
        """
        Get TIFF images.

        Parameters
        ----------
        ifile : str
            Filename to import.
        firstrun : bool, optional
            Option for first time running this routine. The default is False.

        Returns
        -------
        datall : dictionary or None
            Data images

        """
        datall = {}
        ifile = str(ifile)

        dataset = gdal.Open(ifile, gdal.GA_ReadOnly)

        self.pbar2.setMinimum(0)
        self.pbar2.setValue(0)
        self.pbar2.setMaximum(dataset.RasterCount-1)

        gtr = dataset.GetGeoTransform()
        cols = dataset.RasterXSize
        rows = dataset.RasterYSize
        dx = abs(gtr[1])
        dy = abs(gtr[5])
        dxlim = (gtr[0], gtr[0]+gtr[1]*cols)
        dylim = (gtr[3]+gtr[5]*rows, gtr[3])

        axes = self.canvas.ax1

        if firstrun is True:
            axes.set_xlim(dxlim[0], dxlim[1])
            axes.set_ylim(dylim[0], dylim[1])

        ext = (axes.transAxes.transform([(1, 1)]) -
               axes.transAxes.transform([(0, 0)]))[0]

        xlim, ylim = axes.get_xlim(), axes.get_ylim()

        xoff = max(int((xlim[0]-dxlim[0])/dx), 0)
        yoff = max(-int((ylim[1]-dylim[1])/dy), 0)

        xoff1 = min(int((xlim[1]-dxlim[1])/dx), 0)
        yoff1 = min(-int((ylim[0]-dylim[0])/dy), 0)

        xsize = cols-xoff+xoff1
        ysize = rows-yoff+yoff1

        xdim = dx*xsize/int(ext[0])
        ydim = dy*ysize/int(ext[1])

        xbuf = min(xsize, int(ext[0]))
        ybuf = min(ysize, int(ext[1]))

        for i in range(dataset.RasterCount):

            rtmp = dataset.GetRasterBand(i+1)
            nval = rtmp.GetNoDataValue()
            bandid = rtmp.GetDescription()
            if bandid == '':
                bandid = 'Band '+str(i+1)

            dat = Data()
            dat.data = rtmp.ReadAsArray(xoff, yoff, xsize, ysize, xbuf, ybuf)

            if dat.data is None:
                self.showprocesslog('Error: Dataset could not be read '
                                    'properly')

            if dat.data.dtype.kind == 'i':
                if nval is None:
                    nval = 999999
                nval = int(nval)
            elif dat.data.dtype.kind == 'u':
                if nval is None:
                    nval = 0
                nval = int(nval)
            else:
                if nval is None:
                    nval = 1e+20
                nval = float(nval)

            dat.nodata = nval
            dat.xdim = xdim
            dat.ydim = ydim
            dat.wkt = dataset.GetProjection()
            datall[i+1] = dat

            self.pbar2.setValue(i)

        if datall == {}:
            datall = None

        dataset = None

        return datall


def get_shape_coords(sfile, todegrees=False):
    """
    Get coordinates from a shapefile.

    Parameters
    ----------
    sfile : str
        Shapefile name.
    todegrees : bool, optional
        Transform the coordinates to degrees. The default is False.

    Returns
    -------
    ddpoints : numpy array
        Output coordinates.

    """
    gdf = gpd.read_file(sfile)
    gdf = gdf[gdf.geometry != None]

    if todegrees is True:
        gdf = gdf.to_crs(epsg=4326)

    ddpoints = np.array(gdf.geometry.loc[0].exterior.coords)
    ddpoints = ddpoints[:, :2]
    return ddpoints


def get_kml_coords(kml):
    """
    Extract points from kml.

    Parameters
    ----------
    kml : str.
        kml file name.

    Returns
    -------
    coordinates : numpy array
        Coordinate in numpy format.

    """
    ns = "{http://www.opengis.net/kml/2.2}"
    tree = ElementTree.parse(kml)

    coordinates = []
    for placemark in tree.findall(".//" + ns + "Placemark"):
        polygon = placemark.findall(".//" + ns + "Polygon")

        for i in polygon:
            coordstext = i.findtext('.//'+ns+'coordinates')
            coordstext = coordstext.strip()

            for point_text in coordstext.split():
                floats = point_text.split(",")
                coordinates.append([float(floats[0]), float(floats[1])])

    coordinates = np.array(coordinates)
    return coordinates


def _testfn():
    """Test routine."""
    import sys
    sfile = r'd:\Workdata\change\fl35.shp'
    pdir = r'd:\Workdata\change\Planet'

    app = QtWidgets.QApplication(sys.argv)

    CSL = CreateSceneList(None)
    CSL.isrecursive.setChecked(True)
    CSL.shapefile.setText(sfile)
    CSL.scenefile.setText(pdir)
    CSL.settings(True)

    plt.show()


def _testanim():
    """Test for animation."""
    wfile = r'd:\Work\Workdata\change\tmp.gif'

    fig = plt.figure(dpi=150)

    writer = manimation.PillowWriter(fps=4)
    writer.setup(fig, wfile)

    tmp = np.random.rand(100, 100)
    im = plt.imshow(tmp)
    for i in range(20):
        red = np.random.rand(100, 100)
        im.set_data(red)
        fig.suptitle(str(i))
        writer.grab_frame()
    plt.show()
    writer.finish()


if __name__ == "__main__":
    _testfn()
