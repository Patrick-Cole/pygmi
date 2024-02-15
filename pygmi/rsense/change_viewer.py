# -----------------------------------------------------------------------------
# Name:        change_viewer.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2023 Council for Geoscience
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
"""Change Detection Viewer."""

import datetime
from PyQt5 import QtWidgets, QtCore
import pandas as pd
from matplotlib.figure import Figure
import matplotlib.animation as manimation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT

from pygmi.misc import frm
from pygmi.misc import BasicModule
from pygmi.raster.modest_ioimage import imshow


class MyMplCanvas(FigureCanvasQTAgg):
    """Simple canvas with a sine plot."""

    def __init__(self, parent=None, width=10, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)

        self.ax1 = self.fig.add_subplot(111)
        self.im1 = None
        self.bands = [0, 1, 2]
        self.rcid = None
        self.manip = 'RGB Ternary'
        self.cbar = None
        self.capture_active = False
        self.writer = None
        self.piter = parent.piter
        self.showlog = parent.showlog

        super().__init__(self.fig)

        self.setParent(parent)
        self.parent = parent

        FigureCanvasQTAgg.setSizePolicy(self,
                                        QtWidgets.QSizePolicy.Expanding,
                                        QtWidgets.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)

    def capture(self):
        """
        Capture.

        Returns
        -------
        None.

        """
        self.capture_active = not self.capture_active

        if self.capture_active:
            ext = 'GIF (*.gif)'
            wfile, _ = QtWidgets.QFileDialog.getSaveFileName(
                self.parent, 'Save File', '.', ext)
            if wfile == '':
                self.capture_active = not self.capture_active
                return

            self.writer = manimation.PillowWriter(fps=4)
            self.writer.setup(self.fig, wfile)
        else:
            self.writer.finish()

    def compute_initial_figure(self, dat, dates):
        """
        Compute initial figure.

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
        if 'Ternary' in self.manip:
            red = dat.banddict[self.bands[0]]
            green = dat.banddict[self.bands[1]]
            blue = dat.banddict[self.bands[2]]

            data = [red, green, blue]
        else:
            data = [dat.banddict[self.bands[0]]]

        extent = dat.banddict[self.bands[0]].extent

        self.im1 = imshow(self.ax1, data, extent=extent, piter=self.piter,
                          showlog=self.showlog)
        self.im1.rgbmode = self.manip
        self.im1.rgbclip = None
        self.cbar = None
        self.ax1.xaxis.set_major_formatter(frm)
        self.ax1.yaxis.set_major_formatter(frm)

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
        extent = dat.banddata[0].extent
        self.im1.rgbmode = self.manip

        if 'Ternary' in self.manip:
            red = dat.banddict[self.bands[0]]
            green = dat.banddict[self.bands[1]]
            blue = dat.banddict[self.bands[2]]

            data = [red, green, blue]
        else:
            data = [dat.banddict[self.bands[0]]]

        extent = dat.banddict[self.bands[0]].extent

        self.im1.set_data(data)
        self.im1.set_extent(extent)
        self.fig.suptitle(dates)
        self.ax1.xaxis.set_major_formatter(frm)
        self.ax1.yaxis.set_major_formatter(frm)

        self.fig.canvas.draw()


class SceneViewer(BasicModule):
    """Scene Viewer."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.df = None
        self.curimage = 0

        self.canvas = MyMplCanvas(self, width=5, height=4, dpi=100)

        self.mpl_toolbar = NavigationToolbar2QT(self.canvas, self)
        self.slider = QtWidgets.QSlider()
        self.slider.setOrientation(QtCore.Qt.Horizontal)

        self.button1 = QtWidgets.QPushButton('Start Capture')
        self.button2 = QtWidgets.QPushButton('Previous Scene')
        self.button3 = QtWidgets.QPushButton('Next Scene')
        self.cmb_band1 = QtWidgets.QComboBox()
        self.cmb_band2 = QtWidgets.QComboBox()
        self.cmb_band3 = QtWidgets.QComboBox()
        self.cmb_manip = QtWidgets.QComboBox()

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        vbl_1 = QtWidgets.QVBoxLayout()
        vbl_2 = QtWidgets.QVBoxLayout()
        hbl = QtWidgets.QHBoxLayout()
        hblmain = QtWidgets.QHBoxLayout(self)

        self.setWindowTitle("View Change Data")
        self.slider.setTracking(False)

        gbox_1 = QtWidgets.QGroupBox('Display Type')
        vbl_1b = QtWidgets.QVBoxLayout()
        gbox_1.setLayout(vbl_1b)

        gbox_2 = QtWidgets.QGroupBox('Data Bands')
        vbl_2b = QtWidgets.QVBoxLayout()
        gbox_2.setLayout(vbl_2b)

        spacer = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Fixed,
                                       QtWidgets.QSizePolicy.Expanding)

        vbl_2b.addWidget(self.cmb_band1)
        vbl_2b.addWidget(self.cmb_band2)
        vbl_2b.addWidget(self.cmb_band3)

        actions = ['RGB Ternary', 'CMY Ternary', 'Single Colour Map']
        self.cmb_manip.addItems(actions)

        vbl_1b.addWidget(self.cmb_manip)

        hbl.addWidget(self.button2)
        hbl.addWidget(self.button3)

        vbl_1.addWidget(self.canvas)
        vbl_1.addWidget(self.mpl_toolbar)
        vbl_1.addWidget(self.slider)
        vbl_1.addLayout(hbl)

        vbl_2.addWidget(gbox_1)
        vbl_2.addWidget(gbox_2)
        vbl_2.addWidget(self.button1)
        vbl_2.addItem(spacer)

        hblmain.addLayout(vbl_2)
        hblmain.addLayout(vbl_1)

        self.curimage = 0

        self.slider.valueChanged.connect(self.newdata)
        self.button2.clicked.connect(self.prevscene)
        self.button3.clicked.connect(self.nextscene)
        self.button1.clicked.connect(self.capture)
        self.cmb_manip.currentIndexChanged.connect(self.manip_change)
        self.cmb_band1.currentIndexChanged.connect(self.manip_change)
        self.cmb_band2.currentIndexChanged.connect(self.manip_change)
        self.cmb_band3.currentIndexChanged.connect(self.manip_change)

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
        if 'RasterFileList' not in self.indata:
            self.showlog('No batch file list detected.')
            return False

        chk = self.updatescenelist()

        if chk is False:
            return False

        dates = self.df.Datetime[self.curimage]

        self.slider.setMaximum(len(self.df)-1)

        dat = self.df.Filename[self.curimage]
        dat.banddict = {}
        for i in dat.banddata:
            dat.banddict[i.dataid] = i

        bands = dat.bands

        try:
            self.cmb_band1.currentIndexChanged.disconnect()
            self.cmb_band2.currentIndexChanged.disconnect()
            self.cmb_band3.currentIndexChanged.disconnect()
        except TypeError:
            pass

        self.cmb_band1.clear()
        self.cmb_band2.clear()
        self.cmb_band3.clear()

        self.cmb_band1.addItems(bands)
        self.cmb_band2.addItems(bands)
        self.cmb_band3.addItems(bands)

        if len(bands) > 3:
            self.cmb_band1.setCurrentIndex(3)
            self.cmb_band2.setCurrentIndex(2)
            self.cmb_band3.setCurrentIndex(1)
        elif len(bands) == 3:
            self.cmb_band1.setCurrentIndex(2)
            self.cmb_band2.setCurrentIndex(1)
            self.cmb_band3.setCurrentIndex(0)

        self.cmb_band1.currentIndexChanged.connect(self.manip_change)
        self.cmb_band2.currentIndexChanged.connect(self.manip_change)
        self.cmb_band3.currentIndexChanged.connect(self.manip_change)

        self.canvas.bands = [self.cmb_band1.currentText(),
                             self.cmb_band2.currentText(),
                             self.cmb_band3.currentText()]

        self.canvas.manip = self.cmb_manip.currentText()
        self.canvas.compute_initial_figure(dat, dates)

        if not nodialog:
            tmp = self.exec()

            if tmp != 1:
                return tmp
        return True

    def updatescenelist(self):
        """
        Update the scene list.

        Returns
        -------
        bool
            Boolean to indicate success.

        """
        dat = self.indata['RasterFileList']

        subfiles = []
        for i in dat:
            if '.tif' in i.filename.lower():
                subfiles.append(i)

        if not subfiles:
            self.showlog('No GeoTIFF images found.')
            return False

        dtime = []
        flist = []
        nodates = False
        for i in self.piter(subfiles):
            dattmp = i.banddata

            dt = dattmp[0].datetime
            if dt is None:
                dt = datetime.datetime(1900, 1, 1)
                nodates = True

            dtime.append(dt)
            flist.append(i)

        if nodates is True:
            self.showlog('Some of your scenes do not have dates.')

        self.df = pd.DataFrame()
        self.df['Datetime'] = dtime
        self.df['Filename'] = flist

        self.df.sort_values('Datetime', inplace=True)

        return True

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """

    def manip_change(self):
        """
        Change manipulation or bands.

        Returns
        -------
        None.

        """
        maniptxt = self.cmb_manip.currentText()

        if 'Ternary' in maniptxt:
            self.cmb_band2.show()
            self.cmb_band3.show()
        else:
            self.cmb_band2.hide()
            self.cmb_band3.hide()

        self.canvas.bands = [self.cmb_band1.currentText(),
                             self.cmb_band2.currentText(),
                             self.cmb_band3.currentText()]

        self.canvas.manip = maniptxt
        self.newdata(self.curimage)

    def nextscene(self):
        """
        Get next scene.

        Returns
        -------
        None.

        """
        self.slider.setValue(self.slider.value()+1)

    def prevscene(self):
        """
        Get previous scene.

        Returns
        -------
        None.

        """
        self.slider.setValue(self.slider.value()-1)

    def newdata(self, indx):
        """
        Get new dataset.

        Parameters
        ----------
        indx : int
            Current index.

        Returns
        -------
        None.

        """
        self.curimage = indx

        dates = self.df.Datetime[indx]
        dat = self.df.Filename[indx]
        dat.banddict = {}
        for i in dat.banddata:
            dat.banddict[i.dataid] = i

        if self.canvas.bands[0] not in dat.banddict:
            self.showlog('Band name not in dataset.')
            return

        self.canvas.update_plot(dat, dates)

    def capture(self):
        """
        Capture all scenes in the current view as an animation.

        Returns
        -------
        None.

        """
        self.showlog('Starting capture...')
        self.slider.valueChanged.disconnect()

        self.canvas.capture()
        if not self.canvas.capture_active:
            return

        for indx in self.df.index:
            self.slider.setValue(indx)
            QtWidgets.QApplication.processEvents()
            self.newdata(indx)
            self.canvas.writer.grab_frame()

        self.canvas.capture()
        self.slider.valueChanged.connect(self.newdata)
        self.slider.setValue(self.curimage)

        self.showlog('Capture complete.')


def _testfn():
    """Test routine."""
    import sys
    from pygmi.rsense.iodefs import ImportBatch

    idir = r"E:\WorkProjects\ST-2020-1339 Landslides\change\mosaic"
    idir = r"D:\Workdata\change\Planet"
    idir = r"E:\Namaqua_change\namakwa"

    app = QtWidgets.QApplication(sys.argv)

    tmp1 = ImportBatch()
    tmp1.idir = idir
    tmp1.get_sfile(True)
    tmp1.settings()

    tmp2 = SceneViewer()
    tmp2.indata = tmp1.outdata
    tmp2.settings()


if __name__ == "__main__":
    _testfn()
