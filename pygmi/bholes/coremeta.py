# -----------------------------------------------------------------------------
# Name:        hypercore.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2021 Council for Geoscience
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
"""
Core Metadata Routine.

Assigning depths. Click a section and assign depth to that point. Depths are
auto interpolated between assigned depths

Each tray gets a text/xlsx file, with relevant metadata such as:
    tray number, num cores, core number, depth from, depth to,
    tray x (auto get from core number?), tray y from, tray y to

section start depth
section end depth
QC on depth range to see it makes sense
"""

import copy
import os
import sys

import numpy as np
from matplotlib.figure import Figure
from matplotlib import cm
from matplotlib.patches import Polygon as mPolygon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
from PyQt5 import QtWidgets, QtCore

from pygmi.misc import frm
from pygmi.raster.iodefs import get_raster, export_gdal
from pygmi.misc import ProgressBarText


class GraphMap(FigureCanvasQTAgg):
    """
    Graph Map.

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    """

    def __init__(self, parent=None):
        self.figure = Figure()

        super().__init__(self.figure)
        self.setParent(parent)

        self.parent = parent
        self.data = []
        self.mindx = 0
        self.csp = None
        self.subplot = None
        self.numcores = 5

    def init_graph(self):
        """
        Initialise the graph.

        Returns
        -------
        None.

        """
        dat = self.data[self.mindx]
        rows, cols = dat.data.T.shape

        self.figure.clf()
        self.subplot = self.figure.add_subplot(111)
        # self.subplot.get_xaxis().set_visible(False)
        # self.subplot.get_yaxis().set_visible(False)

        ymin = dat.data.mean()-2*dat.data.std()
        ymax = dat.data.mean()+2*dat.data.std()

        self.csp = self.subplot.imshow(dat.data.T, vmin=ymin, vmax=ymax)

        for yline in np.linspace(0, rows, self.numcores+1):
            self.subplot.plot([0, cols], [yline, yline], 'w')

        axes = self.figure.gca()
        axes.set_xlim((0, cols))
        axes.set_ylim((0, rows))

        # axes.set_xlabel('ColumnsEastings')
        # axes.set_ylabel('Northings')

        axes.xaxis.set_major_formatter(frm)
        axes.yaxis.set_major_formatter(frm)

        self.figure.canvas.draw()

    def update_graph(self):
        """
        Update graph.

        Returns
        -------
        None.

        """
        dat = self.data[self.mindx]

        self.csp.set_data(dat.data.T)

        ymin = dat.data.mean()-2*dat.data.std()
        ymax = dat.data.mean()+2*dat.data.std()

        self.csp.set_clim(ymin, ymax)

        self.csp.changed()
        self.figure.canvas.draw()





class CoreMeta(QtWidgets.QDialog):
    """
    Core metadata and depth assignment routine.

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    outdata : dictionary
        dictionary of output datasets
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        if parent is None:
            self.showprocesslog = print
            self.piter = ProgressBarText().iter
        else:
            self.showprocesslog = parent.showprocesslog
            self.piter = parent.pbar.iter

        self.indata = {}
        self.outdata = {}
        self.parent = parent

        self.map = GraphMap(self)
        self.combo = QtWidgets.QComboBox()
        self.mpl_toolbar = NavigationToolbar2QT(self.map, self.parent)
        self.idir = QtWidgets.QLineEdit('')
        self.sb_numcore = QtWidgets.QSpinBox()

        self.setupui()

        self.canvas = self.map.figure.canvas

        self.canvas.mpl_connect('button_press_event',
                                self.button_press_callback)

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        grid_main = QtWidgets.QGridLayout(self)

        buttonbox = QtWidgets.QDialogButtonBox()
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Tray Clipping and Band Selection')
        self.sb_numcore.setMinimum(1)
        self.sb_numcore.setMaximum(10)
        self.sb_numcore.setValue(5)

        lbl_combo = QtWidgets.QLabel('Display Band:')
        lbl_numcore = QtWidgets.QLabel('Number of core per tray:')

        grid_main.addWidget(lbl_combo, 0, 1)
        grid_main.addWidget(self.combo, 0, 2)
        grid_main.addWidget(lbl_numcore, 1, 1)
        grid_main.addWidget(self.sb_numcore, 1, 2)

        grid_main.addWidget(self.map, 0, 0, 10, 1)
        grid_main.addWidget(self.mpl_toolbar, 11, 0)

        grid_main.addWidget(buttonbox, 12, 0, 1, 1, QtCore.Qt.AlignLeft)

        self.sb_numcore.valueChanged.connect(self.ncore_change)
        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)

    def ncore_change(self):
        """
        Number of cores has changed.

        Returns
        -------
        None.

        """
        self.map.numcores = self.sb_numcore.value()
        self.map.init_graph()

    def on_combo(self):
        """
        On combo.

        Returns
        -------
        None.

        """
        self.map.mindx = self.combo.currentIndex()
        self.map.update_graph()

    def settings(self, nodialog=False):
        """
        Settings.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        if 'Raster' not in self.indata:
            self.showprocesslog('Error: You must have a multi-band raster '
                                'dataset in addition to your cluster '
                                'analysis results')
            return False

        self.map.data = self.indata['Raster']

        bands = [i.dataid for i in self.indata['Raster']]

        self.combo.clear()
        self.combo.addItems(bands)
        self.combo.currentIndexChanged.connect(self.on_combo)

        self.ncore_change()

        # self.map.update_graph()

        tmp = self.exec_()

        if tmp == 0:
            return False

        self.acceptall()

        return True

    def loadproj(self, projdata):
        """
        Loads project data into class.

        Parameters
        ----------
        projdata : dictionary
            Project data loaded from JSON project file.

        Returns
        -------
        chk : bool
            A check to see if settings was successfully run.

        """

        # self.combo_class.setCurrentText(projdata['combo_class'])

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

        # projdata['combo_class'] = self.combo_class.currentText()

        return projdata

    def button_press_callback(self, event):
        """
        Button press callback.

        Parameters
        ----------
        event : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if event.inaxes is None:
            return
        if event.button != 1:
            return

        ax = self.map.figure.gca()
        if ax.get_navigate_mode() is not None:
            return

        print(event.xdata, event.ydata)

    def acceptall(self):
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        None.

        """

        data = copy.copy(self.indata['Raster'])
        xy = self.map.polyi.poly.xy

        xy = xy.astype(int)
        xy = np.abs(xy)

        rows = np.unique(xy[:, 1])
        cols = np.unique(xy[:, 0])

        for dat in data:
            dat.data = dat.data[rows.min():rows.max(),
                                cols.min():cols.max()]
        datfin = []
        start = False
        for i in data:
            if i.dataid == self.combostart.currentText():
                start = True
            if not start:
                continue
            if i.dataid == self.comboend.currentText():
                break
            datfin.append(i)

        if self.asave.isChecked():
            meta = 'reflectance scale factor = 10000\n'
            meta += 'wavelength = {\n'
            for i in datfin:
                meta += i.dataid + ',\n'
            meta = meta[:-2]+'}\n'

            odir = os.path.dirname(self.indata['Raster'][0].filename)
            hfile = os.path.basename(self.indata['Raster'][0].filename)
            ofile = os.path.join(odir, 'clip_'+hfile[:-4]+'.hdr')
            export_gdal(ofile, datfin, 'ENVI', envimeta=meta, piter=self.piter)

        self.outdata['Raster'] = datfin
        return True


def dist_point_to_segment(p, s0, s1):
    """
    Dist point to segment.

    Reimplementation of Matplotlib's dist_point_to_segment, after it was
    depreciated. Follows http://geomalgorithms.com/a02-_lines.html

    Parameters
    ----------
    p : numpy array
        Point.
    s0 : numpy array
        Start of segment.
    s1 : numpy array
        End of segment.

    Returns
    -------
    numpy array
        Distance of point to segment.

    """
    p = np.array(p)
    s0 = np.array(s0)
    s1 = np.array(s1)

    v = s1 - s0
    w = p - s0

    c1 = np.dot(w, v)
    if c1 <= 0:
        return np.linalg.norm(p - s0)

    c2 = np.dot(v, v)
    if c2 <= c1:
        return np.linalg.norm(p - s1)

    b = c1/c2
    pb = s0 + b*v

    return np.linalg.norm(p - pb)


def testfn():
    """Main testing routine."""
    import matplotlib.pyplot as plt

    ifile = (r'c:\work\Workdata\HyperspectralScanner\PTest\smile\FENIX\\'
             r'clip_BV1_17_118m16_125m79_2020-06-30_12-43-14.dat')

    pbar = ProgressBarText()
    data = get_raster(ifile, piter=pbar.iter)

    app = QtWidgets.QApplication(sys.argv)  # Necessary to test Qt Classes
    tmp = CoreMeta()
    tmp.indata['Raster'] = data
    tmp.settings()


if __name__ == "__main__":
    testfn()
