# -----------------------------------------------------------------------------
# Name:        raster/cliptozoom.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2024 Council for Geoscience
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
Clip to Zoom.

This module allows a raster dataset to be clipped to the current zoomed
extents.
"""

import numpy as np
from PyQt5 import QtWidgets, QtCore
from matplotlib import colormaps
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
import matplotlib.colors as mcolors
from shapely import Polygon
import geopandas as gpd

from pygmi.misc import frm, BasicModule
from pygmi.raster.modest_image import imshow
from pygmi.raster.dataprep import cut_raster


class MyMplCanvas(FigureCanvasQTAgg):
    """
    Canvas for the actual plot.

    Attributes
    ----------
    axes : matplotlib subplot
    parent : parent
        reference to the parent routine
    """

    def __init__(self, parent=None):
        fig = Figure(layout='constrained', dpi=150)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.fwidth = self.figure.get_figwidth()
        self.fheight = self.figure.get_figheight()

    def update_raster(self, data1, cmap):
        """
        Update the raster plot.

        Parameters
        ----------
        data1 : PyGMI raster Data
            raster dataset to be used in contouring
        cmap : str
            Matplotlib colormap description

        Returns
        -------
        None.

        """
        self.figure.clear()
        self.figure.set_figwidth(self.fwidth)
        self.figure.set_figheight(self.fheight)

        self.axes = self.figure.add_subplot(111)

        rdata = imshow(self.axes, data1.data, extent=data1.extent,
                       cmap=colormaps[cmap], interpolation='nearest')

        if not data1.isrgb:
            rdata.set_clim_std(2.5)
            cbar = self.figure.colorbar(rdata, format=frm)
            cbar.set_label(data1.units)

        if data1.crs is not None and data1.crs.is_geographic:
            self.axes.set_xlabel('Longitude')
            self.axes.set_ylabel('Latitude')
        else:
            self.axes.set_xlabel('Eastings')
            self.axes.set_ylabel('Northings')

        self.axes.xaxis.set_major_formatter(frm)
        self.axes.yaxis.set_major_formatter(frm)

        bbox = self.axes.get_window_extent()
        dpi = self.figure.dpi
        awidth = bbox.width / dpi
        aheight = bbox.height / dpi

        self.figure.set_figwidth(awidth)
        self.figure.set_figheight(aheight)

        self.figure.canvas.draw()


class ClipToZoom(BasicModule):
    """Clip to zoom Class."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle('Clip to Zoom')

        vbl = QtWidgets.QVBoxLayout(self)  # self is where layout is assigned
        hbl = QtWidgets.QHBoxLayout()
        self.mmc = MyMplCanvas(self)
        mpl_toolbar = NavigationToolbar2QT(self.mmc, self.parent)

        self.cmb_1 = QtWidgets.QComboBox()
        lbl_1 = QtWidgets.QLabel('Bands:')
        hbl.addWidget(lbl_1)
        hbl.addWidget(self.cmb_1)

        self.cmb_2 = QtWidgets.QComboBox()
        lbl_2 = QtWidgets.QLabel('Colormap:')
        hbl.addWidget(lbl_2)
        hbl.addWidget(self.cmb_2)
        self.cmb_2.addItems(['viridis', 'jet', 'gray', 'terrain'])

        self.btn_clip = QtWidgets.QPushButton('Clip')
        hbl.addWidget(self.btn_clip)

        vbl.addWidget(self.mmc)
        vbl.addWidget(mpl_toolbar)
        vbl.addLayout(hbl)

        self.setFocus()
        self.cmb_1.currentIndexChanged.connect(self.change_band)
        self.cmb_2.currentIndexChanged.connect(self.change_band)
        self.btn_clip.clicked.connect(self.accept)

    def change_band(self):
        """
        Combo box to choose band.

        Returns
        -------
        None.

        """
        i = self.cmb_1.currentIndex()
        cmap = self.cmb_2.currentText()
        if 'Raster' in self.indata:
            data = self.indata['Raster']
            self.mmc.update_raster(data[i], cmap)

    def settings(self, nodialog=False):
        """
        Run.

        Returns
        -------
        None.

        """
        data = []
        if 'Raster' in self.indata:
            data = self.indata['Raster']
        elif 'Cluster' in self.indata:
            data = self.indata['Cluster']

        for i in data:
            self.cmb_1.addItem(i.dataid)

        tmp = self.exec()

        if tmp == 0:
            return False

        ylim = self.mmc.axes.get_ylim()
        xlim = self.mmc.axes.get_xlim()
        x0, x1 = xlim
        y0, y1 = ylim

        poly = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)])
        gdf = gpd.GeoDataFrame({'geometry': [poly]})

        for datatype in ['Raster', 'Cluster']:
            if datatype not in self.indata:
                continue
            data = self.indata[datatype]

            data = cut_raster(data, gdf, showlog=self.showlog)

            if data is None:
                return False

            self.outdata[datatype] = data

        return True


def _testfn():
    """Test."""
    import sys
    from pygmi.raster.iodefs import get_raster
    import matplotlib.pyplot as plt

    app = QtWidgets.QApplication(sys.argv)

    ifile = r"D:\workdata\PyGMI Test Data\Raster\testdata.hdr"

    data = get_raster(ifile)

    tmp = ClipToZoom()
    tmp.indata['Raster'] = data

    tmp.settings()


if __name__ == "__main__":
    _testfn()
