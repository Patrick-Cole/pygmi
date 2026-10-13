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
Clip to zoom.

This module allows a raster dataset to be clipped to the current zoomed extents.
"""

import geopandas as gpd
from matplotlib import colormaps
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from PySide6 import QtCore, QtWidgets
from shapely import Polygon

from pygmi.maps import CanvasModule
from pygmi.misc import BasicModule
from pygmi.raster.datatypes import Data
from pygmi.raster.misc import cut_raster
from pygmi.raster.modest_image import imshow


class ClipToZoom(BasicModule):
    """
    Clip to zoom GUI Class.

    Parameters
    ----------
    parent
        Reference to the parent routine. The default is None.

    """

    def __init__(
        self,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Clip to Zoom")

        vbl = QtWidgets.QVBoxLayout(self)  # self is where layout is assigned
        hbl = QtWidgets.QHBoxLayout()
        self.mmc = CanvasModule(self)
        mpl_toolbar = NavigationToolbar2QT(self.mmc, self.parent)
        self.axes = self.mmc.figure.add_subplot(111)

        self.buttonbox.htmlfile = "raster.dm.cliptozoom"
        self.buttonbox.buttonbox.hide()
        hbl.addWidget(self.buttonbox)

        self.cmb_1 = QtWidgets.QComboBox()
        lbl_1 = QtWidgets.QLabel("Bands:")
        hbl.addWidget(lbl_1, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        hbl.addWidget(self.cmb_1)

        self.cmb_2 = QtWidgets.QComboBox()
        lbl_2 = QtWidgets.QLabel("Colormap:")
        hbl.addWidget(lbl_2, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        hbl.addWidget(self.cmb_2)
        self.cmb_2.addItems(["viridis", "jet", "gray", "terrain"])

        self.btn_clip = QtWidgets.QPushButton("Clip")
        hbl.addWidget(self.btn_clip)

        vbl.addWidget(self.mmc)
        vbl.addWidget(mpl_toolbar)
        vbl.addLayout(hbl)

        self.setFocus()
        self.cmb_1.currentIndexChanged.connect(self.change_band)
        self.cmb_2.currentIndexChanged.connect(self.change_band)
        self.btn_clip.clicked.connect(self.accept)

    def change_band(self):
        """Combo box to choose band."""
        i = self.cmb_1.currentIndex()
        cmap = self.cmb_2.currentText()
        if "Raster" in self.indata:
            data = self.indata["Raster"]
            self.update_raster(data[i], cmap)

    def settings(
        self,
        nodialog: bool = False,
    ) -> bool:
        """
        Entry point into item.

        Parameters
        ----------
        nodialog
            Run settings without a dialog. The default is False.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        data = []
        if "Raster" in self.indata:
            data = self.indata["Raster"]
        elif "Cluster" in self.indata:
            data = self.indata["Cluster"]

        items = [i.dataid for i in data]
        self.cmb_update(self.cmb_1, items)

        self.change_band()

        tmp = self.exec()

        if tmp == 0:
            return False

        ylim = self.axes.get_ylim()
        xlim = self.axes.get_xlim()
        x0, x1 = xlim
        y0, y1 = ylim

        poly = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)])
        gdf = gpd.GeoDataFrame({"geometry": [poly]})

        for datatype in ["Raster", "Cluster"]:
            if datatype not in self.indata:
                continue
            data = self.indata[datatype]

            data = cut_raster(data, gdf, showlog=self.showlog)

            if data is None:
                return False

            self.outdata[datatype] = data

        return True

    def saveproj(self):
        """Save project data from class."""
        self.saveobj(self.cmb_1)
        self.saveobj(self.cmb_2)

    def update_raster(
        self,
        data1: Data,
        cmap: str,
    ):
        """
        Update the raster plot.

        Parameters
        ----------
        data1
            Raster dataset to be used
        cmap
            Matplotlib colormap description

        """
        self.mmc.figure.clear()

        self.axes = self.mmc.figure.add_subplot(111)

        rdata = imshow(
            self.axes,
            data1.data,
            extent=data1.extent,
            cmap=colormaps[cmap],
            interpolation="nearest",
        )

        if not data1.isrgb:
            rdata.set_clim_std(2.5)

        self.mmc.set_map_axes(self.axes, data1.crs)
        self.mmc.draw_idle()


def _testfn():
    """Test."""
    import sys

    import matplotlib.pyplot as plt

    from pygmi.raster.iodefs import get_raster

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))

    ifile = r"D:\workdata\PyGMI Test Data\Raster\testdata.tif"

    data = get_raster(ifile)

    tmp = ClipToZoom()
    tmp.indata["Raster"] = data

    tmp.settings()

    dat = tmp.outdata["Raster"]
    plt.imshow(dat[0].data)
    plt.show()


if __name__ == "__main__":
    _testfn()
