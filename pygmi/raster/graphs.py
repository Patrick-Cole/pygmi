# -----------------------------------------------------------------------------
# Name:        raster/graphs.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2013 Council for Geoscience
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
Plot Raster Data.

This module provides a variety of methods to plot raster data via the context
menu. The following are supported:

 * Correlation coefficients
 * Images
 * Surfaces
 * Histograms
 * Scatterplots
"""

import matplotlib.colors as mcolors
import numpy as np
import pyvista as pv
from matplotlib import colors
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from numpy.typing import NDArray
from PySide6 import QtCore, QtWidgets
from pyvistaqt import QtInteractor

from pygmi.maps import CanvasModule, frm, set_axes, set_northscale
from pygmi.misc import ContextModule
from pygmi.raster.colormaps import colormaps
from pygmi.raster.datatypes import Data
from pygmi.raster.modest_image import imshow


class MyMplCanvas(CanvasModule):
    """Matplotlib canvas widget for the actual plot."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None

    def format_coord(self, x: float, y: float) -> str:
        """
        Set format coordinate for section plot.

        Parameters
        ----------
        x
            x coordinate.
        y
            y coordinate.

        Returns
        -------
        str
            Output string to display.

        """
        scoords = self.data.metadata["Raster"]["SectionCoords"]
        r1 = scoords[:, 2]
        difference_array = np.absolute(r1 - x)
        idx = difference_array.argmin()
        x1 = scoords[idx, 0]
        y1 = scoords[idx, 1]

        text = f"X={x1:.2f}, Y={y1:.2f}, Z={y:.1f}"
        return text

    def update_ccoef(self, data1: Data, dmat: NDArray):
        """
        Update the correlation coefficient plot.

        Parameters
        ----------
        data1
            raster dataset to be used.
        dmat
            dummy matrix of numbers to be plotted using pcolor.

        """
        cmap = colormaps["viridis"]

        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        rdata = self.axes.pcolormesh(dmat, cmap=cmap)
        self.axes.axis("scaled")
        self.axes.set_title("Correlation Coefficients")
        for i in range(len(data1)):
            for j in range(len(data1)):
                ctmp = np.array([1.0, 1.0, 1.0, 0.0]) - np.array(cmap(dmat[i, j]))
                ctmp = np.abs(ctmp)
                ctmp = ctmp.tolist()

                atext = f"{dmat[i, j]:.2f}"

                self.axes.text(
                    i + 0.5,
                    j + 0.5,
                    atext,
                    c=ctmp,
                    rotation=45,
                    ha="center",
                    va="center",
                )
        dat_mat = [i.dataid for i in data1]
        self.axes.set_xticks(np.array(list(range(len(data1)))) + 0.5)

        self.axes.set_xticklabels(dat_mat, rotation="vertical")
        self.axes.set_yticks(np.array(list(range(len(data1)))) + 0.5)

        self.axes.set_yticklabels(dat_mat, rotation="horizontal")
        self.axes.set_xlim(0, len(data1))
        self.axes.set_ylim(0, len(data1))

        self.figure.colorbar(rdata, format=frm)

        self.figure.canvas.draw()

    def update_raster(self, data1: Data, cmap: str, plotlog: bool, aspect: int = 1):
        """
        Update the raster plot.

        Parameters
        ----------
        data1
            raster dataset to be used in contouring
        cmap
            Matplotlib colormap description
        plotlog
            Use a logarithmic colour scale.
        aspect
            Aspect ratio of colour bar.

        """
        self.data = data1
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        self.custom_resize = True
        is_section = data1.metadata["Raster"]["Section"]

        if data1.isrgb is True:
            rdata = imshow(
                self.axes,
                data1.data,
                extent=data1.extent,
                cmap=colormaps[cmap],
                interpolation="none",
            )
        elif plotlog is True:
            rdata = imshow(
                self.axes,
                data1.data,
                extent=data1.extent,
                cmap=colormaps[cmap],
                interpolation="none",
                norm=colors.LogNorm(vmin=data1.data.min(), vmax=data1.data.max()),
            )
        else:
            rdata = imshow(
                self.axes,
                data1.data,
                extent=data1.extent,
                cmap=colormaps[cmap],
                interpolation="none",
            )
            rdata.set_clim_std(2.5)

        location = "right"
        if not data1.isrgb:
            pad = 0.1
            caspect = 30

            location = "right"
            shrink = 1.0
            anchor = (0.0, 0.5)

            if is_section is True:
                location = "bottom"
                shrink = 0.8
                pad = 0.15
                anchor = (0.5, 1.0)
                caspect = 40

            cbar = self.figure.colorbar(
                rdata,
                format=frm,
                location=location,
                aspect=caspect,
                shrink=shrink,
                anchor=anchor,
                pad=pad,
            )
            cbar.set_label(data1.units)

        if is_section is True:
            self.axes.set_xlabel("Distance")
            self.axes.set_ylabel("Elevation")
            self.axes.format_coord = self.format_coord
            rdata.format_cursor_data = lambda x: f"Data: {x}"

            self.axes.ticklabel_format(style="plain", axis="both", useOffset=False)
            self.axes.tick_params(axis="x", rotation=0)
            self.axes.tick_params(axis="y", rotation=0)

            self.axes.xaxis.set_major_formatter(frm)
            self.axes.yaxis.set_major_formatter(frm)
            self.axes.set_aspect(aspect)
        else:
            set_axes(self.axes, data1.crs)
            set_northscale(self.axes, data1.crs, self.showlog)

        self.draw()

    def update_hexbin(self, data1: Data, data2: Data):
        """
        Update the hexbin plot.

        Parameters
        ----------
        data1
            Raster dataset to be used
        data2
            Raster dataset to be used

        """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)

        self.axes.tick_params(axis="x", rotation=90)
        self.axes.tick_params(axis="y", rotation=0)
        self.axes.set_xlabel(data1.dataid)
        self.axes.set_ylabel(data2.dataid)

        x = data1.data.copy()
        y = data2.data.copy()

        msk = np.logical_or(x.mask, y.mask)
        x.mask = msk
        y.mask = msk
        x = x.compressed()
        y = y.compressed()

        xmin = x.min()
        xmax = x.max()
        ymin = y.min()
        ymax = y.max()

        hbin = self.axes.hexbin(x, y, bins="log", cmap="inferno")
        self.axes.axis([xmin, xmax, ymin, ymax])
        self.axes.set_title("Hexbin Plot")
        cbar = self.figure.colorbar(hbin, format=frm)
        cbar.set_label("log10(N)")

        self.axes.xaxis.set_major_formatter(frm)
        self.axes.yaxis.set_major_formatter(frm)

        if data1.units != "":
            self.axes.set_xlabel(f"{data1.dataid} ({data1.units})")
        if data2.units != "":
            self.axes.set_ylabel(f"{data2.dataid} ({data2.units})")

        self.figure.canvas.draw()

    def update_surface(self, data: Data, icmap: str):
        """
        Update the surface plot.

        Parameters
        ----------
        data
            raster dataset to be used
        icmap
            Matplotlib colormap description

        """
        rows, cols = data.data.shape

        dtlx = data.extent[0]
        dtly = data.extent[-1]
        x = dtlx + np.arange(cols) * data.xdim + data.xdim / 2
        y = dtly - np.arange(rows) * data.ydim - data.ydim / 2
        x, y = np.meshgrid(x, y)
        z = data.data.copy()
        vmin, vmax = np.percentile(z.compressed(), [1, 99])

        if not np.ma.is_masked(z):
            z = np.ma.array(z)

        x = np.ma.array(x, mask=z.mask)
        y = np.ma.array(y, mask=z.mask)

        cmap = colormaps[icmap]

        norml = mcolors.Normalize(vmin=vmin, vmax=vmax)

        z.data[z.mask] = np.nan
        z = z.data

        self.figure.clear()
        self.axes = self.figure.add_subplot(111, projection="3d")

        vmin, vmax = np.percentile(z, [1, 99])

        surf = self.axes.plot_surface(
            x,
            y,
            z,
            cmap=cmap,
            norm=norml,
            vmin=vmin,
            vmax=vmax,
            shade=False,
            antialiased=False,
        )

        self.figure.colorbar(surf, format=frm)

        self.axes.xaxis.set_major_formatter(frm)
        self.axes.yaxis.set_major_formatter(frm)
        self.axes.zaxis.set_major_formatter(frm)

        self.axes.set_title("")
        self.axes.set_xlabel("X")
        self.axes.set_ylabel("Y")
        self.axes.set_zlabel("Z")

        self.figure.canvas.draw()

    def update_hist(self, data1: Data, ylog: bool, iscum: bool):
        """
        Update the histogram plot.

        Parameters
        ----------
        data1
            raster dataset to be used
        ylog
            Boolean for a log scale on y-axis.
        iscum
            Boolean for a cumulative distribution.

        """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)

        self.axes.tick_params(axis="x", rotation=90)
        self.axes.tick_params(axis="y", rotation=0)

        dattmp = data1.data[data1.data.mask == 0].flatten()
        self.axes.hist(
            dattmp, bins="sqrt", cumulative=iscum, histtype="stepfilled", edgecolor="k"
        )
        self.axes.set_title(data1.dataid)
        self.axes.set_xlabel(data1.dataid)
        self.axes.set_ylabel("Counts")

        self.axes.xaxis.set_major_formatter(frm)
        self.axes.yaxis.set_major_formatter(frm)

        if ylog is True:
            self.axes.set_yscale("log")

        self.figure.canvas.draw()


class PlotCCoef(ContextModule):
    """
    Plot 2D Correlation Coefficients.

    Parameters
    ----------
    parent
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle("Correlation Coefficients")

        vbl = QtWidgets.QVBoxLayout(self)
        hbl = QtWidgets.QHBoxLayout()

        self.mmc = MyMplCanvas(self)
        mpl_toolbar = NavigationToolbar2QT(self.mmc, self.parent)
        self.buttonbox.buttonbox.hide()
        self.buttonbox.htmlfile = "raster.cm.showcorr"

        vbl.addWidget(self.mmc)
        hbl.addWidget(mpl_toolbar)
        hbl.addWidget(self.buttonbox)
        vbl.addLayout(hbl)

        self.setMinimumSize(600, 600)

        self.setFocus()

    def run(self):
        """Entry point into the routine, used to run context menu item."""
        if "Raster" in self.indata:
            data = self.indata["Raster"]
        else:
            return

        if data[0].metadata["Raster"]["Section"] is True:
            self.showlog("Sections are not supported.")
            return

        if self.indata["Raster"][0].isrgb:
            self.showlog("RGB images cannot be used in this module.")
            return

        if not check_bands(data):
            QtWidgets.QMessageBox.warning(
                self,
                "Warning",
                "Different size input datasets. "
                "Merge and resample your input data to fix this.",
                QtWidgets.QMessageBox.StandardButton.Ok,
            )
            return

        self.show()

        dummy_mat = [[corr2d(i.data, j.data) for j in data] for i in data]
        dummy_mat = np.array(dummy_mat)

        self.mmc.update_ccoef(data, dummy_mat)


class PlotRaster(ContextModule):
    """
    Plot Raster Class.

    Parameters
    ----------
    parent
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle("Raster Plot (Simple)")

        vbl = QtWidgets.QVBoxLayout(self)
        hbl = QtWidgets.QHBoxLayout()
        self.mmc = MyMplCanvas(parent)
        mpl_toolbar = NavigationToolbar2QT(self.mmc)

        self.buttonbox.buttonbox.hide()
        self.buttonbox.htmlfile = "raster.cm.showsimple"

        hbl.addWidget(self.buttonbox)

        self.cb_log = QtWidgets.QCheckBox("Log Colour Scale")
        hbl.addWidget(self.cb_log)

        self.sb_aspect = QtWidgets.QSpinBox()
        self.sb_aspect.setMinimum(1)
        self.lbl_a = QtWidgets.QLabel("Aspect:")
        hbl.addWidget(self.lbl_a)
        hbl.addWidget(self.sb_aspect)
        self.lbl_a.hide()
        self.sb_aspect.hide()

        self.cmb_1 = QtWidgets.QComboBox()
        lbl_1 = QtWidgets.QLabel("Bands:")
        hbl.addWidget(lbl_1, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        hbl.addWidget(self.cmb_1)

        self.cmb_2 = QtWidgets.QComboBox()
        lbl_2 = QtWidgets.QLabel("Colormap:")
        hbl.addWidget(lbl_2, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        hbl.addWidget(self.cmb_2)
        self.cmb_2.addItems(
            [
                "viridis",
                "jet",
                "gray",
                "terrain",
                "Floyd",
                "MarineCopper",
                "Splash",
                "Wheel",
            ]
        )

        vbl.addWidget(self.mmc)
        vbl.addWidget(mpl_toolbar)
        vbl.addLayout(hbl)

        self.setFocus()
        self.sb_aspect.valueChanged.connect(self.change_band)
        self.cmb_1.currentIndexChanged.connect(self.change_band)
        self.cmb_2.currentIndexChanged.connect(self.change_band)
        self.cb_log.checkStateChanged.connect(self.change_band)

    def change_band(self):
        """Combo box to choose band."""
        i = self.cmb_1.currentIndex()
        cmap = self.cmb_2.currentText()
        plotlog = self.cb_log.isChecked()
        aspect = self.sb_aspect.value()

        if "Raster" in self.indata:
            data = self.indata["Raster"]
            if "Section" in data[i].metadata["Raster"]:
                is_section = data[i].metadata["Raster"]["Section"]
            else:
                is_section = False
            if is_section is True:
                self.sb_aspect.show()
                self.lbl_a.show()
            else:
                self.lbl_a.hide()
                self.sb_aspect.hide()

            self.mmc.update_raster(data[i], cmap, plotlog, aspect)

    def run(self):
        """Entry point into the routine, used to run context menu item."""
        data = []
        if "Raster" in self.indata:
            data = self.indata["Raster"]
        elif "Cluster" in self.indata:
            data = self.indata["Cluster"]

        cols = [i.dataid for i in data]
        self.cmb_update(self.cmb_1, cols)

        self.change_band()
        self.show()


class PlotSurface(ContextModule):
    """
    Plot Surface Class.

    Parameters
    ----------
    parent
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle("Surface Plot")

        vbl = QtWidgets.QVBoxLayout(self)  # self is where layout is assigned
        hbl = QtWidgets.QHBoxLayout()
        hbl2 = QtWidgets.QHBoxLayout()

        self.plotter = QtInteractor(self)
        self.vslider = QtWidgets.QSlider()

        self.buttonbox.buttonbox.hide()
        self.buttonbox.htmlfile = "raster.cm.showsurface"
        hbl.addWidget(self.buttonbox)

        self.cmb_1 = QtWidgets.QComboBox()
        lbl_1 = QtWidgets.QLabel("Bands:")
        hbl.addWidget(lbl_1, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        hbl.addWidget(self.cmb_1)

        self.cmb_2 = QtWidgets.QComboBox()
        lbl_2 = QtWidgets.QLabel("Colormap:")
        hbl.addWidget(lbl_2, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        hbl.addWidget(self.cmb_2)
        self.cmb_2.addItems(
            [
                "viridis",
                "jet",
                "gray",
                "terrain",
                "Floyd",
                "MarineCopper",
                "Splash",
                "Wheel",
            ]
        )

        hbl2.addWidget(self.vslider)
        hbl2.addWidget(self.plotter)

        vbl.addLayout(hbl2)
        vbl.addLayout(hbl)

        self.setFocus()

        self.cmb_1.currentIndexChanged.connect(self.change_band)
        self.cmb_2.currentIndexChanged.connect(self.change_band)
        self.vslider.valueChanged.connect(self.slider)

    def change_band(self):
        """Combo box to choose band."""
        i = self.cmb_1.currentIndex()
        cmap = self.cmb_2.currentText()
        if "Raster" not in self.indata:
            return

        dat = self.indata["Raster"][i]
        dat.data = dat.data.astype(float)

        xmin, xmax, ymin, ymax = dat.extent

        x = np.linspace(xmin, xmax, dat.data.shape[1])
        y = np.linspace(ymax, ymin, dat.data.shape[0])

        x, y = np.meshgrid(x, y)
        z = dat.data.astype(float)

        z = z.filled(np.nan)

        grid = pv.StructuredGrid(x, y, z)
        grid["values"] = z.T.flatten()

        sargs = {
            "title_font_size": 20,
            "label_font_size": 16,
            "shadow": True,
            "n_labels": 5,
            "italic": True,
            "fmt": "%.1f",
            "font_family": "arial",
            "vertical": True,
            "title": dat.units,
        }

        self.plotter.clear()
        self.plotter.add_mesh(
            grid,
            cmap=cmap,
            scalars="values",
            scalar_bar_args=sargs,
        )

        self.plotter.set_scale(zscale=20)

    def run(self):
        """Entry point into the routine, used to run context menu item."""
        if "Raster" in self.indata:
            data = self.indata["Raster"]
        else:
            return

        if data[0].metadata["Raster"]["Section"] is True:
            self.showlog("Sections are not supported.")
            return

        if self.indata["Raster"][0].isrgb:
            self.showlog("RGB images cannot be used in this module.")
            return

        cols = [i.dataid for i in data]
        self.cmb_update(self.cmb_1, cols)

        self.change_band()
        self.show()

    def slider(self):
        """Vertical slider used to scale 3d view."""
        self.plotter.set_scale(zscale=self.vslider.value())


class PlotScatter(ContextModule):
    """
    Plot Hexbin Class.

    A Hexbin is a type of scatter plot which is raster.

    Parameters
    ----------
    parent
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle("Hexbin Plot")

        vbl = QtWidgets.QVBoxLayout(self)  # self is where layout is assigned
        hbl = QtWidgets.QHBoxLayout()
        self.mmc = MyMplCanvas(self)
        mpl_toolbar = NavigationToolbar2QT(self.mmc, self.parent)

        self.buttonbox.buttonbox.hide()
        self.buttonbox.htmlfile = "raster.cm.showhexbin"
        hbl.addWidget(self.buttonbox)

        self.cmb_1 = QtWidgets.QComboBox()
        self.cmb_2 = QtWidgets.QComboBox()
        lbl_1 = QtWidgets.QLabel("X Band:")
        lbl_2 = QtWidgets.QLabel("Y Band:")
        hbl.addWidget(lbl_1, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        hbl.addWidget(self.cmb_1)
        hbl.addWidget(lbl_2, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        hbl.addWidget(self.cmb_2)

        vbl.addWidget(self.mmc)
        vbl.addWidget(mpl_toolbar)
        vbl.addLayout(hbl)

        self.setFocus()

        self.cmb_1.currentIndexChanged.connect(self.change_band)
        self.cmb_2.currentIndexChanged.connect(self.change_band)

    def change_band(self):
        """Combo box to choose band."""
        data = self.indata["Raster"]
        i = self.cmb_1.currentIndex()
        j = self.cmb_2.currentIndex()

        x = data[i]
        y = data[j]
        if x.data.shape != y.data.shape:
            QtWidgets.QMessageBox.warning(
                self,
                "Warning",
                "Different size input datasets. "
                "Merge and resample your input data to fix this.",
                QtWidgets.QMessageBox.StandardButton.Ok,
            )
            return

        self.mmc.update_hexbin(x, y)

    def run(self):
        """Entry point into the routine, used to run context menu item."""
        if "Raster" in self.indata:
            data = self.indata["Raster"]
        else:
            return

        if data[0].metadata["Raster"]["Section"] is True:
            self.showlog("Sections are not supported.")
            return

        if self.indata["Raster"][0].isrgb:
            self.showlog("RGB images cannot be used in this module.")
            return

        cols = [i.dataid for i in data]

        self.cmb_update(self.cmb_1, cols)
        self.cmb_update(self.cmb_2, cols, 1)

        self.change_band()
        self.show()


class PlotHist(ContextModule):
    """
    Plot Histogram Class.

    Parameters
    ----------
    parent
        Reference to the parent routine. The default is None.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle("Histogram")

        vbl = QtWidgets.QVBoxLayout(self)  # self is where layout is assigned
        hbl = QtWidgets.QHBoxLayout()
        self.mmc = MyMplCanvas(self)
        mpl_toolbar = NavigationToolbar2QT(self.mmc, self.parent)

        self.buttonbox.buttonbox.hide()
        self.buttonbox.htmlfile = "raster.cm.showhist"
        hbl.addWidget(self.buttonbox)

        self.cmb_1 = QtWidgets.QComboBox()
        lbl_1 = QtWidgets.QLabel("Bands:")
        self.cb_log = QtWidgets.QCheckBox("Log Y Axis:")
        self.cb_cum = QtWidgets.QCheckBox("Cumulative:")
        hbl.addWidget(self.cb_log)
        hbl.addWidget(self.cb_cum)
        hbl.addWidget(lbl_1, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        hbl.addWidget(self.cmb_1)

        vbl.addWidget(self.mmc)
        vbl.addWidget(mpl_toolbar)
        vbl.addLayout(hbl)

        self.setFocus()

        self.cmb_1.currentIndexChanged.connect(self.change_band)
        self.cb_log.stateChanged.connect(self.change_band)
        self.cb_cum.stateChanged.connect(self.change_band)

    def change_band(self):
        """Combo box to choose band."""
        data = self.indata["Raster"]
        i = self.cmb_1.currentIndex()
        ylog = self.cb_log.isChecked()
        iscum = self.cb_cum.isChecked()
        self.mmc.update_hist(data[i], ylog, iscum)

    def run(self):
        """Entry point into the routine, used to run context menu item."""
        if "Raster" in self.indata:
            data = self.indata["Raster"]
        else:
            return

        if self.indata["Raster"][0].isrgb:
            self.showlog("RGB images cannot be used in this module.")
            return

        cols = [i.dataid for i in data]
        self.cmb_update(self.cmb_1, cols)

        self.change_band()
        self.show()


def check_bands(data: list[Data]) -> bool:
    """
    Check that band sizes are the same.

    Parameters
    ----------
    data
        PyGMI raster dataset.

    Returns
    -------
    chk : bool
        True if sizes are the same, False otherwise.

    """
    chk = True

    dshape = data[0].data.shape
    for i in data:
        if i.data.shape != dshape:
            chk = False

    return chk


def corr2d(idat1: np.ma.MaskedArray, idat2: np.ma.MaskedArray) -> NDArray | None:
    """
    Calculate the 2D correlation.

    Parameters
    ----------
    idat1
        dataset 1 for use in correlation calculation.
    idat2
        dataset 2 for use in correlation calculation.

    Returns
    -------
    ndarray
        array of correlation coefficients
    """
    out = None

    # These next two lines are critical to keep original data safe.
    dat1 = idat1.copy()
    dat2 = idat2.copy()

    if dat1.shape == dat2.shape:
        # These line are to avoid warnings due to powers of large fill values
        mask = np.logical_or(dat1.mask, dat2.mask)
        dat1.mask = mask
        dat2.mask = mask
        dat1 = dat1.compressed()
        dat2 = dat2.compressed()

        mdat1 = dat1 - dat1.mean()
        mdat2 = dat2 - dat2.mean()
        numerator = (mdat1 * mdat2).sum()
        denominator = np.sqrt((mdat1**2).sum() * (mdat2**2).sum())
        out = numerator / denominator

    return out


def _testfn():
    """Test."""
    import sys

    from pygmi.raster.iodefs import get_raster

    ifile = r"D:\workdata\PyGMI Test Data\Raster\testdata.tif"
    ifile = r"D:\workdata\PyGMI Test Data\Raster\landscape.tif"
    ifile = r"D:\workdata\PyGMI Test Data\Magnetics\Matched Filtering\mod400.tif"
    ifile = r"D:\UBC_Files\section.tif"
    ifile = r"D:\Workdata\PyGMI Test Data\Raster\GeoTIFF\Geoscience_000001.tif"

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))

    data = get_raster(ifile)
    data[0].units = "hope"

    tmp = PlotRaster()
    tmp.indata["Raster"] = data
    tmp.run()
    tmp.exec()


if __name__ == "__main__":
    _testfn()
