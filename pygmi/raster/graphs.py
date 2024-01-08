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
"""

import numpy as np
from PyQt5 import QtWidgets, QtCore
from matplotlib import colormaps
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
import matplotlib.colors as mcolors

from pygmi.misc import frm, ContextModule
from pygmi.raster.modest_image import imshow


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

    def update_ccoef(self, data1, dmat):
        """
        Update the correlation coefficient plot.

        Parameters
        ----------
        data1 : PyGMI raster Data
            raster dataset to be used in contouring.
        dmat : numpy array
            dummy matrix of numbers to be plotted using pcolor.

        Returns
        -------
        None.

        """
        cmap = colormaps['viridis']

        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        rdata = self.axes.pcolormesh(dmat, cmap=cmap)
        self.axes.axis('scaled')
        self.axes.set_title('Correlation Coefficients')
        for i in range(len(data1)):
            for j in range(len(data1)):
                ctmp = np.array([1., 1., 1., 0.]) - np.array(cmap(dmat[i, j]))
                ctmp = np.abs(ctmp)
                ctmp = ctmp.tolist()

                if dmat[i, j] < 0.01:
                    atext = f'{dmat[i, j]:.1e}'
                else:
                    atext = f'{dmat[i, j]:.2f}'

                self.axes.text(i+.5, j+.5, atext, c=ctmp, rotation=45,
                               ha='center', va='center')
        dat_mat = [i.dataid for i in data1]
        self.axes.set_xticks(np.array(list(range(len(data1)))) + .5)

        self.axes.set_xticklabels(dat_mat, rotation='vertical')
        self.axes.set_yticks(np.array(list(range(len(data1)))) + .5)

        self.axes.set_yticklabels(dat_mat, rotation='horizontal')
        self.axes.set_xlim(0, len(data1))
        self.axes.set_ylim(0, len(data1))

        cbar = self.figure.colorbar(rdata, format=frm)

        self.figure.canvas.draw()

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
        self.axes = self.figure.add_subplot(111)

        rdata = imshow(self.axes, data1.data, extent=data1.extent,
                       cmap=colormaps[cmap], interpolation='nearest')

        if not data1.isrgb:
            rdata.set_clim_std(2.5)
            cbar = self.figure.colorbar(rdata, format=frm)
            cbar.set_label(data1.units)

        if data1.crs.is_geographic:
            self.axes.set_xlabel('Longitude')
            self.axes.set_ylabel('Latitude')
        else:
            self.axes.set_xlabel('Eastings')
            self.axes.set_ylabel('Northings')

        self.axes.xaxis.set_major_formatter(frm)
        self.axes.yaxis.set_major_formatter(frm)

        self.figure.canvas.draw()

    def update_hexbin(self, data1, data2):
        """
        Update the hexbin plot.

        Parameters
        ----------
        data1 : PyGMI raster Data
            raster dataset to be used
        data2 : PyGMI raster Data
            raster dataset to be used

        Returns
        -------
        None.

        """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
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

        hbin = self.axes.hexbin(x, y, bins='log', cmap='inferno')
        self.axes.axis([xmin, xmax, ymin, ymax])
        self.axes.set_title('Hexbin Plot')
        cbar = self.figure.colorbar(hbin, format=frm)
        cbar.set_label('log10(N)')

        self.axes.xaxis.set_major_formatter(frm)
        self.axes.yaxis.set_major_formatter(frm)

        self.axes.set_xlabel(data1.units)
        self.axes.set_ylabel(data2.units)

        self.figure.canvas.draw()

    def update_surface(self, data, cmap):
        """
        Update the surface plot.

        Parameters
        ----------
        data : PyGMI raster Data
            raster dataset to be used
        cmap : str
            Matplotlib colormap description

        Returns
        -------
        None.

        """
        rows, cols = data.data.shape

        dtlx = data.extent[0]
        dtly = data.extent[-1]
        x = dtlx+np.arange(cols)*data.xdim+data.xdim/2
        y = dtly-np.arange(rows)*data.ydim-data.ydim/2
        x, y = np.meshgrid(x, y)
        z = data.data.copy()
        vmin, vmax = np.percentile(z.compressed(), [1, 99])

        # z[z < vmin] = vmin
        # z[z > vmax] = vmax

        if not np.ma.is_masked(z):
            z = np.ma.array(z)

        x = np.ma.array(x, mask=z.mask)
        y = np.ma.array(y, mask=z.mask)

        cmap = colormaps[cmap]

        norml = mcolors.Normalize(vmin=vmin, vmax=vmax)

        z.data[z.mask] = np.nan
        z = z.data

        self.figure.clear()
        self.axes = self.figure.add_subplot(111, projection='3d')

        vmin, vmax = np.percentile(z, [1, 99])

        surf = self.axes.plot_surface(x, y, z, cmap=cmap,
                                      norm=norml, vmin=vmin, vmax=vmax,
                                      shade=False, antialiased=False)

        self.figure.colorbar(surf, format=frm)

        self.axes.xaxis.set_major_formatter(frm)
        self.axes.yaxis.set_major_formatter(frm)
        self.axes.zaxis.set_major_formatter(frm)

        self.axes.set_title('')
        self.axes.set_xlabel('X')
        self.axes.set_ylabel('Y')
        self.axes.set_zlabel('Z')

        self.figure.canvas.draw()

    def update_hist(self, data1, ylog, iscum):
        """
        Update the histogram plot.

        Parameters
        ----------
        data1 : PyGMI raster Data
            raster dataset to be used
        ylog : bool
            Boolean for a log scale on y-axis.
        iscum : bool
            Boolean for a cumulative distribution.

        Returns
        -------
        None.

        """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)

        dattmp = data1.data[data1.data.mask == 0].flatten()
        self.axes.hist(dattmp, bins='sqrt', cumulative=iscum)
        self.axes.set_title(data1.dataid)
        self.axes.set_xlabel('Data Value')
        self.axes.set_ylabel('Counts')

        self.axes.xaxis.set_major_formatter(frm)
        self.axes.yaxis.set_major_formatter(frm)

        # self.axes.tick_params(axis='x', labelsize=14)
        # self.axes.tick_params(axis='y', labelsize=14)

        if ylog is True:
            self.axes.set_yscale('log')

        self.figure.canvas.draw()


class PlotCCoef(ContextModule):
    """Plot 2D Correlation Coefficients."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle('Correlation Coefficients')

        vbl = QtWidgets.QVBoxLayout(self)  # self is where layout is assigned
        self.mmc = MyMplCanvas(self)
        mpl_toolbar = NavigationToolbar2QT(self.mmc, self.parent)

        vbl.addWidget(self.mmc)
        vbl.addWidget(mpl_toolbar)

        self.setMinimumSize(600, 600)

        self.setFocus()

    def run(self):
        """
        Run.

        Returns
        -------
        None.

        """
        if 'Raster' in self.indata:
            data = self.indata['Raster']
        else:
            return

        if self.indata['Raster'][0].isrgb:
            self.showlog('RGB images cannot be used in this module.')
            return

        if not check_bands(data):
            QtWidgets.QMessageBox.warning(self, 'Warning',
                                          'Different size input datasets. '
                                          'Merge and resample your input data '
                                          'to fix this.',
                                          QtWidgets.QMessageBox.Ok)
            return

        self.show()

        dummy_mat = [[corr2d(i.data, j.data) for j in data] for i in data]
        dummy_mat = np.array(dummy_mat)

        self.mmc.update_ccoef(data, dummy_mat)


class PlotRaster(ContextModule):
    """Plot Raster Class."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle('Raster Plot (Simple)')

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

        vbl.addWidget(self.mmc)
        vbl.addWidget(mpl_toolbar)
        vbl.addLayout(hbl)

        self.setFocus()
        self.cmb_1.currentIndexChanged.connect(self.change_band)
        self.cmb_2.currentIndexChanged.connect(self.change_band)

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

    def run(self):
        """
        Run.

        Returns
        -------
        None.

        """
        self.show()
        data = []
        if 'Raster' in self.indata:
            data = self.indata['Raster']
        elif 'Cluster' in self.indata:
            data = self.indata['Cluster']

        for i in data:
            self.cmb_1.addItem(i.dataid)


class PlotSurface(ContextModule):
    """Plot Surface Class."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle('Surface Plot')

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

        vbl.addWidget(self.mmc)
        vbl.addWidget(mpl_toolbar)
        vbl.addLayout(hbl)

        self.setFocus()

        self.cmb_1.currentIndexChanged.connect(self.change_band)
        self.cmb_2.currentIndexChanged.connect(self.change_band)

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
            self.mmc.update_surface(data[i], cmap)

    def run(self):
        """
        Run.

        Returns
        -------
        None.

        """
        if 'Raster' in self.indata:
            data = self.indata['Raster']
        else:
            return

        if self.indata['Raster'][0].isrgb:
            self.showlog('RGB images cannot be used in this module.')
            return

        self.show()
        for i in data:
            self.cmb_1.addItem(i.dataid)
        self.change_band()


class PlotScatter(ContextModule):
    """
    Plot Hexbin Class.

    A Hexbin is a type of scatter plot which is raster.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle('Hexbin Plot')

        vbl = QtWidgets.QVBoxLayout(self)  # self is where layout is assigned
        hbl = QtWidgets.QHBoxLayout()
        self.mmc = MyMplCanvas(self)
        mpl_toolbar = NavigationToolbar2QT(self.mmc, self.parent)

        self.cmb_1 = QtWidgets.QComboBox()
        self.cmb_2 = QtWidgets.QComboBox()
        lbl_1 = QtWidgets.QLabel('X Band:')
        lbl_2 = QtWidgets.QLabel('Y Band:')
        hbl.addWidget(lbl_1)
        hbl.addWidget(self.cmb_1)
        hbl.addWidget(lbl_2)
        hbl.addWidget(self.cmb_2)

        vbl.addWidget(self.mmc)
        vbl.addWidget(mpl_toolbar)
        vbl.addLayout(hbl)

        self.setFocus()

        self.cmb_1.currentIndexChanged.connect(self.change_band)
        self.cmb_2.currentIndexChanged.connect(self.change_band)

    def change_band(self):
        """
        Combo box to choose band.

        Returns
        -------
        None.

        """
        data = self.indata['Raster']
        i = self.cmb_1.currentIndex()
        j = self.cmb_2.currentIndex()

        x = data[i]
        y = data[j]
        if x.data.shape != y.data.shape:
            QtWidgets.QMessageBox.warning(self, 'Warning',
                                          'Different size input datasets. '
                                          'Merge and resample your input data '
                                          'to fix this.',
                                          QtWidgets.QMessageBox.Ok)
            return

        self.mmc.update_hexbin(x, y)

    def run(self):
        """
        Run.

        Returns
        -------
        None.

        """
        if 'Raster' in self.indata:
            data = self.indata['Raster']
        else:
            return

        if self.indata['Raster'][0].isrgb:
            self.showlog('RGB images cannot be used in this module.')
            return

        self.show()
        for i in data:
            self.cmb_1.addItem(i.dataid)
            self.cmb_2.addItem(i.dataid)

        self.cmb_1.setCurrentIndex(0)
        self.cmb_2.setCurrentIndex(1)


class PlotHist(ContextModule):
    """Plot Histogram Class."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle('Histogram')

        vbl = QtWidgets.QVBoxLayout(self)  # self is where layout is assigned
        hbl = QtWidgets.QHBoxLayout()
        self.mmc = MyMplCanvas(self)
        mpl_toolbar = NavigationToolbar2QT(self.mmc, self.parent)

        self.cmb_1 = QtWidgets.QComboBox()
        lbl_1 = QtWidgets.QLabel('Bands:')
        self.cb_log = QtWidgets.QCheckBox('Log Y Axis:')
        self.cb_cum = QtWidgets.QCheckBox('Cumulative:')
        hbl.addWidget(self.cb_log)
        hbl.addWidget(self.cb_cum)
        hbl.addWidget(lbl_1)
        hbl.addWidget(self.cmb_1)

        vbl.addWidget(self.mmc)
        vbl.addWidget(mpl_toolbar)
        vbl.addLayout(hbl)

        self.setFocus()

        self.cmb_1.currentIndexChanged.connect(self.change_band)
        self.cb_log.stateChanged.connect(self.change_band)
        self.cb_cum.stateChanged.connect(self.change_band)

    def change_band(self):
        """
        Combo box to choose band.

        Returns
        -------
        None.

        """
        data = self.indata['Raster']
        i = self.cmb_1.currentIndex()
        ylog = self.cb_log.isChecked()
        iscum = self.cb_cum.isChecked()
        self.mmc.update_hist(data[i], ylog, iscum)

    def run(self):
        """
        Run.

        Returns
        -------
        None.

        """
        if 'Raster' in self.indata:
            data = self.indata['Raster']
        else:
            return

        if self.indata['Raster'][0].isrgb:
            self.showlog('RGB images cannot be used in this module.')
            return

        self.show()
        for i in data:
            self.cmb_1.addItem(i.dataid)

        self.cmb_1.setCurrentIndex(0)
        self.change_band()


def check_bands(data):
    """
    Check that band sizes are the same.

    Parameters
    ----------
    data : list of PyGMI Data
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


def corr2d(dat1, dat2):
    """
    Calculate the 2D correlation.

    Parameters
    ----------
    dat1 : numpy array
        dataset 1 for use in correlation calculation.
    dat2 : numpy array
        dataset 2 for use in correlation calculation.

    Returns
    -------
    out : numpy array
        array of correlation coefficients
    """
    out = None

    # These next two lines are critical to keep original data safe.
    dat1 = dat1.copy()
    dat2 = dat2.copy()

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
        denominator = np.sqrt((mdat1 ** 2).sum() * (mdat2 ** 2).sum())
        out = numerator / denominator

    return out


def _testfn():
    """Test."""
    from pygmi.raster.iodefs import get_raster
    import matplotlib.pyplot as plt

    ifile = r"d:/Workdata/LULC/2001_stack_norm.tif"

    data = get_raster(ifile)

    cormat = np.array([[corr2d(i.data, j.data) for j in data] for i in data])

    print(cormat)

    plt.figure(dpi=150)
    plt.imshow(cormat, cmap='jet')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    _testfn()
