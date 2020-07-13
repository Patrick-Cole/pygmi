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
import matplotlib.cm as cm
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
import matplotlib.colors as mcolors
from pygmi.raster.modest_image import imshow


class MyMplCanvas(FigureCanvas):
    """
    Canvas for the actual plot.

    Attributes
    ----------
    axes : matplotlib subplot
    parent : parent
        reference to the parent routine
    """

    def __init__(self, parent=None):
        fig = Figure()
        self.axes = fig.add_subplot(111)
        super().__init__(fig)

    def update_pcolor(self, data1, dmat):
        """
        Update the correlation coefficient plot

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
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        self.axes.pcolor(dmat)
        self.axes.axis('scaled')
        self.axes.set_title('Correlation Coefficients')
        for i in range(len(data1)):
            for j in range(len(data1)):
                self.axes.text(i + .1, j + .4, format(float(dmat[i, j]),
                                                      '4.2f'))
        dat_mat = [i.dataid for i in data1]
        self.axes.set_xticks(np.array(list(range(len(data1)))) + .5)

        self.axes.set_xticklabels(dat_mat, rotation='vertical')
        self.axes.set_yticks(np.array(list(range(len(data1)))) + .5)

        self.axes.set_yticklabels(dat_mat, rotation='horizontal')
        self.axes.set_xlim(0, len(data1))
        self.axes.set_ylim(0, len(data1))

        self.figure.tight_layout()
        self.figure.canvas.draw()

    def update_raster(self, data1):
        """
        Update the raster plot.

        Parameters
        ----------
        data1 : PyGMI raster Data
            raster dataset to be used in contouring

        Returns
        -------
        None.

        """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        self.axes.tick_params(axis='x', rotation=90)
        self.axes.tick_params(axis='y', rotation=0)
        self.axes.ticklabel_format(style='plain')

        extent = data1.extent

        # carray = data1.data.compressed()
        # ichk = np.array_equal(carray, carray.astype(int))

        # rdata = self.axes.imshow(data1.data, extent=extent, cmap=cm.jet,
        #                          interpolation='nearest')

        rdata = imshow(self.axes, data1.data, extent=extent, cmap=cm.jet,
                       interpolation='nearest')

        # if ichk and not data1.isrgb:
        #     vals = np.unique(data1.data)
        #     vals = vals.compressed()
        #     bnds = (vals - 0.5).tolist() + [vals.max() + .5]
        #     cbar = self.axes.figure.colorbar(rdata, boundaries=bnds,)
        #                                      # values=vals, ticks=vals)
        #     cbar.set_label(data1.units)
        # elif not data1.isrgb:
        #     cbar = self.figure.colorbar(rdata)
        #     cbar.set_label(data1.units)

        if not data1.isrgb:
            cbar = self.figure.colorbar(rdata)
            cbar.set_label(data1.units)

        self.axes.set_xlabel('Eastings')
        self.axes.set_ylabel('Northings')

        self.figure.tight_layout()
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
        x = data1.copy()
        y = data2.copy()

        msk = np.logical_or(x.mask, y.mask)
        x.mask = msk
        y.mask = msk
        x = x.compressed()
        y = y.compressed()

        xmin = x.min()
        xmax = x.max()
        ymin = y.min()
        ymax = y.max()

        hbin = self.axes.hexbin(x, y, bins='log')
        self.axes.axis([xmin, xmax, ymin, ymax])
        self.axes.set_title('Hexbin Plot')
        cbar = self.figure.colorbar(hbin)
        cbar.set_label('log10(N)')

        self.figure.tight_layout()
        self.figure.canvas.draw()

    def update_wireframe(self, data):
        """
        Update the surface wireframe plot.

        Parameters
        ----------
        data : PyGMI raster Data
            raster dataset to be used

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

        if not np.ma.is_masked(z):
            z = np.ma.array(z)

        x = np.ma.array(x, mask=z.mask)
        y = np.ma.array(y, mask=z.mask)

        cmap = cm.jet

        norml = mcolors.Normalize(vmin=z.min(), vmax=z.max())

        z.data[z.mask] = np.nan
        z = z.data

        self.figure.clear()
        self.axes = self.figure.add_subplot(111, projection='3d')
        ax1 = self.axes

        surf = ax1.plot_surface(x, y, z, cmap=cmap, linewidth=0.1, norm=norml,
                                vmin=z.min(), vmax=z.max(), shade=False,
                                antialiased=False)

        self.figure.colorbar(surf)

        ax1.set_title('')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        self.figure.canvas.draw()

    def update_hist(self, data1):
        """
        Update the hiostogram plot.

        Parameters
        ----------
        data1 : PyGMI raster Data
            raster dataset to be used

        Returns
        -------
        None.

        """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)

        dattmp = data1.data[data1.data.mask == 0].flatten()
        self.axes.hist(dattmp, 50)
        self.axes.set_title(data1.dataid, fontsize=12)
        self.axes.set_xlabel('Data Value', fontsize=8)
        self.axes.set_ylabel('Counts', fontsize=8)

        self.figure.tight_layout()
        self.figure.canvas.draw()


class GraphWindow(QtWidgets.QDialog):
    """
    Graph Window - The QDialog window which will contain our image.

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle('Graph Window')

        vbl = QtWidgets.QVBoxLayout(self)  # self is where layout is assigned
        self.hbl = QtWidgets.QHBoxLayout()
        self.mmc = MyMplCanvas(self)
        mpl_toolbar = NavigationToolbar2QT(self.mmc, self.parent)

        self.combobox1 = QtWidgets.QComboBox()
        self.combobox2 = QtWidgets.QComboBox()
        self.label1 = QtWidgets.QLabel('Bands:')
        self.label2 = QtWidgets.QLabel('Bands:')
        self.hbl.addWidget(self.label1)
        self.hbl.addWidget(self.combobox1)
        self.hbl.addWidget(self.label2)
        self.hbl.addWidget(self.combobox2)

        vbl.addWidget(self.mmc)
        vbl.addWidget(mpl_toolbar)
        vbl.addLayout(self.hbl)

        self.setFocus()

        self.combobox1.currentIndexChanged.connect(self.change_band)
        self.combobox2.currentIndexChanged.connect(self.change_band)

    def change_band(self):
        """
        Combo box to choose band.

        Returns
        -------
        None.

        """


class PlotCCoef(GraphWindow):
    """
    Plot 2D Correlation Coeffiecients

    Attributes
    ----------
    label1 : QLabel
        reference to GraphWindow's label1
    combobox1 : QComboBox
        reference to GraphWindow's combobox1
    label2 : QLabel
        reference to GraphWindow's label2
    combobox2 : QComboBox
        reference to GraphWindow's combobox2
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.label1.hide()
        self.combobox1.hide()
        self.label2.hide()
        self.combobox2.hide()
        self.indata = {}
        self.parent = parent

    def change_band(self):
        """
        Combo box to choose band.

        Returns
        -------
        None.

        """

    def run(self):
        """
        Run.

        Returns
        -------
        None.

        """
        data = self.indata['Raster']

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

        self.mmc.update_pcolor(data, dummy_mat)


def check_bands(data):
    """
    Checks that band sizes are the same.

    Parameters
    ----------
    data : PyGMI Data
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
    if dat1.shape == dat2.shape:
        # These line are to avoid warnings due to powers of large fill values
        mask = np.logical_or(dat1.mask, dat1.mask)
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


class PlotRaster(GraphWindow):
    """
    Plot Raster Class.

    Attributes
    ----------
    label2 : QLabel
        reference to GraphWindow's label2
    combobox2 : QComboBox
        reference to GraphWindow's combobox2
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.label2.hide()
        self.combobox2.hide()
        self.indata = {}
        self.parent = parent

    def change_band(self):
        """
        Combo box to choose band.

        Returns
        -------
        None.

        """
        i = self.combobox1.currentIndex()
        if 'Raster' in self.indata:
            data = self.indata['Raster']
            self.mmc.update_raster(data[i])

    def run(self):
        """
        Run.

        Returns
        -------
        None.

        """
        self.show()
        if 'Raster' in self.indata:
            data = self.indata['Raster']
        elif 'Cluster' in self.indata:
            data = self.indata['Cluster']

        for i in data:
            self.combobox1.addItem(i.dataid)
#        self.change_band()


class PlotSurface(GraphWindow):
    """
    Plot Raster Class.

    Attributes
    ----------
    label2 : QLabel
        reference to GraphWindow's label2
    combobox2 : QComboBox
        reference to GraphWindow's combobox2
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.label2.hide()
        self.combobox2.hide()
        self.indata = {}
        self.parent = parent

    def change_band(self):
        """
        Combo box to choose band.

        Returns
        -------
        None.

        """
        i = self.combobox1.currentIndex()
        if 'Raster' in self.indata:
            data = self.indata['Raster']
            self.mmc.update_wireframe(data[i])

    def run(self):
        """
        Run.

        Returns
        -------
        None.

        """
        if 'Raster' in self.indata:
            self.show()
            data = self.indata['Raster']

            for i in data:
                self.combobox1.addItem(i.dataid)
            self.change_band()


class PlotScatter(GraphWindow):
    """
    Plot Hexbin Class. A Hexbin is a type of scatter plot which is raster.

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    """

    def __init__(self, parent):
        super().__init__(parent)

        self.indata = {}
        self.parent = parent

    def change_band(self):
        """
        Combo box to choose band.

        Returns
        -------
        None.

        """
        data = self.indata['Raster']
        i = self.combobox1.currentIndex()
        j = self.combobox2.currentIndex()

        x = data[i].data
        y = data[j].data
        if x.mask.shape != y.mask.shape:
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
        self.show()
        data = self.indata['Raster']
        for i in data:
            self.combobox1.addItem(i.dataid)
            self.combobox2.addItem(i.dataid)

        self.label1.setText('X Band:')
        self.label2.setText('Y Band:')
        self.combobox1.setCurrentIndex(0)
        self.combobox2.setCurrentIndex(1)


class PlotHist(GraphWindow):
    """
    Plot Hist Class

    Attributes
    ----------
    label2 : QLabel
        reference to GraphWindow's label2
    combobox2 : QComboBox
        reference to GraphWindow's combobox2
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.label2.hide()
        self.combobox2.hide()
        self.indata = {}
        self.parent = parent

    def change_band(self):
        """
        Combo box to choose band.

        Returns
        -------
        None.

        """
        data = self.indata['Raster']
        i = self.combobox1.currentIndex()
        self.mmc.update_hist(data[i])

    def run(self):
        """
        Run.

        Returns
        -------
        None.

        """
        self.show()
        data = self.indata['Raster']
        for i in data:
            self.combobox1.addItem(i.dataid)

        self.label1.setText('Band:')
        self.combobox1.setCurrentIndex(0)
        self.change_band()
