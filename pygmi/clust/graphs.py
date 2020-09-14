# -----------------------------------------------------------------------------
# Name:        clust/graphs.py (part of PyGMI)
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
"""Plot Cluster Data."""

import numpy as np
from PyQt5 import QtWidgets, QtCore
from matplotlib import cm
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.ticker import MaxNLocator
from pygmi.misc import frm


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
        # figure stuff
        fig = Figure()
        self.axes = fig.add_subplot(111)
        self.line = None
        self.ind = None

        super().__init__(fig)

    def update_contour(self, data1):
        """
        Update the plot.

        Parameters
        ----------
        data1 : PyGMI Data.
            Input raster dataset.

        Returns
        -------
        None.

        """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)

        cdat = data1.data
        csp = self.axes.imshow(cdat, cmap=cm.jet, extent=data1.extent)
        vals = np.unique(cdat)
        vals = vals.compressed()
        bnds = (vals - 0.5).tolist() + [vals.max() + .5]
        self.axes.figure.colorbar(csp, boundaries=bnds, values=vals,
                                  ticks=vals)

        self.axes.set_xlabel('Eastings')
        self.axes.set_ylabel('Northings')
        self.axes.xaxis.set_major_formatter(frm)
        self.axes.yaxis.set_major_formatter(frm)
        self.figure.canvas.draw()

    def update_scatter(self, x, y):
        """
        Update the scatter plot.

        Parameters
        ----------
        x : numpy array
            X coordinates (Number of classes).
        y : numpy array
            Y Coordinates.

        Returns
        -------
        None.

        """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)

        xmin = min(x) - 0.1 * np.ptp(x)
        xmax = max(x) + 0.1 * np.ptp(x)
        ymin = min(y) - 0.1 * np.ptp(y)
        ymax = max(y) + 0.1 * np.ptp(y)

        self.axes.scatter(x, y)
        self.axes.axis([xmin, xmax, ymin, ymax])
        self.axes.set_xlabel('Number of Classes')
        self.axes.xaxis.set_ticks(x)
        self.figure.canvas.draw()

    def update_wireframe(self, x, y, z):
        """
        Update wireframe plot.

        Parameters
        ----------
        x : numpy array
            Iteration number.
        y : numpy array
            Number of classes.
        z : numpy array
            z coordinate.

        Returns
        -------
        None.

        """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111, projection='3d')
        self.axes.plot_wireframe(y, x, z)
        self.axes.set_title('log(Objective Function)')
        self.axes.set_ylabel("Number of Classes")
        self.axes.set_xlabel('Iteration')
        self.axes.yaxis.set_major_locator(MaxNLocator(integer=True))
        self.axes.xaxis.set_major_locator(MaxNLocator(integer=True))

        self.figure.canvas.draw()

    def update_membership(self, data1, mem):
        """
        Update membership plot.

        Parameters
        ----------
        data1 : PyGMI Data.
            Raster dataset.
        mem : int
            Membership.

        Returns
        -------
        None.

        """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)

        rdata = self.axes.imshow(data1.metadata['Cluster']['memdat'][mem],
                                 extent=data1.extent, cmap=cm.jet)
        self.figure.colorbar(rdata)
#        self.axes.set_title('Data')
        self.axes.set_xlabel('Eastings')
        self.axes.set_ylabel('Northings')
        self.figure.canvas.draw()


class GraphWindow(QtWidgets.QDialog):
    """
    Graph Window.

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.parent = parent

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle('Graph Window')

        vbl = QtWidgets.QVBoxLayout(self)  # self is where layout is assigned
        hbl = QtWidgets.QHBoxLayout()
        self.mmc = MyMplCanvas(self)
        mpl_toolbar = NavigationToolbar2QT(self.mmc, self.parent)

        self.combobox1 = QtWidgets.QComboBox()
        self.combobox2 = QtWidgets.QComboBox()
        self.label1 = QtWidgets.QLabel('Bands:')
        self.label2 = QtWidgets.QLabel('Bands:')

        hbl.addWidget(self.label1)
        hbl.addWidget(self.combobox1)
        hbl.addWidget(self.label2)
        hbl.addWidget(self.combobox2)

        vbl.addWidget(self.mmc)
        vbl.addWidget(mpl_toolbar)
        vbl.addLayout(hbl)

        self.setFocus()

        self.combobox1.currentIndexChanged.connect(self.change_band)
        self.combobox2.currentIndexChanged.connect(self.change_band)

    def change_band(self):
        """
        Combo to change band.

        Returns
        -------
        None.

        """


class PlotRaster(GraphWindow):
    """
    Plot Raster Class.

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.label2.hide()
        self.combobox2.hide()
        self.indata = {}
        self.parent = parent

    def change_band(self):
        """
        Combo to change band.

        Returns
        -------
        None.

        """
        i = self.combobox1.currentIndex()
        data = self.indata['Cluster']
        self.mmc.update_contour(data[i])

    def run(self):
        """
        Run.

        Returns
        -------
        None.

        """
        self.show()
        data = self.indata['Cluster']

        for i in data:
            self.combobox1.addItem(i.dataid)
        self.change_band()


class PlotMembership(GraphWindow):
    """
    Plot Fuzzy Membership data.

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.indata = {}
        self.parent = parent

    def change_band(self):
        """
        Combo to change band.

        Returns
        -------
        None.

        """
        data = self.indata['Cluster']
        i = self.combobox1.currentIndex()
        self.combobox2.clear()
        self.combobox2.currentIndexChanged.disconnect()

        for j in range(data[i].metadata['Cluster']['no_clusters']):
            self.combobox2.addItem('Membership Map for Cluster ' + str(j + 1))

        self.combobox2.currentIndexChanged.connect(self.change_band_two)
        self.change_band_two()

    def run(self):
        """
        Run.

        Returns
        -------
        None.

        """
        data = self.indata['Cluster']
        if 'memdat' not in data[0].metadata['Cluster'] or len(data[0].metadata['Cluster']['memdat']) == 0:
            return

        self.show()
        for i in data:
            self.combobox1.addItem(i.dataid)

        self.change_band()

    def change_band_two(self):
        """Combo box to choose band."""
        data = self.indata['Cluster']

        i = self.combobox1.currentIndex()
        j = self.combobox2.currentIndex()

        self.mmc.update_membership(data[i], j)


class PlotVRCetc(GraphWindow):
    """
    Plot VRC, NCE, OBJ and XBI.

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.combobox2.hide()
        self.label2.hide()
        self.parent = parent
        self.indata = {}
        if parent is None:
            self.showprocesslog = print
        else:
            self.showprocesslog = parent.showprocesslog

    def change_band(self):
        """
        Combo to change band.

        Returns
        -------
        None.

        """
        data = self.indata['Cluster']

        j = str(self.combobox1.currentText())

        if j == 'Objective Function' and data[0].metadata['Cluster']['obj_fcn'] is not None:
            x = len(data)
            y = 0
            for i in data:
                y = max(y, len(i.metadata['Cluster']['obj_fcn']))

            z = np.zeros([x, y])
            x = list(range(x))
            y = list(range(y))

            for i in x:
                for j in range(len(data[i].metadata['Cluster']['obj_fcn'])):
                    z[i, j] = data[i].metadata['Cluster']['obj_fcn'][j]

            for i in x:
                z[i][z[i] == 0] = z[i][z[i] != 0].min()

            x, y = np.meshgrid(x, y)
            x += data[0].metadata['Cluster']['no_clusters']
            self.mmc.update_wireframe(x.T, y.T, np.log(z))

        if j == 'Variance Ratio Criterion' and data[0].metadata['Cluster']['vrc'] is not None:
            x = [k.metadata['Cluster']['no_clusters'] for k in data]
            y = [k.metadata['Cluster']['vrc'] for k in data]
            self.mmc.update_scatter(x, y)
        if j == 'Normalized Class Entropy' and data[0].metadata['Cluster']['nce'] is not None:
            x = [k.metadata['Cluster']['no_clusters'] for k in data]
            y = [k.metadata['Cluster']['nce'] for k in data]
            self.mmc.update_scatter(x, y)
        if j == 'Xie-Beni Index' and data[0].metadata['Cluster']['xbi'] is not None:
            x = [k.metadata['Cluster']['no_clusters'] for k in data]
            y = [k.metadata['Cluster']['xbi'] for k in data]
            self.mmc.update_scatter(x, y)

    def run(self):
        """
        Run.

        Returns
        -------
        None.

        """
        items = []
        data = self.indata['Cluster']
        meta = data[0].metadata['Cluster']

        if 'obj_fcn' in meta:
            items += ['Objective Function']

        if 'vrc' in meta and len(data) > 1:
            items += ['Variance Ratio Criterion']

        if 'nce' in meta and len(data) > 1:
            items += ['Normalized Class Entropy']

        if 'xbi' in meta and len(data) > 1:
            items += ['Xie-Beni Index']

        if len(items) == 0:
            self.showprocesslog('Your dataset does not qualify')
            return

        self.combobox1.addItems(items)

        self.label1.setText('Graph Type:')
        self.combobox1.setCurrentIndex(0)
        self.show()
