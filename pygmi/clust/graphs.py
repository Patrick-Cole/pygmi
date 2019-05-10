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
""" Plot Cluster Data """

import numpy as np
from PyQt5 import QtWidgets, QtCore
from matplotlib import cm
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT


class MyMplCanvas(FigureCanvas):
    """
    Canvas for the actual plot

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

        FigureCanvas.__init__(self, fig)

    def update_contour(self, data1):
        """ Update the plot """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)

        cdat = data1.data
        csp = self.axes.imshow(cdat, cmap=cm.jet, extent=data1.extent)
        vals = np.unique(cdat)
        vals = vals.compressed()
        bnds = (vals - 0.5).tolist() + [vals.max() + .5]
        self.axes.figure.colorbar(csp, boundaries=bnds, values=vals,
                                  ticks=vals)

        self.axes.set_xlabel("Eastings")
        self.axes.set_ylabel("Northings")
        self.figure.canvas.draw()

    def update_scatter(self, x, y):
        """ Update the plot """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)

        xmin = min(x) - 0.1 * np.ptp(x)
        xmax = max(x) + 0.1 * np.ptp(x)
        ymin = min(y) - 0.1 * np.ptp(y)
        ymax = max(y) + 0.1 * np.ptp(y)

        self.axes.scatter(x, y)
        self.axes.axis([xmin, xmax, ymin, ymax])
        self.axes.set_xlabel("Number of Classes")
        self.axes.xaxis.set_ticks(x)
        self.figure.canvas.draw()


class GraphWindow(QtWidgets.QDialog):
    """
    Graph Window

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Graph Window")

        vbl = QtWidgets.QVBoxLayout(self)  # self is where layout is assigned
        hbl = QtWidgets.QHBoxLayout()
        self.mmc = MyMplCanvas(self)
        mpl_toolbar = NavigationToolbar2QT(self.mmc, self.parent)

        self.combobox1 = QtWidgets.QComboBox()
        self.combobox2 = QtWidgets.QComboBox()
        self.label1 = QtWidgets.QLabel()
        self.label2 = QtWidgets.QLabel()

        self.label1.setText('Bands:')
        self.label2.setText('Bands:')

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
        """ Combo box to choose band """


class PlotRaster(GraphWindow):
    """
    Plot Raster Class

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    """
    def __init__(self, parent):
        GraphWindow.__init__(self, parent)
        self.label2.hide()
        self.combobox2.hide()
        self.indata = {}
        self.parent = parent

    def change_band(self):
        """ Combo box to choose band """
        i = self.combobox1.currentIndex()
        data = self.indata['Cluster']
        self.mmc.update_contour(data[i])

    def run(self):
        """ Run """
        self.show()
        data = self.indata['Cluster']

        for i in data:
            self.combobox1.addItem(i.dataid)
        self.change_band()


class PlotVRCetc(GraphWindow):
    """
    Plot VRC, NCE, OBJ and XBI

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    """
    def __init__(self, parent):
        GraphWindow.__init__(self, parent)
        self.combobox2.hide()
        self.label2.hide()
        self.parent = parent
        self.indata = {}

    def change_band(self):
        """ Combo box to choose band """
        data = self.indata['Cluster']

        j = str(self.combobox1.currentText())

        if j == 'Variance Ratio Criterion' and data[0].vrc is not None:
            x = np.array([k.no_clusters for k in data], dtype=int)
            y = [k.vrc for k in data]
            self.mmc.update_scatter(x, y)

    def run(self):
        """ Run """
        items = []
        data = self.indata['Cluster']

        if data[0].vrc is not None:
            items += ['Variance Ratio Criterion']

        if not items or len(data) == 1:
            self.parent.showprocesslog('Your dataset does not qualify')
            return

        self.combobox1.addItems(items)

        self.label1.setText('Graph Type:')
        self.combobox1.setCurrentIndex(0)
        self.show()
