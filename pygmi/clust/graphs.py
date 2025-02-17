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
"""Routines to plot cluster data."""

import numpy as np
from PyQt5 import QtWidgets, QtCore
from matplotlib import colormaps
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
from matplotlib.ticker import MaxNLocator

from pygmi.misc import frm, ContextModule  # , discrete_colorbar
from pygmi.raster.modest_image import imshow


class MyMplCanvas(FigureCanvasQTAgg):
    """
    Matplotlib canvas widget for the actual plot.

    Parameters
    ----------
    parent : parent
        Reference to the parent routine.
    """

    def __init__(self, parent=None):
        # figure stuff
        fig = Figure(layout='constrained')
        self.axes = fig.add_subplot(111)
        self.line = None
        self.ind = None

        super().__init__(fig)

    def update_classes(self, data1):
        """
        Update the class plot.

        Parameters
        ----------
        data1 : pygmi.raster.datatypes.Data
            Input raster dataset.

        Returns
        -------
        None.

        """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111, label='map')

        cdat = data1.data
        # csp = imshow(self.axes, cdat, cmap=colormaps['jet'],
        #              extent=data1.extent)

        # cannot use modestimage when changing colorbar labels
        csp = self.axes.imshow(cdat, cmap=colormaps['jet'],
                               extent=data1.extent)

        vals = np.unique(cdat)
        vals = vals.compressed()
        bnds = (vals - 0.5).tolist() + [vals.max() + .5]

        if 'labels' in data1.metadata['Cluster']:
            lbls = data1.metadata['Cluster']['labels']
            csp.format_cursor_data = (lambda z: f'{lbls[int(z)-1]}' if not
                                      np.ma.is_masked(z) else 'masked')
        else:
            lbls = None

        if len(vals) > 1:
            cbar = self.axes.figure.colorbar(csp, boundaries=bnds)
            cbar.set_ticks(vals, labels=lbls)

        if data1.crs.is_geographic:
            self.axes.set_xlabel('Longitude')
            self.axes.set_ylabel('Latitude')
        else:
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
        Update the wireframe plot.

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
        data1 : pygmi.raster.datatypes.Data
            Raster dataset.
        mem : int
            Membership.

        Returns
        -------
        None.

        """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)

        rdata = imshow(self.axes, data1.metadata['Cluster']['memdat'][mem],
                       extent=data1.extent, cmap=colormaps['jet'],
                       vmin=0., vmax=1.)

        self.figure.colorbar(rdata)

        if data1.crs.is_geographic:
            self.axes.set_xlabel('Longitude')
            self.axes.set_ylabel('Latitude')
        else:
            self.axes.set_xlabel('Eastings')
            self.axes.set_ylabel('Northings')

        self.figure.canvas.draw()


class GraphWindow(ContextModule):
    """
    Graph Window GUI.

    Parameters
    ----------
    parent : parent
        Reference to the parent routine.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle('Graph Window')

        vbl = QtWidgets.QVBoxLayout(self)  # self is where layout is assigned
        hbl = QtWidgets.QHBoxLayout()
        self.mmc = MyMplCanvas(self)
        mpl_toolbar = NavigationToolbar2QT(self.mmc, self.parent)

        self.cmb_1 = QtWidgets.QComboBox()
        self.cmb_2 = QtWidgets.QComboBox()
        self.lbl_1 = QtWidgets.QLabel('Bands:')
        self.lbl_2 = QtWidgets.QLabel('Bands:')

        hbl.addWidget(self.lbl_1)
        hbl.addWidget(self.cmb_1)
        hbl.addWidget(self.lbl_2)
        hbl.addWidget(self.cmb_2)

        vbl.addWidget(self.mmc)
        vbl.addWidget(mpl_toolbar)
        vbl.addLayout(hbl)

        self.setFocus()

        self.cmb_1.currentIndexChanged.connect(self.change_band)
        self.cmb_2.currentIndexChanged.connect(self.change_band)

    def change_band(self):
        """
        Combo to change band.

        Returns
        -------
        None.

        """


class PlotRaster(GraphWindow):
    """
    Plot Raster Class GUI.

    Parameters
    ----------
    parent : parent
        Reference to the parent routine.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lbl_2.hide()
        self.cmb_2.hide()

    def change_band(self):
        """
        Combo to change band.

        Returns
        -------
        None.

        """
        i = self.cmb_1.currentIndex()
        data = self.indata['Cluster']
        self.mmc.update_classes(data[i])

    def run(self):
        """
        Entry point into the routine, used to run context menu item.

        Returns
        -------
        None.

        """
        data = self.indata['Cluster']

        self.cmb_1.currentIndexChanged.disconnect()
        for i in data:
            self.cmb_1.addItem(i.dataid)

        self.cmb_1.currentIndexChanged.connect(self.change_band)

        self.show()
        self.change_band()


class PlotMembership(GraphWindow):
    """
    Plot Fuzzy Membership data GUI.

    Parameters
    ----------
    parent : parent
        Reference to the parent routine.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.lbl_1.setText('Number of Clusters:')
        self.lbl_2.setText('Membership:')

    def change_band(self):
        """
        Combo to change band.

        Returns
        -------
        None.

        """
        data = self.indata['Cluster']
        i = self.cmb_1.currentIndex()
        self.cmb_2.clear()
        self.cmb_2.currentIndexChanged.disconnect()

        for j in range(data[i].metadata['Cluster']['no_clusters']):
            self.cmb_2.addItem('Membership Map for Cluster ' + str(j + 1))

        self.cmb_2.currentIndexChanged.connect(self.change_band_two)
        self.change_band_two()

    def run(self):
        """
        Entry point into the routine, used to run context menu item.

        Returns
        -------
        None.

        """
        data = self.indata['Cluster']
        if ('memdat' not in data[0].metadata['Cluster'] or
                len(data[0].metadata['Cluster']['memdat']) == 0):
            self.showlog('No membership data.')
            return

        self.show()
        for i in data:
            self.cmb_1.addItem(i.dataid)

        self.change_band()

    def change_band_two(self):
        """Combo box to choose band."""
        data = self.indata['Cluster']

        i = self.cmb_1.currentIndex()
        j = self.cmb_2.currentIndex()

        self.mmc.update_membership(data[i], j)


class PlotVRCetc(GraphWindow):
    """
    Plot VRC, NCE, OBJ and XBI GUI.

    Parameters
    ----------
    parent : parent
        Reference to the parent routine.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cmb_2.hide()
        self.lbl_2.hide()

    def change_band(self):
        """
        Combo to change band.

        Returns
        -------
        None.

        """
        data = self.indata['Cluster']

        j = str(self.cmb_1.currentText())

        if (j == 'Objective Function' and
                data[0].metadata['Cluster']['obj_fcn'] is not None):
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

        if (j == 'Variance Ratio Criterion' and
                data[0].metadata['Cluster']['vrc'] is not None):
            x = [k.metadata['Cluster']['no_clusters'] for k in data]
            y = [k.metadata['Cluster']['vrc'] for k in data]
            self.mmc.update_scatter(x, y)

        # nce and xbi are fuzzy clustering only.
        if (j == 'Normalized Class Entropy' and
                data[0].metadata['Cluster']['nce'] is not None):
            x = [k.metadata['Cluster']['no_clusters'] for k in data]
            y = [k.metadata['Cluster']['nce'] for k in data]
            self.mmc.update_scatter(x, y)
        if (j == 'Xie-Beni Index' and
                data[0].metadata['Cluster']['xbi'] is not None):
            x = [k.metadata['Cluster']['no_clusters'] for k in data]
            y = [k.metadata['Cluster']['xbi'] for k in data]
            self.mmc.update_scatter(x, y)

    def run(self):
        """
        Entry point into the routine, used to run context menu item.

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
            self.showlog('Your dataset does not qualify')
            return

        self.cmb_1.clear()
        self.cmb_1.addItems(items)

        self.lbl_1.setText('Graph Type:')
        self.cmb_1.setCurrentIndex(0)
        self.show()


def _testfn():
    """Test."""
    import sys
    from pygmi.raster.iodefs import get_raster
    from pygmi.clust.cluster import Cluster

    ifile = r"D:\workdata\PyGMI Test Data\Classification\Cut_K_Th_U.ers"

    data = get_raster(ifile)

    app = QtWidgets.QApplication(sys.argv)

    DM = Cluster()
    DM.indata['Raster'] = data
    DM.settings()

    dat = DM.outdata
    dat['Cluster'][0].metadata['Cluster']['labels'] = ['a', 'b', 'c', 'd', 'e']

    tmp2 = PlotRaster()
    tmp2.indata = dat
    tmp2.run()

    app.exec()


if __name__ == "__main__":
    _testfn()