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
from PyQt4 import QtGui, QtCore
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d  # this is used, ignore warning
from matplotlib.backends.backend_qt4agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as \
    NavigationToolbar


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
#        self.background = None
        self.parent = parent

        FigureCanvas.__init__(self, fig)

#        self.figure.canvas.mpl_connect('pick_event', self.onpick)
#        self.figure.canvas.mpl_connect('button_release_event',
#                                       self.button_release_callback)
#        self.figure.canvas.mpl_connect('motion_notify_event',
#                                       self.motion_notify_callback)

#    def button_release_callback(self, event):
#        """ mouse button release """
#        if event.inaxes is None:
#            return
#        if event.button != 1:
#            return
#        self.ind = None

#    def motion_notify_callback(self, event):
#        """ move mouse """
#        if event.inaxes is None:
#            return
#        if event.button != 1:
#            return
#        if self.ind is None:
#            return
#
#        y = event.ydata
#        dtmp = self.line.get_data()
#        dtmp[1][self.ind] = y
#        self.line.set_data(dtmp[0], dtmp[1])
#
##        self.figure.canvas.restore_region(self.background)
#        self.axes.draw_artist(self.line)
##        self.figure.canvas.blit(self.axes.bbox)
#        self.figure.canvas.update()

#    def onpick(self, event):
#        """ Picker event """
#        if event.mouseevent.inaxes is None:
#            return
#        if event.mouseevent.button != 1:
#            return
#        if event.artist != self.line:
#            return True
#
#        self.ind = event.ind
#        self.ind = self.ind[len(self.ind) / 2]  # get center-ish value
#
#        return True

    def update_contour(self, data1):
        """ Update the plot """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)

        extent = (data1.tlx, data1.tlx + data1.cols * data1.xdim,
                  data1.tly - data1.rows * data1.ydim, data1.tly)

        cdat = data1.data + 1
        csp = self.axes.imshow(cdat, cmap=plt.cm.jet, extent=extent)
        vals = np.unique(cdat)
        vals = vals.compressed()
        bnds = (vals - 0.5).tolist() + [vals.max() + .5]
        self.axes.figure.colorbar(csp, boundaries=bnds, values=vals,
                                  ticks=vals)

#        self.axes.set_title('Data')
        self.axes.set_xlabel("Eastings")
        self.axes.set_ylabel("Northings")
        self.figure.canvas.draw()

    def update_raster(self, data1, data2=None):
        """ Update the plot """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)

        extent = (data1.tlx, data1.tlx + data1.cols * data1.xdim,
                  data1.tly - data1.rows * data1.ydim, data1.tly)

        rdata = self.axes.imshow(data1.data, extent=extent,
                                 interpolation='nearest')

        if data2 is not None:
            self.axes.plot(data2.xdata, data2.ydata, '.')

        cbar = self.figure.colorbar(rdata)
        try:
            cbar.set_label(data1.units)
        except AttributeError:
            pass
        self.axes.set_xlabel("Eastings")
        self.axes.set_ylabel("Northings")
        self.figure.canvas.draw()

    def update_membership(self, data1, mem):
        """ Update the plot """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)

        extent = (data1.tlx, data1.tlx + data1.cols * data1.xdim,
                  data1.tly - data1.rows * data1.ydim, data1.tly)

        rdata = self.axes.imshow(data1.memdat[mem], extent=extent)
        self.figure.colorbar(rdata)
#        self.axes.set_title('Data')
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
        self.figure.canvas.draw()

    def update_wireframe(self, x, y, z):
        """ Update the plot """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111, projection='3d')
        self.axes.plot_wireframe(x, y, z)
        self.axes.set_title('log(Objective Function)')
        self.axes.set_xlabel("Number of Classes")
        self.axes.set_ylabel("Iteration")
        self.figure.canvas.draw()


class GraphWindow(QtGui.QDialog):
    """
    Graph Window

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    """
    def __init__(self, parent=None):
        QtGui.QDialog.__init__(self, parent=None)
        self.parent = parent

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Graph Window")

        vbl = QtGui.QVBoxLayout(self)  # self is where layout is assigned
        hbl = QtGui.QHBoxLayout()
        self.mmc = MyMplCanvas(self)
        mpl_toolbar = NavigationToolbar(self.mmc, self.parent)

        self.combobox1 = QtGui.QComboBox()
        self.combobox2 = QtGui.QComboBox()
        self.label1 = QtGui.QLabel()
        self.label2 = QtGui.QLabel()

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
        pass


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
        data2 = None
        if 'Point' in self.indata:
            data2 = self.indata['Point'][0]
        if 'Raster' in self.indata:
            data = self.indata['Raster']
            self.mmc.update_raster(data[i], data2)
        elif 'Cluster' in self.indata:
            data = self.indata['Cluster']
            self.mmc.update_contour(data[i])
        elif 'ProfPic' in self.indata:
            data = self.indata['ProfPic']
            self.mmc.update_rgb(data[i])

    def run(self):
        """ Run """
        self.show()
        if 'Raster' in self.indata:
            data = self.indata['Raster']
        elif 'Cluster' in self.indata:
            data = self.indata['Cluster']
        elif 'ProfPic' in self.indata:
            data = self.indata['ProfPic']

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
    def __init__(self, parent):
        GraphWindow.__init__(self, parent)
        self.indata = {}
        self.parent = parent

    def change_band(self):
        """ Combo box to choose band """
        data = self.indata['Cluster']
        i = self.combobox1.currentIndex()
        self.combobox2.clear()
        self.combobox2.currentIndexChanged.disconnect()

        for j in range(data[i].no_clusters):
            self.combobox2.addItem('Membership Map for Cluster ' + str(j + 1))

        self.combobox2.currentIndexChanged.connect(self.change_band_two)
        self.change_band_two()

    def run(self):
        """ Run """
        data = self.indata['Cluster']
        if len(data[0].memdat) == 0:
            return

        self.show()
        for i in data:
            self.combobox1.addItem(i.dataid)

        self.change_band()

    def change_band_two(self):
        """ Combo box to choose band """
        data = self.indata['Cluster']

        i = self.combobox1.currentIndex()
        j = self.combobox2.currentIndex()

        self.mmc.update_membership(data[i], j)


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

        if j == 'Objective Function' and data[0].obj_fcn is not None:
            x = len(data)
            y = 0
            for i in data:
                y = max(y, len(i.obj_fcn))

            z = np.zeros([x, y])
            x = list(range(x))
            y = list(range(y))

            for i in x:
                for j in range(len(data[i].obj_fcn)):
                    z[i, j] = data[i].obj_fcn[j]

            for i in x:
                z[i][z[i] == 0] = z[i][z[i] != 0].min()

            x, y = np.meshgrid(x, y)
            x += data[0].no_clusters
            self.mmc.update_wireframe(x.T, y.T, np.log(z))

        if j == 'Variance Ratio Criterion' and data[0].vrc is not None:
            x = [k.no_clusters for k in data]
            y = [k.vrc for k in data]
            self.mmc.update_scatter(x, y)
        if j == 'Normalized Class Entropy' and data[0].nce is not None:
            x = [k.no_clusters for k in data]
            y = [k.nce for k in data]
            self.mmc.update_scatter(x, y)
        if j == 'Xie-Beni Index' and data[0].xbi is not None:
            x = [k.no_clusters for k in data]
            y = [k.xbi for k in data]
            self.mmc.update_scatter(x, y)

    def run(self):
        """ Run """
        items = []
        data = self.indata['Cluster']

        if data[0].obj_fcn is not None:
            items += ['Objective Function']

        if data[0].vrc is not None:
            items += ['Variance Ratio Criterion']

        if data[0].nce is not None:
            items += ['Normalized Class Entropy']

        if data[0].xbi is not None:
            items += ['Xie-Beni Index']

        if len(items) == 0:
            self.parent.showprocesslog('Your dataset does not qualify')
            return

        self.combobox1.addItems(items)

        self.label1.setText('Graph Type:')
        self.combobox1.setCurrentIndex(0)
        self.show()
