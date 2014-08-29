# -----------------------------------------------------------------------------
# Name:        graphs.py (part of PyGMI)
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
""" Plot Vector Data """

# pylint: disable=E1101
import numpy as np
from PyQt4 import QtGui, QtCore
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as \
    FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as \
    NavigationToolbar


class MyMplCanvas(FigureCanvas):
    """ MPL Canvas class"""
    def __init__(self, parent=None):
        # figure stuff
        fig = Figure()
        self.axes = fig.add_subplot(111)
        self.line = None
        self.ind = None
        self.background = None
        self.parent = parent

        FigureCanvas.__init__(self, fig)

        self.figure.canvas.mpl_connect('pick_event', self.onpick)
        self.figure.canvas.mpl_connect('button_release_event',
                                       self.button_release_callback)
        self.figure.canvas.mpl_connect('motion_notify_event',
                                       self.motion_notify_callback)

    def button_release_callback(self, event):
        """ mouse button release callback """
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self.ind = None

    def motion_notify_callback(self, event):
        """ move mouse callback """
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        if self.ind is None:
            return

        y = event.ydata
        dtmp = self.line.get_data()
        dtmp[1][self.ind] = y
        self.line.set_data(dtmp[0], dtmp[1])

        self.figure.canvas.restore_region(self.background)
        self.axes.draw_artist(self.line)
#        self.figure.canvas.blit(self.axes.bbox)
        self.figure.canvas.update()

    def onpick(self, event):
        """ Picker event """
        if event.mouseevent.inaxes is None:
            return
        if event.mouseevent.button != 1:
            return
        if event.artist != self.line:
            return True

        self.ind = event.ind
        self.ind = self.ind[len(self.ind) / 2]  # get center-ish value

        return True

    def update_line(self, data1, data2):
        """ Update the plot """
        self.figure.clear()

        ax1 = self.figure.add_subplot(2, 1, 1)
        ax1.set_title(data1.dataid)
        self.axes = ax1

        ax2 = self.figure.add_subplot(2, 1, 2, sharex=ax1)
        ax2.set_title(data2.dataid)

        self.figure.canvas.draw()
        self.background = self.figure.canvas.copy_from_bbox(ax1.bbox)

        ax2.plot(data2.zdata)
        self.line, = ax1.plot(data1.zdata, '.-', picker=5)
        self.figure.canvas.draw()

    def update_vector(self, data):
        """ Update the plot """
        self.figure.clear()

        ax = self.figure.add_subplot(1, 1, 1)
#        ax1.set_title(data1.dataid)
        self.axes = ax

        if data.dtype == 'Line' or data.dtype == 'Poly':
            lc = mc.LineCollection(data.crds)
            ax.add_collection(lc)
            ax.autoscale()
            ax.axis('equal')

        elif data.dtype == 'Point':
            tmp = np.array(data.crds)
            tmp.shape = (tmp.shape[0], tmp.shape[-1])
            ax.plot(tmp[:, 0], tmp[:, 1], 'go')

        self.figure.canvas.draw()

    def update_rose(self, data, rtype, nbins=8):
        """ Update the plot """
        self.figure.clear()

        ax1 = self.figure.add_subplot(121, polar=True)
#        ax1.set_title('Rose')
        ax1.set_theta_direction(-1)
        ax1.set_theta_zero_location('N')
        ax1.yaxis.set_ticklabels([])

        self.axes = ax1

        ax2 = self.figure.add_subplot(1, 2, 2)
        # ax2.set_title('Lineaments')
        # ax2.axis('equal')
        ax2.set_aspect('equal')

        fangle = []
        fcnt = []
        flen = []
        allcrds = data.crds

        for pnts in data.crds:
            pnts = np.transpose(pnts)
            xtmp = pnts[0, 1:]-pnts[0, :-1]
            ytmp = pnts[1, 1:]-pnts[1, :-1]
            ftmp = np.arctan2(xtmp, ytmp)
            ftmp[ftmp < 0] += 2*np.pi
            ftmp[ftmp > np.pi] -= np.pi
            ltmp = np.sqrt(xtmp**2+ytmp**2)

            fangle += [np.sum(ftmp*ltmp)/ltmp.sum()]
            fcnt += ftmp.tolist()
            flen += ltmp.tolist()

        fangle = np.array(fangle)
        fcnt = np.array(fcnt)
        flen = np.array(flen)
        bwidth = np.pi/nbins
        bcols = plt.cm.Set1(np.arange(nbins)/nbins)
        np.random.shuffle(bcols)

        if rtype == 0:
            # Draw rose diagram base on one angle per linear feature

            radii, theta = np.histogram(fangle, bins=np.arange(0, np.pi,
                                                               bwidth))
            xtheta = theta[:-1]  # +(theta[1]-theta[0])/2
            bcols2 = bcols[(xtheta/bwidth).astype(int)]
            ax1.bar(xtheta, radii, width=bwidth, color=bcols2)
            ax1.bar(xtheta+np.pi, radii, width=bwidth, color=bcols2)

            bcols2 = bcols[(fangle/bwidth).astype(int)]
            lc = mc.LineCollection(allcrds, color=bcols2)
            ax2.add_collection(lc)
            ax2.autoscale(enable=True, tight=True)

        else:
            # Draw rose diagram base on one angle per linear segment, normed
            radii, theta = histogram(fcnt, y=flen, xmin=0., xmax=np.pi,
                                     bins=nbins)
            xtheta = theta[:-1]
            bcols2 = bcols[(xtheta/bwidth).astype(int)]
            ax1.bar(xtheta, radii, width=bwidth, color=bcols2)
            ax1.bar(xtheta+np.pi, radii, width=bwidth, color=bcols2)

            bcols2 = bcols[(fcnt/bwidth).astype(int)]
            lc = mc.LineCollection(allcrds, color=bcols2)
            ax2.add_collection(lc)
            ax2.autoscale(enable=True, tight=True)

        self.figure.canvas.draw()


class GraphWindow(QtGui.QDialog):
    """ Graph Window - Main QT Dialog class for graphs."""
    def __init__(self, parent=None):
        QtGui.QDialog.__init__(self, parent=None)
        self.parent = parent

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Graph Window")

        vbl = QtGui.QVBoxLayout(self)  # self is where layout is assigned
        self.hbl = QtGui.QHBoxLayout()
        self.mmc = MyMplCanvas(self)
        mpl_toolbar = NavigationToolbar(self.mmc, self.parent)

        self.combobox1 = QtGui.QComboBox(self)
        self.combobox2 = QtGui.QComboBox(self)
        self.spinbox = QtGui.QSpinBox(self)
        self.label1 = QtGui.QLabel(self)
        self.label2 = QtGui.QLabel(self)
        self.label3 = QtGui.QLabel(self)
        self.label1.setText('Bands:')
        self.label2.setText('Bands:')
        self.label3.setText('Value:')
        self.hbl.addWidget(self.label1)
        self.hbl.addWidget(self.combobox1)
        self.hbl.addWidget(self.label2)
        self.hbl.addWidget(self.combobox2)
        self.hbl.addWidget(self.label3)
        self.hbl.addWidget(self.spinbox)

        vbl.addWidget(self.mmc)
        vbl.addWidget(mpl_toolbar)
        vbl.addLayout(self.hbl)

        self.setFocus()

        self.combobox1.currentIndexChanged.connect(self.change_band)
        self.combobox2.currentIndexChanged.connect(self.change_band)
        self.spinbox.valueChanged.connect(self.change_band)

    def change_band(self):
        """ Combo box to choose band """
        pass


class PlotPoints(GraphWindow):
    """ Plot Raster Class """
    def __init__(self, parent):
        GraphWindow.__init__(self, parent)
        self.indata = {}
        self.parent = parent
        self.spinbox.hide()
        self.label3.hide()

    def change_band(self):
        """ Combo box to choose band """
        data = self.indata['Point']
        i = self.combobox1.currentIndex()
        j = self.combobox2.currentIndex()
        self.mmc.update_line(data[i], data[j])

    def run(self):
        """ Run """
        self.show()
        data = self.indata['Point']
        for i in data:
            self.combobox1.addItem(i.dataid)
            self.combobox2.addItem(i.dataid)

        self.label1.setText('Top Profile:')
        self.label2.setText('Bottom Profile:')
        self.combobox1.setCurrentIndex(0)
        self.combobox2.setCurrentIndex(1)


class PlotVector(GraphWindow):
    """ Plot Raster Class """
    def __init__(self, parent):
        GraphWindow.__init__(self, parent)
        self.indata = {}
        self.parent = parent
        self.combobox1.hide()
        self.label1.hide()
        self.combobox2.hide()
        self.label2.hide()
        self.spinbox.hide()
        self.label3.hide()

    def run(self):
        """ Run """
        self.show()
        data = self.indata['Vector']
        self.mmc.update_vector(data)
#        for i in data:
#            self.combobox1.addItem(i.dataid)
#            self.combobox2.addItem(i.dataid)

#        self.label1.setText('Top Profile:')
#        self.label2.setText('Bottom Profile:')
#        self.combobox1.setCurrentIndex(0)
#        self.combobox2.setCurrentIndex(1)


class PlotRose(GraphWindow):
    """ Plot Raster Class """
    def __init__(self, parent):
        GraphWindow.__init__(self, parent)
        self.indata = {}
        self.parent = parent
        self.combobox2.hide()
        self.label2.hide()
        self.spinbox.setValue(8)
        self.spinbox.setMinimum(2)
        self.spinbox.setMaximum(360)
        self.setWindowTitle("Rose Diagram")

    def change_band(self):
        """ Combo box to choose band """
        if 'Vector' not in self.indata:
            return
        data = self.indata['Vector']
        i = self.combobox1.currentIndex()
        if data.dtype == 'Line':
            self.mmc.update_rose(data, i, self.spinbox.value())

    def run(self):
        """ Run """
        self.show()
        self.combobox1.addItem('Average Angle per Feature')
        self.combobox1.addItem('Angle per segment in Feature')
        self.label1.setText('Rose Diagram Type:')
        self.combobox1.setCurrentIndex(0)


def histogram(x, y=None, xmin=None, xmax=None, bins=10):
    """ Calculate histogram of a set of data. It is different from a
    conventional histogram in that instead of summing elements of
    specific values, this allows the sum of weights/probabilities on a per
    element basis.

    Args:
        x (numpy array): Input data
        y (numpy array): Input data weights. A value of 1 is default behaviour
        xmin (float): Lower value for the bins
        xmax (float): Upper value for the bins
        bins (int): number of bins

    Returns:
        hist (numpy array): The values of the histogram
        bin_edges (numpy array): bin edges of the histogram"""

    radii = np.zeros(bins)
    theta = np.zeros(bins+1)

    if y is None:
        y = np.ones_like(x)
    if xmin is None:
        xmin = x.min()
    if xmax is None:
        xmax = x.max()

    x = np.array(x)
    y = np.array(y)
    theta[-1] = xmax

    xrange = xmax-xmin
    xbin = xrange/bins
    x2 = x/xbin
    x2 = x2.astype(int)

    for i in range(bins):
        radii[i] = y[x2 == i].sum()
        theta[i] = i*xbin

    hist = radii
    bin_edges = theta
    return hist, bin_edges
