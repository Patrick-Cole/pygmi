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
"""Plot Vector Data using Matplotlib."""

import math
import numpy as np
from PyQt5 import QtWidgets, QtCore
from scipy.stats import median_absolute_deviation
import matplotlib.collections as mc
from matplotlib.cm import Set1
from matplotlib.cm import jet
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT


class GraphWindow(QtWidgets.QDialog):
    """Graph Window - Main QT Dialog class for graphs."""

    def __init__(self, parent):
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
        self.spinbox = QtWidgets.QSpinBox()
        self.label1 = QtWidgets.QLabel('Bands:')
        self.label2 = QtWidgets.QLabel('Bands:')
        self.label3 = QtWidgets.QLabel('Value:')
        self.checkbox = QtWidgets.QCheckBox('Option:')

        self.checkbox.hide()

        hbl.addWidget(self.label1)
        hbl.addWidget(self.combobox1)
        hbl.addWidget(self.label2)
        hbl.addWidget(self.combobox2)
        hbl.addWidget(self.label3)
        hbl.addWidget(self.spinbox)

        vbl.addWidget(self.mmc)
        vbl.addWidget(mpl_toolbar)
        vbl.addWidget(self.checkbox)
        vbl.addLayout(hbl)

        self.setFocus()

        self.combobox1.currentIndexChanged.connect(self.change_band)
        self.combobox2.currentIndexChanged.connect(self.change_band)
        self.spinbox.valueChanged.connect(self.change_band)
        self.checkbox.stateChanged.connect(self.change_band)

    def change_band(self):
        """
        Combo box to choose band.

        Returns
        -------
        None.

        """


class MyMplCanvas(FigureCanvas):
    """
    MPL Canvas class.

    This routine will also allow the pciking and movement of nodes of data.
    """

    def __init__(self, parent=None):
        fig = Figure()
        self.axes = fig.add_subplot(111)
        self.line = None
        self.ind = None
        self.background = None

        FigureCanvas.__init__(self, fig)

        self.figure.canvas.mpl_connect('pick_event', self.onpick)
        self.figure.canvas.mpl_connect('button_release_event',
                                       self.button_release_callback)
        self.figure.canvas.mpl_connect('motion_notify_event',
                                       self.motion_notify_callback)

    def button_release_callback(self, event):
        """
        Mouse button release callback.

        Parameters
        ----------
        event : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self.ind = None

    def motion_notify_callback(self, event):
        """
        Move mouse callback.

        Parameters
        ----------
        event : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        if self.ind is None:
            return

        dtmp = self.line.get_data()
        dtmp[1][self.ind] = event.ydata
        self.line.set_data(dtmp[0], dtmp[1])

        self.figure.canvas.restore_region(self.background)
        self.axes.draw_artist(self.line)
        self.figure.canvas.update()

    def onpick(self, event):
        """
        Picker event.

        Parameters
        ----------
        event : TYPE
            DESCRIPTION.

        Returns
        -------
        bool
            Return TRUE if pick succeeded.

        """
        if event.mouseevent.inaxes is None:
            return False
        if event.mouseevent.button != 1:
            return False
        if event.artist != self.line:
            return True

        self.ind = event.ind
        self.ind = self.ind[len(self.ind) // 2]  # get center-ish value

        return True

    def update_line(self, data, ival):
        """
        Update the plot from point data.

        Parameters
        ----------
        data1 : PData object
            Point data
        ival : dictionary key
            dictionary key Point Data

        Returns
        -------
        None.


        """
        data1 = data[ival]

        self.figure.clear()

        ax1 = self.figure.add_subplot(111, label='Profile')

        ax1.set_title(data1.dataid)
        self.axes = ax1

        self.figure.canvas.draw()
        self.background = self.figure.canvas.copy_from_bbox(ax1.bbox)

        self.line, = ax1.plot(data1.zdata, '.-', picker=5)
        self.figure.tight_layout()
        self.figure.canvas.draw()

    def update_lines(self, r, data):
        """
        Update the plot from point data.

        Parameters
        ----------
        r : numpy array
            array of distances, for the x-axis
        data : numpy array
            array of data to be plotted on the y-axis

        Returns
        -------
        None.

        """
        self.figure.clear()

        ax1 = self.figure.add_subplot(111, label='Profile')

        ax1.set_title('Profile')
        self.axes = ax1

        ax1.set_xlabel('Distance')
        ax1.set_ylabel('Value')

        self.figure.canvas.draw()
        self.background = self.figure.canvas.copy_from_bbox(ax1.bbox)

        self.line, = ax1.plot(r, data, '.-', picker=5)
        self.figure.tight_layout()
        self.figure.canvas.draw()

    def update_map(self, data, ival):
        """
        Update the map from point data.

        Parameters
        ----------
        data : PData object
            Point data
        ival : dictionary key
            dictionary key Point Data

        Returns
        -------
        None.

        """
        data1 = data[ival]

        self.figure.clear()

        ax1 = self.figure.add_subplot(111, label='Map')
        ax1.set_title(data1.dataid)
        self.axes = ax1

        self.figure.canvas.draw()
        self.background = self.figure.canvas.copy_from_bbox(ax1.bbox)

        scat = ax1.scatter(data1.xdata, data1.ydata, c=data1.zdata, cmap=jet)
        self.figure.colorbar(scat, ax=ax1)

        self.figure.tight_layout()
        self.figure.canvas.draw()

    def update_lmap(self, data, ival, scale, uselabels):
        """
        Update the plot from line data.

        Parameters
        ----------
        data : LData object
            Line data
        ival : dictionary key
            dictionary key representing the line data channel to be plotted.
        scale: float
            scale of exxageration for the profile data on the map.
        uselabels: bool
            boolean choice whether to use labels or not.

        Returns
        -------
        None.

        """
        self.figure.clear()

        ax1 = self.figure.add_subplot(111, label='Map')
        self.axes = ax1
        ax1.ticklabel_format(useOffset=False)

        self.figure.canvas.draw()
        self.background = self.figure.canvas.copy_from_bbox(ax1.bbox)

        zdata = []
        for line in data.data:
            zdata += data.data[line][ival].tolist()

        zdata = np.array(zdata)
        zdata = zdata[zdata != data.nullvalue]
        zdata = zdata[~np.isnan(zdata)]
        med = np.median(zdata)
        std = 2.5 * median_absolute_deviation(zdata, axis=None)

        if std == 0:
            std = 1

        for line in data.data:
            data1 = data.data[line]
            x = data1[data.xchannel]
            y = data1[data.ychannel]
            z = data1[ival]
            x = x[z != data.nullvalue]
            y = y[z != data.nullvalue]
            z = z[z != data.nullvalue]

            nanchk = np.logical_or(~np.isnan(x), ~np.isnan(y))
            x = x[nanchk]
            y = y[nanchk]
            z = z[nanchk]

            if x.size < 2:
                continue

            spcing = min(x.ptp(), y.ptp())/len(data.data)


            ang = np.arctan2((y[1:]-y[:-1]), (x[1:]-x[:-1]))
            ang = np.append(ang, ang[-1])

            if x.ptp() > y.ptp() and (x[-1]-x[0]) < 0:
                ang += np.pi

            elif y.ptp() > x.ptp() and (y[-1]-y[0]) < 0:
                ang += np.pi

            py = spcing*scale*(z - med)/std

            qx = x - np.sin(ang) * py
            qy = y + np.cos(ang) * py

            if uselabels:
                textang = np.rad2deg(ang[0])
                ax1.text(x[0], y[0], line, rotation=textang)
            ax1.plot(x, y, 'c')
            ax1.plot(qx, qy, 'k')

        self.figure.tight_layout()
        self.figure.canvas.draw()

    def update_vector(self, data):
        """
        Update the plot from vactor data.

        Parameters
        ----------
        data : VData
            Vector data. It can either be 'Line' or 'Poly' and typically is
            imported from a shapefile.

        Returns
        -------
        None.

        """
        self.figure.clear()

        self.axes = self.figure.add_subplot(111, label='map')

        if data.dtype == 'Line' or data.dtype == 'Poly':
            lcol = mc.LineCollection(data.crds)
            self.axes.add_collection(lcol)
            self.axes.autoscale()
            self.axes.axis('equal')

        elif data.dtype == 'Point':
            tmp = np.array(data.crds)
            tmp.shape = (tmp.shape[0], tmp.shape[-1])
            self.axes.plot(tmp[:, 0], tmp[:, 1], 'go')

        self.figure.canvas.draw()

    def update_rose(self, data, rtype, nbins=8):
        """
        Update the rose diagram plot using vector data.

        Parameters
        ----------
        data : VData
            Vector data. It should be 'Line'
        rtype : int
            Rose diagram type. Can be either 0 or 1.
        nbins : int, optional
            Number of bins used in rose diagram.

        Returns
        -------
        None.

        """
        self.figure.clear()

        ax1 = self.figure.add_subplot(121, polar=True, label='Rose')
        ax1.set_theta_direction(-1)
        ax1.set_theta_zero_location('N')
        ax1.yaxis.set_ticklabels([])

        self.axes = ax1

        ax2 = self.figure.add_subplot(122, label='Map')
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
        bcols = Set1(np.arange(nbins+1)/nbins)
        np.random.shuffle(bcols)

        if rtype == 0:
            # Draw rose diagram base on one angle per linear feature

            radii, theta = np.histogram(fangle, bins=np.arange(0, np.pi+bwidth,
                                                               bwidth))
            xtheta = theta[:-1]  # +(theta[1]-theta[0])/2
            bcols2 = bcols[(xtheta/bwidth).astype(int)]
            ax1.bar(xtheta, radii, width=bwidth, color=bcols2)
            ax1.bar(xtheta+np.pi, radii, width=bwidth, color=bcols2)

            bcols2 = bcols[(fangle/bwidth).astype(int)]
            lcol = mc.LineCollection(allcrds, color=bcols2)
            ax2.add_collection(lcol)
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
            lcol = mc.LineCollection(allcrds, color=bcols2)
            ax2.add_collection(lcol)
            ax2.autoscale(enable=True, tight=True)

        self.figure.canvas.draw()


class PlotPoints(GraphWindow):
    """Plot Points Class."""

    def __init__(self, parent):
        GraphWindow.__init__(self, parent)
        self.indata = {}
        self.parent = parent
        self.spinbox.hide()
        self.label3.hide()
        self.combobox2.hide()
        self.label2.hide()

    def change_band(self):
        """
        Combo box to choose band.

        Returns
        -------
        None.

        """
        data = self.indata['Point']
        i = self.combobox1.currentIndex()
        self.mmc.update_line(data, i)

    def run(self):
        """
        Run.

        Returns
        -------
        None.

        """
        self.show()
        data = self.indata['Point']
        for i in data:
            self.combobox1.addItem(i.dataid)
            self.combobox2.addItem(i.dataid)

        self.label1.setText('Editable Profile:')
        self.label2.setText('Normalised Stacked Profile:')
        self.combobox1.setCurrentIndex(0)
        self.combobox2.setCurrentIndex(1)


class PlotPoints2(GraphWindow):
    """Plot Points2 Class."""

    def __init__(self, parent):
        GraphWindow.__init__(self, parent)
        self.indata = {}
        self.parent = parent
        self.spinbox.hide()
        self.label3.hide()
        self.combobox2.hide()
        self.label2.hide()

    def change_band(self):
        """
        Combo box to choose band.

        Returns
        -------
        None.

        """
        data = self.indata['Point']
        i = self.combobox1.currentIndex()
        self.mmc.update_map(data, i)

    def run(self):
        """
        Combo box to choose band.

        Returns
        -------
        None.

        """
        self.show()
        data = self.indata['Point']
        for i in data:
            self.combobox1.addItem(i.dataid)

        self.label1.setText('Map')
        self.combobox1.setCurrentIndex(0)


class PlotLines(GraphWindow):
    """Plot Lines Class."""

    def __init__(self, parent):
        GraphWindow.__init__(self, parent)
        self.indata = {}
        self.parent = parent
        self.spinbox.hide()
        self.label3.hide()
        self.xcol = ''
        self.ycol = ''

    def change_line(self):
        """
        Combo to change line number.

        Returns
        -------
        None.

        """

    def change_band(self):
        """
        Combo box to choose band.

        Returns
        -------
        None.

        """
        data = self.indata['Line'].data
        i = self.combobox1.currentText()
        i2 = self.combobox2.currentText()

        data2 = data[i][i2]
        x = data[i][self.xcol]
        y = data[i][self.ycol]

        r = np.sqrt((x[1:]-x[:-1])**2+(y[1:]-y[:-1])**2)
        r = np.cumsum(r)
        r = np.concatenate(([0.], r))

        self.mmc.update_lines(r, data2)

    def run(self):
        """
        Run.

        Returns
        -------
        None.

        """
        self.combobox1.currentIndexChanged.disconnect()
        self.combobox2.currentIndexChanged.disconnect()

        self.show()
        data = self.indata['Line'].data
        self.xcol = self.indata['Line'].xchannel
        self.ycol = self.indata['Line'].ychannel

        for i in data:
            self.combobox1.addItem(i)
            i2 = i

        for i in data[i2].dtype.names:
            self.combobox2.addItem(i)

        self.label1.setText('Line:')
        self.label2.setText('Column:')

        self.combobox1.setCurrentIndex(0)
        self.combobox2.setCurrentIndex(0)

        self.change_band()

        self.combobox1.currentIndexChanged.connect(self.change_band)
        self.combobox2.currentIndexChanged.connect(self.change_band)


class PlotLines2(GraphWindow):
    """Plot Lines2 Class."""

    def __init__(self, parent):
        GraphWindow.__init__(self, parent)
        self.indata = {}
        self.parent = parent
        self.combobox2.hide()
        self.label2.hide()
        self.checkbox.show()

    def change_band(self):
        """
        Combo box to choose band.

        Returns
        -------
        None.

        """
        data = self.indata['Line']
        scale = self.spinbox.value()
        i = self.combobox1.currentText()
        self.mmc.update_lmap(data, i, scale, self.checkbox.isChecked())

    def run(self):
        """
        Run.

        Returns
        -------
        None.

        """
        self.combobox1.currentIndexChanged.disconnect()
        self.spinbox.valueChanged.disconnect()
        self.checkbox.stateChanged.disconnect()

        self.show()
        data = self.indata['Line'].data

        i2 = list(data.keys())[0]

        for i in data[i2].dtype.names:
            if i == self.indata['Line'].xchannel:
                continue
            if i == self.indata['Line'].ychannel:
                continue
            self.combobox1.addItem(i)

        self.checkbox.setText('Show Line Labels:')
        self.label1.setText('Column:')
        self.label3.setText('Scale:')
        self.spinbox.setMinimum(1)
        self.spinbox.setMaximum(1000000)
        self.spinbox.setValue(100)

        self.combobox1.setCurrentIndex(0)

        self.change_band()

        self.combobox1.currentIndexChanged.connect(self.change_band)
        self.spinbox.valueChanged.connect(self.change_band)
        self.checkbox.stateChanged.connect(self.change_band)


class PlotRose(GraphWindow):
    """Plot Rose Class."""

    def __init__(self, parent):
        GraphWindow.__init__(self, parent)
        self.indata = {}
        self.parent = parent
        self.combobox2.hide()
        self.label2.hide()
        self.spinbox.setValue(8)
        self.spinbox.setMinimum(2)
        self.spinbox.setMaximum(360)
        self.setWindowTitle('Rose Diagram')

    def change_band(self):
        """
        Combo box to choose band.

        Returns
        -------
        None.

        """
        if 'Vector' not in self.indata:
            return
        data = self.indata['Vector']
        i = self.combobox1.currentIndex()
        if data.dtype == 'Line':
            self.mmc.update_rose(data, i, self.spinbox.value())

    def run(self):
        """
        Run.

        Returns
        -------
        None.

        """
        self.show()
        self.combobox1.addItem('Average Angle per Feature')
        self.combobox1.addItem('Angle per segment in Feature')
        self.label1.setText('Rose Diagram Type:')
        self.combobox1.setCurrentIndex(0)


class PlotVector(GraphWindow):
    """Plot Vector Class."""

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
        """
        Run.

        Returns
        -------
        None.

        """
        self.show()
        data = self.indata['Vector']
        self.mmc.update_vector(data)


def histogram(x, y=None, xmin=None, xmax=None, bins=10):
    """
    Histogram.

    Calculate histogram of a set of data. It is different from a
    conventional histogram in that instead of summing elements of
    specific values, this allows the sum of weights/probabilities on a per
    element basis.

    Parameters
    ----------
    x : numpy array
        Input data
    y : numpy array
        Input data weights. A value of 1 is default behaviour
    xmin : float
        Lower value for the bins
    xmax : float
        Upper value for the bins
    bins : int
        number of bins

    Returns
    -------
    hist : numpy array
        The values of the histogram
    bin_edges : numpy array
        bin edges of the histogram

    """
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
    x_2 = x/xbin
    x_2 = x_2.astype(int)

    for i in range(bins):
        radii[i] = y[x_2 == i].sum()
        theta[i] = i*xbin

    hist = radii
    bin_edges = theta
    return hist, bin_edges


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.

    Parameters
    ----------
    origin : list
        List containing origin point (ox, oy)
    point : list
        List containing point to be rotated (px, py)
    angle : float
        Angle in radians.

    Returns
    -------
    qx : float
        Rotated x-coordinate.
    qy : float
        Rotated y-coordinate.

    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy
