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
from scipy.stats import median_abs_deviation
import matplotlib.collections as mc
from matplotlib import colormaps
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
from pandas.api.types import is_numeric_dtype

from pygmi.misc import frm


class GraphWindow(QtWidgets.QDialog):
    """Graph Window - Main QT Dialog class for graphs."""

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


class MyMplCanvas(FigureCanvasQTAgg):
    """
    MPL Canvas class.

    This routine will also allow the picking and movement of nodes of data.
    """

    def __init__(self, parent=None):
        fig = Figure()
        self.axes = fig.add_subplot(111)
        self.line = None
        self.ind = None
        self.background = None
        self.pickevents = False

        super().__init__(fig)

    def button_release_callback(self, event):
        """
        Mouse button release callback.

        Parameters
        ----------
        event : TYPE
            Button release event.

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
            Motion notify event.

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
            Picker event.

        Returns
        -------
        bool
            Return TRUE if pick succeeded, False otherwise.

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

    def resizeline(self, event):
        """
        Resize event.

        Parameters
        ----------
        event : TYPE
            Unused.

        Returns
        -------
        None.

        """
        r, data = self.line.get_data()
        self.update_lines(r, data)

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
        if self.pickevents is False:
            self.figure.canvas.mpl_connect('pick_event', self.onpick)
            self.figure.canvas.mpl_connect('button_release_event',
                                           self.button_release_callback)
            self.figure.canvas.mpl_connect('motion_notify_event',
                                           self.motion_notify_callback)
            self.figure.canvas.mpl_connect('resize_event', self.resizeline)
            self.pickevents = True

        self.figure.clear()

        self.axes = self.figure.add_subplot(111, label='Profile')

        self.axes.set_title('Profile')
        self.axes.set_xlabel('Distance')
        self.axes.set_ylabel('Value')

        self.axes.xaxis.set_major_formatter(frm)
        self.axes.yaxis.set_major_formatter(frm)

        self.figure.canvas.draw()
        self.background = self.figure.canvas.copy_from_bbox(self.axes.bbox)

        self.line, = self.axes.plot(r, data, '.-')
        self.line.set_picker(True)
        self.line.set_pickradius(5)

        self.figure.canvas.draw()

    def update_map(self, xdata, ydata, zdata):
        """
        Update the map from point data.

        Parameters
        ----------
        xdata : numpy array
            1D x-data array.
        ydata : numpy array
            1D y-data array.
        zdata : numpy array
            1D z-data array.

        Returns
        -------
        None.

        """
        self.figure.clear()

        ax1 = self.figure.add_subplot(111, label='Map')
        ax1.ticklabel_format(style='plain')

        self.axes = ax1
        self.axes.tick_params(axis='x', rotation=90)
        self.axes.tick_params(axis='y', rotation=0)

        self.figure.canvas.draw()
        self.background = self.figure.canvas.copy_from_bbox(ax1.bbox)

        if is_numeric_dtype(zdata):
            scat = ax1.scatter(xdata, ydata, c=zdata, cmap=colormaps['jet'])
        else:
            return

        self.figure.colorbar(scat, ax=ax1, format=frm)

        self.axes.xaxis.set_major_formatter(frm)
        self.axes.yaxis.set_major_formatter(frm)

        self.figure.tight_layout()
        self.figure.canvas.draw()

    def update_lmap(self, data, ival, scale, uselabels):
        """
        Update the plot from line data.

        Parameters
        ----------
        data : Pandas dataframe
            Line data
        ival : dictionary key
            dictionary key representing the line data channel to be plotted.
        scale: float
            scale of exaggeration for the profile data on the map.
        uselabels: bool
            boolean choice whether to use labels or not.

        Returns
        -------
        None.

        """
        self.figure.clear()
        ax1 = self.figure.add_subplot(111, label='Map')

        self.axes = ax1
        self.axes.ticklabel_format(useOffset=False, style='plain')
        self.axes.tick_params(axis='x', rotation=90)
        self.axes.tick_params(axis='y', rotation=0)

        self.figure.canvas.draw()
        self.background = self.figure.canvas.copy_from_bbox(ax1.bbox)
        data = data.dropna()

        zdata = data[ival]
        med = np.median(zdata)
        std = 2.5 * median_abs_deviation(zdata, axis=None, scale=1/1.4826)

        if std == 0:
            std = 1

        # Get average spacing between 2 points over the entire survey.
        spcing = np.array([])

        datagrp = data.groupby('line')
        datagrp = list(datagrp)

        for line in datagrp:
            data1 = line[1]
            x = data1.pygmiX.values
            y = data1.pygmiY.values
            z = data1[ival]

            if x.size < 2:
                continue

            spcing = np.append(spcing, np.sqrt(np.diff(x)**2+np.diff(y)**2))

        spcing = spcing.mean()

        for line in datagrp:
            data1 = line[1]

            x = data1.pygmiX.values
            y = data1.pygmiY.values
            z = data1[ival]

            if x.size < 2:
                continue

            ang = np.arctan2((y[1:]-y[:-1]), (x[1:]-x[:-1]))
            ang = np.append(ang, ang[-1])

            if x.ptp() > y.ptp() and (x[-1]-x[0]) < 0:
                ang += np.pi

            elif y.ptp() > x.ptp() and (y[-1]-y[0]) < 0:
                ang += np.pi

            py = spcing*scale*(z - med)/std/100.

            qx = x - np.sin(ang) * py
            qy = y + np.cos(ang) * py

            if uselabels:
                textang = np.rad2deg(ang[0])
                ax1.text(x[0], y[0], line[0], rotation=textang)
            ax1.plot(x, y, 'c')
            ax1.plot(qx, qy, 'k')

        self.axes.xaxis.set_major_formatter(frm)
        self.axes.yaxis.set_major_formatter(frm)

        self.figure.tight_layout()
        self.figure.canvas.draw()

    def update_vector(self, data, col):
        """
        Update the plot from vector data.

        Parameters
        ----------
        data : dictionary
            GeoPandas data in a dictionary.
        col : str
            Label for column to extract.

        Returns
        -------
        None.

        """
        self.figure.clear()

        self.axes = self.figure.add_subplot(111, label='map')
        self.axes.ticklabel_format(style='plain')
        self.axes.tick_params(axis='x', rotation=90)
        self.axes.tick_params(axis='y', rotation=0)

        if 'LineString' in data:
            tmp = []
            for i in data['LineString'].geometry:
                if i.type == 'MultiLineString':
                    for j in i:
                        tmp.append(np.array(j.coords[:]))
                else:
                    tmp.append(np.array(i.coords[:]))

            lcol = mc.LineCollection(tmp)
            self.axes.add_collection(lcol)
            self.axes.autoscale()
            self.axes.axis('equal')
        elif 'Polygon' in data:
            tmp = []
            for i in data['Polygon'].geometry:
                tmp.append(np.array(i.exterior.coords[:])[:, :2].tolist())

            lcol = mc.LineCollection(tmp)
            self.axes.add_collection(lcol)
            self.axes.autoscale()
            self.axes.axis('equal')

        elif 'MultiPolygon' in data:
            tmp = []
            for j in data['MultiPolygon'].geometry:
                for i in list(j.geoms):
                    tmp.append(np.array(i.exterior.coords[:])[:, :2].tolist())

            lcol = mc.LineCollection(tmp)
            self.axes.add_collection(lcol)
            self.axes.autoscale()
            self.axes.axis('equal')

        elif 'Point' in data:
            if col != '':
                self.axes.scatter(data['Point'].geometry.x,
                                  data['Point'].geometry.y,
                                  c=data['Point'][col])
            else:
                self.axes.scatter(data['Point'].geometry.x,
                                  data['Point'].geometry.y)

        self.axes.xaxis.set_major_formatter(frm)
        self.axes.yaxis.set_major_formatter(frm)

        self.figure.tight_layout()
        self.figure.canvas.draw()

    def update_rose(self, data, rtype, nbins=8, equal=False):
        """
        Update the rose diagram plot using vector data.

        Parameters
        ----------
        data : dictionary
            GeoPandas data in a dictionary. It should be 'LineString'
        rtype : int
            Rose diagram type. Can be either 0 or 1.
        nbins : int, optional
            Number of bins used in rose diagram. The default is 8.
        equal : bool, optional
            Option for an equal area rose diagram. The default is False.

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
        ax2.ticklabel_format(useOffset=False, style='plain')
        ax2.tick_params(axis='x', rotation=90)
        ax2.tick_params(axis='y', rotation=0)
        ax2.xaxis.set_major_formatter(frm)
        ax2.yaxis.set_major_formatter(frm)

        fangle = []
        fcnt = []
        flen = []

        allcrds = []
        for i in data['LineString'].geometry:
            if i.type == 'MultiLineString':
                for j in i:
                    allcrds.append(np.array(j.coords[:]))
            else:
                allcrds.append(np.array(i.coords[:]))

        for pnts in allcrds:
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
        Set1 = colormaps['Set1']
        bcols = Set1(np.arange(nbins+1)/nbins)
        np.random.shuffle(bcols)

        if rtype == 0:
            # Draw rose diagram base on one angle per linear feature

            radii, theta = np.histogram(fangle, bins=np.arange(0, np.pi+bwidth,
                                                               bwidth))

            if equal is True:
                radii = .01*radii.max()*np.sqrt(100*radii/radii.max())

            xtheta = theta[:-1]
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
            if equal is True:
                radii = .01*radii.max()*np.sqrt(100*radii/radii.max())

            xtheta = theta[:-1]
            bcols2 = bcols[(xtheta/bwidth).astype(int)]
            ax1.bar(xtheta, radii, width=bwidth, color=bcols2)
            ax1.bar(xtheta+np.pi, radii, width=bwidth, color=bcols2)

            bcols2 = bcols[(fcnt/bwidth).astype(int)]
            lcol = mc.LineCollection(allcrds, color=bcols2)
            ax2.add_collection(lcol)
            ax2.autoscale(enable=True, tight=True)

        self.figure.tight_layout()
        self.figure.canvas.draw()


class PlotPoints(GraphWindow):
    """Plot Points Class."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.indata = {}
        self.parent = parent
        self.spinbox.hide()
        self.label3.hide()
        self.combobox2.hide()
        self.label2.hide()
        self.xdata = None
        self.ydata = None
        if parent is None:
            self.showprocesslog = print
        else:
            self.showprocesslog = parent.showprocesslog

    def change_band(self):
        """
        Combo box to choose band.

        Returns
        -------
        None.

        """
        key = list(self.indata['Line'].keys())[0]
        data = self.indata['Line'][key]
        data = data.dropna()
        i = self.combobox1.currentText()

        self.mmc.update_map(data.pygmiX, data.pygmiY, data[i])

    def run(self):
        """
        Entry point to run class.

        Returns
        -------
        None.

        """
        data = self.indata['Line']
        data = list(data.values())[0]

        filt = ((data.columns != 'geometry') &
                (data.columns != 'line') &
                (data.columns != 'pygmiX') &
                (data.columns != 'pygmiY'))

        cols = list(data.columns[filt])

        if data.pygmiX.isna().min() == True:
            self.showprocesslog('You do not have coordinates in that point '
                                'dataset.')
            return

        self.show()

        self.combobox1.addItems(cols)

        self.label1.setText('Map')
        self.combobox1.setCurrentIndex(0)


class PlotLines(GraphWindow):
    """Plot Lines Class."""

    def __init__(self, parent=None):
        super().__init__(parent)
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
        key = list(self.indata['Line'].keys())[0]
        data = self.indata['Line'][key]
        data = data.dropna()

        line = self.combobox1.currentText()
        col = self.combobox2.currentText()

        data2 = data[data.line == line]
        x = data2.pygmiX.values
        y = data2.pygmiY.values

        data2 = data2[col].values

        r = np.sqrt((x[1:]-x[:-1])**2+(y[1:]-y[:-1])**2)
        r = np.cumsum(r)
        r = np.concatenate(([0.], r))

        self.mmc.update_lines(r, data2)

    def run(self):
        """
        Entry point to run class.

        Returns
        -------
        None.

        """
        self.combobox1.currentIndexChanged.disconnect()
        self.combobox2.currentIndexChanged.disconnect()

        self.show()

        data = self.indata['Line']
        data = list(data.values())[0]
        filt = ((data.columns != 'geometry') &
                (data.columns != 'line') &
                (data.columns != 'pygmiX') &
                (data.columns != 'pygmiY'))
        cols = list(data.columns[filt])
        lines = data.line[data.line != 'nan'].unique()

        self.combobox1.addItems(lines)
        self.combobox2.addItems(cols)

        self.label1.setText('Line:')
        self.label2.setText('Column:')

        self.combobox1.setCurrentIndex(0)
        self.combobox2.setCurrentIndex(0)

        self.change_band()

        self.combobox1.currentIndexChanged.connect(self.change_band)
        self.combobox2.currentIndexChanged.connect(self.change_band)


class PlotLineMap(GraphWindow):
    """Plot Lines Map Class."""

    def __init__(self, parent=None):
        super().__init__(parent)
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
        key = list(self.indata['Line'].keys())[0]
        data = self.indata['Line'][key]
        data = data.dropna()

        scale = self.spinbox.value()
        i = self.combobox1.currentText()
        self.mmc.update_lmap(data, i, scale, self.checkbox.isChecked())

    def run(self):
        """
        Entry point to run class.

        Returns
        -------
        None.

        """
        self.combobox1.currentIndexChanged.disconnect()
        self.spinbox.valueChanged.disconnect()
        self.checkbox.stateChanged.disconnect()

        self.show()

        data = self.indata['Line']
        data = list(data.values())[0]
        filt = ((data.columns != 'geometry') &
                (data.columns != 'line') &
                (data.columns != 'pygmiX') &
                (data.columns != 'pygmiY'))
        cols = list(data.columns[filt])
        self.combobox1.addItems(cols)

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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.indata = {}
        self.parent = parent
        self.combobox2.hide()
        self.label2.hide()
        self.spinbox.setValue(8)
        self.spinbox.setMinimum(2)
        self.spinbox.setMaximum(360)
        self.checkbox.show()

        self.setWindowTitle('Rose Diagram')
        if parent is None:
            self.showprocesslog = print
        else:
            self.showprocesslog = parent.showprocesslog

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
        equal = self.checkbox.isChecked()
        if 'LineString' in data:
            self.mmc.update_rose(data, i, self.spinbox.value(), equal)
        else:
            self.showprocesslog('No line type data.')
            return

    def run(self):
        """
        Entry point to run class.

        Returns
        -------
        None.

        """
        self.show()
        self.combobox1.addItem('Average Angle per Feature')
        self.combobox1.addItem('Angle per segment in Feature')
        self.checkbox.setText('Equal Area Rose Diagram')
        self.label1.setText('Rose Diagram Type:')
        self.combobox1.setCurrentIndex(0)


class PlotVector(GraphWindow):
    """Plot Vector Class."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.indata = {}
        self.parent = parent
        self.label1.hide()
        self.combobox2.hide()
        self.label2.hide()
        self.spinbox.hide()
        self.label3.hide()

    def change_band(self):
        """
        Combo box to choose band.

        Returns
        -------
        None.

        """
        data = self.indata['Vector']
        i = self.combobox1.currentText()

        self.mmc.update_vector(data, i)

    def run(self):
        """
        Entry point to run class.

        Returns
        -------
        None.

        """
        data = self.indata['Vector']
        key = list(data.keys())[0]
        data = data[key]
        data = data.select_dtypes(include='number')

        filt = ((data.columns != 'geometry') &
                (data.columns != 'line') &
                (data.columns != 'pygmiX') &
                (data.columns != 'pygmiY'))

        cols = list(data.columns[filt])
        if len(cols) > 0:
            self.combobox1.addItems(cols)
            self.combobox1.setCurrentIndex(0)
        else:
            self.combobox1.hide()

        self.show()

        self.change_band()


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
        Input data weights. The default is None.
    xmin : float
        Lower value for the bins. The default is None.
    xmax : float
        Upper value for the bins. The default is None.
    bins : int
        number of bins. The default is 10.

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
