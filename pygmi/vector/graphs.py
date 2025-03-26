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
from PyQt6 import QtWidgets, QtCore
from scipy.stats import median_abs_deviation
import matplotlib.collections as mc
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import BoundaryNorm
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.ticker import StrMethodFormatter
from matplotlib import rcParams
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.cluster import KMeans

from pygmi.misc import frm, ContextModule, discrete_colorbar

rcParams['savefig.dpi'] = 300

copper = np.array([[255., 236., 184.],
                   [255., 230., 172.],
                   [255., 225., 164.],
                   [255., 221., 158.],
                   [255., 218., 153.],
                   [254., 216., 149.],
                   [253., 213., 146.],
                   [252., 212., 143.],
                   [251., 210., 140.],
                   [250., 208., 138.],
                   [249., 206., 136.],
                   [248., 205., 133.],
                   [247., 203., 131.],
                   [246., 202., 129.],
                   [245., 201., 128.],
                   [245., 199., 126.],
                   [244., 198., 124.],
                   [243., 197., 122.],
                   [242., 196., 121.],
                   [241., 195., 120.],
                   [240., 194., 118.],
                   [239., 193., 117.],
                   [238., 192., 115.],
                   [238., 191., 114.],
                   [237., 190., 113.],
                   [236., 189., 112.],
                   [235., 188., 110.],
                   [234., 187., 109.],
                   [234., 186., 108.],
                   [233., 185., 107.],
                   [232., 184., 106.],
                   [231., 183., 105.],
                   [230., 182., 104.],
                   [230., 181., 103.],
                   [229., 181., 102.],
                   [228., 180., 101.],
                   [227., 179., 100.],
                   [227., 178.,  99.],
                   [226., 177.,  98.],
                   [225., 177.,  97.],
                   [224., 176.,  96.],
                   [224., 175.,  95.],
                   [223., 174.,  94.],
                   [222., 174.,  93.],
                   [222., 173.,  93.],
                   [221., 172.,  92.],
                   [220., 171.,  91.],
                   [220., 171.,  90.],
                   [219., 170.,  89.],
                   [218., 169.,  88.],
                   [217., 168.,  88.],
                   [217., 168.,  87.],
                   [216., 167.,  86.],
                   [215., 166.,  85.],
                   [215., 166.,  85.],
                   [214., 165.,  84.],
                   [213., 164.,  83.],
                   [213., 164.,  82.],
                   [212., 163.,  82.],
                   [211., 162.,  81.],
                   [211., 162.,  80.],
                   [210., 161.,  80.],
                   [210., 160.,  79.],
                   [209., 160.,  78.],
                   [208., 159.,  77.],
                   [208., 158.,  77.],
                   [207., 158.,  76.],
                   [206., 157.,  75.],
                   [206., 156.,  75.],
                   [205., 156.,  74.],
                   [204., 155.,  74.],
                   [204., 154.,  73.],
                   [203., 154.,  72.],
                   [203., 153.,  72.],
                   [202., 153.,  71.],
                   [201., 152.,  70.],
                   [201., 151.,  70.],
                   [200., 151.,  69.],
                   [199., 150.,  68.],
                   [199., 149.,  68.],
                   [198., 149.,  67.],
                   [197., 148.,  67.],
                   [197., 148.,  66.],
                   [196., 147.,  66.],
                   [196., 147.,  65.],
                   [195., 146.,  64.],
                   [194., 145.,  64.],
                   [194., 145.,  63.],
                   [193., 144.,  63.],
                   [192., 144.,  62.],
                   [192., 143.,  62.],
                   [191., 142.,  61.],
                   [190., 142.,  60.],
                   [190., 141.,  60.],
                   [189., 141.,  59.],
                   [189., 140.,  59.],
                   [188., 139.,  58.],
                   [187., 139.,  58.],
                   [187., 138.,  57.],
                   [186., 138.,  57.],
                   [186., 137.,  56.],
                   [185., 137.,  56.],
                   [184., 136.,  55.],
                   [184., 136.,  55.],
                   [183., 135.,  54.],
                   [182., 134.,  53.],
                   [182., 134.,  53.],
                   [181., 133.,  52.],
                   [181., 133.,  52.],
                   [180., 132.,  51.],
                   [179., 132.,  51.],
                   [179., 131.,  50.],
                   [178., 130.,  50.],
                   [177., 130.,  49.],
                   [177., 129.,  49.],
                   [176., 129.,  49.],
                   [176., 128.,  48.],
                   [175., 128.,  48.],
                   [174., 127.,  47.],
                   [174., 127.,  47.],
                   [173., 126.,  46.],
                   [172., 125.,  46.],
                   [172., 125.,  45.],
                   [171., 124.,  45.],
                   [170., 124.,  44.],
                   [170., 123.,  44.],
                   [169., 123.,  43.],
                   [169., 122.,  43.],
                   [168., 121.,  42.],
                   [167., 121.,  42.],
                   [167., 120.,  41.],
                   [166., 120.,  41.],
                   [165., 119.,  41.],
                   [165., 119.,  40.],
                   [164., 118.,  40.],
                   [163., 117.,  39.],
                   [163., 117.,  39.],
                   [162., 116.,  38.],
                   [161., 116.,  38.],
                   [161., 115.,  37.],
                   [160., 115.,  37.],
                   [159., 114.,  37.],
                   [159., 113.,  36.],
                   [158., 113.,  36.],
                   [157., 112.,  35.],
                   [157., 112.,  35.],
                   [156., 111.,  34.],
                   [155., 110.,  34.],
                   [155., 110.,  34.],
                   [154., 109.,  33.],
                   [153., 109.,  33.],
                   [152., 108.,  32.],
                   [152., 107.,  32.],
                   [151., 107.,  31.],
                   [150., 106.,  31.],
                   [150., 106.,  31.],
                   [149., 105.,  30.],
                   [148., 104.,  30.],
                   [147., 104.,  29.],
                   [147., 103.,  29.],
                   [146., 102.,  28.],
                   [145., 102.,  28.],
                   [144., 101.,  28.],
                   [144., 101.,  27.],
                   [143., 100.,  27.],
                   [142.,  99.,  26.],
                   [141.,  99.,  26.],
                   [140.,  98.,  26.],
                   [140.,  97.,  25.],
                   [139.,  97.,  25.],
                   [138.,  96.,  25.],
                   [137.,  95.,  24.],
                   [136.,  95.,  24.],
                   [136.,  94.,  23.],
                   [135.,  93.,  23.],
                   [134.,  93.,  23.],
                   [133.,  92.,  22.],
                   [132.,  91.,  22.],
                   [131.,  90.,  21.],
                   [131.,  90.,  21.],
                   [130.,  89.,  21.],
                   [129.,  88.,  20.],
                   [128.,  88.,  20.],
                   [127.,  87.,  20.],
                   [126.,  86.,  19.],
                   [125.,  85.,  19.],
                   [124.,  85.,  19.],
                   [123.,  84.,  18.],
                   [123.,  83.,  18.],
                   [122.,  82.,  17.],
                   [121.,  82.,  17.],
                   [120.,  81.,  17.],
                   [119.,  80.,  16.],
                   [118.,  79.,  16.],
                   [117.,  79.,  16.],
                   [116.,  78.,  15.],
                   [115.,  77.,  15.],
                   [114.,  76.,  15.],
                   [113.,  75.,  14.],
                   [112.,  75.,  14.],
                   [111.,  74.,  14.],
                   [110.,  73.,  13.],
                   [109.,  72.,  13.],
                   [108.,  71.,  12.],
                   [107.,  70.,  12.],
                   [106.,  69.,  12.],
                   [104.,  68.,  11.],
                   [103.,  68.,  11.],
                   [102.,  67.,  11.],
                   [101.,  66.,  10.],
                   [100.,  65.,  10.],
                   [99.,  64.,  10.],
                   [97.,  63.,   9.],
                   [96.,  62.,   9.],
                   [95.,  61.,   9.],
                   [94.,  60.,   8.],
                   [92.,  59.,   8.],
                   [91.,  58.,   8.],
                   [90.,  58.,   7.],
                   [89.,  57.,   7.],
                   [87.,  56.,   7.],
                   [86.,  55.,   6.],
                   [85.,  54.,   6.],
                   [83.,  53.,   6.],
                   [82.,  52.,   5.],
                   [80.,  51.,   5.],
                   [79.,  49.,   5.],
                   [77.,  48.,   4.],
                   [76.,  47.,   4.],
                   [74.,  46.,   4.],
                   [72.,  45.,   3.],
                   [71.,  43.,   3.],
                   [69.,  42.,   3.],
                   [67.,  41.,   2.],
                   [66.,  40.,   2.],
                   [64.,  38.,   2.],
                   [62.,  37.,   1.],
                   [60.,  35.,   1.],
                   [58.,  34.,   1.],
                   [56.,  32.,   0.],
                   [54.,  31.,   0.],
                   [52.,  29.,   0.],
                   [50.,  28.,   0.],
                   [47.,  26.,   0.],
                   [45.,  24.,   0.],
                   [43.,  22.,   0.],
                   [40.,  20.,   0.],
                   [37.,  19.,   0.],
                   [34.,  17.,   0.],
                   [31.,  15.,   0.],
                   [28.,  13.,   0.],
                   [24.,  10.,   0.],
                   [21.,   8.,   0.],
                   [16.,   6.,   0.],
                   [11.,   3.,   0.],
                   [0.,   0.,   0.]])

class GraphWindow(ContextModule):
    """
    Graph Window - Main QT Dialog class for graphs.

    Parameters
    ----------
    parent : parent, optional
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle('Graph Window')

        self.data = None

        vbl = QtWidgets.QVBoxLayout(self)  # self is where layout is assigned
        hbl = QtWidgets.QHBoxLayout()
        self.mmc = MyMplCanvas(self)

        mpl_toolbar = NavigationToolbar2QT(self.mmc, self.parent)

        self.cmb_1 = QtWidgets.QComboBox()
        self.cmb_2 = QtWidgets.QComboBox()
        self.cmb_c = QtWidgets.QComboBox()
        self.spinbox = QtWidgets.QSpinBox()
        self.lbl_1 = QtWidgets.QLabel('Bands:')
        self.lbl_2 = QtWidgets.QLabel('Bands:')
        self.lbl_3 = QtWidgets.QLabel('Value:')
        self.lbl_c = QtWidgets.QLabel('Colour Bar:')
        self.cb_1 = QtWidgets.QCheckBox('Option:')

        tmp = sorted(m for m in colormaps())

        self.cmb_c.addItem('jet')
        self.cmb_c.addItem('viridis')
        self.cmb_c.addItem('terrain')
        self.cmb_c.addItem('MarineCopper')
        self.cmb_c.addItems(tmp)

        if 'MarineCopper' not in colormaps():
            newcmp = ListedColormap(copper/255, 'MarineCopper')
            colormaps.register(newcmp)

        self.cb_1.hide()
        self.lbl_c.hide()
        self.cmb_c.hide()

        hbl.addWidget(self.lbl_1)
        hbl.addWidget(self.cmb_1)
        hbl.addWidget(self.lbl_2)
        hbl.addWidget(self.cmb_2)
        hbl.addWidget(self.lbl_c)
        hbl.addWidget(self.cmb_c)
        hbl.addWidget(self.lbl_3)
        hbl.addWidget(self.spinbox)

        vbl.addWidget(self.mmc)
        vbl.addWidget(mpl_toolbar)
        vbl.addWidget(self.cb_1)
        vbl.addLayout(hbl)

        self.setFocus()

        self.cmb_1.currentIndexChanged.connect(self.change_band)
        self.cmb_2.currentIndexChanged.connect(self.change_band)
        self.cmb_c.currentIndexChanged.connect(self.change_band)
        self.spinbox.valueChanged.connect(self.change_band)
        self.cb_1.stateChanged.connect(self.change_band)

    def change_band(self):
        """
        Combo box to choose band.

        Returns
        -------
        None.

        """


class MyMplCanvas(FigureCanvasQTAgg):
    """
    Matplotlib canvas widget for the actual plot.

    This routine will also allow the picking and movement of nodes of data.

    Parameters
    ----------
    parent : parent, optional
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None):
        fig = Figure(layout='tight')
        self.axes = fig.add_subplot(111)
        self.line = None
        self.ind = None
        self.background = None
        self.pickevents = False
        self.cmap = colormaps['viridis']

        self.ccoeflbls = None
        self.dmat = None
        self.texts = None

        super().__init__(fig)

    def button_release_callback(self, event):
        """
        Mouse button release callback.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
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

    def format_coord(self, x, y):
        """
        Set format coordinate for correlation coefficient plot.

        Parameters
        ----------
        x : float
            x coordinate.
        y : float
            y coordinate.

        Returns
        -------
        str
            Output string to display.

        """
        col = int(x + 0.5)
        row = int(y + 0.5)
        numcols = len(self.ccoeflbls)
        numrows = len(self.ccoeflbls)

        if 0 <= col < numcols and 0 <= row < numrows:
            xlbl = self.ccoeflbls[col]
            ylbl = self.ccoeflbls[::-1][row]
            z = self.dmat[row, col]
            if np.ma.is_masked(z):
                return ''

            return f'{xlbl}, {ylbl} correlation: {z}%'

    def motion_notify_callback(self, event):
        """
        Move mouse callback.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
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
        event : matplotlib.backend_bases.PickEvent
            Pick event.

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
        event : matplotlib.backend_bases.ResizeEvent
            Resize event.

        Returns
        -------
        None.

        """
        r, data = self.line.get_data()
        self.update_lines(r, data)

    def textresize(self, axes):
        """
        Resize the text on a correlation plot when zooming.

        Parameters
        ----------
        axes : Matplotlib axes
            Current Matplotlib axes.

        Returns
        -------
        None.

        """
        if self.texts is None:
            return

        xsize = np.ptp(axes.get_xlim())
        ysize = np.ptp(axes.get_ylim())
        xmin, xmax = axes.get_xlim()
        ymin, ymax = axes.get_ylim()

        size = 35 / np.sqrt(min(xsize, ysize))
        for i in self.texts:
            i.set_size(size)

            xpos, ypos = i.get_position()
            if xmin <= xpos <= xmax and ymin <= ypos <= ymax:
                i.set_visible(True)
            else:
                i.set_visible(False)

    def update_ccoef(self, data, style='Normal'):
        """
        Update the plot from point data.

        Parameters
        ----------
        data : dictionary
            GeoPandas data in a dictionary.
        style : str
            Style of colour mapping.

        Returns
        -------
        None.

        """
        self.figure.clear()

        self.axes = self.figure.add_subplot(111, label='map')
        cb_registry = self.axes.callbacks
        cb_registry.connect('ylim_changed', self.textresize)
        cb_registry.connect('xlim_changed', self.textresize)

        self.axes.tick_params(axis='x', rotation=90)
        self.axes.tick_params(axis='y', rotation=0)
        self.axes.axis('scaled')
        self.axes.set_title('Correlation Coefficients')

        # self.figure.set_figwidth(7)
        # self.figure.set_figheight(7)

        # calculate correlations
        corr = data.corr(numeric_only=True)
        corr = (corr*100).round(0)

        corr = corr.dropna(axis=0, how='all').dropna(axis=1, how='all')
        corr = corr.replace(np.nan, 0)
        corr = corr.astype(int)

        dmat = corr.to_numpy()
        dmat = np.ma.array(dmat, mask=~np.tri(*dmat.shape).astype(bool))

        self.axes.format_coord = self.format_coord
        self.ccoeflbls = corr.columns.tolist()
        self.dmat = dmat[::-1]

        annot_kws = {"size": 35 / np.sqrt(len(corr))}

        if style == 'Normal':
            cmap = colormaps['jet']
            norm = None
        else:
            cmap = colormaps['jet']
            cmap.set_under('w')
            cmap.set_over('w')

            bounds = [50, 60, 70, 80, 90, 99]
            norm = BoundaryNorm(bounds, cmap.N, extend='min')

        im, cbar = heatmap(self.dmat, self.ccoeflbls[::-1], self.ccoeflbls,
                           self.axes, cmap=cmap, cbarlabel='Correlation',
                           norm=norm)

        im.format_cursor_data = lambda x: ''

        self.texts = annotate_heatmap(im, valfmt="{x:.0f}", **annot_kws)

        self.figure.canvas.draw()
        # self.draw_idle()

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
            data1 = data1.dropna(subset=ival)
            x = data1.geometry.x.values
            y = data1.geometry.y.values
            z = data1[ival]

            if x.size < 2:
                continue

            spcing = np.append(spcing, np.sqrt(np.diff(x)**2+np.diff(y)**2))

        spcing = spcing.mean()

        for line in datagrp:
            data1 = line[1]

            x = data1.geometry.x.values
            y = data1.geometry.y.values
            z = data1[ival]

            if x.size < 2:
                continue

            ang = np.arctan2((y[1:]-y[:-1]), (x[1:]-x[:-1]))
            ang = np.append(ang, ang[-1])

            if np.ptp(x) > np.ptp(y) and (x[-1]-x[0]) < 0:
                ang += np.pi

            elif np.ptp(y) > np.ptp(x) and (y[-1]-y[0]) < 0:
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

        # self.figure.tight_layout()
        self.figure.canvas.draw()

    def update_vector(self, data, col, style=None):
        """
        Update the plot from vector data.

        Parameters
        ----------
        data : dictionary
            GeoPandas data in a dictionary.
        col : str
            Label for column to extract.
        style : str or None
            Style of colour mapping.

        Returns
        -------
        None.

        """
        self.figure.clear()

        self.axes = self.figure.add_subplot(111, label='map')
        self.axes.ticklabel_format(style='plain')
        self.axes.tick_params(axis='x', rotation=90)
        self.axes.tick_params(axis='y', rotation=0)
        self.axes.axis('equal')

        if 'LineString' in data.geom_type.iloc[0]:
            tmp = []
            for i in data.geometry:
                tmp.append(np.array(i.coords[:]))

            lcol = mc.LineCollection(tmp)
            self.axes.add_collection(lcol)
            self.axes.autoscale()

        elif 'Polygon' in data.geom_type.iloc[0]:
            tmp = []
            for j in data.geometry:
                tmp.append(np.array(j.exterior.coords[:])[:, :2].tolist())

            lcol = mc.LineCollection(tmp)
            self.axes.add_collection(lcol)
            self.axes.autoscale()

        elif 'Point' in data.geom_type.iloc[0]:
            if col != '' and style is None or 'Normal' in style:
                dstd = data[col].std()
                dmean = data[col].mean()
                vmin = max(dmean-2*dstd, data[col].min())
                vmax = min(dmean+2*dstd, data[col].max())

                # data.plot(ax=self.axes, column=col, aspect='equal',
                #           legend=True, cmap=self.cmap, vmin=vmin,
                #           vmax=vmax)

                scat = self.axes.scatter(data.geometry.x,
                                         data.geometry.y,
                                         c=data[col], vmin=vmin, vmax=vmax,
                                         cmap=self.cmap, marker='.')
                self.figure.colorbar(scat, ax=self.axes, format=frm)
            elif col != '' and 'Standard' in style:
                m3 = data[col].mean()
                s3 = data[col].std()
                x1 = data[col].min()
                x2 = data[col].max()
                eps = (x2-x1)*0.000001

                r2 = int((x2-m3)//s3)
                r2 = min(3, r2)

                bnds = [m3+i*s3 for i in range(0, r2+1)]
                lbls = [f'{i} to {i+1}' for i in range(0, r2)]
                bnds = [x1-eps] + bnds + [x2]
                lbls[0] = 'mean'+lbls[0][1:]
                lbls = ['min to mean']+lbls+[f'{r2} to max']

                z3 = pd.cut(data[col], bnds, labels=False)
                scat = self.axes.scatter(data.geometry.x,
                                         data.geometry.y,
                                         c=z3, cmap=self.cmap, marker='.')
                discrete_colorbar(self.axes, scat, z3, lbls)

            elif col != '' and 'Quartile' in style:
                z3 = pd.qcut(data[col], 4, labels=False, duplicates='drop')

                scat = self.axes.scatter(data.geometry.x,
                                         data.geometry.y,
                                         c=z3+1, cmap=self.cmap, marker='.')
                discrete_colorbar(self.axes, scat, z3+1)

            elif col != '' and 'K-Means' in style:
                z1 = np.array(data[col]).reshape(-1, 1)
                z3 = KMeans(n_clusters=5, n_init='auto').fit_predict(z1)
                scat = self.axes.scatter(data.geometry.x,
                                         data.geometry.y,
                                         c=z3+1, cmap=self.cmap, marker='.')
                discrete_colorbar(self.axes, scat, z3+1)
            else:
                self.axes.scatter(data.geometry.x, data.geometry.y)

        self.axes.xaxis.set_major_formatter(frm)
        self.axes.yaxis.set_major_formatter(frm)

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
        for i in data.geometry:
            if i.geom_type == 'MultiLineString':
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
            ax2.autoscale(enable=True)  # , tight=True)

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
            ax2.autoscale(enable=True)

        self.figure.canvas.draw()

    def update_hist(self, data, col, ylog, iscum):
        """
        Update the histogram plot.

        Parameters
        ----------
        data : dictionary
            GeoPandas data in a dictionary.
        col : str
            Label for column to extract.
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

        dattmp = data.loc[:, col]

        self.axes.hist(dattmp, bins='sqrt', cumulative=iscum,
                       histtype='stepfilled', edgecolor='k')
        self.axes.set_title(col)
        self.axes.set_xlabel('Data Value')
        self.axes.set_ylabel('Counts')

        self.axes.xaxis.set_major_formatter(frm)
        self.axes.yaxis.set_major_formatter(frm)

        if ylog is True:
            self.axes.set_yscale('log')

        self.figure.canvas.draw()


class PlotCCoef(GraphWindow):
    """
    GUI to plot correlation coefficients.

    Parameters
    ----------
    parent : parent, optional
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cmb_2.hide()
        self.lbl_2.hide()
        self.spinbox.hide()
        self.lbl_3.hide()
        self.setWindowTitle('Vector Plot')

    def change_band(self):
        """
        Combo box to choose band.

        Returns
        -------
        None.

        """
        if self.data is None:
            return

        style = self.cmb_1.currentText()
        self.mmc.update_ccoef(self.data, style)

    def run(self):
        """
        Entry point into the routine, used to run context menu item.

        Returns
        -------
        None.

        """
        self.data = self.indata['Vector'][0]

        self.cmb_1.currentIndexChanged.disconnect()
        self.cmb_1.clear()
        self.cmb_1.addItems(['Normal', 'Positive correlation highlights'])
        self.lbl_1.setText('Style:')
        self.cmb_1.currentIndexChanged.connect(self.change_band)

        self.change_band()

        self.show()


class PlotHist(ContextModule):
    """
    GUI to plot histogram from vectors.

    Parameters
    ----------
    parent : parent, optional
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
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
        data = self.indata['Vector'][0]
        col = self.cmb_1.currentText()
        ylog = self.cb_log.isChecked()
        iscum = self.cb_cum.isChecked()
        self.mmc.update_hist(data, col, ylog, iscum)

    def run(self):
        """
        Entry point into the routine, used to run context menu item.

        Returns
        -------
        None.

        """
        data = self.indata['Vector'][0]
        cols = data.select_dtypes(include=np.number).columns.tolist()

        if not cols:
            self.showlog('No numerical columns.')
            return

        self.show()
        self.cmb_1.clear()
        self.cmb_1.addItems(cols)

        self.cmb_1.setCurrentIndex(0)
        self.change_band()


class PlotLines(GraphWindow):
    """
    GUI to plot lines.

    Parameters
    ----------
    parent : parent, optional
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.spinbox.hide()
        self.lbl_3.hide()
        self.xcol = ''
        self.ycol = ''
        self.setWindowTitle('Plot Profiles')

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
        if self.data is None:
            return

        line = self.cmb_1.currentText()
        col = self.cmb_2.currentText()

        data2 = self.data[self.data.line == line]
        data2 = data2.dropna(subset=col)
        x = data2.geometry.x.values
        y = data2.geometry.y.values

        data2 = data2[col].values

        r = np.sqrt((x[1:]-x[:-1])**2+(y[1:]-y[:-1])**2)
        r = np.cumsum(r)
        r = np.concatenate(([0.], r))

        self.mmc.update_lines(r, data2)

    def run(self):
        """
        Entry point into the routine, used to run context menu item.

        Returns
        -------
        None.

        """
        self.data = None
        for i in self.indata['Vector']:
            if i.geom_type.iloc[0] == 'Point':
                self.data = i
                break

        if self.data is None:
            self.showlog('No point type data.')
            return

        self.cmb_1.currentIndexChanged.disconnect()
        self.cmb_2.currentIndexChanged.disconnect()

        self.show()

        filt = ((self.data.columns != 'geometry') &
                (self.data.columns != 'line'))
        cols = list(self.data.columns[filt])
        lines = self.data.line[self.data.line != 'nan'].unique()

        self.cmb_1.clear()
        self.cmb_2.clear()
        self.cmb_1.addItems(lines)
        self.cmb_2.addItems(cols)

        self.lbl_1.setText('Line:')
        self.lbl_2.setText('Column:')

        self.cmb_1.setCurrentIndex(0)
        self.cmb_2.setCurrentIndex(0)

        self.change_band()

        self.cmb_1.currentIndexChanged.connect(self.change_band)
        self.cmb_2.currentIndexChanged.connect(self.change_band)


class PlotLineMap(GraphWindow):
    """
    GUI to plot a line map.

    Parameters
    ----------
    parent : parent, optional
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cmb_2.hide()
        self.lbl_2.hide()
        self.cb_1.show()
        self.setWindowTitle('Profile Map')

    def change_band(self):
        """
        Combo box to choose band.

        Returns
        -------
        None.

        """
        if self.data is None:
            return

        scale = self.spinbox.value()
        i = self.cmb_1.currentText()
        data = self.data.dropna(subset=i)

        self.mmc.update_lmap(data, i, scale, self.cb_1.isChecked())

    def run(self):
        """
        Entry point into the routine, used to run context menu item.

        Returns
        -------
        None.

        """
        self.data = None
        for i in self.indata['Vector']:
            if i.geom_type.iloc[0] == 'Point':
                self.data = i
                break

        if self.data is None:
            self.showlog('No point type data.')
            return

        self.cmb_1.currentIndexChanged.disconnect()
        self.spinbox.valueChanged.disconnect()
        self.cb_1.stateChanged.disconnect()

        self.show()

        data = self.indata['Vector'][0]
        filt = ((data.columns != 'geometry') &
                (data.columns != 'line'))
        cols = list(data.columns[filt])
        self.cmb_1.clear()
        self.cmb_1.addItems(cols)

        self.cb_1.setText('Show Line Labels:')
        self.lbl_1.setText('Column:')
        self.lbl_3.setText('Scale:')
        self.spinbox.setMinimum(1)
        self.spinbox.setMaximum(1000000)
        self.spinbox.setValue(100)

        self.cmb_1.setCurrentIndex(0)

        self.change_band()

        self.cmb_1.currentIndexChanged.connect(self.change_band)
        self.spinbox.valueChanged.connect(self.change_band)
        self.cb_1.stateChanged.connect(self.change_band)


class PlotRose(GraphWindow):
    """
    GUI to plot rose diagrams.

    Parameters
    ----------
    parent : parent, optional
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cmb_2.hide()
        self.lbl_2.hide()
        self.spinbox.setValue(8)
        self.spinbox.setMinimum(2)
        self.spinbox.setMaximum(360)
        self.cb_1.show()

        self.setWindowTitle('Rose Diagram')
        if parent is None:
            self.showlog = print
        else:
            self.showlog = parent.showlog

    def change_band(self):
        """
        Combo box to choose band.

        Returns
        -------
        None.

        """
        if self.data is None:
            return

        i = self.cmb_1.currentIndex()
        equal = self.cb_1.isChecked()

        self.mmc.update_rose(self.data, i, self.spinbox.value(), equal)

    def run(self):
        """
        Entry point into the routine, used to run context menu item.

        Returns
        -------
        None.

        """
        self.data = None
        if 'Vector' not in self.indata:
            return
        for i in self.indata['Vector']:
            if i.geom_type.iloc[0] == 'LineString':
                self.data = i
                break

        if self.data is None:
            self.showlog('No line type data.')
            return

        self.show()
        self.cmb_1.addItem('Average Angle per Feature')
        self.cmb_1.addItem('Angle per segment in Feature')
        self.cb_1.setText('Equal Area Rose Diagram')
        self.lbl_1.setText('Rose Diagram Type:')
        self.cmb_1.setCurrentIndex(0)


class PlotVector(GraphWindow):
    """
    GUI to plot vectors.

    Parameters
    ----------
    parent : parent, optional
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.spinbox.hide()
        self.lbl_3.hide()
        self.cmb_c.show()
        self.lbl_c.show()
        self.setWindowTitle('Vector Plot')

    def change_band(self):
        """
        Combo box to choose band.

        Returns
        -------
        None.

        """
        if self.data is None:
            return

        i = self.cmb_1.currentText()

        if i == '':
            data = self.data
        else:
            data = self.data.dropna(subset=i)
        if data.size == 0:
            i = ''
            data = self.data

        style = self.cmb_2.currentText()

        txt = str(self.cmb_c.currentText())
        self.mmc.cmap = colormaps[txt]

        self.mmc.update_vector(data, i, style)

    def run(self):
        """
        Entry point into the routine, used to run context menu item.

        Returns
        -------
        None.

        """
        self.data = self.indata['Vector'][0]

        cols = list(self.data.select_dtypes(include=np.number).columns)
        if len(cols) > 0 and 'Point' in self.data.geom_type.iloc[0]:
            self.cmb_1.clear()
            self.cmb_1.addItems(cols)
            self.cmb_1.setCurrentIndex(0)
        else:
            self.cmb_1.hide()
            self.lbl_1.hide()
            self.cmb_2.hide()
            self.lbl_2.hide()
            self.cmb_c.hide()
            self.lbl_c.hide()

        self.cmb_2.clear()
        self.cmb_2.addItems(['Normal',
                             'Group using Standard Deviations above Mean (0)',
                             'Group by Quartile',
                             'Group into K-Means Classes'])
        self.lbl_1.setText('Channel:')
        self.lbl_2.setText('Style:')

        self.show()

        self.change_band()


def heatmap(data, row_labels, col_labels, ax, *,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    From Matplotlib.org

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """
    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
    #          rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    Annotate a heatmap.

    From Matplotlib.org.

    Parameters
    ----------
    im
        The AxesImage to be labelled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colours.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colours from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = {'horizontalalignment': 'center',
          'verticalalignment': 'center'}
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.ma.is_masked(data[i, j]):
                continue
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            # tp = TextPath((j-.4, i), valfmt(data[i, j], None), size=0.4)
            # text = im.axes.add_patch(PathPatch(tp, color='black'))
            texts.append(text)

    return texts


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


def _testfn():
    """Calculate structural complexity."""
    import sys
    import os
    from pygmi.vector.iodefs import ImportVector

    sfile = r"D:\Workdata\PyGMI Test Data\Vector\Rose\2329AC_lin_wgs84sutm35.shp"
    sfile = r"D:\buglet_bugs\RS_lineaments_fracturesOnly.shp"
    sfile = r'D:\Work\Programming\geochem\all_geochem.shp'
    sfile = r"D:\temp\geochem.shp"

    app = QtWidgets.QApplication(sys.argv)
    os.chdir(os.path.dirname(sfile))

    IO = ImportVector()
    IO.ifile = sfile
    # IO.cmb_bounds.setCurrentText('SA Mapsheet')
    IO.settings(True)

    # SC = PlotVector()
    SC = PlotCCoef()
    SC.indata = IO.outdata
    SC.run()

    app.exec()


if __name__ == "__main__":
    _testfn()
