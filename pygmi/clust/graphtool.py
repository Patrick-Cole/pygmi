# -----------------------------------------------------------------------------
# Name:        graph_tool.py (part of PyGMI)
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
Graph tool is a multi-function graphing tool for use with cluster analysis.
"""

import numpy as np
from PyQt5 import QtWidgets, QtCore
from matplotlib.figure import Figure
from matplotlib import cm
from matplotlib.artist import Artist
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.ticker import NullFormatter
from matplotlib.backends.backend_qt5agg import FigureCanvas


class GraphHist(FigureCanvas):
    """Graph Hist."""

    def __init__(self, parent=None):
        self.figure = Figure()

        super().__init__(self.figure)

        self.setParent(parent)

        self.nullfmt = NullFormatter()
        self.pntxy = None
        self.polyi = None
        self.axhistx = None
        self.axhisty = None
        self.axscatter = None
        self.histx = None
        self.histy = None
        self.xcoord = None
        self.ycoord = None
        self.data = []
        self.cindx = [0, 1, 0]
        self.cdata = []
        self.csp = None

    def get_hist(self, bins):
        """
        Routine to get the scattergram with histogram overlay.

        Parameters
        ----------
        bins : int
            Number of bins.

        Returns
        -------
        xymahist : numpy array
            Output data.

        """
        xyhist = np.zeros((bins + 1, bins + 1))
        xxx = self.xcoord.compressed()
        yyy = self.ycoord.compressed()

        xyhist = np.histogram2d(xxx, yyy, bins + 1)

        xymahist = np.ma.masked_equal(xyhist[0], 0)
        return xymahist

    def get_clust_scat(self, bins, dattmp, ctmp):
        """
        Routine to get the scattergram with cluster overlay.

        Parameters
        ----------
        bins : int
            Number of bins.
        dattmp : list
            Data.
        ctmp : list
            Cluster indices.

        Returns
        -------
        xymahist : numpy array
            Output data.

        """
        clust = np.ma.array(dattmp[ctmp[2] - 1].data.flatten())
        clust.mask = np.ma.getmaskarray(self.xcoord)
        clust = clust.compressed()
        xxx = self.xcoord.compressed()
        yyy = self.ycoord.compressed()
        xyhist = np.zeros((bins + 1, bins + 1))

        xyhist[xxx, yyy] = (clust + 1)

        xymahist = np.ma.masked_equal(xyhist, 0)
        return xymahist

    def init_graph(self):
        """
        Initialize the Graph.

        Returns
        -------
        None.

        """
        self.figure.clf()

        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = bottom + height + 0.02
        left_h = left + width + 0.02

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.2]
        rect_histy = [left_h, bottom, 0.2, height]

        self.axscatter = self.figure.add_axes(rect_scatter, label='s')
        self.axhistx = self.figure.add_axes(rect_histx, label='x')
        self.axhisty = self.figure.add_axes(rect_histy, label='y')

        # Setup the coordinates
        self.setup_coords()

        # setup 1d histograms
        self.setup_hist()

# Compressed eliminates the masked values so that hist
        xymahist = self.get_hist(50)

        self.axscatter.get_xaxis().set_visible(False)
        self.axscatter.get_yaxis().set_visible(False)

        self.csp = self.axscatter.imshow(xymahist.T, interpolation='nearest',
                                         cmap=cm.jet, aspect='auto')

        self.csp.set_clim(xymahist.min(), xymahist.max())
        self.csp.changed()
        self.figure.canvas.draw()

    def polyint(self):
        """
        Polygon Interactor routine.

        Returns
        -------
        None.

        """
        pntxy = np.transpose([self.xcoord, self.ycoord])
        self.polyi = PolygonInteractor(self.axscatter, pntxy)
        self.polyi.ishist = True

    def setup_coords(self):
        """
        Routine to setup the coordinates for the scattergram.

        Returns
        -------
        None.

        """
        self.xcoord = self.data[self.cindx[0]].data.flatten()
        self.ycoord = self.data[self.cindx[1]].data.flatten()
        self.xcoord -= self.xcoord.min()
        self.ycoord -= self.ycoord.min()
        xptp = self.xcoord.ptp()
        yptp = self.ycoord.ptp()
        xstep = xptp / 50
        ystep = yptp / 50
        self.xcoord /= xstep
        self.ycoord /= ystep
        self.xcoord = self.xcoord.astype(int)
        self.ycoord = self.ycoord.astype(int)

    def setup_hist(self):
        """
        Routine to setup the 1D histograms.

        Returns
        -------
        None.

        """
        self.axhistx.xaxis.set_major_formatter(self.nullfmt)
        self.axhisty.yaxis.set_major_formatter(self.nullfmt)
        self.axhistx.yaxis.set_major_formatter(self.nullfmt)
        self.axhisty.xaxis.set_major_formatter(self.nullfmt)
        xrng = [self.xcoord.min(), self.xcoord.max()]
        yrng = [self.ycoord.min(), self.ycoord.max()]
        self.histx = self.axhistx.hist(self.xcoord.compressed(), 50)
        self.histy = self.axhisty.hist(self.ycoord.compressed(), 50,
                                       orientation='horizontal')
        self.axhistx.set_xlim(xrng)
        self.axhisty.set_ylim(yrng[::-1])

    def update_graph(self, clearaxis=False):
        """
        Draw Routine.

        Parameters
        ----------
        clearaxis : bool, optional
            True to clear the axis. The default is False.

        Returns
        -------
        None.

        """
        if clearaxis is True:
            self.axhistx.cla()
            self.axhisty.cla()

            self.setup_coords()
            self.polyi.pntxy = np.array([self.xcoord, self.ycoord]).T
            self.setup_hist()

        if self.cindx[2] > 0:
            xymahist = self.get_clust_scat(50, self.cdata, self.cindx)
        else:
            xymahist = self.get_hist(50)

        if self.csp is None:
            return

        self.csp.set_data(xymahist.T)
        self.csp.set_clim(xymahist.min(), xymahist.max())
        self.csp.changed()
        self.figure.canvas.draw()
        self.polyi.draw_callback()


class GraphMap(FigureCanvas):
    """
    Graph Map.

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    """

    def __init__(self, parent=None):
        self.figure = Figure()

        super().__init__(self.figure)
        self.setParent(parent)

        self.parent = parent
        self.polyi = None
        self.data = []
        self.cdata = []
        self.mindx = [0, 0]
        self.csp = None
        self.subplot = None

    def init_graph(self):
        """
        Initialize the Graph.

        Returns
        -------
        None.

        """
        mtmp = self.mindx
        dat = self.data[mtmp[0]]

        self.figure.clf()
        self.subplot = self.figure.add_subplot(111)
        self.subplot.get_xaxis().set_visible(False)
        self.subplot.get_yaxis().set_visible(False)

        self.csp = self.subplot.imshow(dat.data, cmap=cm.jet)
        self.subplot.figure.colorbar(self.csp)

        self.figure.canvas.draw()

    def polyint(self):
        """
        Polygon Integrator.

        Returns
        -------
        None.

        """
        mtmp = self.mindx
        dat = self.data[mtmp[0]].data

        xtmp = np.arange(dat.shape[1])
        ytmp = np.arange(dat.shape[0])
        xmesh, ymesh = np.meshgrid(xtmp, ytmp)
        xmesh = np.ma.array(xmesh, dtype=float, mask=dat.mask)
        ymesh = np.ma.array(ymesh, dtype=float, mask=dat.mask)
        xmesh = xmesh.flatten()
        ymesh = ymesh.flatten()
        xmesh = xmesh.filled(np.nan)
        ymesh = ymesh.filled(np.nan)
        pntxy = np.transpose([xmesh, ymesh])
        self.polyi = PolygonInteractor(self.subplot, pntxy)
        self.polyi.ishist = False

    def update_graph(self):
        """
        Draw routine.

        Returns
        -------
        None.

        """
        mtmp = self.mindx
        dat = self.data[mtmp[0]]

        if mtmp[1] > 0:
            cdat = self.cdata[mtmp[1] - 1].data
            self.csp.set_data(cdat)
            self.csp.set_clim(cdat.min(), cdat.max())
        else:
            self.csp.set_data(dat.data)
            self.csp.set_clim(dat.data.min(), dat.data.max())

        self.csp.changed()
        self.figure.canvas.draw()
        self.polyi.draw_callback()


class PolygonInteractor(QtCore.QObject):
    """Polygon Interactor."""

    showverts = True
    epsilon = 5  # max pixel distance to count as a vertex hit
    polyi_changed = QtCore.pyqtSignal(list)

    def __init__(self, axtmp, pntxy):
        super().__init__()
        self.ax = axtmp
        self.poly = Polygon([(1, 1)], animated=True)
        self.ax.add_patch(self.poly)
        self.canvas = self.poly.figure.canvas
        self.poly.set_alpha(0.5)
        self.pntxy = pntxy
        self.ishist = True
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)

        xtmp, ytmp = list(zip(*self.poly.xy))

        self.line = Line2D(xtmp, ytmp, marker='o', markerfacecolor='r',
                           color='y', animated=True)
        self.ax.add_line(self.line)

        self.poly.add_callback(self.poly_changed)
        self._ind = None  # the active vert

        self.canvas.mpl_connect('button_press_event',
                                self.button_press_callback)
        self.canvas.mpl_connect('button_release_event',
                                self.button_release_callback)
        self.canvas.mpl_connect('motion_notify_event',
                                self.motion_notify_callback)

    def draw_callback(self):
        """
        Draw callback.

        Returns
        -------
        None.

        """
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        QtWidgets.QApplication.processEvents()

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.canvas.update()

    def new_poly(self, npoly):
        """
        New Polygon.

        Parameters
        ----------
        npoly : list
            New polygon coordinates.

        Returns
        -------
        None.

        """
        self.poly.set_xy(npoly)
        self.line.set_data(list(zip(*self.poly.xy)))

        self.canvas.draw()
        self.update_plots()

    def poly_changed(self, poly):
        """
        Polygon changed.

        Parameters
        ----------
        poly : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # this method is called whenever the polygon object is called
        # only copy the artist props to the line (except visibility)
        vis = self.line.get_visible()
        Artist.update_from(self.line, poly)
        self.line.set_visible(vis)  # don't use the poly visibility state

    def get_ind_under_point(self, event):
        """
        Get the index of vertex under point if within epsilon tolerance.

        Parameters
        ----------
        event : TYPE
            DESCRIPTION.

        Returns
        -------
        ind : int or None
            Index of vertex under point.

        """
        # display coords
        xytmp = np.asarray(self.poly.xy)
        xyt = self.poly.get_transform().transform(xytmp)
        xtt, ytt = xyt[:, 0], xyt[:, 1]
        dtt = np.sqrt((xtt - event.x) ** 2 + (ytt - event.y) ** 2)
        indseq = np.nonzero(np.equal(dtt, np.amin(dtt)))[0]
        ind = indseq[0]

        if dtt[ind] >= self.epsilon:
            ind = None

        return ind

    def button_press_callback(self, event):
        """
        Button press callback.

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
        self._ind = self.get_ind_under_point(event)

        if self._ind is None:
            xys = self.poly.get_transform().transform(self.poly.xy)
            ptmp = self.poly.get_transform().transform([event.xdata,
                                                        event.ydata])
#            ptmp = event.x, event.y  # display coords

            if len(xys) == 1:
                self.poly.xy = np.array(
                    [(event.xdata, event.ydata)] +
                    [(event.xdata, event.ydata)])
                self.line.set_data(list(zip(*self.poly.xy)))

                self.canvas.restore_region(self.background)
                self.ax.draw_artist(self.poly)
                self.ax.draw_artist(self.line)
                self.canvas.update()
                return
            dmin = -1
            imin = -1
            for i in range(len(xys) - 1):
                s0tmp = xys[i]
                s1tmp = xys[i + 1]
                dtmp = dist_point_to_segment(ptmp, s0tmp, s1tmp)

                if dmin == -1:
                    dmin = dtmp
                    imin = i
                elif dtmp < dmin:
                    dmin = dtmp
                    imin = i
            i = imin

            self.poly.xy = np.array(list(self.poly.xy[:i + 1]) +
                                    [(event.xdata, event.ydata)] +
                                    list(self.poly.xy[i + 1:]))
            self.line.set_data(list(zip(*self.poly.xy)))

            self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.poly)
            self.ax.draw_artist(self.line)
            self.canvas.update()

    def button_release_callback(self, event):
        """
        Button release callback.

        Parameters
        ----------
        event : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if event.button != 1:
            return
        self._ind = None
        self.update_plots()

    def update_plots(self):
        """
        Update plots.

        Returns
        -------
        None.

        """
        polymask = Path(self.poly.xy).contains_points(self.pntxy)
        self.polyi_changed.emit(polymask.tolist())

    def motion_notify_callback(self, event):
        """
        Mouse notify callback.

        Parameters
        ----------
        event : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        xtmp, ytmp = event.xdata, event.ydata

        self.poly.xy[self._ind] = xtmp, ytmp
        if self._ind == 0:
            self.poly.xy[-1] = xtmp, ytmp

        self.line.set_data(list(zip(*self.poly.xy)))

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.canvas.update()


class ScatterPlot(QtWidgets.QDialog):
    """
    Main Graph Tool Routine.

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    outdata : dictionary
        dictionary of output datasets
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.indata = {}
        self.outdata = {}
        self.parent = parent
        self.m1 = 0
        self.c = [0, 1, 0]
        self.m = [0, 0]
        self.dat_tmp = None
        if parent is None:
            self.showprocesslog = print
        else:
            self.showprocesslog = parent.showprocesslog

        self.map = GraphMap(self)
        self.hist = GraphHist(self)

        self.cp_dpoly = QtWidgets.QPushButton('Delete Polygon')
        self.cp_combo = QtWidgets.QComboBox()
        self.cp_combo2 = QtWidgets.QComboBox()
        self.cp_combo3 = QtWidgets.QComboBox()
        self.map_dpoly = QtWidgets.QPushButton('Delete Polygon')
        self.map_combo = QtWidgets.QComboBox()
        self.map_combo2 = QtWidgets.QComboBox()

        self.setupui()

        self.hist.cindx = self.c
        self.map.mindx = self.m

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        grid_main = QtWidgets.QGridLayout(self)
        group_cp = QtWidgets.QGroupBox('Cross Plot Settings')
        grid_left = QtWidgets.QGridLayout(group_cp)
        group_map = QtWidgets.QGroupBox('Map Settings')
        grid_right = QtWidgets.QGridLayout(group_map)

        self.setWindowTitle('Graph Window')

        lbl_combo_left = QtWidgets.QLabel('X Data Band:')
        lbl_combo2_left = QtWidgets.QLabel('Y Data Band:')
        lbl_combo3_left = QtWidgets.QLabel('Cluster Overlay:')
        lbl_combo_right = QtWidgets.QLabel('Data Band:')
        lbl_combo2_right = QtWidgets.QLabel('Cluster Overlay:')

        grid_left.addWidget(lbl_combo_left, 0, 0, 1, 1)
        grid_left.addWidget(lbl_combo2_left, 1, 0, 1, 1)
        grid_left.addWidget(lbl_combo3_left, 2, 0, 1, 1)
        grid_left.addWidget(self.cp_dpoly, 0, 2, 1, 1)
        grid_left.addWidget(self.cp_combo, 0, 1, 1, 1)
        grid_left.addWidget(self.cp_combo2, 1, 1, 1, 1)
        grid_left.addWidget(self.cp_combo3, 2, 1, 1, 1)
        grid_right.addWidget(lbl_combo_right, 0, 0, 1, 1)
        grid_right.addWidget(lbl_combo2_right, 1, 0, 1, 1)
        grid_right.addWidget(self.map_dpoly, 0, 2, 1, 1)
        grid_right.addWidget(self.map_combo, 0, 1, 1, 1)
        grid_right.addWidget(self.map_combo2, 1, 1, 1, 1)
        grid_main.addWidget(self.hist, 0, 0, 1, 1)
        grid_main.addWidget(self.map, 0, 1, 1, 1)
        grid_main.addWidget(group_cp, 1, 0, 1, 1)
        grid_main.addWidget(group_map, 1, 1, 1, 1)

        self.cp_dpoly.clicked.connect(self.on_cp_dpoly)
        self.map_dpoly.clicked.connect(self.on_map_dpoly)

    def on_cp_dpoly(self):
        """
        On cp dpoly.

        Returns
        -------
        None.

        """
        self.hist.polyi.new_poly([[10, 10]])

        mtmp = self.map_combo.currentIndex()
        mask = self.indata['Raster'][mtmp].data.mask

        dattmp = self.map.csp.get_array()
        dattmp.mask = mask
        self.map.csp.changed()
        self.map.figure.canvas.draw()

    def on_map_dpoly(self):
        """
        On map dpoly.

        Returns
        -------
        None.

        """
        self.map.polyi.new_poly([[10, 10]])
        dattmp = self.hist.csp.get_array()
        dattmp.mask = np.ma.getmaskarray(np.ma.masked_equal(dattmp.data, 0.))
        self.hist.csp.changed()
        self.hist.figure.canvas.draw()

    def on_cp_combo(self):
        """
        On cp combo.

        Returns
        -------
        None.

        """
        gstmp = self.cp_combo.currentIndex()
        if gstmp != self.c[0]:
            self.c[0] = gstmp
            self.hist.update_graph(clearaxis=True)
            self.map.polyi.update_plots()

    def on_cp_combo2(self):
        """
        On cp combo 2.

        Returns
        -------
        None.

        """
        gstmp = self.cp_combo2.currentIndex()
        if gstmp != self.c[1]:
            self.c[1] = gstmp
            self.hist.update_graph(clearaxis=True)
            self.map.polyi.update_plots()

    def on_cp_combo3(self):
        """
        On cp combo 3.

        Returns
        -------
        None.

        """
        self.c[2] = self.cp_combo3.currentIndex()
        self.hist.update_graph()
        self.map.polyi.update_plots()

    def on_map_combo(self):
        """
        On map combo.

        Returns
        -------
        None.

        """
        self.m[0] = self.map_combo.currentIndex()
        self.map.update_graph()
        self.hist.polyi.update_plots()

    def on_map_combo2(self):
        """
        On map combo 2.

        Returns
        -------
        None.

        """
        self.m[1] = self.map_combo2.currentIndex()
        self.map.update_graph()
        self.hist.polyi.update_plots()

    def settings(self, nodialog=False):
        """
        Run.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        if 'Raster' not in self.indata:
            self.showprocesslog('Error: You must have a multi-band raster dataset in '
                  'addition to your cluster analysis results')
            return False

        self.dat_tmp = self.indata['Raster']
        self.map.data = self.indata['Raster']
        self.hist.data = self.indata['Raster']

        bands = [i.dataid for i in self.indata['Raster']]

        self.cp_combo.clear()
        self.cp_combo2.clear()
        self.map_combo.clear()

        self.cp_combo.addItems(bands)
        self.cp_combo2.addItems(bands)
        self.map_combo.addItems(bands)
        self.cp_combo2.setCurrentIndex(1)
        self.cp_combo.currentIndexChanged.connect(self.on_cp_combo)
        self.cp_combo2.currentIndexChanged.connect(self.on_cp_combo2)
        self.map_combo.currentIndexChanged.connect(self.on_map_combo)

        cbands = ['Scatter Amplitudes']
        mbands = ['None']

        if 'Cluster' in self.indata:
            self.hist.cdata = self.indata['Cluster']
            self.map.cdata = self.indata['Cluster']
            cbands += [i.dataid for i in self.indata['Cluster']]
            mbands += [i.dataid for i in self.indata['Cluster']]

        self.cp_combo3.clear()
        self.map_combo2.clear()

        self.cp_combo3.addItems(cbands)
        self.map_combo2.addItems(mbands)
        self.cp_combo3.currentIndexChanged.connect(self.on_cp_combo3)
        self.map_combo2.currentIndexChanged.connect(self.on_map_combo2)

        self.hist.init_graph()
        self.map.init_graph()

        self.show()

        self.hist.polyint()
        self.map.polyint()

        self.hist.polyi.polyi_changed.connect(self.update_map)
        self.map.polyi.polyi_changed.connect(self.update_hist)

        self.hist.update_graph(clearaxis=True)
        self.map.update_graph()

        return True

    def loadproj(self, projdata):
        """
        Loads project data into class.

        Parameters
        ----------
        projdata : dictionary
            Project data loaded from JSON project file.

        Returns
        -------
        chk : bool
            A check to see if settings was successfully run.

        """

        return False

    def saveproj(self):
        """
        Save project data from class.


        Returns
        -------
        projdata : dictionary
            Project data to be saved to JSON project file.

        """
        projdata = {}

#        projdata['ftype'] = '2D Mean'

        return projdata


    def update_map(self, polymask):
        """
        Update map.

        Parameters
        ----------
        polymask : numpy array
            Polygon mask.

        Returns
        -------
        None.

        """
        if max(polymask) is False:
            return

        mtmp = self.map_combo.currentIndex()
        mask = self.indata['Raster'][mtmp].data.mask

        polymask = np.array(polymask)
        polymask.shape = mask.shape
        polymask = np.logical_or(~polymask, mask)

        dattmp = self.map.csp.get_array()
        dattmp.mask = polymask
        self.map.csp.changed()
        self.map.figure.canvas.draw()

    def update_hist(self, polymask):
        """
        Update histogram.

        Parameters
        ----------
        polymask : numpy array
            Polygon mask.

        Returns
        -------
        None.

        """
        if max(polymask) is False:
            return

        polymask = np.array(polymask)
        dattmp = self.hist.csp.get_array()
        atmp = np.array([self.hist.xcoord[polymask],
                         self.hist.ycoord[polymask]]).T
        dattmp.mask = np.ones_like(np.ma.getmaskarray(dattmp))
        for i in atmp:
            dattmp.mask[i[1], i[0]] = False
        self.hist.csp.changed()
        self.hist.figure.canvas.draw()


def dist_point_to_segment(p, s0, s1):
    """
    Dist point to segment.

    Reimplementation of Matplotlib's dist_point_to_segment, after it was
    depreciated. Follows http://geomalgorithms.com/a02-_lines.html

    Parameters
    ----------
    p : numpy array
        Point.
    s0 : numpy array
        Start of segment.
    s1 : numpy array
        End of segment.

    Returns
    -------
    numpy array
        Distance of point to segment.

    """
    p = np.array(p)
    s0 = np.array(s0)
    s1 = np.array(s1)

    v = s1 - s0
    w = p - s0

    c1 = np.dot(w, v)
    if c1 <= 0:
        return np.linalg.norm(p - s0)

    c2 = np.dot(v, v)
    if c2 <= c1:
        return np.linalg.norm(p - s1)

    b = c1/c2
    pb = s0 + b*v

    return np.linalg.norm(p - pb)
