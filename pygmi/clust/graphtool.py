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
""" Graph tool is a multi-function graphing tool """

import numpy as np
from PyQt4 import QtGui, QtCore
from matplotlib.figure import Figure
import matplotlib.pylab as plt
from matplotlib.path import Path
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as \
    FigureCanvas


class GraphHist(FigureCanvas):
    """Plots several lines in distinct colors."""
    def __init__(self, parent):
        self.figure = Figure()

        FigureCanvas.__init__(self, self.figure)
        self.setParent(parent)

        self.nullfmt = plt.NullFormatter()
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
        """ Routine to get the scattergram with histogram overlay """
        xyhist = np.zeros((bins + 1, bins + 1))
        xxx = self.xcoord.compressed()
        yyy = self.ycoord.compressed()

        xyhist = np.histogram2d(xxx, yyy, bins + 1)

        xymahist = np.ma.masked_equal(xyhist[0], 0)
        return xymahist

    def get_clust_scat(self, bins, dattmp, ctmp):
        """ Routine to get the scattergram with cluster overlay """
        clust = np.ma.array(dattmp[ctmp[2] - 1].data.flatten())
        clust.mask = self.xcoord.mask
        clust = clust.compressed()
        xxx = self.xcoord.compressed()
        yyy = self.ycoord.compressed()
        xyhist = np.zeros((bins + 1, bins + 1))

        xyhist[xxx, yyy] = (clust + 1)

        xymahist = np.ma.masked_equal(xyhist, 0)
        return xymahist

    def init_graph(self):
        """ Initialize the Graph """
        # definitions for the axes
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
                                         cmap=plt.cm.jet, aspect='auto')

        self.figure.canvas.draw()

    def polyint(self):
        """ Polygon Interactor routine """
        pntxy = np.transpose([self.xcoord, self.ycoord])
        self.polyi = PolygonInteractor(self.axscatter, pntxy)
        self.polyi.ishist = True

    def setup_coords(self):
        """ Routine to setup the coordinates for the scattergram """
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
        """ Routine to setup the 1d histograms """
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
        """ Draw Routine """
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
    """Plots several lines in distinct colors."""
    def __init__(self, parent):
        self.figure = Figure()

        FigureCanvas.__init__(self, self.figure)
        self.setParent(parent)

        self.parent = parent
        self.polyi = None
        self.data = []
        self.cdata = []
        self.mindx = [0, 0]
        self.csp = None
        self.subplot = None

#    def dat_extent(self, dat):
#        """ Gets the extend of the dat variable """
#        left = dat.tlx
#        top = dat.tly
#        right = left + dat.cols*dat.xdim
#        bottom = top - dat.rows*dat.ydim
#        return (left, right, bottom, top)

    def init_graph(self):
        """ Initialize the Graph """
        mtmp = self.mindx
        dat = self.data[mtmp[0]]

        self.figure.clf()
        self.subplot = self.figure.add_subplot(111)
        self.subplot.get_xaxis().set_visible(False)
        self.subplot.get_yaxis().set_visible(False)

        self.csp = self.subplot.imshow(dat.data, cmap=plt.cm.jet)
        self.subplot.figure.colorbar(self.csp)

        self.figure.canvas.draw()

    def polyint(self):
        """ Polygon Integrator """
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
        """ Draw routine """
        mtmp = self.mindx
        dat = self.data[mtmp[0]]

        self.figure.clf()
        self.subplot = self.figure.add_subplot(111)
        self.subplot.get_xaxis().set_visible(False)
        self.subplot.get_yaxis().set_visible(False)

        if mtmp[1] > 0:
            cdat = self.cdata[mtmp[1] - 1].data + 1
            self.csp = self.subplot.imshow(cdat, cmap=plt.cm.jet)
            vals = np.unique(cdat)
            vals = vals.compressed()
            bnds = (vals - 0.5).tolist() + [vals.max() + .5]
            self.subplot.figure.colorbar(self.csp, boundaries=bnds,
                                         values=vals, ticks=vals)
        else:
            self.csp = self.subplot.imshow(dat.data, cmap=plt.cm.jet)
            self.subplot.figure.colorbar(self.csp)

        self.figure.canvas.draw()
        self.polyi.draw_callback()


class PolygonInteractor(QtCore.QObject):
    """ Polygon Interactor """
    showverts = True
    epsilon = 5  # max pixel distance to count as a vertex hit
    polyi_changed = QtCore.pyqtSignal(list)

    def __init__(self, axtmp, pntxy):
        QtCore.QObject.__init__(self)
        self.ax = axtmp
        self.poly = plt.Polygon([(1, 1)], animated=True)
        self.ax.add_patch(self.poly)
        self.canvas = self.poly.figure.canvas
        self.poly.set_alpha(0.5)
        self.pntxy = pntxy
        self.ishist = True
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)

        xtmp, ytmp = list(zip(*self.poly.xy))

        self.line = plt.Line2D(xtmp, ytmp, marker='o', markerfacecolor='r',
                               color='y', animated=True)
        self.ax.add_line(self.line)
#        self._update_line(poly)

        self.poly.add_callback(self.poly_changed)
        self._ind = None  # the active vert

        self.canvas.mpl_connect('button_press_event',
                                self.button_press_callback)
        self.canvas.mpl_connect('button_release_event',
                                self.button_release_callback)
        self.canvas.mpl_connect('motion_notify_event',
                                self.motion_notify_callback)

    def draw_callback(self):
        """ Draw callback """
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        QtGui.QApplication.processEvents()

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
#        self.canvas.blit(self.ax.bbox)
        self.canvas.update()

    def new_poly(self, npoly):
        """ New Polygon """
        self.poly.set_xy(npoly)
        self.line.set_data(list(zip(*self.poly.xy)))

        self.canvas.draw()
        self.update_plots()

    def poly_changed(self, poly):
        """ Changed Polygon """
        # this method is called whenever the polygon object is called
        # only copy the artist props to the line (except visibility)
        vis = self.line.get_visible()
        plt.Artist.update_from(self.line, poly)
        self.line.set_visible(vis)  # don't use the poly visibility state

    def get_ind_under_point(self, event):
        """get the index of vertex under point if within epsilon tolerance"""

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
        """whenever a mouse button is pressed"""
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self._ind = self.get_ind_under_point(event)

        if self._ind is None:
            xys = self.poly.get_transform().transform(self.poly.xy)
            ptmp = event.x, event.y  # display coords

            if len(xys) == 1:
                self.poly.xy = np.array(
                    [(event.xdata, event.ydata)] +
                    [(event.xdata, event.ydata)])
                self.line.set_data(list(zip(*self.poly.xy)))

                self.canvas.restore_region(self.background)
                self.ax.draw_artist(self.poly)
                self.ax.draw_artist(self.line)
#                self.canvas.blit(self.ax.bbox)
                self.canvas.update()
                return
            dmin = -1
            imin = -1
            for i in range(len(xys) - 1):
                s0tmp = xys[i]
                s1tmp = xys[i + 1]
                dtmp = plt.dist_point_to_segment(ptmp, s0tmp, s1tmp)
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
#            self.canvas.blit(self.ax.bbox)
            self.canvas.update()

    def button_release_callback(self, event):
        """Whenever a mouse button is released"""
        if event.button != 1:
            return
        self._ind = None
        self.update_plots()

    def update_plots(self):
        """ Update Plots """
        polymask = Path(self.poly.xy).contains_points(self.pntxy)
        self.polyi_changed.emit(polymask.tolist())

    def motion_notify_callback(self, event):
        """on mouse movement"""
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
#        self.canvas.blit(self.ax.bbox)
        self.canvas.update()


class ScatterPlot(QtGui.QDialog):
    """ Main Graph Tool Routine """
    def __init__(self, parent):
        QtGui.QDialog.__init__(self, parent)
        self.indata = {}
        self.outdata = {}
        self.parent = parent
        self.m1 = 0
        self.c = [0, 1, 0]
        self.m = [0, 0]
        self.dat_tmp = None

        self.grid_main = QtGui.QGridLayout(self)
        self.map = GraphMap(self)
        self.hist = GraphHist(self)

        self.group_cp = QtGui.QGroupBox(self)
        self.grid_left = QtGui.QGridLayout(self.group_cp)
        self.cp_dpoly = QtGui.QPushButton(self)
        self.cp_combo = QtGui.QComboBox(self)
        self.cp_combo2 = QtGui.QComboBox(self)
        self.cp_combo3 = QtGui.QComboBox(self)

        self.group_map = QtGui.QGroupBox(self)
        self.grid_right = QtGui.QGridLayout(self.group_map)
        self.map_dpoly = QtGui.QPushButton(self)
        self.map_combo = QtGui.QComboBox(self)
        self.map_combo2 = QtGui.QComboBox(self)

        self.setupui()

        self.hist.cindx = self.c
        self.map.mindx = self.m

    def setupui(self):
        """ Setup UI """

        self.setWindowTitle("Graph Window")
        self.group_map.setTitle("Map Settings")
        self.group_cp.setTitle('Cross Plot Settings')

        lbl_combo_left = QtGui.QLabel(self)
        lbl_combo2_left = QtGui.QLabel(self)
        lbl_combo3_left = QtGui.QLabel(self)
        lbl_combo_left.setText('X Data Band:')
        lbl_combo2_left.setText('Y Data Band:')
        lbl_combo3_left.setText('Cluster Overlay:')
        self.cp_dpoly.setText('Delete Polygon')

        lbl_combo_right = QtGui.QLabel(self)
        lbl_combo2_right = QtGui.QLabel(self)
        lbl_combo_right.setText('Data Band:')
        lbl_combo2_right.setText('Cluster Overlay:')
        self.map_dpoly.setText('Delete Polygon')

        self.grid_left.addWidget(lbl_combo_left, 0, 0, 1, 1)
        self.grid_left.addWidget(lbl_combo2_left, 1, 0, 1, 1)
        self.grid_left.addWidget(lbl_combo3_left, 2, 0, 1, 1)
        self.grid_left.addWidget(self.cp_dpoly, 0, 2, 1, 1)
        self.grid_left.addWidget(self.cp_combo, 0, 1, 1, 1)
        self.grid_left.addWidget(self.cp_combo2, 1, 1, 1, 1)
        self.grid_left.addWidget(self.cp_combo3, 2, 1, 1, 1)

        self.grid_right.addWidget(lbl_combo_right, 0, 0, 1, 1)
        self.grid_right.addWidget(lbl_combo2_right, 1, 0, 1, 1)
        self.grid_right.addWidget(self.map_dpoly, 0, 2, 1, 1)
        self.grid_right.addWidget(self.map_combo, 0, 1, 1, 1)
        self.grid_right.addWidget(self.map_combo2, 1, 1, 1, 1)

        self.grid_main.addWidget(self.hist, 0, 0, 1, 1)
        self.grid_main.addWidget(self.map, 0, 1, 1, 1)
        self.grid_main.addWidget(self.group_cp, 1, 0, 1, 1)
        self.grid_main.addWidget(self.group_map, 1, 1, 1, 1)

        self.cp_dpoly.clicked.connect(self.on_cp_dpoly)
        self.map_dpoly.clicked.connect(self.on_map_dpoly)

    def on_cp_dpoly(self):
        """ cp dpoly """
        self.hist.polyi.new_poly([[10, 10]])
        dattmp = self.map.csp.get_array()
        dattmp.mask = np.zeros_like(dattmp.mask)
        self.map.csp.changed()
        self.map.figure.canvas.draw()

    def on_map_dpoly(self):
        """ map dpoly """
        self.map.polyi.new_poly([[10, 10]])
        dattmp = self.hist.csp.get_array()
        dattmp.mask = np.ma.masked_equal(dattmp.data, 0.).mask
        self.hist.csp.changed()
        self.hist.figure.canvas.draw()

    def on_cp_combo(self):
        """ On Combo """
        gstmp = self.cp_combo.currentIndex()
        if gstmp != self.c[0]:
            self.c[0] = gstmp
            self.hist.update_graph(clearaxis=True)

    def on_cp_combo2(self):
        """ On Combo 2 """
        gstmp = self.cp_combo2.currentIndex()
        if gstmp != self.c[1]:
            self.c[1] = gstmp
            self.hist.update_graph(clearaxis=True)

    def on_cp_combo3(self):
        """ On combo 3 """
        self.c[2] = self.cp_combo3.currentIndex()
        self.hist.update_graph()

    def on_map_combo(self):
        """ On map combo """
        self.m[0] = self.map_combo.currentIndex()
        self.map.update_graph()

    def on_map_combo2(self):
        """ On map combo 2 """
        self.m[1] = self.map_combo2.currentIndex()
        self.map.update_graph()

    def settings(self):
        """ run """
        if 'Raster' not in self.indata:
            self.parent.showprocesslog('Error: You must have a multi-band ' +
                                       'raster dataset in addition to your' +
                                       ' cluster analysis results')
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
            cbands += ['Clusters: ' + str(i.data.ptp()) for i in
                       self.indata['Cluster']]
            mbands += ['Clusters: ' + str(i.data.ptp()) for i in
                       self.indata['Cluster']]

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

        return True

    def update_map(self, polymask):
        """ update """
        if max(polymask) is False:
            return
        polymask = np.array(polymask)
        dattmp = self.map.csp.get_array()
        dattmp.mask = polymask
        self.map.csp.changed()
        self.map.figure.canvas.draw()

    def update_hist(self, polymask):
        """ update """
        if max(polymask) is False:
            return

        polymask = np.array(polymask)
        dattmp = self.hist.csp.get_array()
        atmp = np.array([self.hist.xcoord[polymask],
                         self.hist.ycoord[polymask]]).T
        dattmp.mask = np.ones_like(dattmp.mask)
        for i in atmp:
            dattmp.mask[i[1], i[0]] = False
        self.hist.csp.changed()
        self.hist.figure.canvas.draw()
