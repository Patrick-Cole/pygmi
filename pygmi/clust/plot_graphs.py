# -----------------------------------------------------------------------------
# Name:        plot_graphs.py (part of PyGMI)
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
""" Plot Raster Data """

# pylint: disable=E1101
import numpy as np
from PySide import QtGui, QtCore
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d  # this is used, ignore pylint warning
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as \
    FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as \
    NavigationToolbar

import matplotlib
rcParams = matplotlib.rcParams

import matplotlib.image as mi
import matplotlib.colors as mcolors
import matplotlib.cbook as cbook


class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None):
        # figure stuff
        fig = Figure()
        self.axes = fig.add_subplot(111)
        self.line = None
        self.ind = None
        self.background = None
        self.parent = parent

        FigureCanvas.__init__(self, fig)

#        self.setParent(parent)

#        FigureCanvas.setSizePolicy(self,
#                                   QtGui.QSizePolicy.Expanding,
#                                   QtGui.QSizePolicy.Expanding)
#        FigureCanvas.updateGeometry(self)

        self.figure.canvas.mpl_connect('pick_event', self.onpick)
        self.figure.canvas.mpl_connect('button_release_event',
                                       self.button_release_callback)
        self.figure.canvas.mpl_connect('motion_notify_event',
                                       self.motion_notify_callback)
#        self.figure.canvas.mpl_connect('draw_event', self.draw_callback)

#    def draw_callback(self, event):
#        """ draw callback """
#        self.background = self.figure.canvas.copy_from_bbox(self.axes.bbox)
#        self.axes.draw_artist(self.line)
#        self.figure.canvas.blit(self.axes.bbox)

    def button_release_callback(self, event):
        """ mouse button release """
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self.ind = None

    def motion_notify_callback(self, event):
        """ move mouse """
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

    def update_pcolor(self, data1, dmat):
        """ Update the plot """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        self.axes.pcolor(dmat)
        self.axes.axis('scaled')
        self.axes.set_title('Correlation Coefficients')
        for i in range(len(data1)):
            for j in range(len(data1)):
                self.axes.text(i + .1, j + .4, format(float(dmat[i, j]),
                               '4.2f'))
        dat_mat = [i.bandid for i in data1]
        self.axes.set_xticks(np.array(list(range(len(data1)))) + .5)

        self.axes.set_xticklabels(dat_mat, rotation='vertical')
        self.axes.set_yticks(np.array(list(range(len(data1)))) + .5)

        self.axes.set_yticklabels(dat_mat, rotation='horizontal')
        self.axes.set_xlim(0, len(data1))
        self.axes.set_ylim(0, len(data1))
#        self.figure.colorbar()
        self.figure.canvas.draw()

    def update_raster(self, data1, data2=None):
        """ Update the plot """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)

        extent = (data1.tlx, data1.tlx + data1.cols * data1.xdim,
                  data1.tly - data1.rows * data1.ydim, data1.tly)
#        extent = (0, data1.cols * data1.xdim,
#                  0, data1.rows * data1.ydim)

#        rdata = self.axes.imshow(data1.data, extent=extent,
#                                 interpolation='nearest')

        rdata = imshow(self.axes, data1.data.astype(np.float32), extent=extent,
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

    def update_rgb(self, data1):
        """ Update the plot """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)

        self.axes.imshow(data1.data)
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

    def update_hexbin(self, data1, data2):
        """ Update the plot """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        x = data1.copy()
        y = data2.copy()
        msk = np.logical_or(x.mask, y.mask)
        x.mask = msk
        y.mask = msk
        x = x.compressed()
        y = y.compressed()

        xmin = x.min()
        xmax = x.max()
        ymin = y.min()
        ymax = y.max()

        hbin = self.axes.hexbin(x, y, bins='log')
        self.axes.axis([xmin, xmax, ymin, ymax])
        self.axes.set_title('Hexbin Plot')
        cbar = self.figure.colorbar(hbin)
        cbar.set_label('log10(N)')
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

    def update_wireframe(self, x, y, z):
        """ Update the plot """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111, projection='3d')
        self.axes.plot_wireframe(x, y, z)
        self.axes.set_title('log(Objective Function)')
        self.axes.set_xlabel("Number of Classes")
        self.axes.set_ylabel("Iteration")
        self.figure.canvas.draw()

    def update_hist(self, data1):
        """ Update the plot """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)

        dattmp = data1.data[data1.data.mask == 0].flatten()
        self.axes.hist(dattmp, 50)
        self.axes.set_title(data1.bandid, fontsize=12)
        self.axes.set_xlabel("Data Value", fontsize=8)
        self.axes.set_ylabel("Counts", fontsize=8)
        self.figure.canvas.draw()


class GraphWindow(QtGui.QDialog):
    """ Graph Window """
    def __init__(self, parent=None):
        QtGui.QDialog.__init__(self, parent=None)
        self.parent = parent

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Graph Window")

        vbl = QtGui.QVBoxLayout(self)  # self is where layout is assigned
        self.hbl = QtGui.QHBoxLayout()
        self.mmc = MyMplCanvas(self)
        mpl_toolbar = NavigationToolbar(self.mmc, self)

        self.combobox1 = QtGui.QComboBox(self)
        self.combobox2 = QtGui.QComboBox(self)
        self.label1 = QtGui.QLabel(self)
        self.label2 = QtGui.QLabel(self)
        self.label1.setText('Bands:')
        self.label2.setText('Bands:')
        self.hbl.addWidget(self.label1)
        self.hbl.addWidget(self.combobox1)
        self.hbl.addWidget(self.label2)
        self.hbl.addWidget(self.combobox2)

        vbl.addWidget(self.mmc)
        vbl.addWidget(mpl_toolbar)
        vbl.addLayout(self.hbl)

        self.setFocus()

        self.combobox1.currentIndexChanged.connect(self.change_band)
        self.combobox2.currentIndexChanged.connect(self.change_band)

    def change_band(self):
        """ Combo box to choose band """
        pass


class PlotCCoef(GraphWindow):
    """ Plot 2D Correlation Coeffiecients """
    def __init__(self, parent):
        GraphWindow.__init__(self, parent)
        self.label1.hide()
        self.combobox1.hide()
        self.label2.hide()
        self.combobox2.hide()
        self.indata = {}
        self.parent = parent

    def change_band(self):
        """ Combo box to choose band """
        pass

    def run(self):
        """ Run """
        self.show()
        data = self.indata['Raster']

        dummy_mat = [[self.corr2d(i.data, j.data) for j in data] for i in data]
        dummy_mat = np.array(dummy_mat)

        self.mmc.update_pcolor(data, dummy_mat)

    def corr2d(self, dat1, dat2):
        """ Calculate the 2D correlation """
        if dat1.shape == dat2.shape:
            mdat1 = dat1 - dat1.mean()
            mdat2 = dat2 - dat2.mean()
            numerator = (mdat1 * mdat2).sum()
            denominator = np.sqrt((mdat1 ** 2).sum() * (mdat2 ** 2).sum())
            out = numerator / denominator
            return out


class PlotRaster(GraphWindow):
    """ Plot Raster Class """
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
            self.combobox1.addItem(i.bandid)
        self.change_band()


class PlotMembership(GraphWindow):
    """ Plot Fuzzy Membership data."""
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
            self.combobox1.addItem(i.bandid)

        self.change_band()

    def change_band_two(self):
        """ Combo box to choose band """
        data = self.indata['Cluster']

        i = self.combobox1.currentIndex()
        j = self.combobox2.currentIndex()

        self.mmc.update_membership(data[i], j)


class PlotVRCetc(GraphWindow):
    """ Plot VRC, NCE, OBJ and XBI """
    def __init__(self, parent):
        GraphWindow.__init__(self, parent)
        self.combobox2.hide()
        self.label2.hide()
        self.parent = parent
        self.indata = {}

    def change_band(self):
        """ Combo box to choose band """
        data = self.indata['Cluster']

        j = self.combobox1.currentText()

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


class PlotScatter(GraphWindow):
    """ Plot Raster Class """
    def __init__(self, parent):
        GraphWindow.__init__(self, parent=None)
        self.indata = {}
        self.parent = parent

    def change_band(self):
        """ Combo box to choose band """
        data = self.indata['Raster']
        i = self.combobox1.currentIndex()
        j = self.combobox2.currentIndex()
        self.mmc.update_hexbin(data[i].data, data[j].data)

    def run(self):
        """ Run """
        self.show()
        data = self.indata['Raster']
        for i in data:
            self.combobox1.addItem(i.bandid)
            self.combobox2.addItem(i.bandid)

        self.label1.setText('X Band:')
        self.label2.setText('Y Band:')
        self.combobox1.setCurrentIndex(0)
        self.combobox2.setCurrentIndex(1)


class PlotPoints(GraphWindow):
    """ Plot Raster Class """
    def __init__(self, parent):
        GraphWindow.__init__(self, parent)
        self.indata = {}
        self.parent = parent

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


class PlotHist(GraphWindow):
    """ Plot Hist Class """
    def __init__(self, parent):
        GraphWindow.__init__(self, parent)
        self.label2.hide()
        self.combobox2.hide()
        self.indata = {}
        self.parent = parent

    def change_band(self):
        """ Combo box to choose band """
        data = self.indata['Raster']
        i = self.combobox1.currentIndex()
        self.mmc.update_hist(data[i])

    def run(self):
        """ Run """
        self.show()
        data = self.indata['Raster']
        for i in data:
            self.combobox1.addItem(i.bandid)

        self.label1.setText('Band:')
        self.combobox1.setCurrentIndex(0)
        self.change_band()


class ModestImage(mi.AxesImage):
    """
    Computationally modest image class.

    ModestImage is an extension of the Matplotlib AxesImage class
    better suited for the interactive display of larger images. Before
    drawing, ModestImage resamples the data array based on the screen
    resolution and view window. This has very little affect on the
    appearance of the image, but can substantially cut down on
    computation since calculations of unresolved or clipped pixels
    are skipped.

    The interface of ModestImage is the same as AxesImage. However, it
    does not currently support setting the 'extent' property. There
    may also be weird coordinate warping operations for images that
    I'm not aware of. Don't expect those to work either.
    """
    def __init__(self, *args, **kwargs):
        if 'extent' in kwargs and kwargs['extent'] is not None:
            raise NotImplementedError("ModestImage does not support extents")

        self._full_res = None
        self._sx, self._sy = None, None
        self._bounds = (None, None, None, None)
        super(ModestImage, self).__init__(*args, **kwargs)

    def set_data(self, A):
        """
        Set the image array

        ACCEPTS: numpy/PIL Image A
        """
        self._full_res = A
        self._A = A

        if self._A.dtype != np.uint8 and not np.can_cast(self._A.dtype,
                                                         np.float):
            raise TypeError("Image data can not convert to float")

        if (self._A.ndim not in (2, 3) or
                (self._A.ndim == 3 and self._A.shape[-1] not in (3, 4))):
            raise TypeError("Invalid dimensions for image data")

        self._imcache = None
        self._rgbacache = None
        self._oldxslice = None
        self._oldyslice = None
        self._sx, self._sy = None, None

    def get_array(self):
        """Override to return the full-resolution array"""
        return self._full_res

    def _scale_to_res(self):
        """ Change self._A and _extent to render an image whose
        resolution is matched to the eventual rendering."""

        ax = self.axes

        fx0, fy0, fx1, fy1 = ax.dataLim.extents
        rows, cols = self._full_res.shape
        ddx = (fx1-fx0)/cols
        ddy = (fy1-fy0)/rows

        ext = ax.transAxes.transform([1, 1]) - ax.transAxes.transform([0, 0])
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        dx, dy = xlim[1] - xlim[0], ylim[1] - ylim[0]

        y0 = max(0, (ylim[0]-fy0)/ddy)
        y1 = min(rows, (ylim[1]-fy0)/ddy)
        x0 = max(0, (xlim[0]-fx0)/ddx)
        x1 = min(cols, (xlim[1]-fx0)/ddx)

        if y1 == y0:
            y1 = y0+1

        if x1 == x0:
            x1 = x0+1

#        y0 = max(0, ylim[0] - 5)
#        y1 = min(self._full_res.shape[0], ylim[1] + 5)
#        x0 = max(0, xlim[0] - 5)
#        x1 = min(self._full_res.shape[1], xlim[1] + 5)

        y0, y1, x0, x1 = map(int, [y0, y1, x0, x1])

#        sy = int(max(1, min((y1 - y0), np.ceil(dy / ext[1]))))
#        sx = int(max(1, min((x1 - x0), np.ceil(dx / ext[0]))))

        sy = int(np.ceil(dy/(ddy*ext[1])))
        sx = int(np.ceil(dx/(ddx*ext[0])))

        # have we already calculated what we need?

        if self._sx is None:
            pass
        elif (sx >= self._sx and sy >= self._sy and
              x0 >= self._bounds[0] and x1 <= self._bounds[1] and
              y0 >= self._bounds[2] and y1 <= self._bounds[3]):
            return

        self._A = self._full_res[(rows-y1):(rows-y0):sy, x0:x1:sx]
        self._A = cbook.safe_masked_invalid(self._A)
#        x1 = x0 + self._A.shape[1] * sx
#        y1 = y0 + self._A.shape[0] * sy

        y0 = ylim[0]
        y1 = ylim[1]
        x0 = xlim[0]
        x1 = xlim[1]

        self.set_extent([x0 - .5, x1 - .5, y0 - .5, y1 - .5])
        self._sx = sx
        self._sy = sy
        self._bounds = (x0, x1, y0, y1)
        self.changed()

    def draw(self, renderer, *args, **kwargs):
        self._scale_to_res()
        super(ModestImage, self).draw(renderer, *args, **kwargs)


# def main():
#    from time import time
#    import matplotlib.pyplot as plt
#    import numpy as np
#    x, y = np.mgrid[0:2000, 0:2000]
#    data = np.sin(x / 10.) * np.cos(y / 30.)
#
#    f = plt.figure()
#    ax = f.add_subplot(111)
#
#    #try switching between
#    artist = ModestImage(ax, data=data)
#    #artist = mi.AxesImage(ax, data=data)
#
#    ax.set_aspect('equal')
#    artist.norm.vmin = -1
#    artist.norm.vmax = 1
#
#    ax.add_artist(artist)
#    ax.set_xlim(0, 1000)
#    ax.set_ylim(0, 1000)
#
#    t0 = time()
#    plt.gcf().canvas.draw()
#    t1 = time()
#
#    print "Draw time for %s: %0.1f ms" % (artist.__class__.__name__,
#                                          (t1 - t0) * 1000)
#
#    plt.show()


def imshow(axes, X, cmap=None, norm=None, aspect=None,
           interpolation=None, alpha=None, vmin=None, vmax=None,
           origin=None, extent=None, shape=None, filternorm=1,
           filterrad=4.0, imlim=None, resample=None, url=None, **kwargs):
    """Similar to matplotlib's imshow command, but produces a ModestImage

    Unlike matplotlib version, must explicitly specify axes
    """

    if not axes._hold:
        axes.cla()
    if norm is not None:
        assert isinstance(norm, mcolors.Normalize)
    if aspect is None:
        aspect = rcParams['image.aspect']
    axes.set_aspect(aspect)
    im = ModestImage(axes, cmap, norm, interpolation, origin, extent,
                     filternorm=filternorm, filterrad=filterrad,
                     resample=resample, **kwargs)

    im.set_data(X)
    im.set_alpha(alpha)
    axes._set_artist_props(im)

    if im.get_clip_path() is None:
        # image does not already have clipping set, clip to axes patch
        im.set_clip_path(axes.patch)

    # if norm is None and shape is None:
    #    im.set_clim(vmin, vmax)
    if vmin is not None or vmax is not None:
        im.set_clim(vmin, vmax)
    else:
        im.autoscale_None()
    im.set_url(url)

    # update ax.dataLim, and, if autoscaling, set viewLim
    # to tightly fit the image, regardless of dataLim.
    im.set_extent(im.get_extent())

    axes.images.append(im)
    im._remove_method = lambda h: axes.images.remove(h)

    return im

# if __name__ == "__main__":
#    main()
