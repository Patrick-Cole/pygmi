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
    """ Graph Window """
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
            self.combobox1.addItem(i.dataid)
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
            self.combobox1.addItem(i.dataid)

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

    ModestImage
    Copyright (c) 2013 Chris Beaumont

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
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
