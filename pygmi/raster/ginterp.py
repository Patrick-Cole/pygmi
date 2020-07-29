# -----------------------------------------------------------------------------
# Name:        ginterp.py (part of PyGMI)
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
# Licence for original ModestImage code (modified version below)
# ModestImage
# Copyright (c) 2013 Chris Beaumont
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# -----------------------------------------------------------------------------
"""
Plot Raster Data.

This is the raster data interpretation module.  This module allows for the
display of raster data in a variety of modes, as well as the export of that
display to GeoTiff format.

Currently the following is supported
 * Pseudo Color - data mapped to a color map
 * Contours with solid contours
 * RGB ternary images
 * CMYK ternary images
 * Sun shaded or hill shaded images

It can be very effectively used in conjunction with a GIS package which
supports GeoTiff files.
"""

import os
import copy
from math import cos, sin, tan
import numpy as np
import numexpr as ne
from PyQt5 import QtWidgets, QtCore
from scipy import ndimage
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.image as mi
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
from matplotlib import rcParams
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import pygmi.raster.iodefs as iodefs
import pygmi.raster.dataprep as dataprep
import pygmi.menu_default as menu_default


class ModestImage(mi.AxesImage):
    """
    Computationally modest image class - modified for use in PyGMI.

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
            raise NotImplementedError('ModestImage does not support extents')

        self._full_res = None
        self._sx, self._sy = None, None
        self._bounds = (None, None, None, None)
        super().__init__(*args, **kwargs)

        self.smallres = None
        self.cbar = cm.jet
        self.htype = 'Linear'
        self.hstype = 'Linear'
        self.dtype = 'Single Color Map'
        self.cell = 100.
        self.phi = -np.pi/4.
        self.theta = np.pi/4.
        self.alpha = .0
        self.kval = 0.01
        self._rgbacache = None
        self._oldxslice = None
        self._oldyslice = None

    def set_data(self, A):
        """
        Set the image array.

        Parameters
        ----------
        A : numpy/PIL Image A
            numpy/PIL Image A.

        Returns
        -------
        None.

        """
        self._full_res = A
        self._A = A
        self.smallres = A

        self._imcache = None
        self._rgbacache = None
        self._oldxslice = None
        self._oldyslice = None
        self._sx, self._sy = None, None
        if self.axes.dataLim.x0 != np.inf:
            self._scale_to_res()

    def _scale_to_res(self):
        """
        Scale to resolution.

        Change self._A and _extent to render an image whose
        resolution is matched to the eventual rendering.

        Returns
        -------
        None.

        """
        ax = self.axes

        fx0, fy0, fx1, fy1 = ax.dataLim.extents
        try:
            tmp = self._full_res.shape
            rows = tmp[0]
            cols = tmp[1]
        except AttributeError:
            tmp = self._full_res[0].shape
            rows = tmp[0]
            cols = tmp[1]

        ddx = (fx1-fx0)/cols
        ddy = (fy1-fy0)/rows

        ext = ax.transAxes.transform([1, 1]) - ax.transAxes.transform([0, 0])
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        dx, dy = xlim[1] - xlim[0], ylim[1] - ylim[0]

        y0 = max(0, (ylim[0]-fy0)/ddy)
        y1 = min(rows, (ylim[1]-fy0)/ddy)
        x0 = max(0, (xlim[0]-fx0)/ddx)
        x1 = min(cols, (xlim[1] - fx0)/ddx)

        if y1 == y0:
            y1 = y0+1

        if x1 == x0:
            x1 = x0+1

        y0, y1, x0, x1 = [int(i) for i in [y0, y1, x0, x1]]

        # This divisor is to slightly increase the resolution of sunshaded
        # images to get optimal detail.
        divtmp = 1.0
        if self.dtype == 'Sunshade':
            divtmp = 1.5

        sy = max(int(np.ceil(dy/(ddy*ext[1]))/divtmp), 1)
        sx = max(int(np.ceil(dx/(ddx*ext[0]))/divtmp), 1)

        if self._sx is None:
            pass
        elif (sx >= self._sx and sy >= self._sy and
              x0 >= self._bounds[0] and x1 <= self._bounds[1] and
              y0 >= self._bounds[2] and y1 <= self._bounds[3]):
            return

        if self.dtype == 'Single Color Map':
            pseudo = self._full_res[(rows-y1):(rows-y0):sy, x0:x1:sx]
            mask = np.ma.getmaskarray(pseudo)

            if self.htype == '90% Linear, 10% Compact':
                pseudo = histcomp(pseudo, perc=10.)

            if self.htype == '95% Linear, 5% Compact':
                pseudo = histcomp(pseudo)

            if self.htype == '98% Linear, 2% Compact':
                pseudo = histcomp(pseudo, perc=2.)

            if self.htype == 'Histogram Equalization':
                pseudo = histeq(pseudo)

            self.smallres = pseudo.copy()

            pnorm = norm2(pseudo)

            colormap = self.cbar(pnorm)
            colormap[:, :, 3] = np.logical_not(mask)

            self._A = colormap

        elif self.dtype == 'Sunshade':
            pseudo = self._full_res[0][(rows-y1):(rows-y0):sy, x0:x1:sx]
            sun = self._full_res[1][(rows-y1):(rows-y0):sy, x0:x1:sx]
            mask = np.logical_or(pseudo.mask, sun.mask)

            if self.htype == '95% Linear, 5% Compact':
                pseudo = histcomp(pseudo)

            if self.htype == '98% Linear, 2% Compact':
                pseudo = histcomp(pseudo, perc=2.)

            if self.htype == '90% Linear, 10% Compact':
                pseudo = histcomp(pseudo, perc=10.)

            if self.htype == 'Histogram Equalization':
                pseudo = histeq(pseudo)

            if self.hstype == '95% Linear, 5% Compact':
                sun = histcomp(sun)

            if self.hstype == '98% Linear, 2% Compact':
                sun = histcomp(sun, perc=2.)

            if self.hstype == '90% Linear, 10% Compact':
                sun = histcomp(sun, perc=10.)

            if self.hstype == 'Histogram Equalization':
                sun = histeq(sun)

            self.smallres = np.ma.ones((sun.shape[0], sun.shape[1], 2))
            self.smallres[:, :, 0] = pseudo
            self.smallres[:, :, 1] = sun

            sunshader = currentshader(sun.data, self.cell, self.theta,
                                      self.phi, self.alpha)

            snorm = norm2(sunshader)
            pnorm = norm2(pseudo)

            colormap = self.cbar(pnorm)

            colormap[:, :, 0] *= snorm  # red
            colormap[:, :, 1] *= snorm  # green
            colormap[:, :, 2] *= snorm  # blue
            colormap[:, :, 3] = np.logical_not(mask)

            self._A = colormap

        elif 'Ternary' in self.dtype:
            red = self._full_res[0][(rows-y1):(rows-y0):sy, x0:x1:sx]
            green = self._full_res[1][(rows-y1):(rows-y0):sy, x0:x1:sx]
            blue = self._full_res[2][(rows-y1):(rows-y0):sy, x0:x1:sx]
            mask = np.logical_or(red.mask, green.mask)
            mask = np.logical_or(mask, blue.mask)

            if self.htype == '95% Linear, 5% Compact':
                red = histcomp(red)
                green = histcomp(green)
                blue = histcomp(blue)

            if self.htype == '98% Linear, 2% Compact':
                red = histcomp(red, perc=2.)
                green = histcomp(green, perc=2.)
                blue = histcomp(blue, perc=2.)

            if self.htype == '90% Linear, 10% Compact':
                red = histcomp(red, perc=10.)
                green = histcomp(green, perc=10.)
                blue = histcomp(blue, perc=10.)

            if self.htype == 'Histogram Equalization':
                red = histeq(red)
                green = histeq(green)
                blue = histeq(blue)

            self.smallres = np.ma.ones((red.shape[0], red.shape[1], 3))
            self.smallres[:, :, 0] = red
            self.smallres[:, :, 1] = green
            self.smallres[:, :, 2] = blue

            colormap = np.ma.ones((red.shape[0], red.shape[1], 4))
            colormap[:, :, 0] = norm2(red)
            colormap[:, :, 1] = norm2(green)
            colormap[:, :, 2] = norm2(blue)
            colormap[:, :, 3] = np.logical_not(mask)

            if 'CMY' in self.dtype:
                colormap[:, :, 0] = (1-colormap[:, :, 0])*(1-self.kval)
                colormap[:, :, 1] = (1-colormap[:, :, 1])*(1-self.kval)
                colormap[:, :, 2] = (1-colormap[:, :, 2])*(1-self.kval)

            self._A = colormap

        y0 = ylim[0]
        y1 = ylim[1]
        x0 = xlim[0]
        x1 = xlim[1]

        self.set_extent([x0, x1, y0, y1])
        self._sx = sx
        self._sy = sy
        self._bounds = (x0, x1, y0, y1)
        self.changed()

    def draw(self, renderer, *args, **kwargs):
        """
        Draw.

        Parameters
        ----------
        renderer : Matplotlib renderer.
            Matplotlib renderer.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        None.

        """
        # This loop forces the histograms to remain static
        for argb in self.figure.axes[1:]:
            if np.inf in argb.dataLim.extents:
                continue
            if np.nan in argb.dataLim.extents:
                continue
            argb.set_xlim(argb.dataLim.x0, argb.dataLim.x1)
            argb.set_ylim(argb.dataLim.y0, argb.dataLim.y1*1.2)

        # The next command runs the original draw for this class.
        super().draw(renderer, *args, **kwargs)


def imshow(axes, X, cmap=None, norm=None, aspect=None,
           interpolation=None, alpha=None, vmin=None, vmax=None,
           origin=None, extent=None, shape=None, filternorm=1,
           filterrad=4.0, imlim=None, resample=None, url=None, **kwargs):
    """
    Similar to matplotlib's imshow command, but produces a ModestImage.

    Unlike matplotlib version, must explicitly specify axes

    Parameters
    ----------
    axes : TYPE
        DESCRIPTION.
    X : numpy array or PIL image.
        The image data.
    cmap : str or Colormap, optional
        Colormap instance. The default is None.
    norm : Normalize, optional
        Normalize instanc used to scale data. The default is None.
    aspect : {'equal', 'auto'} or float, optional
        Controls the aspect ratio of the axes.. The default is None.
    interpolation : str, optional
        The interpolation method used. The default is None.
    alpha : scaler, optional
        The alpha blending value, between 0 (transparent) and 1 (opaque). The default is None.
    vmin : scalar, optional
        Minimum data value. The default is None.
    vmax : scalar, optional
        Maximum data value. The default is None.
    origin : {'upper', 'lower'}, optional
        Origin location. The default is None.
    extent : scalars (left, right, bottom, top), optional
        The bounding box in data coordinates that the image will fill. The default is None.
    shape : TYPE, optional
        DESCRIPTION. The default is None.
    filternorm : float, optional
        A parameter for the antigrain image resize filter. The default is 1.
    filterrad : float > 0, optional
        The filter radius for filters that have a radius parameter. The default is 4.0.
    imlim : TYPE, optional
        DESCRIPTION. The default is None.
    resample : bool, optional
        When True, use a full resampling method. The default is None.
    url : str, optional
        URL. The default is None.
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    im : pygmi.raster.ginterp.ModestImage
        ModestImage output.

    """

#    if not axes._hold:
#        axes.cla()
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


class MyMplCanvas(FigureCanvas):
    """
    Canvas for the actual plot.

    Attributes
    ----------
    htype : str
        string indicating the histogram stretch to apply to the data
    hstype : str
        string indicating the histogram stretch to apply to the sun data
    cbar : matplotlib color map
        color map to be used for pseudo color bars
    data : list
        list of PyGMI raster data objects - used for color images
    sdata : list
        list of PyGMI raster data objects - used for shaded images
    gmode : str
        string contaning the graphics mode - Contour, Ternary, Sunshade,
        Single Color Map.
    argb : list
        list of matplotlib subplots. There are up to three.
    hhist : list
        matplotlib hist associated with argb
    hband: list
        list of strings containing the band names to be used.
    htxt : list
        list of strings associated with hhist, denoting a raster value (where
        mouse is currently hovering over on image)
    image : Modestimage imshow
        imshow instance - this is the primary way of displaying an image.
    cnt : matplotlib contour
        contour instance - used for the contour image
    cntf : matplotlib contourf
        contourf instance - used for the contour image
    background : matplotlib bounding box
        image bounding box - used in blitting
    bbox_hist_red :  matplotlib bounding box
        red histogram bounding box
    bbox_hist_green :  matplotlib bounding box
        green histogram bounding box
    bbox_hist_blue :  matplotlib bounding box
        blue histogram bounding box
    axes : matpolotlib axes
        axes for the plot
    pinit : numpy array
        calculated with aspect - used in sunshading
    qinit : numpy array
        calculated with aspect - used in sunshading
    phi : float
        azimuth (sunshading)
    theta : float
        sun elevation (sunshading)
    cell : float
        between 1 and 100 - controls sunshade detail.
    alpha : float
        how much incident light is reflected (0 to 1)
    kval : float
        k value for cmyk mode
    """

    def __init__(self, parent=None):
        fig = Figure()
        super().__init__(fig)

        # figure stuff
        self.htype = 'Linear'
        self.hstype = 'Linear'
        self.cbar = cm.jet
        self.data = []
        self.sdata = []
        self.gmode = None
        self.argb = [None, None, None]
        self.hhist = [None, None, None]
        self.hband = [None, None, None]
        self.htxt = [None, None, None]
        self.image = None
        self.cnt = None
        self.cntf = None
        self.background = None
        self.bbox_hist_red = None
        self.bbox_hist_green = None
        self.bbox_hist_blue = None

        gspc = gridspec.GridSpec(3, 4)
        self.axes = fig.add_subplot(gspc[0:, 1:])
        self.axes.xaxis.set_visible(False)
        self.axes.yaxis.set_visible(False)

        for i in range(3):
            self.argb[i] = fig.add_subplot(gspc[i, 0])
            self.argb[i].xaxis.set_visible(False)
            self.argb[i].yaxis.set_visible(False)
            self.argb[i].autoscale(False)

        fig.subplots_adjust(bottom=0.05)
        fig.subplots_adjust(top=.95)
        fig.subplots_adjust(left=0.05)
        fig.subplots_adjust(right=.95)
        fig.subplots_adjust(wspace=0.05)
        fig.subplots_adjust(hspace=0.05)

        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        self.figure.canvas.mpl_connect('motion_notify_event', self.move)
        self.cid = self.figure.canvas.mpl_connect('resize_event',
                                                  self.init_graph)

# sun shading stuff
        self.pinit = None
        self.qinit = None
        self.phi = -np.pi/4.
        self.theta = np.pi/4.
        self.cell = 100.
        self.alpha = .0

# cmyk stuff
        self.kval = 0.01

    def init_graph(self, event=None):
        """
        Initialize the graph.

        Parameters
        ----------
        event : TYPE, optional
            Event. The default is None.

        Returns
        -------
        None.

        """
        self.figure.canvas.mpl_disconnect(self.cid)

        self.axes.clear()
        for i in range(3):
            self.argb[i].clear()

        x_1, x_2, y_1, y_2 = self.data[0].extent

        self.axes.set_xlim(x_1, x_2)
        self.axes.set_ylim(y_1, y_2)
        self.axes.set_aspect('equal')

        self.figure.canvas.draw()
        QtWidgets.QApplication.processEvents()

        self.background = self.figure.canvas.copy_from_bbox(self.axes.bbox)
        self.bbox_hist_red = self.figure.canvas.copy_from_bbox(
            self.argb[0].bbox)
        self.bbox_hist_green = self.figure.canvas.copy_from_bbox(
            self.argb[1].bbox)
        self.bbox_hist_blue = self.figure.canvas.copy_from_bbox(
            self.argb[2].bbox)

        self.image = imshow(self.axes, self.data[0].data, origin='upper',
                            extent=(x_1, x_2, y_1, y_2))

        # This line prevents imshow from generating color values on the
        # toolbar
        self.image.format_cursor_data = lambda x: ""
        self.update_graph()

        self.cid = self.figure.canvas.mpl_connect('resize_event',
                                                  self.init_graph)

    def move(self, event):
        """
        Mouse is moving.

        Parameters
        ----------
        event : TYPE
            Event.

        Returns
        -------
        None.

        """
        if not self.data or self.gmode == 'Contour':
            return

        if event.inaxes == self.axes:
            zval = [-999, -999, -999]

            for i in self.data:
                itlx = i.extent[0]
                itly = i.extent[-1]
                for j in range(3):
                    if i.dataid == self.hband[j]:
                        col = int((event.xdata - itlx)/i.xdim)
                        row = int((itly - event.ydata)/i.ydim)
                        zval[j] = i.data[row, col]

            if self.gmode == 'Single Color Map':
                bnum = self.update_hist_single(zval[0])
                self.figure.canvas.restore_region(self.bbox_hist_red)
                self.argb[0].draw_artist(self.htxt[0])
                self.argb[0].draw_artist(self.hhist[0][2][bnum])
                self.figure.canvas.update()

            if 'Ternary' in self.gmode:
                bnum = self.update_hist_rgb(zval)
                self.figure.canvas.restore_region(self.bbox_hist_red)
                self.figure.canvas.restore_region(self.bbox_hist_green)
                self.figure.canvas.restore_region(self.bbox_hist_blue)

                for j in range(3):
                    self.argb[j].draw_artist(self.htxt[j])
                    self.argb[j].draw_artist(self.hhist[j][2][bnum[j]])

                self.figure.canvas.update()

            if self.gmode == 'Sunshade':
                for i in self.sdata:
                    itlx = i.extent[0]
                    itly = i.extent[-1]
                    for j in [1]:
                        if i.dataid == self.hband[j]:
                            col = int((event.xdata - itlx)/i.xdim)
                            row = int((itly - event.ydata)/i.ydim)
                            zval[j] = i.data[row, col]
                bnum = self.update_hist_sun(zval)
                self.figure.canvas.restore_region(self.bbox_hist_red)
                self.figure.canvas.restore_region(self.bbox_hist_green)
                for j in range(2):
                    self.argb[j].draw_artist(self.htxt[j])
                    self.argb[j].draw_artist(self.hhist[j][2][bnum[j]])

                self.figure.canvas.update()

    def update_contour(self):
        """
        Update contours.

        Returns
        -------
        None.

        """
        self.image.dtype = 'Single Color Map'

        x1, x2, y1, y2 = self.data[0].extent
        self.image.set_visible(False)

        for i in self.data:
            if i.dataid == self.hband[0]:
                dat = i.data

        self.image.set_data(dat)
        dat = norm2(self.image.smallres)

        xdim = (x2-x1)/dat.data.shape[1]/2
        ydim = (y2-y1)/dat.data.shape[0]/2
        xi = np.linspace(x1+xdim, x2-xdim, dat.data.shape[1])
        yi = np.linspace(y2-ydim, y1+ydim, dat.data.shape[0])

        self.cnt = self.axes.contour(xi, yi, dat, extent=(x1, x2, y1, y2),
                                     linewidths=1, colors='k')
        self.cntf = self.axes.contourf(xi, yi, dat, extent=(x1, x2, y1, y2),
                                       cmap=self.cbar)

        self.figure.canvas.draw()

    def update_graph(self):
        """
        Update plot.

        Returns
        -------
        None.

        """
        if not self.data or self.gmode is None:
            return

        self.image.cbar = self.cbar
        self.image.htype = self.htype
        self.image.hstype = self.hstype
        self.image.alpha = self.alpha
        self.image.cell = self.cell
        self.image.theta = self.theta
        self.image.phi = self.phi
        self.image.kval = self.kval

        for i in range(3):
            self.argb[i].clear()

        if self.gmode == 'Single Color Map':
            self.update_single_color_map()

        if self.gmode == 'Contour':
            self.update_contour()

        if 'Ternary' in self.gmode:
            self.update_rgb()

        if self.gmode == 'Sunshade':
            self.update_shade_plot()

    def update_hist_rgb(self, zval):
        """
        Update the rgb histograms.

        Parameters
        ----------
        zval : numpy array
            Data values.

        Returns
        -------
        bnum : list
            Bin numbers.

        """
        hcol = ['r', 'g', 'b']
        if 'CMY' in self.gmode:
            hcol = ['c', 'm', 'y']

        hst = self.hhist
        bnum = []

        for i in range(3):
            bins, patches = hst[i][1:]
            for j in patches:
                j.set_color(hcol[i])

            if np.ma.is_masked(zval[i]) is True:
                bnum.append(0)
                continue

            binnum = (bins < zval[i]).sum()-1

            if -1 < binnum < len(patches):
                patches[binnum].set_color('k')
                bnum.append(binnum)
            else:
                bnum.append(0)
            self.update_hist_text(self.htxt[i], zval[i])
        return bnum

    def update_hist_single(self, zval, hno=0):
        """
        Update the color on a single histogram.

        Parameters
        ----------
        zval : numpy array
            Data values.
        hno : int, optional
            Histogram number. The default is 0.

        Returns
        -------
        binnum : int
            Number of bins.

        """
        hst = self.hhist[hno]
        bins, patches = hst[1:]
        binave = np.arange(0, 1, 1/(bins.size-2))

        if hno == 0:
            bincol = self.cbar(binave)
        else:
            bincol = cm.gray(binave)

        for j, _ in enumerate(patches):
            patches[j].set_color(bincol[j])

# This section draws the black line.
        if np.ma.is_masked(zval) is True:
            return 0
        binnum = (bins < zval).sum()-1

        if binnum < 0 or binnum >= len(patches):
            return 0

        patches[binnum].set_color('k')
        self.update_hist_text(self.htxt[hno], zval)
        return binnum

    def update_hist_sun(self, zval=None):
        """
        Updates a sunshade histogram.

        Parameters
        ----------
        zval : numpy array
            Data values.

        Returns
        -------
        bnum : TYPE
            DESCRIPTION.

        """
        if zval is None:
            zval = [0.0, 0.0]

        bnum = [None, None]
        bnum[0] = self.update_hist_single(zval[0], 0)
        bnum[1] = self.update_hist_single(zval[1], 1)
        return bnum

    def update_hist_text(self, hst, zval):
        """
        Update the value on the histogram

        Parameters
        ----------
        hst : histogram
            Histogram.
        zval : float
            Data value.

        Returns
        -------
        None.

        """
        xmin, xmax, ymin, ymax = hst.axes.axis()
        xnew = 0.95*(xmax-xmin)+xmin
        ynew = 0.95*(ymax-ymin)+ymin
        hst.set_position((xnew, ynew))
        hst.set_text(str(zval))

    def update_rgb(self):
        """
        Update the RGB Ternary Map.

        Returns
        -------
        None.

        """
        self.image.dtype = self.gmode
        dat = [None, None, None]
        for i in self.data:
            for j in range(3):
                if i.dataid == self.hband[j]:
                    dat[j] = i.data

        self.image.set_data(dat)
        hdata = self.image.smallres

        for i in range(3):
            self.hhist[i] = self.argb[i].hist(hdata[:, :, i].compressed(), 50,
                                              ec='none')
            self.htxt[i] = self.argb[i].text(0., 0., '', ha='right', va='top')

            self.argb[i].set_xlim(self.hhist[i][1].min(),
                                  self.hhist[i][1].max())
            self.argb[i].set_ylim(0, self.hhist[i][0].max()*1.2)

        self.update_hist_rgb([-999, -999, -999])

        self.figure.canvas.restore_region(self.background)
        self.figure.canvas.restore_region(self.bbox_hist_red)
        self.figure.canvas.restore_region(self.bbox_hist_green)
        self.figure.canvas.restore_region(self.bbox_hist_blue)

        self.axes.draw_artist(self.image)

        for j in range(3):
            for i in self.hhist[j][2]:
                self.argb[j].draw_artist(i)

        self.figure.canvas.update()

        self.bbox_hist_red = self.figure.canvas.copy_from_bbox(
            self.argb[0].bbox)
        self.bbox_hist_green = self.figure.canvas.copy_from_bbox(
            self.argb[1].bbox)
        self.bbox_hist_blue = self.figure.canvas.copy_from_bbox(
            self.argb[2].bbox)

        for j in range(3):
            self.argb[j].draw_artist(self.htxt[j])

        self.figure.canvas.update()
        self.figure.canvas.flush_events()

    def update_single_color_map(self):
        """
        Updates the single color map.

        Returns
        -------
        None.

        """
        self.image.dtype = 'Single Color Map'
        for i in self.data:
            if i.dataid == self.hband[0]:
                dat = i.data

        self.image.set_data(dat)
        dat = self.image.smallres

        self.hhist[0] = self.argb[0].hist(dat.compressed(), 50, ec='none')
        self.htxt[0] = self.argb[0].text(0.0, 0.0, '', ha='right', va='top')
        self.argb[0].set_xlim(self.hhist[0][1].min(), self.hhist[0][1].max())
        self.argb[0].set_ylim(0, self.hhist[0][0].max()*1.2)

        self.update_hist_single(0.0)

        self.figure.canvas.restore_region(self.background)
        self.figure.canvas.restore_region(self.bbox_hist_red)

        self.axes.draw_artist(self.image)

        for i in self.hhist[0][2]:
            self.argb[0].draw_artist(i)

        self.figure.canvas.update()
        self.bbox_hist_red = self.figure.canvas.copy_from_bbox(
            self.argb[0].bbox)

        self.argb[0].draw_artist(self.htxt[0])
        self.figure.canvas.update()

    def update_shade_plot(self):
        """
        Update sun shade plot.

        Returns
        -------
        None.

        """
        self.image.dtype = 'Sunshade'
        data = [None, None]

        for i in self.data:
            if i.dataid == self.hband[0]:
                data[0] = i.data

        for i in self.sdata:
            if i.dataid == self.hband[1]:
                data[1] = i.data

        self.image.set_data(data)

        hdata = self.image.smallres

        for i in range(2):
            self.hhist[i] = self.argb[i].hist(hdata[:, :, i].compressed(), 50,
                                              ec='none')
            self.htxt[i] = self.argb[i].text(0., 0., '', ha='right', va='top')
            self.argb[i].set_xlim(self.hhist[i][1].min(),
                                  self.hhist[i][1].max())
            self.argb[i].set_ylim(0, self.hhist[i][0].max()*1.2)

        zval = [data[0].data.min(), data[1].data.min()]
        self.update_hist_sun(zval)

        self.figure.canvas.restore_region(self.background)
        self.figure.canvas.restore_region(self.bbox_hist_red)
        self.figure.canvas.restore_region(self.bbox_hist_green)

        self.axes.draw_artist(self.image)

        for j in range(2):
            for i in self.hhist[j][2]:
                self.argb[j].draw_artist(i)

        self.figure.canvas.update()

        self.bbox_hist_red = self.figure.canvas.copy_from_bbox(
            self.argb[0].bbox)
        self.bbox_hist_green = self.figure.canvas.copy_from_bbox(
            self.argb[1].bbox)

        for j in range(2):
            self.argb[j].draw_artist(self.htxt[j])

        self.figure.canvas.update()
        self.figure.canvas.flush_events()


class MySunCanvas(FigureCanvas):
    """
    Canvas for the sunshading tool.

    Attributes
    ----------
    sun: matplotlib plot instance
        plot of a circle 'o' showing where the sun is
    axes: matplotlib axes instance
        axes on which the sun is drawn
    """

    def __init__(self, parent=None):
        fig = Figure()
        super().__init__(fig)

        self.sun = None
        self.axes = fig.add_subplot(111, polar=True)

        self.setParent(parent)
        self.setMaximumSize(120, 120)
        self.setMinimumSize(120, 120)

    def init_graph(self):
        """
        Initialise graph.

        Returns
        -------
        None.

        """
        self.axes.clear()
        self.axes.set_xticklabels(self.axes.get_xticklabels(), fontsize=8)
        self.axes.set_yticklabels(self.axes.get_yticklabels(), visible=False)

        self.axes.set_autoscaley_on(False)
        self.axes.set_rmax(1.0)
        self.axes.set_rmin(0.0)

        self.sun, = self.axes.plot(np.pi/4., cos(np.pi/4.), 'o')
        self.figure.canvas.draw()


class PlotInterp(QtWidgets.QDialog):
    """
    This is the primary class for the raster data interpretation module. The
    main interface is set up from here, as well as monitoring of the mouse
    over the sunshading.

    The PlotInterp class allows for the display of raster data in a variety of
    modes, as well as the export of that display to GeoTiff format.

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    outdata : dictionary
        dictionary of output datasets
    self.mmc : pygmi.raster.ginterp.MyMplCanvas, FigureCanvas
        main canvas containing the image
    self.msc : pygmi.raster.ginterp.MySunCanvas, FigureCanvas
        small canvas containing the sunshading control
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.indata = {}
        self.outdata = {}
        self.parent = parent
        self.units = {}

        self.mmc = MyMplCanvas(self)
        self.msc = MySunCanvas(self)
        self.btn_saveimg = QtWidgets.QPushButton('Save GeoTiff')
        self.cbox_dtype = QtWidgets.QComboBox()
        self.cbox_band1 = QtWidgets.QComboBox()
        self.cbox_band2 = QtWidgets.QComboBox()
        self.cbox_band3 = QtWidgets.QComboBox()
        self.cbox_htype = QtWidgets.QComboBox()
        self.cbox_hstype = QtWidgets.QComboBox()
        self.cbox_cbar = QtWidgets.QComboBox(self)
        self.kslider = QtWidgets.QSlider(QtCore.Qt.Horizontal)  # cmyK
        self.sslider = QtWidgets.QSlider(QtCore.Qt.Horizontal)  # sunshade
        self.aslider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slabel = QtWidgets.QLabel('Sunshade Stretch:')
        self.labels = QtWidgets.QLabel('Sunshade Detail')
        self.labela = QtWidgets.QLabel('Light Reflectance')
        self.labelc = QtWidgets.QLabel('Color Bar:')
        self.labelk = QtWidgets.QLabel('K value:')

        self.setupui()

        self.change_cbar()
        self.setFocus()

        self.mmc.gmode = 'Single Color Map'
        self.mmc.argb[0].set_visible(True)
        self.mmc.argb[1].set_visible(False)
        self.mmc.argb[2].set_visible(False)

        self.slabel.hide()
        self.cbox_hstype.hide()
        self.cbox_band1.show()
        self.cbox_band2.hide()
        self.cbox_band3.hide()
        self.sslider.hide()
        self.aslider.hide()
        self.kslider.hide()
        self.msc.hide()
        self.labela.hide()
        self.labels.hide()
        self.labelk.hide()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        helpdocs = menu_default.HelpButton('pygmi.raster.ginterp')
        label1 = QtWidgets.QLabel('Display Type:')
        label2 = QtWidgets.QLabel('Data Bands:')
        label3 = QtWidgets.QLabel('Histogram Stretch:')

        vbl_raster = QtWidgets.QVBoxLayout()
        hbl_all = QtWidgets.QHBoxLayout(self)
        vbl_right = QtWidgets.QVBoxLayout()

        mpl_toolbar = NavigationToolbar2QT(self.mmc, self)
        spacer = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Fixed,
                                       QtWidgets.QSizePolicy.Expanding)
        self.sslider.setMinimum(1)
        self.sslider.setMaximum(100)
        self.sslider.setValue(25)
        self.aslider.setMinimum(1)
        self.aslider.setMaximum(100)
        self.aslider.setSingleStep(1)
        self.aslider.setValue(75)
        self.kslider.setMinimum(1)
        self.kslider.setMaximum(100)
        self.kslider.setValue(1)

        self.sslider.setSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                   QtWidgets.QSizePolicy.Fixed)
        self.aslider.setSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                   QtWidgets.QSizePolicy.Fixed)
        self.kslider.setSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                   QtWidgets.QSizePolicy.Fixed)
#        tmp = sorted(cm.datad.keys())
        tmp = sorted(m for m in cm.cmap_d.keys() if not
                     m.startswith(('spectral', 'Vega', 'jet')))

        self.cbox_cbar.addItem('jet')
        self.cbox_cbar.addItems(tmp)
        self.cbox_dtype.addItems(['Single Color Map', 'Contour', 'RGB Ternary',
                                  'CMY Ternary', 'Sunshade'])
        self.cbox_htype.addItems(['Linear',
                                  '90% Linear, 10% Compact',
                                  '95% Linear, 5% Compact',
                                  '98% Linear, 2% Compact',
                                  'Histogram Equalization'])
        self.cbox_hstype.addItems(['Linear',
                                   '90% Linear, 10% Compact',
                                   '95% Linear, 5% Compact',
                                   '98% Linear, 2% Compact',
                                   'Histogram Equalization'])

        self.setWindowTitle('Raster Data Interpretation')

        vbl_raster.addWidget(label1)
        vbl_raster.addWidget(self.cbox_dtype)
        vbl_raster.addWidget(label2)
        vbl_raster.addWidget(self.cbox_band1)
        vbl_raster.addWidget(self.cbox_band2)
        vbl_raster.addWidget(self.cbox_band3)
        vbl_raster.addWidget(label3)
        vbl_raster.addWidget(self.cbox_htype)
        vbl_raster.addWidget(self.slabel)
        vbl_raster.addWidget(self.cbox_hstype)
        vbl_raster.addWidget(self.labelc)
        vbl_raster.addWidget(self.cbox_cbar)
        vbl_raster.addWidget(self.msc)
        vbl_raster.addWidget(self.labels)
        vbl_raster.addWidget(self.sslider)
        vbl_raster.addWidget(self.labela)
        vbl_raster.addWidget(self.aslider)
        vbl_raster.addWidget(self.labelk)
        vbl_raster.addWidget(self.kslider)
        vbl_raster.addItem(spacer)
        vbl_raster.addWidget(self.btn_saveimg)
        vbl_raster.addWidget(helpdocs)
        vbl_right.addWidget(self.mmc)
        vbl_right.addWidget(mpl_toolbar)

        hbl_all.addLayout(vbl_raster)
        hbl_all.addLayout(vbl_right)

        self.cbox_cbar.currentIndexChanged.connect(self.change_cbar)
        self.cbox_dtype.currentIndexChanged.connect(self.change_dtype)
        self.cbox_htype.currentIndexChanged.connect(self.change_htype)
        self.cbox_hstype.currentIndexChanged.connect(self.change_hstype)

        self.sslider.sliderReleased.connect(self.change_dtype)
        self.aslider.sliderReleased.connect(self.change_dtype)
        self.kslider.sliderReleased.connect(self.change_dtype)
        self.msc.figure.canvas.mpl_connect('button_press_event', self.move)
        self.btn_saveimg.clicked.connect(self.save_img)

        self.resize(self.parent.width(), self.parent.height())

    def change_blue(self):
        """
        Change the blue or third display band.

        Returns
        -------
        None.

        """
        txt = str(self.cbox_band3.currentText())
        self.mmc.hband[2] = txt
        self.mmc.init_graph()

    def change_cbar(self):
        """
        Change the color map for the color bar.

        Returns
        -------
        None.

        """
        txt = str(self.cbox_cbar.currentText())
        self.mmc.cbar = cm.get_cmap(txt)
        self.mmc.update_graph()

    def change_dtype(self):
        """
        Change display type.

        Returns
        -------
        None.

        """
        txt = str(self.cbox_dtype.currentText())
        self.mmc.gmode = txt
        self.cbox_band1.show()

        if txt == 'Single Color Map':
            self.slabel.hide()
            self.labelc.show()
            self.labelk.hide()
            self.cbox_hstype.hide()
            self.cbox_band2.hide()
            self.cbox_band3.hide()
            self.cbox_cbar.show()
            self.mmc.argb[0].set_visible(True)
            self.mmc.argb[1].set_visible(False)
            self.mmc.argb[2].set_visible(False)
            self.sslider.hide()
            self.aslider.hide()
            self.kslider.hide()
            self.msc.hide()
            self.labela.hide()
            self.labels.hide()
            self.mmc.init_graph()

        if txt == 'Contour':
            self.labelk.hide()
            self.slabel.hide()
            self.labelc.show()
            self.cbox_hstype.hide()
            self.cbox_band2.hide()
            self.cbox_band3.hide()
            self.cbox_cbar.show()
            self.mmc.argb[0].set_visible(False)
            self.mmc.argb[1].set_visible(False)
            self.mmc.argb[2].set_visible(False)
            self.sslider.hide()
            self.aslider.hide()
            self.kslider.hide()
            self.msc.hide()
            self.labela.hide()
            self.labels.hide()
            self.mmc.init_graph()

        if 'Ternary' in txt:
            self.labelk.hide()
            self.slabel.hide()
            self.labelc.hide()
            self.cbox_hstype.hide()
            self.cbox_band2.show()
            self.cbox_band3.show()
            self.cbox_cbar.hide()
            self.mmc.argb[0].set_visible(True)
            self.mmc.argb[1].set_visible(True)
            self.mmc.argb[2].set_visible(True)
            self.sslider.hide()
            self.aslider.hide()
            self.kslider.hide()
            if 'CMY' in txt:
                self.kslider.show()
                self.labelk.show()
                self.mmc.kval = float(self.kslider.value())/100.
            self.msc.hide()
            self.labela.hide()
            self.labels.hide()
            self.mmc.init_graph()

        if txt == 'Sunshade':
            self.labelc.show()
            self.labelk.hide()
            self.msc.show()
            self.sslider.show()
            self.aslider.show()
            self.kslider.hide()
            self.labela.show()
            self.labels.show()
            self.slabel.show()
            self.cbox_hstype.show()
            self.cbox_band2.show()
            self.cbox_band3.hide()
            self.cbox_cbar.show()
            self.mmc.argb[0].set_visible(True)
            self.mmc.argb[1].set_visible(True)
            self.mmc.argb[2].set_visible(False)
            self.mmc.cell = self.sslider.value()
            self.mmc.alpha = float(self.aslider.value())/100.
#            QtWidgets.QApplication.processEvents()
            self.msc.init_graph()
            self.mmc.init_graph()

    def change_green(self):
        """
        Change the greed or second band.

        Returns
        -------
        None.

        """
        txt = str(self.cbox_band2.currentText())
        self.mmc.hband[1] = txt
        self.mmc.init_graph()

    def change_hstype(self):
        """
        Change the histogram stretch to apply to the sun shaded data.

        Returns
        -------
        None.

        """
        txt = str(self.cbox_hstype.currentText())
        self.mmc.hstype = txt
        self.mmc.init_graph()

    def change_htype(self):
        """
        Change the histogram stretch to apply to the normal data.

        Returns
        -------
        None.

        """
        txt = str(self.cbox_htype.currentText())
        self.mmc.htype = txt
        self.mmc.init_graph()

    def change_red(self):
        """
        Change the red or first band.

        Returns
        -------
        None.

        """
        txt = str(self.cbox_band1.currentText())
        self.mmc.hband[0] = txt
        self.mmc.init_graph()

    def data_init(self):
        """
        Data initialise.

        Entry point into routine. This entry point exists for
        the case  where data must be initialised before entering at the
        standard 'settings' sub module.

        Returns
        -------
        None.

        """
        if 'Cluster' in self.indata:
            self.indata = copy.deepcopy(self.indata)
            self.indata = dataprep.cluster_to_raster(self.indata)

        if 'Raster' not in self.indata:
            return

        self.indata['Raster'] = dataprep.merge(self.indata['Raster'])

        data = self.indata['Raster']
        sdata = self.indata['Raster']

        for i in data:
            self.units[i.dataid] = i.units

        self.mmc.data = data
        self.mmc.sdata = sdata
        self.mmc.hband[0] = data[0].dataid
        self.mmc.hband[1] = data[0].dataid
        self.mmc.hband[2] = data[0].dataid

        blist = []
        for i in data:
            blist.append(i.dataid)

        try:
            self.cbox_band1.currentIndexChanged.disconnect()
            self.cbox_band2.currentIndexChanged.disconnect()
            self.cbox_band3.currentIndexChanged.disconnect()
        except TypeError:
            pass

        self.cbox_band1.clear()
        self.cbox_band2.clear()
        self.cbox_band3.clear()
        self.cbox_band1.addItems(blist)
        self.cbox_band2.addItems(blist)
        self.cbox_band3.addItems(blist)

        self.cbox_band1.currentIndexChanged.connect(self.change_red)
        self.cbox_band2.currentIndexChanged.connect(self.change_green)
        self.cbox_band3.currentIndexChanged.connect(self.change_blue)

    def move(self, event):
        """
        Move event is used to track changes to the sunshading.

        Parameters
        ----------
        event : matplotlib button press event
            Event returned by matplotlib when a button is pressed

        Returns
        -------
        None.

        """
        if event.inaxes == self.msc.axes:
            self.msc.sun.set_xdata(event.xdata)
            self.msc.sun.set_ydata(event.ydata)
            self.msc.figure.canvas.draw()

            phi = -event.xdata
            theta = np.pi/2. - np.arccos(event.ydata)
            self.mmc.phi = phi
            self.mmc.theta = theta
            self.mmc.update_graph()

    def save_img(self):
        """
        Save image as a GeoTiff.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """

        ext = 'GeoTiff (*.tif)'
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.parent, 'Save File', '.', ext)
        if filename == '':
            return False

        text, okay = QtWidgets.QInputDialog.getText(
            self, 'Colorbar', 'Enter length in inches:',
            QtWidgets.QLineEdit.Normal, '8')

        if not okay:
            return False

        blen = float(text)
        bwid = blen/16.

        dtype = str(self.cbox_dtype.currentText())

        if 'Ternary' not in dtype:
            text, okay = QtWidgets.QInputDialog.getText(
                self, 'Colorbar', 'Enter colorbar unit label:',
                QtWidgets.QLineEdit.Normal,
                self.units[str(self.cbox_band1.currentText())])

            if not okay:
                return False
        else:
            units = str(self.cbox_band1.currentText())
            rtext, okay = QtWidgets.QInputDialog.getText(
                self, 'Ternary Colorbar', 'Enter red/cyan label:',
                QtWidgets.QLineEdit.Normal, units)

            if not okay:
                return False

            units = str(self.cbox_band2.currentText())
            gtext, okay = QtWidgets.QInputDialog.getText(
                self, 'Ternary Colorbar', 'Enter green/magenta label:',
                QtWidgets.QLineEdit.Normal, units)

            if not okay:
                return False

            units = str(self.cbox_band3.currentText())
            btext, okay = QtWidgets.QInputDialog.getText(
                self, 'Ternary Colorbar', 'Enter blue/yelow label:',
                QtWidgets.QLineEdit.Normal, units)

            if not okay:
                return False

        img = self.mmc.image.get_array()
        htype = str(self.cbox_htype.currentText())
        hstype = str(self.cbox_hstype.currentText())
        cell = self.mmc.cell
        alpha = self.mmc.alpha
        phi = self.mmc.phi
        theta = self.mmc.theta

        if dtype == 'Single Color Map':
            pseudo = self.mmc.image._full_res.copy()
            psmall = self.mmc.image.smallres
            pmask = pseudo.mask.copy()

            pseudo[pseudo < psmall.min()] = psmall.min()
            pseudo[pseudo > psmall.max()] = psmall.max()
            pseudo.mask = pmask

            if htype == '95% Linear, 5% Compact':
                pseudo = histcomp(pseudo)

            if htype == '98% Linear, 2% Compact':
                pseudo = histcomp(pseudo, perc=2.)

            if htype == '90% Linear, 10% Compact':
                pseudo = histcomp(pseudo, perc=10.)

            if htype == 'Histogram Equalization':
                pseudo = histeq(pseudo)

            cmin = pseudo.min()
            cmax = pseudo.max()

            # The function below normalizes as well.
            img = img2rgb(pseudo, self.mmc.cbar)

            pseudo = None

        elif dtype == 'Sunshade':
            pseudo = self.mmc.image._full_res[0]
            sun = self.mmc.image._full_res[1]

            if htype == '90% Linear, 10% Compact':
                pseudo = histcomp(pseudo, perc=10.)

            if htype == '95% Linear, 5% Compact':
                pseudo = histcomp(pseudo)

            if htype == '98% Linear, 2% Compact':
                pseudo = histcomp(pseudo, perc=2.)

            if htype == 'Histogram Equalization':
                pseudo = histeq(pseudo)

            if hstype == '90% Linear, 10% Compact':
                sun = histcomp(sun, perc=10.)

            if hstype == '95% Linear, 5% Compact':
                sun = histcomp(sun)

            if hstype == '98% Linear, 2% Compact':
                sun = histcomp(pseudo, perc=2.)

            if hstype == 'Histogram Equalization':
                sun = histeq(sun)

            cmin = pseudo.min()
            cmax = pseudo.max()

            sunshader = currentshader(sun.data, cell, theta, phi, alpha)
            snorm = norm2(sunshader)

            img = img2rgb(pseudo, self.mmc.cbar)
            pseudo = None
            sunshader = None

            img[:, :, 0] = img[:, :, 0]*snorm  # red
            img[:, :, 1] = img[:, :, 1]*snorm  # green
            img[:, :, 2] = img[:, :, 2]*snorm  # blue
            img = img.astype(np.uint8)

        elif 'Ternary' in dtype:
            red = self.mmc.image._full_res[0]
            green = self.mmc.image._full_res[1]
            blue = self.mmc.image._full_res[2]
            mask = np.logical_or(red.mask, green.mask)
            mask = np.logical_or(mask, blue.mask)
            mask = np.logical_not(mask)

            if htype == '95% Linear, 5% Compact':
                red = histcomp(red)
                green = histcomp(green)
                blue = histcomp(blue)

            if htype == '98% Linear, 2% Compact':
                red = histcomp(red, perc=2.)
                green = histcomp(green, perc=2.)
                blue = histcomp(blue, perc=2.)

            if htype == '90% Linear, 10% Compact':
                red = histcomp(red, perc=10.)
                green = histcomp(green, perc=10.)
                blue = histcomp(blue, perc=10.)

            if htype == 'Histogram Equalization':
                red = histeq(red)
                green = histeq(green)
                blue = histeq(blue)

            cmin = red.min()
            cmax = red.max()

            colormap = np.ones((red.shape[0], red.shape[1], 4), dtype=np.uint8)
            colormap[:, :, 3] = mask*254+1

            if 'CMY' in dtype:
                colormap[:, :, 0] = (1-norm2(red))*254+1
                colormap[:, :, 1] = (1-norm2(green))*254+1
                colormap[:, :, 2] = (1-norm2(blue))*254+1
            else:
                colormap[:, :, 0] = norm255(red)
                colormap[:, :, 1] = norm255(green)
                colormap[:, :, 2] = norm255(blue)

            img = colormap

        elif dtype == 'Contour':
            pseudo = self.mmc.image._full_res.copy()
            psmall = self.mmc.image.smallres
            pmask = np.ma.getmaskarray(pseudo)

            pseudo[pseudo < psmall.min()] = psmall.min()
            pseudo[pseudo > psmall.max()] = psmall.max()
            pseudo.mask = pmask

            if htype == '95% Linear, 5% Compact':
                pseudo = histcomp(pseudo)

            if htype == '98% Linear, 2% Compact':
                pseudo = histcomp(pseudo, perc=2.)

            if htype == '90% Linear, 10% Compact':
                pseudo = histcomp(pseudo, perc=10.)

            if htype == 'Histogram Equalization':
                pseudo = histeq(pseudo)

            cmin = pseudo.min()
            cmax = pseudo.max()

            self.mmc.figure.set_frameon(False)
            self.mmc.axes.set_axis_off()
            tmpsize = self.mmc.figure.get_size_inches()
            self.mmc.figure.set_size_inches(tmpsize*3)
            self.mmc.figure.canvas.draw()
            img = np.fromstring(self.mmc.figure.canvas.tostring_argb(),
                                dtype=np.uint8, sep='')
            w, h = self.mmc.figure.canvas.get_width_height()

            self.mmc.figure.set_size_inches(tmpsize)
            self.mmc.figure.set_frameon(True)
            self.mmc.axes.set_axis_on()
            self.mmc.figure.canvas.draw()

            img.shape = (h, w, 4)
            img = np.roll(img, 3, axis=2)

            cmask = np.ones(img.shape[1], dtype=bool)
            for i in range(img.shape[1]):
                if img[:, i, 3].mean() == 0:
                    cmask[i] = False
            img = img[:, cmask]
            rmask = np.ones(img.shape[0], dtype=bool)
            for i in range(img.shape[0]):
                if img[i, :, 3].mean() == 0:
                    rmask[i] = False
            img = img[rmask]

            mask = img[:, :, 3]

        # export = iodefs.ExportData(self.parent)

        os.chdir(os.path.dirname(filename))

        newimg = [copy.deepcopy(self.mmc.data[0]),
                  copy.deepcopy(self.mmc.data[0]),
                  copy.deepcopy(self.mmc.data[0]),
                  copy.deepcopy(self.mmc.data[0])]

        newimg[0].data = img[:, :, 0]
        newimg[1].data = img[:, :, 1]
        newimg[2].data = img[:, :, 2]
        newimg[3].data = img[:, :, 3]

        mask = img[:, :, 3]
        newimg[0].data[newimg[0].data == 0] = 1
        newimg[1].data[newimg[1].data == 0] = 1
        newimg[2].data[newimg[2].data == 0] = 1

        newimg[0].data[mask <= 1] = 0
        newimg[1].data[mask <= 1] = 0
        newimg[2].data[mask <= 1] = 0

        newimg[0].nullvalue = 0
        newimg[1].nullvalue = 0
        newimg[2].nullvalue = 0
        newimg[3].nullvalue = 0

        iodefs.export_gdal(str(filename), newimg, 'GTiff')

# Section for colorbars
        if 'Ternary' not in dtype:
            txt = str(self.cbox_cbar.currentText())
            cmap = cm.get_cmap(txt)
            norm = mcolors.Normalize(vmin=cmin, vmax=cmax)

# Horizontal Bar
            fig = Figure()
            canvas = FigureCanvas(fig)
            fig.set_figwidth(blen)
            fig.set_figheight(bwid+0.75)
            fig.set_tight_layout(True)
            ax = fig.gca()

            cb = mcolorbar.ColorbarBase(ax, cmap=cmap, norm=norm,
                                        orientation='horizontal')
            cb.set_label(text)

            fname = filename[:-4]+'_hcbar.png'
            canvas.print_figure(fname, dpi=300)

# Vertical Bar
            fig = Figure()
            canvas = FigureCanvas(fig)
            fig.set_figwidth(bwid+1)
            fig.set_figheight(blen)
            fig.set_tight_layout(True)
            ax = fig.gca()

            cb = mcolorbar.ColorbarBase(ax, cmap=cmap, norm=norm,
                                        orientation='vertical')
            cb.set_label(text)

            fname = filename[:-4]+'_vcbar.png'
            canvas.print_figure(fname, dpi=300)
        else:
            fig = Figure()
            canvas = FigureCanvas(fig)
            fig.set_tight_layout(True)

            redlabel = rtext
            greenlabel = gtext
            bluelabel = btext

            tmp = np.array([[list(range(255))]*255])
            tmp.shape = (255, 255)
            tmp = np.transpose(tmp)

            red = ndimage.rotate(tmp, 0)
            green = ndimage.rotate(tmp, 120)
            blue = ndimage.rotate(tmp, -120)

            tmp = np.zeros((blue.shape[0], 90))
            blue = np.hstack((tmp, blue))
            green = np.hstack((green, tmp))

            rtmp = np.zeros_like(blue)
            j = 92
            rtmp[:255, j:j+255] = red
            red = rtmp

            if 'RGB' in dtype:
                red = red.max()-red
                green = green.max()-green
                blue = blue.max()-blue

            data = np.transpose([red.flatten(), green.flatten(),
                                 blue.flatten()])
            data.shape = (red.shape[0], red.shape[1], 3)

            data = data[:221, 90:350]

            ax = fig.gca()
            ax.set_xlim((-100, 355))
            ax.set_ylim((-100, 322))

            path = Path([[0, 0], [127.5, 222], [254, 0], [0, 0]])
            patch = PathPatch(path, facecolor='none')
            ax.add_patch(patch)

            data = data.astype(int)

            im = ax.imshow(data, extent=(0, 255, 0, 222), clip_path=patch,
                           clip_on=True)
            im.set_clip_path(patch)

            ax.text(0, -5, greenlabel, horizontalalignment='center',
                    verticalalignment='top', size=20)
            ax.text(254, -5, bluelabel, horizontalalignment='center',
                    verticalalignment='top', size=20)
            ax.text(127.5, 225, redlabel, horizontalalignment='center',
                    size=20)
            ax.tick_params(top='off', right='off', bottom='off', left='off',
                           labelbottom='off', labelleft='off')

            ax.axis('off')
            fname = filename[:-4]+'_tern.png'
            canvas.print_figure(fname, dpi=300)

        QtWidgets.QMessageBox.information(self, 'Information',
                                          'Save to GeoTiff is complete!',
                                          QtWidgets.QMessageBox.Ok)

    def settings(self, nodialog=False):
        """
        Settings.

        This is called when the used double clicks the routine from the
        main PyGMI interface.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """

        if nodialog:
            return True

        if 'Raster' not in self.indata:
            return False
        if self.indata['Raster'][0].isrgb:
            print('RGB images cannot be used in this module.')
            return False

        self.show()
        QtWidgets.QApplication.processEvents()

        self.mmc.init_graph()
        self.msc.init_graph()
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


def aspect2(data):
    """
    Aspect of a dataset.

    Parameters
    ----------
    data : numpy MxN array
        input data used for the aspect calculation

    Returns
    -------
    adeg : numpy masked array
        aspect in degrees
    dzdx : numpy array
        gradient in x direction
    dzdy : numpy array
        gradient in y direction
    """
    cdy = np.array([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])
    cdx = np.array([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]])

    dzdx = ndimage.convolve(data, cdx)  # Use convolve: matrix filtering
    dzdy = ndimage.convolve(data, cdy)  # 'valid' gets reduced array

    dzdx = ne.evaluate('dzdx/8.')
    dzdy = ne.evaluate('dzdy/8.')

# Aspect Section
    pi = np.pi
    adeg = ne.evaluate('90-arctan2(dzdy, -dzdx)*180./pi')
    adeg = np.ma.masked_invalid(adeg)
    adeg[np.ma.less(adeg, 0.)] += 360.
    adeg[np.logical_and(dzdx == 0, dzdy == 0)] = -1.

    return [adeg, dzdx, dzdy]


def currentshader(data, cell, theta, phi, alpha):
    """
    Blinn shader - used for sun shading.

    Parameters
    ----------
    data : numpy array
        Dataset to be shaded.
    cell : float
        between 1 and 100 - controls sunshade detail.
    theta : float
        sun elevation (also called g in code below)
    phi : float
        azimuth
    alpha : float
        how much incident light is reflected (0 to 1)

    Returns
    -------
    R : numpy array
        array containg the shaded results.

    """
    asp = aspect2(data)
    n = 2
    pinit = asp[1]
    qinit = asp[2]
    p = ne.evaluate('pinit/cell')
    q = ne.evaluate('qinit/cell')
    sqrt_1p2q2 = ne.evaluate('sqrt(1+p**2+q**2)')

    cosg2 = cos(theta/2)
    p0 = -cos(phi)*tan(theta)
    q0 = -sin(phi)*tan(theta)
    sqrttmp = ne.evaluate('(1+sqrt(1+p0**2+q0**2))')
    p1 = ne.evaluate('p0 / sqrttmp')
    q1 = ne.evaluate('q0 / sqrttmp')

    cosi = ne.evaluate('((1+p0*p+q0*q)/(sqrt_1p2q2*sqrt(1+p0**2+q0**2)))')
    coss = ne.evaluate('((1+p1*p+q1*q)/(sqrt_1p2q2*sqrt(1+p1**2+q1**2)))')
    Ps = ne.evaluate('coss**n')
    R = np.ma.masked_invalid(ne.evaluate('((1-alpha)+alpha*Ps)*cosi/cosg2'))

    return R


def histcomp(img, nbr_bins=256, perc=5.):
    """
    Histogram Compaction

    This compacts a % of the outliers in data, allowing for a cleaner, linear
    representation of the data.

    Parameters
    ----------
    img : numpy array
        data to compact
    nbr_bins : int
        number of bins to use in compaction

    Returns
    -------
    img2 : numpy array
        compacted array
    """
# get image histogram
    imask = np.ma.getmaskarray(img)
    tmp = img.compressed()
    imhist, bins = np.histogram(tmp, nbr_bins)

    cdf = imhist.cumsum()  # cumulative distribution function
    cdf = cdf / float(cdf[-1])  # normalize

    perc = perc/100.

    sindx = np.arange(nbr_bins)[cdf > perc][0]
    if cdf[0] > (1-perc):
        eindx = 1
    else:
        eindx = np.arange(nbr_bins)[cdf < (1-perc)][-1]+1
    svalue = bins[sindx]
    evalue = bins[eindx]

    scnt = perc*(nbr_bins-1)
    if scnt > sindx:
        scnt = sindx

    ecnt = perc*(nbr_bins-1)
    if ecnt > ((nbr_bins-1)-eindx):
        ecnt = (nbr_bins-1)-eindx

    img2 = np.empty_like(img, dtype=np.float32)
    np.copyto(img2, img)

    filt = np.ma.less(img2, svalue)
    img2[filt] = svalue

    filt = np.ma.greater(img2, evalue)
    img2[filt] = evalue

    img2 = np.ma.array(img2, mask=imask)
# use linear interpolation of cdf to find new pixel values
    return img2


def histeq(img, nbr_bins=32768):
    """
    Histogram Equalization.

    Equalizes the histogram to colors. This allows for seeing as much data as
    possible in the image, at the expense of knowing the real value of the
    data at a point. It bins the data equally - flattening the distribution.

    Parameters
    ----------
    img : numpy array
        input data to be equalised
    nbr_bins : integer
        number of bins to be used in the calculation

    Returns
    -------
    im2 : numpy array
        output data
    """
# get image histogram
    imhist, bins = np.histogram(img.compressed(), nbr_bins)
    bins = (bins[1:]-bins[:-1])/2+bins[:-1]

    cdf = imhist.cumsum()  # cumulative distribution function
    cdf = cdf - cdf[0]  # subtract min, which is first val in cdf
    cdf = cdf.astype(np.int64)
    cdf = nbr_bins * cdf / cdf[-1]  # norm to nbr_bins

# use linear interpolation of cdf to find new pixel values
    im2 = np.interp(img, bins, cdf)
    im2 = np.ma.array(im2, mask=img.mask)

    return im2


def img2rgb(img, cbar=cm.jet):
    """
    Image to RGB.

    convert image to 4 channel rgba color image.

    Parameters
    ----------
    img : numpy array
        array to be converted to rgba image.
    cbar : matplotlib color map
        colormap to apply to the image

    Returns
    -------
    im2 : numpy array
        output rgba image
    """
    im2 = img.copy()
    im2 = norm255(im2)
    cbartmp = cbar(range(255))
    cbartmp = np.array([[0., 0., 0., 1.]]+cbartmp.tolist())*255
    cbartmp = cbartmp.round()
    cbartmp = cbartmp.astype(np.uint8)
    im2 = cbartmp[im2]
    im2[:, :, 3] = np.logical_not(img.mask)*254+1

    return im2


def norm2(dat):
    """
    Normalise array vector between 0 and 1.

    Parameters
    ----------
    dat : numpy array
        array to be normalised

    Returns
    -------
    out : numpy array of floats
        normalised array
    """
    datmin = float(dat.min())
    datptp = float(dat.ptp())
    out = np.ma.array(ne.evaluate('(dat-datmin)/datptp'))
    out.mask = np.ma.getmaskarray(dat)
    return out


def norm255(dat):
    """
    Normalise array vector between 1 and 255.

    Parameters
    ----------
    dat : numpy array
        array to be normalised

    Returns
    -------
    out : numpy array of 8 bit integers
        normalised array
    """
    datmin = float(dat.min())
    datptp = float(dat.ptp())
    out = ne.evaluate('254*(dat-datmin)/datptp+1')
    out = out.round()
    out = out.astype(np.uint8)
    return out
