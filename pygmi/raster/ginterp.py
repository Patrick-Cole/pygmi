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
import sys
import copy
from math import cos, sin, tan
import numpy as np
import numexpr as ne
from PyQt5 import QtWidgets, QtCore
from scipy import ndimage
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.pyplot import colormaps
from matplotlib.colors import ListedColormap

import pygmi.raster.iodefs as iodefs
import pygmi.raster.dataprep as dataprep
import pygmi.menu_default as menu_default
from pygmi.raster.modest_image import imshow


class MyMplCanvas(FigureCanvasQTAgg):
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
        string containing the graphics mode - Contour, Ternary, Sunshade,
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
    image : imshow
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
    axes : matplotlib axes
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
        self.cbar = cm.get_cmap('jet')
        self.newcmp = self.cbar
        self.fullhist = False
        self.data = []
        self.sdata = []
        self.gmode = None
        self.argb = [None, None, None]
        self.bgrgb = [None, None, None]
        self.hhist = [None, None, None]
        self.hband = [None, None, None, None]
        self.htxt = [None, None, None]
        self.image = None
        self.cnt = None
        self.cntf = None
        self.background = None
        self.bbox_hist_red = None
        self.bbox_hist_green = None
        self.bbox_hist_blue = None
        self.shade = False
        self.ccbar = None
        self.clippercu = 0.0
        self.clippercl = 0.0
        self.flagresize = False
        self.clipvalu = [None, None, None]
        self.clipvall = [None, None, None]

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

        FigureCanvasQTAgg.setSizePolicy(self,
                                        QtWidgets.QSizePolicy.Expanding,
                                        QtWidgets.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)

        self.figure.canvas.mpl_connect('motion_notify_event', self.move)
        self.cid = self.figure.canvas.mpl_connect('resize_event',
                                                  self.revent)
        # self.zid = self.figure.canvas.mpl_connect('button_release_event',
        #                                           self.zevent)

    # sun shading stuff
        self.pinit = None
        self.qinit = None
        self.phi = -np.pi/4.
        self.theta = np.pi/4.
        self.cell = 100.
        self.alpha = .0

    # cmyk stuff
        self.kval = 0.01

    # def zevent(self, event):
    #     """
    #     Event check for zooming.

    #     Parameters
    #     ----------
    #     event : TYPE
    #         DESCRIPTION.

    #     Returns
    #     -------
    #     None.

    #     """
    #     nmode = event.inaxes.get_navigate_mode()

        # if nmode == 'ZOOM' and self.gmode == 'Contour':
        #     self.init_graph()
        #     print(111)

    def revent(self, event):
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
        self.flagresize = True

    def init_graph(self, event=None):
        """
        Initialize the graph.

        Parameters
        ----------
        event : TYPE, optional
            Unused. The default is None.

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

        self.bgrgb[0] = self.figure.canvas.copy_from_bbox(self.argb[0].bbox)
        self.bgrgb[1] = self.figure.canvas.copy_from_bbox(self.argb[1].bbox)
        self.bgrgb[2] = self.figure.canvas.copy_from_bbox(self.argb[2].bbox)

        self.background = self.figure.canvas.copy_from_bbox(self.axes.bbox)

        # self.image = self.axes.imshow(self.data[0].data, origin='upper',
        #                               extent=(x_1, x_2, y_1, y_2))

        tmp = np.ma.array([[np.nan]])
        self.image = imshow(self.axes, tmp, origin='upper',
                            extent=(x_1, x_2, y_1, y_2))

        # This line prevents imshow from generating color values on the
        # toolbar
        self.image.format_cursor_data = lambda x: ""
        self.update_graph()
        # self.figure.canvas.draw()

        self.cid = self.figure.canvas.mpl_connect('resize_event',
                                                  self.revent)

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
            if self.flagresize is True:
                self.flagresize = False
                # self.init_graph()

                self.update_graph()

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
                self.argb[0].draw_artist(self.clipvalu[0])
                self.argb[0].draw_artist(self.clipvall[0])
                self.figure.canvas.update()

            if 'Ternary' in self.gmode:
                bnum = self.update_hist_rgb(zval)
                self.figure.canvas.restore_region(self.bbox_hist_red)
                self.figure.canvas.restore_region(self.bbox_hist_green)
                self.figure.canvas.restore_region(self.bbox_hist_blue)

                for j in range(3):
                    self.argb[j].draw_artist(self.htxt[j])
                    self.argb[j].draw_artist(self.hhist[j][2][bnum[j]])
                    if self.clipvalu[j] is not None:
                        self.argb[j].draw_artist(self.clipvalu[j])
                    if self.clipvall[j] is not None:
                        self.argb[j].draw_artist(self.clipvall[j])

                self.figure.canvas.update()

    def update_contour(self):
        """
        Update contours.

        Returns
        -------
        None.

        """
        x1, x2, y1, y2 = self.data[0].extent
        self.image.set_visible(False)

        for i in self.data:
            if i.dataid == self.hband[0]:
                dat = i.data.copy()

        # self.image.set_data(dat)
        # self.image._scale_to_res()
        # dat = self.image._A

        # x1, x2, y1, y2 = self.image.get_extent()

        if self.htype == 'Histogram Equalization':
            dat = histeq(dat)
        elif self.clippercl > 0. or self.clippercu > 0.:
            dat, _, _ = histcomp(dat, perc=self.clippercl,
                                 uperc=self.clippercu)

        # self.image.set_data(dat)

        xdim = (x2-x1)/dat.data.shape[1]/2
        ydim = (y2-y1)/dat.data.shape[0]/2
        xi = np.linspace(x1+xdim, x2-xdim, dat.data.shape[1])
        yi = np.linspace(y2-ydim, y1+ydim, dat.data.shape[0])

        self.cnt = self.axes.contour(xi, yi, dat, extent=(x1, x2, y1, y2),
                                     linewidths=1, colors='k',
                                     linestyles='solid')
        self.cntf = self.axes.contourf(xi, yi, dat, extent=(x1, x2, y1, y2),
                                       cmap=self.cbar)

        self.ccbar = self.figure.colorbar(self.cntf, ax=self.axes)
        self.figure.canvas.draw()

    def update_graph(self):
        """
        Update plot.

        Returns
        -------
        None.

        """
        if self.ccbar is not None:
            self.ccbar.remove()
            self.ccbar = None

        if not self.data or self.gmode is None:
            return

        for i in range(3):
            self.argb[i].clear()

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

        self.bgrgb[0] = self.figure.canvas.copy_from_bbox(self.argb[0].bbox)
        self.bgrgb[1] = self.figure.canvas.copy_from_bbox(self.argb[1].bbox)
        self.bgrgb[2] = self.figure.canvas.copy_from_bbox(self.argb[2].bbox)

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

            if np.ma.is_masked(zval[i]) is True or zval[i] is None:
                bnum.append(0)
                self.update_hist_text(self.htxt[i], None)
                continue

            binnum = (bins < zval[i]).sum()-1

            if (-1 < binnum < len(patches) and
                    self.htype != 'Histogram Equalization'):
                patches[binnum].set_color('k')
                bnum.append(binnum)
            else:
                bnum.append(0)
            self.update_hist_text(self.htxt[i], zval[i])
        return bnum

    def update_hist_single(self, zval=None, hno=0):
        """
        Update the color on a single histogram.

        Parameters
        ----------
        zval : float
            Data value.
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
            # bincol = self.cbar(binave)
            bincol = self.newcmp(binave)
        else:
            bincol = cm.get_cmap('gray')(binave)

        for j, _ in enumerate(patches):
            patches[j].set_color(bincol[j])

        # This section draws the black line.
        if zval is None or np.ma.is_masked(zval) is True:
            self.update_hist_text(self.htxt[hno], None)
            return 0

        binnum = (bins < zval).sum()-1
        if binnum < 0 or binnum >= len(patches):
            self.update_hist_text(self.htxt[hno], zval)
            return 0

        self.update_hist_text(self.htxt[hno], zval)
        if self.htype == 'Histogram Equalization':
            return 0
        patches[binnum].set_color('k')

        return binnum

    def update_hist_text(self, hst, zval):
        """
        Update the value on the histogram.

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

        if zval is None:
            hst.set_text('')
        else:
            hst.set_text(f'{zval:.4f}')

    def update_rgb(self):
        """
        Update the RGB Ternary Map.

        Returns
        -------
        None.

        """
        self.clipvalu = [None, None, None]
        self.clipvall = [None, None, None]

        self.image.rgbmode = self.gmode
        self.image.kval = self.kval

        dat = [None, None, None]
        for i in self.data:
            if i.dataid == self.hband[3]:
                sun = i.data
            for j in range(3):
                if i.dataid == self.hband[j]:
                    dat[j] = i.data

        if self.shade is True:
            self.image.shade = [self.cell, self.theta, self.phi, self.alpha]
            dat.append(sun)
        else:
            self.image.shade = None

        dat = np.ma.array(dat)

        dat = np.moveaxis(dat, 0, -1)

        self.image.set_data(dat)
        self.image._scale_to_res()

        if self.image._A.ndim == 3:
            dat = self.image._A
        else:
            dat = self.image._A[:, :, :3]

        lclip = [0, 0, 0]
        uclip = [0, 0, 0]

        if self.htype == 'Histogram Equalization':
            self.image.dohisteq = True
        else:
            self.image.dohisteq = False
            lclip[0], uclip[0] = np.percentile(dat[:, :, 0].compressed(),
                                               [self.clippercl,
                                                100-self.clippercu])
            lclip[1], uclip[1] = np.percentile(dat[:, :, 1].compressed(),
                                               [self.clippercl,
                                                100-self.clippercu])
            lclip[2], uclip[2] = np.percentile(dat[:, :, 2].compressed(),
                                               [self.clippercl,
                                                100-self.clippercu])

            self.image.rgbclip = [[lclip[0], uclip[0]],
                                  [lclip[1], uclip[1]],
                                  [lclip[2], uclip[2]]]

        for i in range(3):
            hdata = dat[:, :, i]
            if ((self.clippercu > 0. or self.clippercl > 0.) and
                    self.fullhist is True and
                    self.htype != 'Histogram Equalization'):
                self.hhist[i] = self.argb[i].hist(hdata.compressed(), 50,
                                                  ec='none',
                                                  range=(lclip[i], uclip[i]))
                self.clipvall[i] = self.argb[i].axvline(lclip[i], ls='--')
                self.clipvalu[i] = self.argb[i].axvline(uclip[i], ls='--')

            elif self.htype == 'Histogram Equalization':
                hdata = histeq(hdata)
                hdata = hdata.compressed()
                self.hhist[i] = self.argb[i].hist(hdata, 50, ec='none')
            else:
                self.hhist[i] = self.argb[i].hist(hdata.compressed(), 50,
                                                  ec='none',
                                                  range=(lclip[i], uclip[i]))
            self.htxt[i] = self.argb[i].text(0., 0., '', ha='right', va='top')

            self.argb[i].set_xlim(self.hhist[i][1].min(),
                                  self.hhist[i][1].max())
            self.argb[i].set_ylim(0, self.hhist[i][0].max()*1.2)

        self.figure.canvas.restore_region(self.bgrgb[0])
        self.figure.canvas.restore_region(self.bgrgb[1])
        self.figure.canvas.restore_region(self.bgrgb[2])

        self.update_hist_rgb([None, None, None])

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
            if self.clipvalu[j] is not None:
                self.argb[j].draw_artist(self.clipvalu[j])
            if self.clipvall[j] is not None:
                self.argb[j].draw_artist(self.clipvall[j])

        self.figure.canvas.update()
        self.figure.canvas.flush_events()

    def update_single_color_map(self):
        """
        Update the single color map.

        Returns
        -------
        None.

        """
        self.clipvalu = [None, None, None]
        self.clipvall = [None, None, None]
        self.image.rgbmode = self.gmode

        for i in self.data:
            if i.dataid == self.hband[0]:
                pseudo = i.data
            if i.dataid == self.hband[3]:
                sun = i.data

        if self.shade is True:
            self.image.shade = [self.cell, self.theta, self.phi, self.alpha]
            pseudo = np.ma.stack([pseudo, sun])
            pseudo = np.moveaxis(pseudo, 0, -1)
        else:
            self.image.shade = None

        self.image.set_data(pseudo)
        self.image._scale_to_res()

        if self.image._A.ndim == 2:
            pseudo = self.image._A
        else:
            pseudo = self.image._A[:, :, 0]

        lclip = None
        uclip = None
        if self.htype == 'Histogram Equalization':
            self.image.dohisteq = True
            pseudo = histeq(pseudo)
            pseudoc = pseudo.compressed()
            lclip = pseudoc.min()
            uclip = pseudoc.max()
        else:
            self.image.dohisteq = False
            pseudoc = pseudo.compressed()
            lclip, uclip = np.percentile(pseudoc, [self.clippercl,
                                                   100-self.clippercu])

        self.image.cmap = self.cbar
        self.image.set_clim(lclip, uclip)

        self.newcmp = self.cbar
        if ((self.clippercu > 0. or self.clippercl > 0.) and
                self.fullhist is True and
                self.htype != 'Histogram Equalization'):
            self.hhist[0] = self.argb[0].hist(pseudoc, 50, ec='none')
            tmp = self.hhist[0][1]
            filt = (tmp > lclip) & (tmp < uclip)
            bcnt = np.sum(filt)

            cols = self.cbar(np.linspace(0, 1, bcnt))
            tmp = np.nonzero(filt)

            tmp1 = cols.copy()
            if tmp[0][0] > 0:
                tmp1 = np.vstack(([cols[0]]*tmp[0][0], tmp1))
            if tmp[0][-1] < 49:
                tmp1 = np.vstack((tmp1, [cols[-1]]*(49-tmp[0][-1])))
            self.newcmp = ListedColormap(tmp1)
        else:
            self.hhist[0] = self.argb[0].hist(pseudoc, 50, ec='none',
                                              range=(lclip, uclip))

        self.htxt[0] = self.argb[0].text(0.0, 0.0, '', ha='right', va='top')
        self.argb[0].set_xlim(self.hhist[0][1].min(), self.hhist[0][1].max())
        self.argb[0].set_ylim(0, self.hhist[0][0].max()*1.2)

        self.clipvall[0] = self.argb[0].axvline(lclip, ls='--')
        self.clipvalu[0] = self.argb[0].axvline(uclip, ls='--')

        self.figure.canvas.restore_region(self.bgrgb[0])
        self.update_hist_single()
        self.axes.draw_artist(self.image)

        for i in self.hhist[0][2]:
            self.argb[0].draw_artist(i)

        self.figure.canvas.update()

        self.bbox_hist_red = self.figure.canvas.copy_from_bbox(
            self.argb[0].bbox)

        self.argb[0].draw_artist(self.htxt[0])
        self.argb[0].draw_artist(self.clipvalu[0])
        self.argb[0].draw_artist(self.clipvall[0])
        self.figure.canvas.update()

    def update_shade(self):
        """
        Update sun shade plot.

        Returns
        -------
        None.

        """
        pseudo = self.image._full_res
        for i in self.data:
            if i.dataid == self.hband[3]:
                sun = i.data

        if pseudo.ndim == 2:
            tmp = np.ma.stack([pseudo, sun])
            tmp = np.moveaxis(tmp, 0, -1)
            self.image.set_data(tmp)
            self.image.set_data(tmp)
        elif pseudo.ndim == 2 and pseudo.shape[-1] == 3:
            tmp = np.ma.concatenate((pseudo, sun), axis=-1)
            self.image.set_data(tmp)
        else:
            pseudo[:, :, -1] = sun
            self.image.set_data(pseudo)

        self.image.shade = [self.cell, self.theta, self.phi, self.alpha]
        self.axes.draw_artist(self.image)
        self.figure.canvas.update()

    def update_shade_plot(self):
        """
        Update shade plot for export.

        Returns
        -------
        numpy array
            Sunshader data.

        """
        if self.shade is not True:
            return 1

        for i in self.sdata:
            if i.dataid == self.hband[3]:
                sun = i.data

        sunshader = currentshader(sun.data, self.cell, self.theta,
                                  self.phi, self.alpha)

        snorm = norm2(sunshader)

        return snorm


class MySunCanvas(FigureCanvasQTAgg):
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
        self.setMaximumSize(200, 200)
        self.setMinimumSize(120, 120)

    def init_graph(self):
        """
        Initialise graph.

        Returns
        -------
        None.

        """
        self.axes.clear()
        # self.axes.xaxis.set_tick_params(labelsize=8)
        self.axes.tick_params(labelleft=False, labelright=False)
        self.axes.set_autoscaley_on(False)
        self.axes.set_rmax(1.0)
        self.axes.set_rmin(0.0)
        self.axes.set_xticklabels([])

        self.sun, = self.axes.plot(np.pi/4., cos(np.pi/4.), 'o')
        self.figure.tight_layout()
        self.figure.canvas.draw()


class PlotInterp(QtWidgets.QDialog):
    """
    The primary class for the raster data interpretation module.

    The main interface is set up from here, as well as monitoring of the mouse
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
        if parent is None:
            self.showprocesslog = print
        else:
            self.showprocesslog = parent.showprocesslog

        self.indata = {}
        self.outdata = {}
        self.parent = parent
        self.units = {}

        self.mmc = MyMplCanvas(self)
        self.msc = MySunCanvas(self)
        self.btn_saveimg = QtWidgets.QPushButton('Save GeoTiff')
        self.chk_histtype = QtWidgets.QCheckBox('Full histogram with clip '
                                                'lines')
        self.cbox_dtype = QtWidgets.QComboBox()
        self.cbox_band1 = QtWidgets.QComboBox()
        self.cbox_band2 = QtWidgets.QComboBox()
        self.cbox_band3 = QtWidgets.QComboBox()
        self.cbox_bands = QtWidgets.QComboBox()
        self.cbox_htype = QtWidgets.QComboBox()
        self.lineclipu = QtWidgets.QLineEdit()
        self.lineclipl = QtWidgets.QLineEdit()
        self.cbox_cbar = QtWidgets.QComboBox(self)
        self.kslider = QtWidgets.QSlider(QtCore.Qt.Horizontal)  # cmyK
        self.sslider = QtWidgets.QSlider(QtCore.Qt.Horizontal)  # sunshade
        self.aslider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.label4 = QtWidgets.QLabel('Sunshade Data:')
        self.labels = QtWidgets.QLabel('Sunshade Detail')
        self.labela = QtWidgets.QLabel('Light Reflectance')
        self.labelc = QtWidgets.QLabel('Color Bar:')
        self.labelk = QtWidgets.QLabel('K value:')
        self.gbox_sun = QtWidgets.QGroupBox('Sunshading')

        self.setupui()

        txt = str(self.cbox_cbar.currentText())
        self.mmc.cbar = cm.get_cmap(txt)

        # self.change_cbar()
        self.setFocus()

        self.mmc.gmode = 'Single Color Map'
        self.mmc.argb[0].set_visible(True)
        self.mmc.argb[1].set_visible(False)
        self.mmc.argb[2].set_visible(False)

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
        self.label4.hide()
        self.cbox_bands.hide()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        helpdocs = menu_default.HelpButton('pygmi.raster.ginterp')
        btn_apply = QtWidgets.QPushButton('Apply Histogram')

        gbox1 = QtWidgets.QGroupBox('Display Type')
        v1 = QtWidgets.QVBoxLayout()
        gbox1.setLayout(v1)

        gbox2 = QtWidgets.QGroupBox('Data Bands')
        v2 = QtWidgets.QVBoxLayout()
        gbox2.setLayout(v2)

        gbox3 = QtWidgets.QGroupBox('Histogram Stretch')
        v3 = QtWidgets.QVBoxLayout()
        gbox3.setLayout(v3)

        gbox1.setSizePolicy(QtWidgets.QSizePolicy.Fixed,
                            QtWidgets.QSizePolicy.Preferred)
        gbox2.setSizePolicy(QtWidgets.QSizePolicy.Fixed,
                            QtWidgets.QSizePolicy.Preferred)
        gbox3.setSizePolicy(QtWidgets.QSizePolicy.Fixed,
                            QtWidgets.QSizePolicy.Preferred)
        self.gbox_sun.setSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                    QtWidgets.QSizePolicy.Preferred)

        v4 = QtWidgets.QVBoxLayout()
        self.gbox_sun.setLayout(v4)
        self.gbox_sun.setCheckable(True)
        self.gbox_sun.setChecked(False)

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

        # self.lineclip.setInputMask('00.0')
        self.lineclipu.setPlaceholderText('% of high values to exclude')
        self.lineclipl.setPlaceholderText('% of low values to exclude')
        self.btn_saveimg.setAutoDefault(False)
        helpdocs.setAutoDefault(False)
        btn_apply.setAutoDefault(False)

        tmp = sorted(m for m in colormaps())

        self.cbox_cbar.addItem('jet')
        self.cbox_cbar.addItem('viridis')
        self.cbox_cbar.addItem('terrain')
        self.cbox_cbar.addItems(tmp)
        self.cbox_dtype.addItems(['Single Color Map', 'Contour', 'RGB Ternary',
                                  'CMY Ternary'])
        self.cbox_htype.addItems(['Linear with Percent Clip',
                                  'Histogram Equalization'])

        self.setWindowTitle('Raster Data Display')

        v1.addWidget(self.cbox_dtype)
        v1.addWidget(self.labelk)
        v1.addWidget(self.kslider)
        vbl_raster.addWidget(gbox1)

        v2.addWidget(self.cbox_band1)
        v2.addWidget(self.cbox_band2)
        v2.addWidget(self.cbox_band3)
        vbl_raster.addWidget(gbox2)

        v3.addWidget(self.cbox_htype)
        v3.addWidget(self.lineclipl)
        v3.addWidget(self.lineclipu)
        v3.addWidget(self.chk_histtype)
        v3.addWidget(btn_apply)
        v3.addWidget(self.labelc)
        v3.addWidget(self.cbox_cbar)
        vbl_raster.addWidget(gbox3)

        vbl_raster.addWidget(self.gbox_sun)
        v4.addWidget(self.label4)
        v4.addWidget(self.cbox_bands)
        v4.addWidget(self.msc)
        v4.addWidget(self.labels)
        v4.addWidget(self.sslider)
        v4.addWidget(self.labela)
        v4.addWidget(self.aslider)
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

        self.sslider.sliderReleased.connect(self.change_sunsliders)
        self.aslider.sliderReleased.connect(self.change_sunsliders)
        self.kslider.sliderReleased.connect(self.change_kval)
        self.msc.figure.canvas.mpl_connect('button_press_event', self.move)
        self.btn_saveimg.clicked.connect(self.save_img)
        # self.gbox_sun.clicked.connect(self.change_dtype)
        self.gbox_sun.clicked.connect(self.change_sun_checkbox)
        btn_apply.clicked.connect(self.change_lclip)
        # self.lineclipu.returnPressed.connect(self.change_lclip_upper)
        # self.lineclipl.returnPressed.connect(self.change_lclip_lower)
        self.chk_histtype.clicked.connect(self.change_dtype)

        if self.parent is not None:
            self.resize(self.parent.width(), self.parent.height())

    def change_lclip(self):
        """
        Change the linear clip percentage.

        Returns
        -------
        None.

        """
        txt = self.lineclipu.text()
        try:
            uclip = float(txt)
        except ValueError:
            if txt == '':
                uclip = 0.0
            else:
                uclip = self.mmc.clippercu
            self.lineclipu.setText(str(uclip))

        if uclip < 0.0 or uclip >= 100.0:
            uclip = self.mmc.clippercu
            self.lineclipu.setText(str(uclip))

        txt = self.lineclipl.text()
        try:
            lclip = float(txt)
        except ValueError:
            if txt == '':
                lclip = 0.0
            else:
                lclip = self.mmc.clippercl
            self.lineclipl.setText(str(lclip))

        if lclip < 0.0 or lclip >= 100.0:
            lclip = self.mmc.clippercl
            self.lineclipl.setText(str(lclip))

        if (lclip+uclip) >= 100.:
            clip = self.mmc.clippercu
            self.lineclipu.setText(str(clip))
            clip = self.mmc.clippercl
            self.lineclipl.setText(str(clip))
            return

        self.mmc.clippercu = uclip
        self.mmc.clippercl = lclip

        # self.change_lclip_lower()
        # self.change_lclip_upper()
        self.change_dtype()

    # def change_lclip_upper(self):
    #     """
    #     Change the linear clip percentage.

    #     Returns
    #     -------
    #     None.

    #     """
    #     txt = self.lineclipu.text()

    #     try:
    #         clip = float(txt)
    #     except ValueError:
    #         if txt == '':
    #             clip = 0.0
    #         else:
    #             clip = self.mmc.clippercu
    #         self.lineclipu.setText(str(clip))

    #     if clip < 0.0 or clip >= 100.0:
    #         clip = self.mmc.clippercu
    #         self.lineclipu.setText(str(clip))
    #     self.mmc.clippercu = clip

    #     # self.change_dtype()

    # def change_lclip_lower(self):
    #     """
    #     Change the linear clip percentage.

    #     Returns
    #     -------
    #     None.

    #     """
    #     txt = self.lineclipl.text()

    #     try:
    #         clip = float(txt)
    #     except ValueError:
    #         if txt == '':
    #             clip = 0.0
    #         else:
    #             clip = self.mmc.clippercl
    #         self.lineclipl.setText(str(clip))

    #     if clip < 0.0 or clip >= 100.0:
    #         clip = self.mmc.clippercl
    #         self.lineclipl.setText(str(clip))
    #     self.mmc.clippercl = clip

        # self.change_dtype()

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
        self.mmc.figure.canvas.mpl_disconnect(self.mmc.cid)

        txt = str(self.cbox_dtype.currentText())
        self.mmc.gmode = txt
        self.cbox_band1.show()
        self.mmc.fullhist = self.chk_histtype.isChecked()

        if txt == 'Single Color Map':
            # self.slabel.hide()
            self.labelc.show()
            self.labelk.hide()
            self.cbox_band2.hide()
            self.cbox_band3.hide()
            self.cbox_cbar.show()
            self.mmc.argb[0].set_visible(True)
            self.mmc.argb[1].set_visible(False)
            self.mmc.argb[2].set_visible(False)
            self.sslider.hide()
            self.aslider.hide()
            self.kslider.hide()

        if txt == 'Contour':
            self.labelk.hide()
            self.labelc.show()
            self.cbox_band2.hide()
            self.cbox_band3.hide()
            self.cbox_cbar.show()
            self.mmc.argb[0].set_visible(False)
            self.mmc.argb[1].set_visible(False)
            self.mmc.argb[2].set_visible(False)
            self.sslider.hide()
            self.aslider.hide()
            self.kslider.hide()
            self.gbox_sun.setChecked(False)

        if 'Ternary' in txt:
            self.labelk.hide()
            self.labelc.hide()
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

        if self.gbox_sun.isChecked():
            self.msc.show()
            self.label4.show()
            self.cbox_bands.show()
            self.sslider.show()
            self.aslider.show()
            self.labela.show()
            self.labels.show()
            self.mmc.cell = self.sslider.value()
            self.mmc.alpha = float(self.aslider.value())/100.
            self.mmc.shade = True
            # self.cbox_bands.setCurrentText(self.cbox_band1.currentText())
            self.msc.init_graph()
        else:
            self.msc.hide()
            self.labela.hide()
            self.labels.hide()
            self.label4.hide()
            self.cbox_bands.hide()
            self.mmc.shade = False

        self.mmc.cid = self.mmc.figure.canvas.mpl_connect('resize_event',
                                                          self.mmc.revent)
        self.mmc.init_graph()

    def change_kval(self):
        """
        Change the CMYK K value.

        Returns
        -------
        None.

        """
        self.mmc.kval = float(self.kslider.value())/100.
        self.mmc.update_graph()

    def change_sun_checkbox(self):
        """
        Use when sunshading checkbox is clicked.

        Returns
        -------
        None.

        """
        self.mmc.figure.canvas.mpl_disconnect(self.mmc.cid)

        if self.gbox_sun.isChecked():
            self.msc.show()
            self.label4.show()
            self.cbox_bands.show()
            self.sslider.show()
            self.aslider.show()
            self.labela.show()
            self.labels.show()
            self.mmc.cell = self.sslider.value()
            self.mmc.alpha = float(self.aslider.value())/100.
            self.mmc.shade = True
            self.msc.init_graph()
            QtWidgets.QApplication.processEvents()
        else:
            self.msc.hide()
            self.labela.hide()
            self.labels.hide()
            self.label4.hide()
            self.cbox_bands.hide()
            self.sslider.hide()
            self.aslider.hide()
            self.mmc.shade = False
            QtWidgets.QApplication.processEvents()
        self.mmc.update_graph()

        self.mmc.cid = self.mmc.figure.canvas.mpl_connect('resize_event',
                                                          self.mmc.revent)

    def change_sunsliders(self):
        """
        Change the sun shading sliders.

        Returns
        -------
        None.

        """
        self.mmc.cell = self.sslider.value()
        self.mmc.alpha = float(self.aslider.value())/100.
        self.mmc.update_shade()

    def change_green(self):
        """
        Change the green or second band.

        Returns
        -------
        None.

        """
        txt = str(self.cbox_band2.currentText())
        self.mmc.hband[1] = txt
        self.mmc.init_graph()

    def change_htype(self):
        """
        Change the histogram stretch to apply to the normal data.

        Returns
        -------
        None.

        """
        txt = str(self.cbox_htype.currentText())

        if txt == 'Histogram Equalization':
            self.lineclipl.hide()
            self.lineclipu.hide()
        else:
            self.lineclipl.show()
            self.lineclipu.show()

        self.mmc.htype = txt
        self.mmc.update_graph()

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

    def change_sun(self):
        """
        Change the sunshade band.

        Returns
        -------
        None.

        """
        txt = str(self.cbox_bands.currentText())
        self.mmc.hband[3] = txt
        self.mmc.update_graph()

    def data_init(self):
        """
        Initialise Data.

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

        self.indata['Raster'] = dataprep.lstack(self.indata['Raster'])

        if 'Cluster' in self.indata:
            data = self.indata['Cluster']
            newdat = copy.copy(self.indata['Raster'])
            for i in data:
                if 'memdat' not in i.metadata['Cluster']:
                    continue
                for j, val in enumerate(i.metadata['Cluster']['memdat']):
                    tmp = copy.deepcopy(i)
                    tmp.memdat = None
                    tmp.data = val
                    tmp.dataid = ('Membership of class ' + str(j+1)
                                  + ': '+tmp.dataid)
                    newdat.append(tmp)
            data = newdat
            sdata = newdat
        else:
            data = self.indata['Raster']
            sdata = self.indata['Raster']

        for i in data:
            self.units[i.dataid] = i.units

        self.mmc.data = data
        self.mmc.sdata = sdata
        self.mmc.hband[0] = data[0].dataid
        self.mmc.hband[1] = data[0].dataid
        self.mmc.hband[2] = data[0].dataid
        self.mmc.hband[3] = data[0].dataid

        blist = []
        for i in data:
            blist.append(i.dataid)

        try:
            self.cbox_band1.currentIndexChanged.disconnect()
            self.cbox_band2.currentIndexChanged.disconnect()
            self.cbox_band3.currentIndexChanged.disconnect()
            self.cbox_bands.currentIndexChanged.disconnect()
        except TypeError:
            pass

        self.cbox_band1.clear()
        self.cbox_band2.clear()
        self.cbox_band3.clear()
        self.cbox_bands.clear()
        self.cbox_band1.addItems(blist)
        self.cbox_band2.addItems(blist)
        self.cbox_band3.addItems(blist)
        self.cbox_bands.addItems(blist)

        self.cbox_band1.currentIndexChanged.connect(self.change_red)
        self.cbox_band2.currentIndexChanged.connect(self.change_green)
        self.cbox_band3.currentIndexChanged.connect(self.change_blue)
        self.cbox_bands.currentIndexChanged.connect(self.change_sun)

    def move(self, event):
        """
        Move event is used to track changes to the sunshading.

        Parameters
        ----------
        event : TYPE
            Unused.

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
            self.mmc.update_shade()

    def save_img(self):
        """
        Save image as a GeoTiff.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        snorm = self.mmc.update_shade_plot()

        ext = 'GeoTiff (*.tif)'
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.parent, 'Save File', '.', ext)
        if filename == '':
            return False

        text, okay = QtWidgets.QInputDialog.getText(
            self, 'Colorbar', 'Enter length in inches:',
            QtWidgets.QLineEdit.Normal, '4')

        if not okay:
            return False

        try:
            blen = float(text)
        except ValueError:
            QtWidgets.QMessageBox.warning(self.parent, 'Error',
                                          'Invalid value.',
                                          QtWidgets.QMessageBox.Ok)
            return False

        bwid = blen/16.

        dtype = str(self.cbox_dtype.currentText())

        rtext = 'Red'
        gtext = 'Green'
        btext = 'Blue'

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

        htype = str(self.cbox_htype.currentText())
        clippercl = self.mmc.clippercl
        clippercu = self.mmc.clippercu

        if dtype == 'Single Color Map':

            for i in self.mmc.data:
                if i.dataid == self.mmc.hband[0]:
                    pseudo = i.data

            if htype == 'Histogram Equalization':
                pseudo = histeq(pseudo)
            elif clippercl > 0. or clippercu > 0.:
                pseudo, _, _ = histcomp(pseudo, perc=clippercl,
                                        uperc=clippercu)

            cmin = pseudo.min()
            cmax = pseudo.max()

            # The function below normalizes as well.
            img = img2rgb(pseudo, self.mmc.cbar)

            pseudo = None

            img[:, :, 0] = img[:, :, 0]*snorm  # red
            img[:, :, 1] = img[:, :, 1]*snorm  # green
            img[:, :, 2] = img[:, :, 2]*snorm  # blue
            img = img.astype(np.uint8)

        elif 'Ternary' in dtype:

            dat = [None, None, None]
            for i in self.mmc.data:
                for j in range(3):
                    if i.dataid == self.mmc.hband[j]:
                        dat[j] = i.data

            red = dat[0]
            green = dat[1]
            blue = dat[2]

            mask = np.logical_or(red.mask, green.mask)
            mask = np.logical_or(mask, blue.mask)
            mask = np.logical_not(mask)

            if htype == 'Histogram Equalization':
                red = histeq(red)
                green = histeq(green)
                blue = histeq(blue)
            elif clippercl > 0. or clippercu > 0.:
                red, _, _ = histcomp(red, perc=clippercl, uperc=clippercu)
                green, _, _ = histcomp(green, perc=clippercl, uperc=clippercu)
                blue, _, _ = histcomp(blue, perc=clippercl, uperc=clippercu)

            cmin = red.min()
            cmax = red.max()

            img = np.ones((red.shape[0], red.shape[1], 4), dtype=np.uint8)
            img[:, :, 3] = mask*254+1

            if 'CMY' in dtype:
                img[:, :, 0] = (1-norm2(red))*254+1
                img[:, :, 1] = (1-norm2(green))*254+1
                img[:, :, 2] = (1-norm2(blue))*254+1
            else:
                img[:, :, 0] = norm255(red)
                img[:, :, 1] = norm255(green)
                img[:, :, 2] = norm255(blue)

            img[:, :, 0] = img[:, :, 0]*snorm  # red
            img[:, :, 1] = img[:, :, 1]*snorm  # green
            img[:, :, 2] = img[:, :, 2]*snorm  # blue
            img = img.astype(np.uint8)

        elif dtype == 'Contour':
            pseudo = self.mmc.image._full_res.copy()
            if htype == 'Histogram Equalization':
                pseudo = histeq(pseudo)
            elif clippercl > 0. or clippercu > 0.:
                pseudo, _, _ = histcomp(pseudo, perc=clippercl,
                                        uperc=clippercu)

            cmin = pseudo.min()
            cmax = pseudo.max()

            if self.mmc.ccbar is not None:
                self.mmc.ccbar.remove()
                self.mmc.ccbar = None

            self.mmc.figure.set_frameon(False)
            self.mmc.axes.set_axis_off()
            tmpsize = self.mmc.figure.get_size_inches()
            self.mmc.figure.set_size_inches(tmpsize*3)
            self.mmc.figure.canvas.draw()
            img = np.frombuffer(self.mmc.figure.canvas.tostring_argb(),
                                dtype=np.uint8)
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

        newimg[0].nodata = 0
        newimg[1].nodata = 0
        newimg[2].nodata = 0
        newimg[3].nodata = 0

        newimg[0].dataid = rtext
        newimg[1].dataid = gtext
        newimg[2].dataid = btext
        newimg[3].dataid = 'Alpha'

        iodefs.export_raster(str(filename), newimg, 'GTiff')

# Section for colorbars
        if 'Ternary' not in dtype:
            txt = str(self.cbox_cbar.currentText())
            cmap = cm.get_cmap(txt)
            norm = mcolors.Normalize(vmin=cmin, vmax=cmax)

# Horizontal Bar
            fig = Figure()
            canvas = FigureCanvasQTAgg(fig)
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
            canvas = FigureCanvasQTAgg(fig)
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
            fig = Figure(figsize=[blen, blen])
            canvas = FigureCanvasQTAgg(fig)
            fig.set_tight_layout(True)

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

            data = np.transpose([red.flatten(),
                                 green.flatten(),
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

            ax.text(0, -5, gtext, horizontalalignment='center',
                    verticalalignment='top', size=20)
            ax.text(254, -5, btext, horizontalalignment='center',
                    verticalalignment='top', size=20)
            ax.text(127.5, 225, rtext, horizontalalignment='center',
                    size=20)
            ax.tick_params(top='off', right='off', bottom='off', left='off',
                           labelbottom='off', labelleft='off')

            ax.axis('off')
            fname = filename[:-4]+'_tern.png'
            canvas.print_figure(fname, dpi=300)

        QtWidgets.QMessageBox.information(self, 'Information',
                                          'Save to GeoTiff is complete!',
                                          QtWidgets.QMessageBox.Ok)

        return True

    def settings(self, nodialog=False):
        """
        Entry point into item.

        Parameters
        ----------
        nodialog : bool, optional
            Run settings without a dialog. The default is False.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        if nodialog:
            return True

        if 'Raster' not in self.indata:
            self.showprocesslog('No Raster Data.')
            return False

        if self.indata['Raster'][0].isrgb:
            self.showprocesslog('RGB images cannot be used in this module.')
            return False

        self.show()
        self.mmc.init_graph()
        self.msc.init_graph()

        tmp = self.exec_()

        if tmp == 0:
            return False

        return True

    def loadproj(self, projdata):
        """
        Load project data into class.

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
        array containing the shaded results.

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


def histcomp(img, nbr_bins=None, perc=5., uperc=None):
    """
    Histogram Compaction.

    This compacts a % of the outliers in data, allowing for a cleaner, linear
    representation of the data.

    Parameters
    ----------
    img : numpy array
        data to compact
    nbr_bins : int
        number of bins to use in compaction
    perc : float
        percentage of histogram to clip. If uperc is not None, then this is
        the lower percentage
    uperc : float
        upper percentage to clip. If uperc is None, then it is set to the
        same value as perc

    Returns
    -------
    img2 : numpy array
        compacted array
    """
    if uperc is None:
        uperc = perc

    if nbr_bins is None:
        nbr_bins = max(img.shape)
        nbr_bins = max(nbr_bins, 256)

# get image histogram
    imask = np.ma.getmaskarray(img)
    # tmp = img.compressed()
    # imhist, bins = np.histogram(tmp, nbr_bins)

    # cdf = imhist.cumsum()  # cumulative distribution function
    # if cdf[-1] == 0:
    #     return img
    # cdf = cdf / float(cdf[-1])  # normalize

    # perc = perc/100.
    # uperc = uperc/100.

    # sindx = np.arange(nbr_bins)[cdf > perc][0]
    # if cdf[0] > (1-uperc):
    #     eindx = 1
    # else:
    #     eindx = np.arange(nbr_bins)[cdf < (1-uperc)][-1]+1
    # svalue = bins[sindx]
    # evalue = bins[eindx]

    svalue, evalue = np.percentile(img.compressed(), (perc, 100-uperc))

    img2 = np.empty_like(img, dtype=np.float32)
    np.copyto(img2, img)

    filt = np.ma.less(img2, svalue)
    img2[filt] = svalue

    filt = np.ma.greater(img2, evalue)
    img2[filt] = evalue

    img2 = np.ma.array(img2, mask=imask)

    return img2, svalue, evalue


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
    bins = (bins[1:]-bins[:-1])/2+bins[:-1]  # get bin center point

    cdf = imhist.cumsum()  # cumulative distribution function
    cdf = cdf - cdf[0]  # subtract min, which is first val in cdf
    cdf = cdf.astype(np.int64)
    cdf = nbr_bins * cdf / cdf[-1]  # norm to nbr_bins

    # use linear interpolation of cdf to find new pixel values
    im2 = np.interp(img, bins, cdf)
    im2 = np.ma.array(im2, mask=img.mask)

    return im2


def img2rgb(img, cbar=cm.get_cmap('jet')):
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


def _testfn():
    """Test routine."""
    import matplotlib
    matplotlib.interactive(False)

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 '..//..')))
    app = QtWidgets.QApplication(sys.argv)



    # data = iodefs.get_raster(r"C:\Workdata\MagMerge\NC_reg_highres_merge_wgs84dd.tif")
    # data = iodefs.get_raster(r'c:\WorkData\testdata.hdr')
    data = iodefs.get_raster(r"E:\Workdata\people\mikedentith\perth_surf_win.grd")

    tmp = PlotInterp()
    tmp.indata['Raster'] = data
    tmp.data_init()

    tmp.settings()


if __name__ == "__main__":
    _testfn()
