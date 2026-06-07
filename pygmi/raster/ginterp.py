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
Raster data display.

This is the raster data display module.  This module allows for the
display of raster data in a variety of modes, as well as the export of that
display to GeoTIFF format.

Currently the following is supported
 * Pseudo Colour - data mapped to a colour map
 * Contours with solid contours
 * RGB ternary images
 * CMYK ternary images
 * Sun shaded or hill shaded images

It can be very effectively used in conjunction with a GIS package which
supports GeoTIFF files.
"""

import copy
import io
import os
import sys
from math import cos

import matplotlib.colorbar as mcolorbar
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import numpy as np
from matplotlib import gridspec
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PySide6 import QtCore, QtGui, QtWidgets
from scipy import ndimage

from pygmi.maps import frm
from pygmi.misc import BasicModule
from pygmi.raster import dataprep, iodefs
from pygmi.raster.colormaps import colormaps
from pygmi.raster.misc import (
    currentshader,
    histcomp,
    histeq,
    img2rgb,
    lstack,
    norm2,
    norm255,
)
from pygmi.raster.modest_image import imshow


class MyMplCanvas(FigureCanvasQTAgg):
    """
    Matplotlib canvas widget for the actual plot.

    Attributes
    ----------
    htype : str
        string indicating the histogram stretch to apply to the data
    cbar : matplotlib colour map
        colour map to be used for pseudo colour bars
    data : list of pygmi.raster.datatypes.Data
        list of PyGMI raster data objects - used for colour images
    sdata : list of pygmi.raster.datatypes.Data
        list of PyGMI raster data objects - used for shaded images
    gmode : str
        string containing the graphics mode - Contour, Ternary, Sunshade,
        Single Colour Map.
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
        k value for CMYK mode
    """

    def __init__(self):
        fig = Figure(figsize=(12, 8))
        super().__init__(fig)

        # figure stuff
        self.htype = "Linear with Percent Clip"
        self.cbar = colormaps["jet"]
        self.newcmp = self.cbar
        self.fullhist = False
        self.data = []
        self.sdata = []
        self.gmode = None
        self.axes = None
        self.argb = [None, None, None]
        self.argbvis = [True, False, False]
        self.argbunit = ["", "", ""]
        self.bgrgb = [None, None, None]
        self.hhist = [[], [], []]
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
        self.scbar = None
        self.clippercu = {}
        self.clippercl = {}
        self.clipmin = {}
        self.clipmax = {}
        self.flagresize = False
        self.clipvalu = [None, None, None]
        self.clipvall = [None, None, None]
        self.levels = 10

        # gspc = gridspec.GridSpec(3, 4)
        # self.axes = fig.add_subplot(gspc[0:, 1:])
        # self.axes.tick_params(axis='x', rotation=90)
        # self.axes.tick_params(axis='y', rotation=0)
        # self.axes.ticklabel_format(style='plain', axis='both')
        # self.axes.xaxis.set_visible(False)
        # self.axes.yaxis.set_visible(False)

        # for i in range(3):
        #     self.argb[i] = fig.add_subplot(gspc[i, 0])
        #     self.argb[i].xaxis.set_visible(False)
        #     self.argb[i].yaxis.set_visible(False)
        #     self.argb[i].autoscale(False)

        # fig.subplots_adjust(bottom=0.05)
        # fig.subplots_adjust(top=.95)
        # fig.subplots_adjust(left=0.05)
        # fig.subplots_adjust(right=.95)
        # fig.subplots_adjust(wspace=0.05)
        # fig.subplots_adjust(hspace=0.05)

        FigureCanvasQTAgg.setSizePolicy(
            self,
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        FigureCanvasQTAgg.updateGeometry(self)

        self.figure.canvas.mpl_connect("motion_notify_event", self.move)
        self.cid = self.figure.canvas.mpl_connect("resize_event", self.revent)

        # sun shading stuff
        self.pinit = None
        self.qinit = None
        self.phi = -np.pi / 4.0
        self.theta = np.pi / 4.0
        self.cell = 100.0
        self.alpha = 0.0

        # cmyk stuff
        self.kval = 0.01

    def revent(self, event):
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
        self.flagresize = True

    def init_graph(self):
        """
        Initialize the graph.

        Returns
        -------
        None.

        """
        if self.ccbar is not None:
            self.ccbar.remove()
            self.ccbar = None

        self.figure.canvas.mpl_disconnect(self.cid)
        self.figure.clear()

        gspc = gridspec.GridSpec(3, 4)
        self.axes = self.figure.add_subplot(gspc[0:, 1:])
        self.axes.tick_params(axis="x", rotation=90)
        self.axes.tick_params(axis="y", rotation=0)
        self.axes.ticklabel_format(style="plain", axis="both")
        self.axes.xaxis.set_visible(False)
        self.axes.yaxis.set_visible(False)

        for i in range(3):
            self.argb[i] = self.figure.add_subplot(gspc[i, 0])
            self.argb[i].xaxis.set_visible(False)
            self.argb[i].yaxis.set_visible(False)
            self.argb[i].autoscale(False)
            self.argb[i].set_visible(self.argbvis[i])

        x_1, x_2, y_1, y_2 = self.data[0].extent

        self.axes.set_xlim(x_1, x_2)
        self.axes.set_ylim(y_1, y_2)
        self.axes.set_aspect("equal")

        self.figure.tight_layout()
        self.figure.canvas.draw()

        self.bgrgb[0] = self.figure.canvas.copy_from_bbox(self.argb[0].bbox)
        self.bgrgb[1] = self.figure.canvas.copy_from_bbox(self.argb[1].bbox)
        self.bgrgb[2] = self.figure.canvas.copy_from_bbox(self.argb[2].bbox)

        self.background = self.figure.canvas.copy_from_bbox(self.axes.bbox)

        tmp = np.ma.array([[np.nan]])
        self.image = imshow(self.axes, tmp, origin="upper", extent=(x_1, x_2, y_1, y_2))

        # This line prevents imshow from generating colour values on the
        # toolbar
        self.image.format_cursor_data = lambda x: ""
        self.update_graph()

        self.cid = self.figure.canvas.mpl_connect("resize_event", self.revent)

    def move(self, event):
        """
        Mouse is moving over canvas.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Mouse event.

        Returns
        -------
        None.

        """
        if not self.data or self.gmode == "Contour":
            return

        if event.inaxes == self.axes:
            if self.flagresize is True:
                self.flagresize = False

                self.update_graph()

            zval = [-999, -999, -999]
            for i in self.data:
                itlx = i.extent[0]
                itly = i.extent[-1]
                for j in range(3):
                    if i.dataid == self.hband[j]:
                        col = int((event.xdata - itlx) / i.xdim)
                        row = int((itly - event.ydata) / i.ydim)

                        if row == i.data.shape[0] or col == i.data.shape[1]:
                            return

                        zval[j] = i.data[row, col]

            if self.gmode == "Single Colour Map":
                bnum = self.update_hist_single(zval[0])
                self.figure.canvas.restore_region(self.bbox_hist_red)
                self.argb[0].draw_artist(self.htxt[0])
                self.argb[0].draw_artist(self.hhist[0][2][bnum])
                self.argb[0].draw_artist(self.clipvalu[0])
                self.argb[0].draw_artist(self.clipvall[0])
                self.figure.canvas.update()

            if "Ternary" in self.gmode:
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
        clippercu = self.clippercu[self.hband[0]]
        clippercl = self.clippercl[self.hband[0]]

        dat = self.data[0].data

        for i in self.data:
            if i.dataid == self.hband[0]:
                dat = i.data.copy()

        if self.htype == "Histogram Equalization":
            dat = histeq(dat)
        elif clippercl > 0.0 or clippercu > 0.0:
            dat, _, _ = histcomp(dat, perc=clippercl, uperc=clippercu)

        xdim = (x2 - x1) / dat.data.shape[1] / 2
        ydim = (y2 - y1) / dat.data.shape[0] / 2
        xi = np.linspace(x1 + xdim, x2 - xdim, dat.data.shape[1])
        yi = np.linspace(y2 - ydim, y1 + ydim, dat.data.shape[0])

        self.cnt = self.axes.contour(
            xi,
            yi,
            dat,
            extent=(x1, x2, y1, y2),
            linewidths=2,
            colors="k",
            levels=self.levels,
            linestyles="solid",
        )
        self.cntf = self.axes.contourf(
            xi, yi, dat, extent=(x1, x2, y1, y2), levels=self.levels, cmap=self.cbar
        )

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

        if self.gmode == "Single Colour Map":
            self.update_single_color_map()

        if self.gmode == "Contour":
            self.update_contour()

        if "Ternary" in self.gmode:
            self.update_rgb()

        if self.gmode == "Sunshade":
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
        hcol = ["r", "g", "b"]
        if "CMY" in self.gmode:
            hcol = ["c", "m", "y"]

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

            binnum = (bins < zval[i]).sum() - 1

            if -1 < binnum < len(patches) and self.htype != "Histogram Equalization":
                patches[binnum].set_color("k")
                bnum.append(binnum)
            else:
                bnum.append(0)
            self.update_hist_text(self.htxt[i], zval[i])
        return bnum

    def update_hist_single(self, zval=None, hno=0):
        """
        Update the colour on a single histogram.

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
        binave = np.arange(0, 1, 1 / (bins.size - 2))

        if hno == 0:
            bincol = self.newcmp(binave)
        else:
            bincol = colormaps["gray"](binave)

        for j, patchesj in enumerate(patches):
            patchesj.set_color(bincol[j])

        # This section draws the black line.
        if zval is None or np.ma.is_masked(zval) is True:
            self.update_hist_text(self.htxt[hno], None)
            return 0

        binnum = (bins < zval).sum() - 1
        if binnum < 0 or binnum >= len(patches):
            self.update_hist_text(self.htxt[hno], zval)
            return 0

        self.update_hist_text(self.htxt[hno], zval)
        if self.htype == "Histogram Equalization":
            return 0
        patches[binnum].set_color("k")

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
        xnew = 0.95 * (xmax - xmin) + xmin
        ynew = 0.95 * (ymax - ymin) + ymin
        hst.set_position((xnew, ynew))

        if zval is None:
            hst.set_text("")
        else:
            hst.set_text(f"{zval:.4f}")

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

        sun = None
        dat = [None, None, None]
        for i in self.data:
            if i.dataid == self.hband[3]:
                sun = i.data
            for j in range(3):
                if i.dataid == self.hband[j]:
                    dat[j] = i.data
                    self.argbunit[j] = i.units

        self.image.set_shade(self.shade, self.cell, self.theta, self.phi, self.alpha)

        if self.shade is True:
            dat.append(sun)

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

        if self.htype == "Histogram Equalization":
            self.image.dohisteq = True
        elif self.htype == "Linear with Percent Clip":
            self.image.dohisteq = False
            clippercu = self.clippercu[self.hband[0]]
            clippercl = self.clippercl[self.hband[0]]
            lclip[0], uclip[0] = np.percentile(
                dat[:, :, 0].compressed(), [clippercl, 100 - clippercu]
            )
            clippercu = self.clippercu[self.hband[1]]
            clippercl = self.clippercl[self.hband[1]]
            lclip[1], uclip[1] = np.percentile(
                dat[:, :, 1].compressed(), [clippercl, 100 - clippercu]
            )
            clippercu = self.clippercu[self.hband[2]]
            clippercl = self.clippercl[self.hband[2]]
            lclip[2], uclip[2] = np.percentile(
                dat[:, :, 2].compressed(), [clippercl, 100 - clippercu]
            )

            self.image.rgbclip = [
                [lclip[0], uclip[0]],
                [lclip[1], uclip[1]],
                [lclip[2], uclip[2]],
            ]
        else:
            self.image.dohisteq = False
            lclip[0] = self.clipmin[self.hband[0]]
            uclip[0] = self.clipmax[self.hband[0]]
            lclip[1] = self.clipmin[self.hband[1]]
            uclip[1] = self.clipmax[self.hband[1]]
            lclip[2] = self.clipmin[self.hband[2]]
            uclip[2] = self.clipmax[self.hband[2]]
            self.image.rgbclip = [
                [lclip[0], uclip[0]],
                [lclip[1], uclip[1]],
                [lclip[2], uclip[2]],
            ]

        for i in range(3):
            hdata = dat[:, :, i]
            clippercu = self.clippercu[self.hband[i]]
            clippercl = self.clippercl[self.hband[i]]

            if (
                (clippercu > 0.0 or clippercl > 0.0)
                and self.fullhist is True
                and self.htype == "Linear with Percent Clip"
            ):
                self.hhist[i] = self.argb[i].hist(hdata.compressed(), 50, ec="none")
                self.clipvall[i] = self.argb[i].axvline(lclip[i], ls="--")
                self.clipvalu[i] = self.argb[i].axvline(uclip[i], ls="--")

            elif self.htype == "Histogram Equalization":
                hdata = histeq(hdata)
                hdata = hdata.compressed()
                self.hhist[i] = self.argb[i].hist(hdata, 50, ec="none")
            else:
                self.hhist[i] = self.argb[i].hist(
                    hdata.compressed(), 50, ec="none", range=(lclip[i], uclip[i])
                )
            self.htxt[i] = self.argb[i].text(0.0, 0.0, "", ha="right", va="top")

            self.argb[i].set_xlim(self.hhist[i][1].min(), self.hhist[i][1].max())
            self.argb[i].set_ylim(0, self.hhist[i][0].max() * 1.2)

        self.figure.canvas.restore_region(self.bgrgb[0])
        self.figure.canvas.restore_region(self.bgrgb[1])
        self.figure.canvas.restore_region(self.bgrgb[2])

        self.update_hist_rgb([None, None, None])

        self.axes.draw_artist(self.image)

        for j in range(3):
            for i in self.hhist[j][2]:
                self.argb[j].draw_artist(i)

        self.figure.canvas.update()

        self.bbox_hist_red = self.figure.canvas.copy_from_bbox(self.argb[0].bbox)
        self.bbox_hist_green = self.figure.canvas.copy_from_bbox(self.argb[1].bbox)
        self.bbox_hist_blue = self.figure.canvas.copy_from_bbox(self.argb[2].bbox)

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
        Update the single colour map.

        Returns
        -------
        None.

        """
        self.clipvalu = [None, None, None]
        self.clipvall = [None, None, None]
        self.image.rgbmode = self.gmode

        clippercu = self.clippercu[self.hband[0]]
        clippercl = self.clippercl[self.hband[0]]

        sun = None
        pseudo = self.data[0].data

        for i in self.data:
            if i.dataid == self.hband[0]:
                pseudo = i.data
                self.argbunit[0] = i.units
            if i.dataid == self.hband[3]:
                sun = i.data

        self.image.set_shade(self.shade, self.cell, self.theta, self.phi, self.alpha)
        if self.shade is True:
            pseudo = np.ma.stack([pseudo, sun])
            pseudo = np.moveaxis(pseudo, 0, -1)

        self.image.set_data(pseudo)
        self.image._scale_to_res()

        if self.image._A.ndim == 2:
            pseudo = self.image._A
        else:
            pseudo = self.image._A[:, :, 0]

        lclip = None
        uclip = None
        if self.htype == "Histogram Equalization":
            self.image.dohisteq = True
            pseudo = histeq(pseudo)
            pseudoc = pseudo.compressed()
            lclip = pseudoc.min()
            uclip = pseudoc.max()
        elif self.htype == "Linear with Percent Clip":
            self.image.dohisteq = False
            pseudoc = pseudo.compressed()
            lclip, uclip = np.percentile(pseudoc, [clippercl, 100 - clippercu])
        else:
            self.image.dohisteq = False
            pseudoc = pseudo.compressed()
            lclip = self.clipmin[self.hband[0]]
            uclip = self.clipmax[self.hband[0]]

        self.image.cmap = self.cbar
        self.image.set_clim(lclip, uclip)
        self.image.set_clim(lclip, uclip)

        self.newcmp = self.cbar
        if (
            (clippercu > 0.0 or clippercl > 0.0)
            and self.fullhist is True
            and self.htype == "Linear with Percent Clip"
        ):
            self.hhist[0] = self.argb[0].hist(pseudoc, 50, ec="none")
            tmp = self.hhist[0][1]
            filt = (tmp > lclip) & (tmp < uclip)
            bcnt = np.sum(filt)

            cols = self.cbar(np.linspace(0, 1, bcnt))
            tmp = np.nonzero(filt)

            tmp1 = cols.copy()
            if tmp[0][0] > 0:
                tmp1 = np.vstack(([cols[0]] * tmp[0][0], tmp1))
            if tmp[0][-1] < 49:
                tmp1 = np.vstack((tmp1, [cols[-1]] * (49 - tmp[0][-1])))
            self.newcmp = ListedColormap(tmp1)
        else:
            self.hhist[0] = self.argb[0].hist(
                pseudoc, 50, ec="none", range=(lclip, uclip)
            )

        self.htxt[0] = self.argb[0].text(0.0, 0.0, "", ha="right", va="top")
        self.argb[0].set_xlim(self.hhist[0][1].min(), self.hhist[0][1].max())
        self.argb[0].set_ylim(0, self.hhist[0][0].max() * 1.2)

        self.clipvall[0] = self.argb[0].axvline(lclip, ls="--")
        self.clipvalu[0] = self.argb[0].axvline(uclip, ls="--")

        self.figure.canvas.restore_region(self.bgrgb[0])
        self.update_hist_single()
        self.axes.draw_artist(self.image)

        for i in self.hhist[0][2]:
            self.argb[0].draw_artist(i)

        self.figure.canvas.update()

        self.bbox_hist_red = self.figure.canvas.copy_from_bbox(self.argb[0].bbox)

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
        sun = None

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

        self.image.set_shade(True, self.cell, self.theta, self.phi, self.alpha)
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

        sun = None
        for i in self.sdata:
            if i.dataid == self.hband[3]:
                sun = i.data

        sunshader = currentshader(sun.data, self.cell, self.theta, self.phi, self.alpha)

        snorm = norm2(sunshader)

        return snorm


class MySunCanvas(FigureCanvasQTAgg):
    """
    Canvas widget for the sunshading tool.

    Attributes
    ----------
    sun: matplotlib plot instance
        plot of a circle 'o' showing where the sun is
    axes: matplotlib axes instance
        axes on which the sun is drawn
    """

    def __init__(self):
        fig = Figure(layout="tight")
        super().__init__(fig)

        self.sun = None
        self.axes = fig.add_subplot(111, polar=True)

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
        self.axes.tick_params(labelleft=False, labelright=False)
        self.axes.set_autoscaley_on(False)
        self.axes.set_rmax(1.0)
        self.axes.set_rmin(0.0)
        self.axes.set_xticklabels([])

        (self.sun,) = self.axes.plot(np.pi / 4.0, cos(np.pi / 4.0), "o")
        self.figure.canvas.draw()


class PlotInterp(BasicModule):
    """
    The primary GUI class for the raster data interpretation module.

    The main interface is set up from here, as well as monitoring of the mouse
    over the sunshading.

    The PlotInterp class allows for the display of raster data in a variety of
    modes, as well as the export of that display to GeoTIFF format.

    Parameters
    ----------
    parent : pygmi.main.MainWidget, optional
        Reference to the parent routine. The default is None.

    Attributes
    ----------
    self.mmc : pygmi.raster.ginterp.MyMplCanvas, FigureCanvas
        main canvas containing the image
    self.msc : pygmi.raster.ginterp.MySunCanvas, FigureCanvas
        small canvas containing the sunshading control
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.units = {}
        self.clippercu = {}
        self.clippercl = {}
        self.clipmin = {}
        self.clipmax = {}

        self.mmc = MyMplCanvas()
        self.msc = MySunCanvas()
        self.btn_saveimg = QtWidgets.QPushButton("Save GeoTIFF")
        self.btn_savepng = QtWidgets.QPushButton("Save PNG")
        self.cb_histtype = QtWidgets.QCheckBox("Full histogram with clip lines")
        self.cmb_dtype = QtWidgets.QComboBox()
        self.cmb_band1 = QtWidgets.QComboBox()
        self.cmb_band2 = QtWidgets.QComboBox()
        self.cmb_band3 = QtWidgets.QComboBox()
        self.cmb_bands = QtWidgets.QComboBox()
        self.cmb_bandh = QtWidgets.QComboBox()
        self.cmb_htype = QtWidgets.QComboBox()
        self.le_contours = QtWidgets.QLineEdit()
        self.dsb_lineclipl = QtWidgets.QDoubleSpinBox()
        self.dsb_lineclipu = QtWidgets.QDoubleSpinBox()
        self.dsb_linemin = QtWidgets.QDoubleSpinBox()
        self.dsb_linemax = QtWidgets.QDoubleSpinBox()
        self.cmb_cbar = QtWidgets.QComboBox(self)
        self.kslider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)  # CMYK
        self.sslider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)  # sunshade
        self.aslider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.lbl_4 = QtWidgets.QLabel("Sunshade Data:")
        self.lbl_s = QtWidgets.QLabel("Sunshade Detail")
        self.lbl_a = QtWidgets.QLabel("Light Reflectance")
        self.lbl_c = QtWidgets.QLabel("Colour Bar:")
        self.lbl_k = QtWidgets.QLabel("K value:")
        self.gbox_sun = QtWidgets.QGroupBox("Sunshading")

        self.btn_allclipperc = QtWidgets.QPushButton(
            "Set current exclusion % to all bands"
        )

        self.setupui()

        txt = str(self.cmb_cbar.currentText())
        self.mmc.cbar = colormaps[txt]

        self.setFocus()

        self.mmc.gmode = "Single Colour Map"
        self.cmb_band1.show()
        self.cmb_band2.hide()
        self.cmb_band3.hide()
        self.sslider.hide()
        self.aslider.hide()
        self.kslider.hide()
        self.msc.hide()
        self.lbl_a.hide()
        self.lbl_s.hide()
        self.lbl_k.hide()
        self.lbl_4.hide()
        self.cmb_bands.hide()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        self.buttonbox.buttonbox.hide()
        self.buttonbox.htmlfile = "raster.dm.rasterdisplay"
        btn_apply = QtWidgets.QPushButton("Apply Histogram")

        self.btn_allclipperc.setDefault(False)
        self.btn_allclipperc.setAutoDefault(False)

        gbox_1 = QtWidgets.QGroupBox("Display Type")
        vbl_1 = QtWidgets.QVBoxLayout()
        gbox_1.setLayout(vbl_1)

        gbox_2 = QtWidgets.QGroupBox("Data Bands")
        vbl_2 = QtWidgets.QVBoxLayout()
        gbox_2.setLayout(vbl_2)

        gbox_3 = QtWidgets.QGroupBox("Histogram Stretch")
        vbl_3 = QtWidgets.QVBoxLayout()
        gbox_3.setLayout(vbl_3)

        vbl_4 = QtWidgets.QVBoxLayout()
        self.gbox_sun.setLayout(vbl_4)
        self.gbox_sun.setCheckable(True)
        self.gbox_sun.setChecked(False)

        vbl_raster = QtWidgets.QVBoxLayout()
        hbl_all = QtWidgets.QHBoxLayout(self)
        vbl_right = QtWidgets.QVBoxLayout()

        widget = QtWidgets.QWidget()
        widget.setLayout(vbl_raster)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        scroll.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Preferred
        )

        mpl_toolbar = NavigationToolbar2QT(self.mmc, self)
        spacer = QtWidgets.QSpacerItem(
            20,
            40,
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
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

        self.le_contours.setPlaceholderText("Number of contour levels (10 default)")
        self.le_contours.hide()
        self.le_contours.setValidator(QtGui.QIntValidator(1, 2147483647))
        self.btn_saveimg.setAutoDefault(False)
        btn_apply.setAutoDefault(False)

        tmp = sorted(m for m in colormaps())

        self.cmb_cbar.addItem("jet")
        self.cmb_cbar.addItem("viridis")
        self.cmb_cbar.addItem("terrain")
        self.cmb_cbar.addItem("Floyd")
        self.cmb_cbar.addItem("MarineCopper")
        self.cmb_cbar.addItem("Splash")
        self.cmb_cbar.addItem("Wheel")
        self.cmb_cbar.addItems(tmp)
        self.cmb_dtype.addItems(
            ["Single Colour Map", "Contour", "RGB Ternary", "CMY Ternary"]
        )
        self.cmb_htype.addItems(
            ["Linear with Percent Clip", "Linear with Range", "Histogram Equalization"]
        )

        self.setWindowTitle("Raster Data Display")
        self.dsb_lineclipl.setPrefix("Low Exclude %: ")
        self.dsb_lineclipu.setPrefix("High Exclude %: ")
        self.dsb_linemin.setPrefix("Minimum: ")
        self.dsb_linemin.setRange(-1e20, 1e20)
        self.dsb_linemax.setPrefix("Maximum: ")
        self.dsb_linemax.setRange(-1e20, 1e20)

        self.dsb_linemax.hide()
        self.dsb_linemin.hide()

        vbl_1.addWidget(self.cmb_dtype)
        vbl_1.addWidget(self.le_contours)
        vbl_1.addWidget(self.lbl_k)
        vbl_1.addWidget(self.kslider)
        vbl_raster.addWidget(gbox_1)

        vbl_2.addWidget(self.cmb_band1)
        vbl_2.addWidget(self.cmb_band2)
        vbl_2.addWidget(self.cmb_band3)
        vbl_raster.addWidget(gbox_2)

        vbl_3.addWidget(self.cmb_htype)
        vbl_3.addWidget(self.cmb_bandh)
        vbl_3.addWidget(self.dsb_lineclipl)
        vbl_3.addWidget(self.dsb_lineclipu)
        vbl_3.addWidget(self.dsb_linemin)
        vbl_3.addWidget(self.dsb_linemax)
        vbl_3.addWidget(self.cb_histtype)
        vbl_3.addWidget(self.btn_allclipperc)
        vbl_3.addWidget(btn_apply)
        vbl_3.addWidget(self.lbl_c)
        vbl_3.addWidget(self.cmb_cbar)
        vbl_raster.addWidget(gbox_3)

        vbl_raster.addWidget(self.gbox_sun)
        vbl_4.addWidget(self.lbl_4)
        vbl_4.addWidget(self.cmb_bands)
        vbl_4.addWidget(self.msc)
        vbl_4.addWidget(self.lbl_s)
        vbl_4.addWidget(self.sslider)
        vbl_4.addWidget(self.lbl_a)
        vbl_4.addWidget(self.aslider)
        vbl_raster.addItem(spacer)
        vbl_raster.addWidget(self.btn_saveimg)
        vbl_raster.addWidget(self.btn_savepng)
        vbl_raster.addWidget(self.buttonbox)
        vbl_right.addWidget(self.mmc)
        vbl_right.addWidget(mpl_toolbar)

        hbl_all.addWidget(scroll)
        # hbl_all.addLayout(vbl_raster)
        hbl_all.addLayout(vbl_right)

        self.cmb_cbar.currentIndexChanged.connect(self.change_cbar)
        self.cmb_dtype.currentIndexChanged.connect(self.change_dtype)
        self.cmb_htype.currentIndexChanged.connect(self.change_htype)

        self.sslider.sliderReleased.connect(self.change_sunsliders)
        self.aslider.sliderReleased.connect(self.change_sunsliders)
        self.kslider.sliderReleased.connect(self.change_kval)
        self.msc.figure.canvas.mpl_connect("button_press_event", self.move)
        self.btn_saveimg.clicked.connect(self.save_img)
        self.btn_savepng.clicked.connect(self.save_png)
        self.gbox_sun.clicked.connect(self.change_sun_checkbox)
        btn_apply.clicked.connect(self.change_lclip)
        self.cb_histtype.clicked.connect(self.change_dtype)
        self.btn_allclipperc.clicked.connect(self.change_allclip)
        self.le_contours.textChanged.connect(self.change_dtype)
        self.cmb_band1.currentIndexChanged.connect(self.change_red)
        self.cmb_band2.currentIndexChanged.connect(self.change_green)
        self.cmb_band3.currentIndexChanged.connect(self.change_blue)
        self.cmb_bands.currentIndexChanged.connect(self.change_sun)
        self.cmb_bandh.currentIndexChanged.connect(self.change_clipband)

        if self.parent is not None:
            self.resize(self.parent.width(), self.parent.height())

    def change_allclip(self):
        """
        Change all clip percentages to the current one.

        Returns
        -------
        None.

        """
        uclip = self.dsb_lineclipu.value()
        lclip = self.dsb_lineclipl.value()

        for key in self.clippercl:
            self.clippercl[key] = lclip
            self.clippercu[key] = uclip

        self.mmc.clippercu = self.clippercu
        self.mmc.clippercl = self.clippercl

        self.change_lclip()

    def change_blue(self):
        """
        Change the blue or third display band.

        Returns
        -------
        None.

        """
        txt = str(self.cmb_band3.currentText())
        self.cmb_bandh.setCurrentText(txt)
        self.mmc.hband[2] = txt
        self.mmc.init_graph()

    def change_cbar(self):
        """
        Change the colour map for the colour bar.

        Returns
        -------
        None.

        """
        txt = str(self.cmb_cbar.currentText())
        self.mmc.cbar = colormaps[txt]
        self.mmc.update_graph()

    def change_clipband(self):
        """
        Change the clip percentage band.

        Returns
        -------
        None.

        """
        dattxt = self.cmb_bandh.currentText()

        self.dsb_lineclipu.setValue(self.clippercu[dattxt])
        self.dsb_lineclipl.setValue(self.clippercl[dattxt])

        self.set_minmax()

    def change_dtype(self):
        """
        Change display type.

        Returns
        -------
        None.

        """
        self.mmc.figure.canvas.mpl_disconnect(self.mmc.cid)

        txt = str(self.cmb_dtype.currentText())
        self.mmc.gmode = txt
        self.cmb_band1.show()
        self.mmc.fullhist = self.cb_histtype.isChecked()

        if txt == "Single Colour Map":
            self.lbl_c.show()
            self.lbl_k.hide()
            self.cmb_band2.hide()
            self.cmb_band3.hide()
            self.cmb_cbar.show()
            self.mmc.argbvis = [True, False, False]
            self.sslider.hide()
            self.aslider.hide()
            self.kslider.hide()
            self.le_contours.hide()
            self.cmb_bandh.hide()

        if txt == "Contour":
            self.lbl_k.hide()
            self.lbl_c.show()
            self.cmb_band2.hide()
            self.cmb_band3.hide()
            self.cmb_cbar.show()
            self.mmc.argbvis = [False, False, False]
            self.sslider.hide()
            self.aslider.hide()
            self.kslider.hide()
            self.gbox_sun.setChecked(False)
            self.le_contours.show()
            self.cmb_bandh.hide()

            try:
                self.mmc.levels = int(self.le_contours.text())
            except ValueError:
                self.mmc.levels = 10

        if "Ternary" in txt:
            self.lbl_k.hide()
            self.lbl_c.hide()
            self.cmb_band2.show()
            self.cmb_band3.show()
            self.cmb_cbar.hide()
            self.mmc.argbvis = [True, True, True]
            self.sslider.hide()
            self.aslider.hide()
            self.kslider.hide()
            self.le_contours.hide()
            self.cmb_bandh.show()
            if "CMY" in txt:
                self.kslider.show()
                self.lbl_k.show()
                self.mmc.kval = float(self.kslider.value()) / 100.0

        if self.gbox_sun.isChecked():
            self.msc.show()
            self.lbl_4.show()
            self.cmb_bands.show()
            self.sslider.show()
            self.aslider.show()
            self.lbl_a.show()
            self.lbl_s.show()
            self.mmc.cell = self.sslider.value()
            self.mmc.alpha = float(self.aslider.value()) / 100.0
            self.mmc.shade = True
            self.msc.init_graph()
        else:
            self.msc.hide()
            self.lbl_a.hide()
            self.lbl_s.hide()
            self.lbl_4.hide()
            self.cmb_bands.hide()
            self.mmc.shade = False

        self.mmc.cid = self.mmc.figure.canvas.mpl_connect(
            "resize_event", self.mmc.revent
        )
        self.mmc.init_graph()

    def change_green(self):
        """
        Change the green or second band.

        Returns
        -------
        None.

        """
        txt = str(self.cmb_band2.currentText())
        self.cmb_bandh.setCurrentText(txt)
        self.mmc.hband[1] = txt
        self.mmc.init_graph()

    def change_htype(self):
        """
        Change the histogram stretch to apply to the normal data.

        Returns
        -------
        None.

        """
        txt = str(self.cmb_htype.currentText())

        if txt == "Histogram Equalization":
            self.dsb_lineclipl.hide()
            self.dsb_lineclipu.hide()
            self.dsb_linemin.hide()
            self.dsb_linemax.hide()
            self.cmb_bandh.hide()
            self.btn_allclipperc.hide()
            self.cb_histtype.hide()
        elif txt == "Linear with Percent Clip":
            self.dsb_lineclipl.show()
            self.dsb_lineclipu.show()
            self.dsb_linemin.hide()
            self.dsb_linemax.hide()
            self.cmb_bandh.show()
            self.btn_allclipperc.show()
            self.cb_histtype.show()
        else:
            self.dsb_lineclipl.hide()
            self.dsb_lineclipu.hide()
            self.dsb_linemin.show()
            self.dsb_linemax.show()
            self.cmb_bandh.show()
            self.btn_allclipperc.hide()
            self.cb_histtype.hide()

        self.mmc.htype = txt
        self.mmc.update_graph()

    def change_kval(self):
        """
        Change the CMYK K value.

        Returns
        -------
        None.

        """
        self.mmc.kval = float(self.kslider.value()) / 100.0
        self.mmc.update_graph()

    def change_lclip(self):
        """
        Change the linear clip percentage.

        Returns
        -------
        None.

        """
        dattxt = self.cmb_bandh.currentText()

        uclip = self.dsb_lineclipu.value()
        lclip = self.dsb_lineclipl.value()
        clipmax = self.dsb_linemax.value()
        clipmin = self.dsb_linemin.value()

        if (lclip + uclip) >= 100.0:
            clip = self.mmc.clippercu[dattxt]
            self.dsb_lineclipu.setValue(clip)
            clip = self.mmc.clippercl[dattxt]
            self.dsb_lineclipl.setValue(clip)
            return

        self.clippercl[dattxt] = lclip
        self.clippercu[dattxt] = uclip
        self.mmc.clippercu = self.clippercu
        self.mmc.clippercl = self.clippercl
        self.clipmax[dattxt] = clipmax
        self.clipmin[dattxt] = clipmin
        self.mmc.clipmax = self.clipmax
        self.mmc.clipmin = self.clipmin

        self.change_dtype()
        # self.set_minmax()

    def change_red(self):
        """
        Change the red or first band.

        Returns
        -------
        None.

        """
        txt = str(self.cmb_band1.currentText())
        self.cmb_bandh.setCurrentText(txt)
        self.mmc.hband[0] = txt
        self.mmc.init_graph()

    def change_sun(self):
        """
        Change the sunshade band.

        Returns
        -------
        None.

        """
        txt = str(self.cmb_bands.currentText())
        self.mmc.hband[3] = txt
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
            self.lbl_4.show()
            self.cmb_bands.show()
            self.sslider.show()
            self.aslider.show()
            self.lbl_a.show()
            self.lbl_s.show()
            self.mmc.cell = self.sslider.value()
            self.mmc.alpha = float(self.aslider.value()) / 100.0
            self.mmc.shade = True
            self.msc.init_graph()
            QtWidgets.QApplication.processEvents()
        else:
            self.msc.hide()
            self.lbl_a.hide()
            self.lbl_s.hide()
            self.lbl_4.hide()
            self.cmb_bands.hide()
            self.sslider.hide()
            self.aslider.hide()
            self.mmc.shade = False
            QtWidgets.QApplication.processEvents()
        self.mmc.update_graph()

        self.mmc.cid = self.mmc.figure.canvas.mpl_connect(
            "resize_event", self.mmc.revent
        )

    def change_sunsliders(self):
        """
        Change the sun shading sliders.

        Returns
        -------
        None.

        """
        self.mmc.cell = self.sslider.value()
        self.mmc.alpha = float(self.aslider.value()) / 100.0
        self.mmc.update_shade()

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
        if "Cluster" in self.indata:
            self.indata = copy.deepcopy(self.indata)
            self.indata = dataprep.cluster_to_raster(self.indata)

        if "Raster" not in self.indata:
            return

        # Get rid of RGB bands.
        indata = []
        for i in self.indata["Raster"]:
            if i.isrgb is True:
                continue
            indata.append(i)

        if not indata:
            return

        indata = lstack(indata, showlog=self.showlog, piter=self.piter)

        # Add membership data.
        if "Cluster" in self.indata:
            newdat = copy.copy(indata)
            for i in self.indata["Cluster"]:
                if "memdat" not in i.metadata["Cluster"]:
                    continue
                for j, val in enumerate(i.metadata["Cluster"]["memdat"]):
                    tmp = copy.deepcopy(i)
                    tmp.memdat = None
                    tmp.data = val
                    tmp.dataid = "Membership of class " + str(j + 1) + ": " + tmp.dataid
                    newdat.append(tmp)
            data = newdat
            sdata = newdat
        else:
            data = indata
            sdata = indata

        for i in data:
            self.units[i.dataid] = i.units
            self.clipmin[i.dataid] = i.data.min()
            self.clipmax[i.dataid] = i.data.max()

        self.mmc.data = data
        self.mmc.sdata = sdata
        self.mmc.hband[0] = data[0].dataid
        self.mmc.hband[1] = data[0].dataid
        self.mmc.hband[2] = data[0].dataid
        self.mmc.hband[3] = data[0].dataid

        blist = []

        for i in data:
            blist.append(i.dataid)
            if i.dataid not in self.clippercl:
                self.clippercu[i.dataid] = 0.0
                self.clippercl[i.dataid] = 0.0

        self.mmc.clippercu = self.clippercu
        self.mmc.clippercl = self.clippercl
        self.mmc.clipmin = self.clipmin
        self.mmc.clipmax = self.clipmax

        self.cmb_update(self.cmb_band1, blist)
        self.cmb_update(self.cmb_band2, blist)
        self.cmb_update(self.cmb_band3, blist)
        self.cmb_update(self.cmb_bands, blist)
        self.cmb_update(self.cmb_bandh, blist)

    def set_minmax(self):
        """Get minimum and maximum of histogram band."""
        dattxt = self.cmb_bandh.currentText()
        uclip = self.dsb_lineclipu.value()
        lclip = self.dsb_lineclipl.value()

        dat = self.mmc.data[0]
        for i in self.mmc.data:
            if i.dataid == dattxt:
                dat = i

        lmin, lmax = np.percentile(dat.data.compressed(), [lclip, 100 - uclip])

        self.dsb_linemin.setValue(lmin)
        self.dsb_linemax.setValue(lmax)

    def move(self, event):
        """
        Move event is used to track changes to the sunshading.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Mouse event.

        Returns
        -------
        None.

        """
        if event.inaxes == self.msc.axes:
            self.msc.sun.set_xdata([event.xdata])
            self.msc.sun.set_ydata([event.ydata])
            self.msc.figure.canvas.draw()

            phi = -event.xdata
            theta = np.pi / 2.0 - np.arccos(event.ydata)
            self.mmc.phi = phi
            self.mmc.theta = theta
            self.mmc.update_shade()

    def run(self):
        """Entry point into the routine, used to run context menu item."""
        self.data_init()
        self.settings()

    def save_png(self):
        """
        Save image as a GeoTIFF.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        ext = "PNG (*.png)"
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.parent, "Save File", ".", ext
        )
        if filename == "":
            return False

        dtype = str(self.cmb_dtype.currentText())

        fig = self.mmc.figure
        axes = self.mmc.axes

        divider = make_axes_locatable(axes)
        axes.xaxis.set_visible(True)
        axes.yaxis.set_visible(True)
        axes.set_xlabel("Eastings")
        axes.set_ylabel("Northings")
        axes.xaxis.set_major_formatter(frm)
        axes.yaxis.set_major_formatter(frm)

        if dtype == "Single Colour Map":
            cax = divider.append_axes("right", size="7%", pad=0.05)
            cbar = fig.colorbar(self.mmc.image, cax=cax)

            text = self.mmc.argbunit[0]
            if text == "":
                text, okay = QtWidgets.QInputDialog.getText(
                    self,
                    "Colorbar",
                    "Enter colorbar unit label:",
                    QtWidgets.QLineEdit.EchoMode.Normal,
                    self.units[str(self.cmb_band1.currentText())],
                )

                if not okay:
                    self.change_dtype()
                    return

            cbar.set_label(text)
        elif "Ternary" in dtype:
            cax = divider.append_axes("right", size="50%", pad=0.05)
            rtext = self.mmc.argbunit[0]
            gtext = self.mmc.argbunit[1]
            btext = self.mmc.argbunit[2]

            if rtext == "":
                if "RGB" in dtype:
                    rtext, okay = QtWidgets.QInputDialog.getText(
                        self,
                        "Ternary Colorbar",
                        "Enter red label:",
                        QtWidgets.QLineEdit.EchoMode.Normal,
                        "red",
                    )
                else:
                    rtext, okay = QtWidgets.QInputDialog.getText(
                        self,
                        "Ternary Colorbar",
                        "Enter cyan label:",
                        QtWidgets.QLineEdit.EchoMode.Normal,
                        "cyan",
                    )

                if not okay:
                    self.change_dtype()
                    return
            if gtext == "":
                if "RGB" in dtype:
                    gtext, okay = QtWidgets.QInputDialog.getText(
                        self,
                        "Ternary Colorbar",
                        "Enter green label:",
                        QtWidgets.QLineEdit.EchoMode.Normal,
                        "green",
                    )
                else:
                    gtext, okay = QtWidgets.QInputDialog.getText(
                        self,
                        "Ternary Colorbar",
                        "Enter magenta label:",
                        QtWidgets.QLineEdit.EchoMode.Normal,
                        "magenta",
                    )

                if not okay:
                    self.change_dtype()
                    return

            if btext == "":
                if "RGB" in dtype:
                    btext, okay = QtWidgets.QInputDialog.getText(
                        self,
                        "Ternary Colorbar",
                        "Enter blue label:",
                        QtWidgets.QLineEdit.EchoMode.Normal,
                        "blue",
                    )
                else:
                    btext, okay = QtWidgets.QInputDialog.getText(
                        self,
                        "Ternary Colorbar",
                        "Enter yellow label:",
                        QtWidgets.QLineEdit.EchoMode.Normal,
                        "yellow",
                    )

                if not okay:
                    self.change_dtype()
                    return

            tmp = np.array([[list(range(255))] * 255])
            tmp = tmp.reshape(255, 255)
            tmp = np.transpose(tmp)

            red = ndimage.rotate(tmp, 0)
            green = ndimage.rotate(tmp, 120)
            blue = ndimage.rotate(tmp, -120)

            tmp = np.zeros((blue.shape[0], 90))
            blue = np.hstack((tmp, blue))
            green = np.hstack((green, tmp))

            rtmp = np.zeros_like(blue)
            j = 92
            rtmp[:255, j : j + 255] = red
            red = rtmp

            if "RGB" in dtype:
                red = red.max() - red
                green = green.max() - green
                blue = blue.max() - blue

            data = np.transpose([red.flatten(), green.flatten(), blue.flatten()])
            data = data.reshape(red.shape[0], red.shape[1], 3)

            data = data[:221, 90:350]

            ax = cax
            ax.set_xlim((-100, 355))
            ax.set_ylim((-100, 322))

            path = Path([[0, 0], [127.5, 222], [254, 0], [0, 0]])
            patch = PathPatch(path, facecolor="none")
            ax.add_patch(patch)

            data = data.astype(int)

            im = ax.imshow(data, extent=(0, 255, 0, 222), clip_path=patch, clip_on=True)
            im.set_clip_path(patch)

            ax.text(0, -5, gtext, horizontalalignment="center", verticalalignment="top")
            ax.text(
                254, -5, btext, horizontalalignment="center", verticalalignment="top"
            )
            ax.text(127.5, 225, rtext, horizontalalignment="center")
            ax.tick_params(
                top="off",
                right="off",
                bottom="off",
                left="off",
                labelbottom="off",
                labelleft="off",
            )

            ax.axis("off")

        for i in range(3):
            self.mmc.argb[i].set_visible(False)

        fig.tight_layout()

        fig.savefig(filename, bbox_inches="tight", dpi=300)

        self.change_dtype()

        return

    def save_img(self):
        """
        Save image as a GeoTIFF.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        snorm = self.mmc.update_shade_plot()

        ext = "GeoTIFF (*.tif)"
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.parent, "Save File", ".", ext
        )
        if filename == "":
            return False

        dtype = str(self.cmb_dtype.currentText())

        if "Ternary" not in dtype:
            text, okay = QtWidgets.QInputDialog.getText(
                self,
                "Colorbar",
                "Enter length and width in inches:",
                QtWidgets.QLineEdit.EchoMode.Normal,
                "4, 0.25",
            )

            if not okay:
                return False

            try:
                text = text.split(",")
                blen = float(text[0])
                bwid = float(text[1])
            except ValueError:
                QtWidgets.QMessageBox.warning(
                    self.parent,
                    "Error",
                    "Invalid value.",
                    QtWidgets.QMessageBox.StandardButton.Ok,
                )
                return False
        else:
            text, okay = QtWidgets.QInputDialog.getText(
                self,
                "Colorbar",
                "Enter length in inches:",
                QtWidgets.QLineEdit.EchoMode.Normal,
                "4",
            )

            if not okay:
                return False

            try:
                blen = float(text)
                bwid = blen
            except ValueError:
                QtWidgets.QMessageBox.warning(
                    self.parent,
                    "Error",
                    "Invalid value.",
                    QtWidgets.QMessageBox.StandardButton.Ok,
                )
                return False

        rtext = "Red"
        gtext = "Green"
        btext = "Blue"

        if "Ternary" not in dtype:
            text, okay = QtWidgets.QInputDialog.getText(
                self,
                "Colorbar",
                "Enter colorbar unit label:",
                QtWidgets.QLineEdit.EchoMode.Normal,
                self.units[str(self.cmb_band1.currentText())],
            )

            if not okay:
                return False
        else:
            units = str(self.cmb_band1.currentText())
            rtext, okay = QtWidgets.QInputDialog.getText(
                self,
                "Ternary Colorbar",
                "Enter red/cyan label:",
                QtWidgets.QLineEdit.EchoMode.Normal,
                units,
            )

            if not okay:
                return False

            units = str(self.cmb_band2.currentText())
            gtext, okay = QtWidgets.QInputDialog.getText(
                self,
                "Ternary Colorbar",
                "Enter green/magenta label:",
                QtWidgets.QLineEdit.EchoMode.Normal,
                units,
            )

            if not okay:
                return False

            units = str(self.cmb_band3.currentText())
            btext, okay = QtWidgets.QInputDialog.getText(
                self,
                "Ternary Colorbar",
                "Enter blue/yellow label:",
                QtWidgets.QLineEdit.EchoMode.Normal,
                units,
            )

            if not okay:
                return False

        htype = str(self.cmb_htype.currentText())
        cmin = None
        cmax = None
        img = np.array([])

        if dtype == "Single Colour Map":
            clippercu = self.mmc.clippercu[self.mmc.hband[0]]
            clippercl = self.mmc.clippercl[self.mmc.hband[0]]
            pseudo = self.mmc.data[0].data
            for i in self.mmc.data:
                if i.dataid == self.mmc.hband[0]:
                    pseudo = i.data

            if htype == "Histogram Equalization":
                pseudo = histeq(pseudo)
            elif clippercl > 0.0 or clippercu > 0.0:
                pseudo, _, _ = histcomp(pseudo, perc=clippercl, uperc=clippercu)

            cmin = pseudo.min()
            cmax = pseudo.max()

            # The function below normalizes as well.
            img = img2rgb(pseudo, self.mmc.cbar)
            pseudo = None

            img[:, :, 0] = img[:, :, 0] * snorm  # red
            img[:, :, 1] = img[:, :, 1] * snorm  # green
            img[:, :, 2] = img[:, :, 2] * snorm  # blue
            img = img.astype(np.uint8)

        elif "Ternary" in dtype:
            dat = [None, None, None]
            for i in self.mmc.data:
                for j in range(3):
                    if i.dataid == self.mmc.hband[j]:
                        dat[j] = i.data

            red = dat[0]
            green = dat[1]
            blue = dat[2]

            mask = np.logical_and(red.mask, green.mask)
            mask = np.logical_and(mask, blue.mask)
            mask = np.logical_not(mask)

            if htype == "Histogram Equalization":
                red = histeq(red)
                green = histeq(green)
                blue = histeq(blue)
            else:
                clippercu = self.mmc.clippercu[self.mmc.hband[0]]
                clippercl = self.mmc.clippercl[self.mmc.hband[0]]
                red, _, _ = histcomp(red, perc=clippercl, uperc=clippercu)
                clippercu = self.mmc.clippercu[self.mmc.hband[1]]
                clippercl = self.mmc.clippercl[self.mmc.hband[1]]
                green, _, _ = histcomp(green, perc=clippercl, uperc=clippercu)
                clippercu = self.mmc.clippercu[self.mmc.hband[2]]
                clippercl = self.mmc.clippercl[self.mmc.hband[2]]
                blue, _, _ = histcomp(blue, perc=clippercl, uperc=clippercu)

            red = red.filled(red.min())
            green = green.filled(green.min())
            blue = blue.filled(blue.min())
            red = np.ma.array(red, mask=dat[0].mask)
            green = np.ma.array(green, mask=dat[1].mask)
            blue = np.ma.array(blue, mask=dat[2].mask)

            img = np.zeros((red.shape[0], red.shape[1], 4), dtype=np.uint8)
            img[:, :, 3] = mask * 254 + 1

            if "CMY" in dtype:
                img[:, :, 0] = (1 - norm2(red)) * 254 + 1
                img[:, :, 1] = (1 - norm2(green)) * 254 + 1
                img[:, :, 2] = (1 - norm2(blue)) * 254 + 1
            else:
                img[:, :, 0] = norm255(red)
                img[:, :, 1] = norm255(green)
                img[:, :, 2] = norm255(blue)

            img[:, :, 0] = img[:, :, 0] * snorm  # red
            img[:, :, 1] = img[:, :, 1] * snorm  # green
            img[:, :, 2] = img[:, :, 2] * snorm  # blue
            img = img.astype(np.uint8)

        elif dtype == "Contour":
            cmin = self.mmc.cnt.zmin
            cmax = self.mmc.cnt.zmax

            # if self.mmc.ccbar is not None:
            #     self.mmc.ccbar.remove()
            #     self.mmc.ccbar = None

            self.mmc.axes.set_axis_off()
            tmpsize = self.mmc.figure.get_size_inches()
            self.mmc.figure.set_size_inches(tmpsize * 3)
            self.mmc.figure.canvas.draw()

            buf = io.BytesIO()
            extent = self.mmc.axes.get_window_extent().transformed(
                self.mmc.figure.dpi_scale_trans.inverted()
            )
            self.mmc.figure.savefig(buf, format="png", bbox_inches=extent, pad_inches=0)

            # buf.seek(0)
            # img = mpimg.imread(buf)

            img = np.asarray(self.mmc.figure.canvas.buffer_rgba())

            self.mmc.figure.set_size_inches(tmpsize)
            self.mmc.axes.set_axis_on()
            self.mmc.figure.canvas.draw()

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
            mask[mask < 255] = 0
            tmp = (img[:, :, 0] == 255) & (img[:, :, 1] == 255) & (img[:, :, 2] == 255)
            mask[tmp] = 0
            img[:, :, 3] = mask

        os.chdir(os.path.dirname(filename))

        newimg = [
            copy.deepcopy(self.mmc.data[0]),
            copy.deepcopy(self.mmc.data[0]),
            copy.deepcopy(self.mmc.data[0]),
            copy.deepcopy(self.mmc.data[0]),
        ]

        xmin, xmax, ymin, ymax = newimg[0].extent
        ydim = (ymax - ymin) / img.shape[0]
        xdim = (xmax - xmin) / img.shape[1]

        newimg[0].data = img[:, :, 0]
        newimg[1].data = img[:, :, 1]
        newimg[2].data = img[:, :, 2]
        newimg[3].data = img[:, :, 3]

        newimg[0].set_transform(xdim, xmin, ydim, ymax)
        newimg[1].set_transform(xdim, xmin, ydim, ymax)
        newimg[2].set_transform(xdim, xmin, ydim, ymax)
        newimg[3].set_transform(xdim, xmin, ydim, ymax)

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
        newimg[3].dataid = "Alpha"

        iodefs.export_raster(
            str(filename),
            newimg,
            drv="GTiff",
            piter=self.piter,
            bandsort=False,
            updatestats=True,
            showlog=self.showlog,
            compression="DEFLATE",
        )

        # Section for colorbars
        if "Ternary" not in dtype:
            txt = str(self.cmb_cbar.currentText())
            cmap = colormaps[txt]
            norm = mcolors.Normalize(vmin=cmin, vmax=cmax)

            # Horizontal Bar
            fig = Figure(layout="tight")
            canvas = FigureCanvasQTAgg(fig)
            fig.set_figwidth(blen)
            fig.set_figheight(bwid + 0.75)
            ax = fig.gca()

            if "Contour" in dtype:
                cb = mcolorbar.ColorbarBase(ax, self.mmc.cntf, orientation="horizontal")
            else:
                cb = mcolorbar.ColorbarBase(
                    ax, cmap=cmap, norm=norm, orientation="horizontal"
                )
            cb.set_label(text)

            fname = filename[:-4] + "_hcbar.png"
            canvas.print_figure(fname, dpi=300)

            # Vertical Bar
            fig = Figure(layout="tight")
            canvas = FigureCanvasQTAgg(fig)
            fig.set_figwidth(bwid + 1)
            fig.set_figheight(blen)
            ax = fig.gca()
            if "Contour" in dtype:
                cb = mcolorbar.ColorbarBase(ax, self.mmc.cntf, orientation="vertical")
            else:
                cb = mcolorbar.ColorbarBase(
                    ax, cmap=cmap, norm=norm, orientation="vertical"
                )
            cb.set_label(text)

            fname = filename[:-4] + "_vcbar.png"
            canvas.print_figure(fname, dpi=300)
        else:
            fig = Figure(figsize=[blen, blen], layout="tight")
            canvas = FigureCanvasQTAgg(fig)

            tmp = np.array([[list(range(255))] * 255])
            tmp = tmp.reshape(255, 255)
            tmp = np.transpose(tmp)

            red = ndimage.rotate(tmp, 0)
            green = ndimage.rotate(tmp, 120)
            blue = ndimage.rotate(tmp, -120)

            tmp = np.zeros((blue.shape[0], 90))
            blue = np.hstack((tmp, blue))
            green = np.hstack((green, tmp))

            rtmp = np.zeros_like(blue)
            j = 92
            rtmp[:255, j : j + 255] = red
            red = rtmp

            if "RGB" in dtype:
                red = red.max() - red
                green = green.max() - green
                blue = blue.max() - blue

            data = np.transpose([red.flatten(), green.flatten(), blue.flatten()])
            data = data.reshape(red.shape[0], red.shape[1], 3)

            data = data[:221, 90:350]

            ax = fig.gca()
            ax.set_xlim((-100, 355))
            ax.set_ylim((-100, 322))

            path = Path([[0, 0], [127.5, 222], [254, 0], [0, 0]])
            patch = PathPatch(path, facecolor="none")
            ax.add_patch(patch)

            data = data.astype(int)

            im = ax.imshow(data, extent=(0, 255, 0, 222), clip_path=patch, clip_on=True)
            im.set_clip_path(patch)

            ax.text(
                0,
                -5,
                gtext,
                horizontalalignment="center",
                verticalalignment="top",
                fontsize="xx-large",
            )
            ax.text(
                254,
                -5,
                btext,
                horizontalalignment="center",
                verticalalignment="top",
                fontsize="xx-large",
            )
            ax.text(
                127.5, 225, rtext, horizontalalignment="center", fontsize="xx-large"
            )
            ax.tick_params(
                top="off",
                right="off",
                bottom="off",
                left="off",
                labelbottom="off",
                labelleft="off",
            )

            ax.axis("off")
            fname = filename[:-4] + "_tern.png"
            canvas.print_figure(fname, dpi=300)

        QtWidgets.QMessageBox.information(
            self,
            "Information",
            "Save to GeoTIFF is complete!",
            QtWidgets.QMessageBox.StandardButton.Ok,
        )

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

        if "Raster" not in self.indata:
            self.showlog("No Raster Data.")
            return False

        if self.indata["Raster"][0].isrgb:
            self.showlog("RGB images cannot be used in this module.")
            return False

        self.mmc.hband[0] = str(self.cmb_band1.currentText())
        self.mmc.hband[1] = str(self.cmb_band2.currentText())
        self.mmc.hband[2] = str(self.cmb_band3.currentText())

        self.change_dtype()

        self.mmc.init_graph()
        self.msc.init_graph()

        self.set_minmax()

        tmp = self.exec()

        if tmp == 0:
            return False

        return True

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.cmb_dtype)
        self.saveobj(self.cmb_band1)
        self.saveobj(self.cmb_band2)
        self.saveobj(self.cmb_band3)
        self.saveobj(self.cmb_bands)
        self.saveobj(self.cmb_bandh)
        self.saveobj(self.cmb_htype)
        self.saveobj(self.dsb_lineclipu)
        self.saveobj(self.dsb_lineclipl)
        self.saveobj(self.cmb_cbar)
        self.saveobj(self.kslider)
        self.saveobj(self.sslider)
        self.saveobj(self.aslider)
        self.saveobj(self.cb_histtype)
        self.saveobj(self.gbox_sun)
        self.saveobj(self.clippercl)
        self.saveobj(self.clippercu)


def _testfn():
    """Test routine."""

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..//..")))
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))

    # ifile = r"D:\workdata\PyGMI Test Data\Raster\testdata.tif"
    ifile = r"C:\Work\PyGMI Test Data\Raster\testdata.tif"

    data = iodefs.get_raster(ifile)

    tmp = PlotInterp()
    tmp.indata["Raster"] = data
    tmp.data_init()

    tmp.settings()

    import matplotlib.pyplot as plt

    from pygmi.raster.iodefs import get_raster

    dat = get_raster(r"c:/work/aaa.tif")

    plt.imshow(dat[0].data, extent=dat[0].extent)
    plt.show()


if __name__ == "__main__":
    _testfn()
