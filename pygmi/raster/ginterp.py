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
""" Plot Raster Data """

# pylint: disable=C0103, W0612, E1101
import os
import copy
import numpy as np
import numexpr as ne
from math import cos, sin, tan
from PyQt4 import QtGui, QtCore
from scipy import ndimage
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.image as mi
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
from matplotlib import rcParams
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as \
    FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as \
    NavigationToolbar
import pygmi.raster.iodefs as iodefs
import pygmi.raster.dataprep as dataprep


class MySunCanvas(FigureCanvas):
    """Canvas for the sunshading tool."""
    def __init__(self, parent):
        fig = Figure()
        super(MySunCanvas, self).__init__(fig)

        self.sun = None
        self.axes = fig.add_subplot(111, polar=True)

        self.setParent(parent)
        self.setMaximumSize(120, 120)
        self.setMinimumSize(120, 120)

    def init_graph(self):
        """ Init graph """
        self.axes.clear()
        plt.setp(self.axes.get_xticklabels(), fontsize=8)
        plt.setp(self.axes.get_yticklabels(), visible=False)

        self.axes.set_autoscaley_on(False)
        self.axes.set_rmax(1.0)
        self.axes.set_rmin(0.0)

        self.sun, = self.axes.plot(np.pi/4., cos(np.pi/4.), 'o')
        self.figure.canvas.draw()


class MyMplCanvas(FigureCanvas):
    """Canvas for the actual plot"""
    def __init__(self, parent):
        fig = Figure()
        super(MyMplCanvas, self).__init__(fig)

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
        self.patches = []
        self.lines = []
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
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        self.figure.canvas.mpl_connect('motion_notify_event', self.move)

# sun shading stuff
        self.shademode = 'color'
        self.pinit = None
        self.qinit = None
        self.phi = -np.pi/4.
        self.theta = np.pi/4.
        self.cell = 100.
        self.n = 2.
        self.alpha = .0

    def dat_extent(self, dat):
        """ Gets the extend of the dat variable """
        left = dat.tlx
        top = dat.tly
        right = left + dat.cols*dat.xdim
        bottom = top - dat.rows*dat.ydim
        return (left, right, bottom, top)

    def init_graph(self):
        """ Initialize the graph """

        self.axes.clear()
        for i in range(3):
            self.argb[i].clear()

        x1, x2, y1, y2 = self.dat_extent(self.data[0])
        self.axes.set_xlim(x1, x2)
        self.axes.set_ylim(y1, y2)
        self.axes.set_aspect('equal')

        self.figure.canvas.draw()
        QtGui.QApplication.processEvents()

        self.background = self.figure.canvas.copy_from_bbox(self.axes.bbox)
        self.bbox_hist_red = self.figure.canvas.copy_from_bbox(
            self.argb[0].bbox)
        self.bbox_hist_green = self.figure.canvas.copy_from_bbox(
            self.argb[1].bbox)
        self.bbox_hist_blue = self.figure.canvas.copy_from_bbox(
            self.argb[2].bbox)

        self.image = imshow(self.axes, self.data[0].data, origin='upper',
                            extent=(x1, x2, y1, y2))

        self.update_graph()

    def lamb_horn(self):
        """ Lambert by horn """
        R = ((1+self.p0*self.p+self.q0*self.q) /
             (self.sqrt_1p2q2+np.sqrt(1+self.p0**2+self.q0**2)))
        return R

    def move(self, event):
        """ Mouse is moving """
        if len(self.data) == 0 or self.gmode == 'Contour':
            return

        if event.inaxes == self.axes:
            zval = [-999, -999, -999]

            for i in self.data:
                for j in range(3):
                    if i.bandid == self.hband[j]:
                        col = int((event.xdata - i.tlx)/i.xdim)
                        row = int((i.tly - event.ydata)/i.ydim)
                        zval[j] = i.data[row, col]

            if self.gmode == 'Single Color Map':
                bnum = self.update_hist_single(zval[0])
                self.figure.canvas.restore_region(self.bbox_hist_red)
                self.argb[0].draw_artist(self.htxt[0])
                self.argb[0].draw_artist(self.hhist[0][2][bnum])
                self.figure.canvas.update()

            if self.gmode == 'RGB Ternary':
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
                    for j in [1]:
                        if i.bandid == self.hband[j]:
                            col = int((event.xdata - i.tlx)/i.xdim)
                            row = int((i.tly - event.ydata)/i.ydim)
                            zval[j] = i.data[row, col]
                bnum = self.update_hist_sun(zval)
                self.figure.canvas.restore_region(self.bbox_hist_red)
                self.figure.canvas.restore_region(self.bbox_hist_green)
                for j in range(2):
                    self.argb[j].draw_artist(self.htxt[j])
                    self.argb[j].draw_artist(self.hhist[j][2][bnum[j]])

                self.figure.canvas.update()

    def update_contour(self):
        """ Updates the contour map """
        self.image.dtype = 'Single Color Map'

        x1, x2, y1, y2 = self.dat_extent(self.data[0])
        self.image.set_visible(False)

        for i in self.data:
            if i.bandid == self.hband[0]:
                dat = i.data

        self.image.set_data(dat)
        dat = norm2(self.image.smallres)

        xdim = (x2-x1)/dat.shape[1]/2
        ydim = (y2-y1)/dat.shape[0]/2
        xi = np.linspace(x1+xdim, x2-xdim, dat.shape[1])
        yi = np.linspace(y2-ydim, y1+ydim, dat.shape[0])

        self.cnt = self.axes.contour(xi, yi, dat, extent=(x1, x2, y1, y2),
                                     linewidths=0.5, colors='k')
        self.cntf = self.axes.contourf(xi, yi, dat, extent=(x1, x2, y1, y2),
                                       cmap=self.cbar)

        self.figure.canvas.draw()

    def update_graph(self):
        """ Update the plot """
        if len(self.data) == 0 or self.gmode is None:
            return

        self.image.cbar = self.cbar
        self.image.htype = self.htype
        self.image.hstype = self.hstype
        self.image.alpha = self.alpha
        self.image.cell = self.cell
        self.image.theta = self.theta
        self.image.phi = self.phi

        for i in range(3):
            self.argb[i].clear()

        if self.gmode == 'Single Color Map':
            self.update_single_color_map()

        if self.gmode == 'Contour':
            self.update_contour()

        if self.gmode == 'RGB Ternary':
            self.update_rgb()

        if self.gmode == 'Sunshade':
            self.update_shade_plot()

        for i in self.patches:
            self.axes.add_patch(i)

        for i in self.lines:
            self.axes.add_line(i)

    def update_hist_rgb(self, zval):
        """ updates the rgb histograms """
        hcol = ['r', 'g', 'b']
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

            if binnum > -1 and binnum < len(patches):
                patches[binnum].set_color('k')
                bnum.append(binnum)
            else:
                bnum.append(0)
            self.update_hist_text(self.htxt[i], zval[i])
        return bnum

    def update_hist_single(self, zval, hno=0):
        """ updates the color on a single histogram """
        hst = self.hhist[hno]
        bins, patches = hst[1:]
        binave = np.arange(0, 1, 1/(bins.size-2))

        if hno == 0:
            bincol = self.cbar(binave)
        else:
            bincol = cm.gray(binave)

        for j in range(len(patches)):
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
        """ Updates a sunshade histogram """
        if zval is None:
            zval = [0.0, 0.0]

        bnum = [None, None]
        bnum[0] = self.update_hist_single(zval[0], 0)
        bnum[1] = self.update_hist_single(zval[1], 1)
        return bnum

    def update_hist_text(self, hst, zval):
        """ Update the value on the histogram """
        xmin, xmax, ymin, ymax = hst.axes.axis()
        xnew = 0.95*(xmax-xmin)+xmin
        ynew = 0.95*(ymax-ymin)+ymin
        hst.set_position((xnew, ynew))
        hst.set_text(str(zval))

    def update_rgb(self):
        """ Updates the RGB Ternary Map """
        self.image.dtype = 'RGB Ternary'
        dat = [None, None, None]
        for i in self.data:
            for j in range(3):
                if i.bandid == self.hband[j]:
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
        """ Updates the single color map """
        self.image.dtype = 'Single Color Map'
        for i in self.data:
            if i.bandid == self.hband[0]:
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
        """ Updates sun shade plot """
        self.image.dtype = 'Sunshade'
        data = [None, None]

        for i in self.data:
            if i.bandid == self.hband[0]:
                data[0] = i.data

        for i in self.sdata:
            if i.bandid == self.hband[1]:
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


class PlotInterp(QtGui.QDialog):
    """ Graph Window """
    def __init__(self, parent=None):
        super(PlotInterp, self).__init__(parent)
        self.indata = {}
        self.outdata = {}
        self.parent = parent
        self.equation = ''
        self.bands = {}
        self.bandsall = []

        self.mmc = MyMplCanvas(self)
        self.msc = MySunCanvas(self)
        self.tabwidget = QtGui.QTabWidget(self)
        self.btn_saveimg = QtGui.QPushButton(self)
        self.cbox_dtype = QtGui.QComboBox(self)
        self.cbox_band1 = QtGui.QComboBox(self)
        self.cbox_band2 = QtGui.QComboBox(self)
        self.cbox_band3 = QtGui.QComboBox(self)
        self.cbox_htype = QtGui.QComboBox(self)
        self.cbox_hstype = QtGui.QComboBox(self)
        self.cbox_cbar = QtGui.QComboBox(self)
        self.sslider = QtGui.QSlider(QtCore.Qt.Horizontal, self)  # sunshade
        self.aslider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.slabel = QtGui.QLabel(self)
        self.labels = QtGui.QLabel(self)
        self.labela = QtGui.QLabel(self)
        self.labelc = QtGui.QLabel(self)

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
        self.msc.hide()
        self.labela.hide()
        self.labels.hide()

    def setupui(self):
        """ Setup UI """
        sizepolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred,
                                       QtGui.QSizePolicy.Expanding)

        self.setWindowTitle("Graph Window")
        mpl_toolbar = NavigationToolbar(self.mmc, self)

        rwidget = QtGui.QWidget()
        vbl_raster = QtGui.QVBoxLayout(rwidget)

        self.tabwidget.setSizePolicy(sizepolicy)

        label1 = QtGui.QLabel(self)
        label1.setText('Display Type:')
        vbl_raster.addWidget(label1)
        vbl_raster.addWidget(self.cbox_dtype)

        label2 = QtGui.QLabel(self)
        label2.setText('Data Bands:')
        vbl_raster.addWidget(label2)
        vbl_raster.addWidget(self.cbox_band1)
        vbl_raster.addWidget(self.cbox_band2)
        vbl_raster.addWidget(self.cbox_band3)

        label3 = QtGui.QLabel(self)
        label3.setText('Histogram Stretch:')
        vbl_raster.addWidget(label3)
        vbl_raster.addWidget(self.cbox_htype)

        self.slabel.setText('Sunshade Stretch:')
        vbl_raster.addWidget(self.slabel)
        vbl_raster.addWidget(self.cbox_hstype)

        self.labelc.setText('Color Bar:')
        vbl_raster.addWidget(self.labelc)
        vbl_raster.addWidget(self.cbox_cbar)

        self.labels.setText('Sunshade Detail')
        self.labela.setText('Light Reflectance')
        vbl_raster.addWidget(self.msc)
        vbl_raster.addWidget(self.labels)
        vbl_raster.addWidget(self.sslider)
        vbl_raster.addWidget(self.labela)
        vbl_raster.addWidget(self.aslider)

        spacer = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum,
                                   QtGui.QSizePolicy.Expanding)
        vbl_raster.addItem(spacer)

        self.btn_saveimg.setText('Save GeoTiff')
        vbl_raster.addWidget(self.btn_saveimg)

# Right Vertical Layout
        vbl_right = QtGui.QVBoxLayout()  # self is where layout is assigned
        vbl_right.addWidget(self.mmc)
        vbl_right.addWidget(mpl_toolbar)

# Combined Layout
        self.tabwidget.addTab(rwidget, "Raster")
#        self.tabwidget.addTab(vwidget, "Vector")

        hbl_all = QtGui.QHBoxLayout(self)
        hbl_all.addWidget(self.tabwidget)
        hbl_all.addLayout(vbl_right)

        self.cbox_dtype.addItems(['Single Color Map', 'Contour', 'RGB Ternary',
                                 'Sunshade'])
        self.cbox_htype.addItems(['Linear', '95% Linear, 5% Compact',
                                 'Histogram Equalization'])
        self.cbox_hstype.addItems(['Linear', '95% Linear, 5% Compact',
                                  'Histogram Equalization'])

        self.sslider.setMinimum(1)
        self.sslider.setMaximum(100)
        self.sslider.setValue(25)
        self.aslider.setMinimum(1)
        self.aslider.setMaximum(100)
        self.aslider.setSingleStep(1)
        self.aslider.setValue(75)

        tmp = sorted(cm.datad.keys())
        self.cbox_cbar.addItem('jet')
        self.cbox_cbar.addItems(tmp)

        self.cbox_cbar.currentIndexChanged.connect(self.change_cbar)
        self.cbox_dtype.currentIndexChanged.connect(self.change_dtype)
        self.cbox_htype.currentIndexChanged.connect(self.change_htype)
        self.cbox_hstype.currentIndexChanged.connect(self.change_hstype)

        self.sslider.sliderReleased.connect(self.change_dtype)
        self.aslider.sliderReleased.connect(self.change_dtype)
        self.msc.figure.canvas.mpl_connect('button_press_event', self.move)
        self.btn_saveimg.clicked.connect(self.save_img)

    def save_img(self):
        """Save GeoTiff """
        ext = "GeoTiff (*.tif)"
        filename, filt = QtGui.QFileDialog.getSaveFileName(
            self.parent, 'Save File', '.', ext)
        if filename == '':
            return False

        text, ok = QtGui.QInputDialog.getText(
            self, "Colorbar", "Enter length in inches:",
            QtGui.QLineEdit.Normal, "8")

        if not ok:
            return

        blen = float(text)
        bwid = blen/16.

        text, ok = QtGui.QInputDialog.getText(
            self, "Colorbar", "Enter colorbar unit label:",
            QtGui.QLineEdit.Normal, "Some Units")

        if not ok:
            return

        img = self.mmc.image.get_array()
        dtype = self.cbox_dtype.currentText()
        htype = str(self.cbox_htype.currentText())
        hstype = str(self.cbox_hstype.currentText())
        cell = self.mmc.cell
        alpha = self.mmc.alpha
        phi = self.mmc.phi
        theta = self.mmc.theta

        if dtype == 'Single Color Map':
            pseudo = self.mmc.image._full_res
#            mask = np.logical_not(pseudo.mask)
#            pseudo += pseudo.min()

            if htype == '95% Linear, 5% Compact':
                pseudo = histcomp(pseudo)

            if htype == 'Histogram Equalization':
                pseudo = histeq(pseudo)

            # The function below normalizes as well.
            img = img2rgb(pseudo, self.mmc.cbar, self.mmc.image.smallres)

            pseudo = None

        elif dtype == 'Sunshade':
            pseudo = self.mmc.image._full_res[0]
            sun = self.mmc.image._full_res[1]

            if htype == '95% Linear, 5% Compact':
                pseudo = histcomp(pseudo)

            if htype == 'Histogram Equalization':
                pseudo = histeq(pseudo)

            if hstype == '95% Linear, 5% Compact':
                sun = histcomp(sun)

            if hstype == 'Histogram Equalization':
                sun = histeq(sun)

            sunshader = currentshader(sun.data, cell, theta, phi, alpha)
            snorm = norm2(sunshader)
#            pnorm = norm2(pseudo)

            img = img2rgb(pseudo, self.mmc.cbar,
                          self.mmc.image.smallres[:, :, 0])
            pseudo = None
            sunshader = None

            img[:, :, 0] = img[:, :, 0]*snorm  # red
            img[:, :, 1] = img[:, :, 1]*snorm  # green
            img[:, :, 2] = img[:, :, 2]*snorm  # blue
            img = img.astype(np.uint8)
#            mask = np.logical_or(pseudo.mask, sun.mask)
#            mask = np.logical_not(mask)
#            img[:, :, 3] = mask

        elif dtype == 'RGB Ternary':
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

            if htype == 'Histogram Equalization':
                red = histeq(red)
                green = histeq(green)
                blue = histeq(blue)

            colormap = np.ones((red.shape[0], red.shape[1], 4), dtype=np.uint8)
            colormap[:, :, 0] = norm255(red)
            colormap[:, :, 1] = norm255(green)
            colormap[:, :, 2] = norm255(blue)
            colormap[:, :, 3] = mask*254+1

            img = colormap

        elif dtype == 'Contour':
            self.mmc.figure.set_frameon(False)
            self.mmc.axes.set_axis_off()
            self.mmc.figure.canvas.draw()
#            fcol = int(self.mmc.figure.get_facecolor()[0]*255)
            img = np.fromstring(self.mmc.figure.canvas.tostring_argb(),
                                dtype=np.uint8, sep='')
            self.mmc.figure.set_frameon(True)
            self.mmc.axes.set_axis_on()
            self.mmc.figure.canvas.draw()

            w, h = self.mmc.figure.canvas.get_width_height()
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

        export = iodefs.ExportData(self.parent)

        os.chdir(filename.rpartition('/')[0])

        newimg = [copy.deepcopy(self.mmc.data[0]),
                  copy.deepcopy(self.mmc.data[0]),
                  copy.deepcopy(self.mmc.data[0]),
                  copy.deepcopy(self.mmc.data[0])]

        newimg[0].data = img[:, :, 0]
        newimg[1].data = img[:, :, 1]
        newimg[2].data = img[:, :, 2]
        newimg[3].data = img[:, :, 3]

        mask = img[:, :, 3]
        newimg[0].data[mask <= 1] = 0
        newimg[1].data[mask <= 1] = 0
        newimg[2].data[mask <= 1] = 0
#        newimg[3].data[mask == 0] = 0

        newimg[0].nullvalue = 0
        newimg[1].nullvalue = 0
        newimg[2].nullvalue = 0
        newimg[3].nullvalue = 0

        newimg[0].cols = img.shape[1]
        newimg[1].cols = img.shape[1]
        newimg[2].cols = img.shape[1]
        newimg[3].cols = img.shape[1]

        newimg[0].rows = img.shape[0]
        newimg[1].rows = img.shape[0]
        newimg[2].rows = img.shape[0]
        newimg[3].rows = img.shape[0]

        export.ifile = str(filename)
        export.ext = filename[-3:]
        export.export_gdal(newimg, 'GTiff')

# Section for colorbars

        txt = str(self.cbox_cbar.currentText())
        cmap = plt.get_cmap(txt)
        cmin = self.mmc.data[0].data.min()
        cmax = self.mmc.data[0].data.max()
        norm = mcolors.Normalize(vmin=cmin, vmax=cmax)

# Horizontal Bar
        fig = plt.figure(figsize=(blen, (bwid+0.75)), tight_layout=True)
        ax = fig.add_subplot(111)

        cb = mcolorbar.ColorbarBase(ax, cmap=cmap, norm=norm,
                                    orientation='horizontal')
        cb.set_label(text)

        fname = filename[:-4]+'_hcbar.tif'
        fig.savefig(fname, dpi=300)

# Vertical Bar
        fig = plt.figure(figsize=((bwid + 1), blen), tight_layout=True)
        ax = fig.add_subplot(111)

        cb = mcolorbar.ColorbarBase(ax, cmap=cmap, norm=norm,
                                    orientation='vertical')
        cb.set_label(text)

        fname = filename[:-4]+'_vcbar.tif'
        fig.savefig(fname, dpi=300)

        QtGui.QMessageBox.information(self, "Information",
                                      "Save to GeoTiff is complete!",
                                      QtGui.QMessageBox.Ok,
                                      QtGui.QMessageBox.Ok)

    def change_blue(self):
        """ Combo box to change display bands """
        txt = str(self.cbox_band3.currentText())
        self.mmc.hband[2] = txt
        self.mmc.init_graph()

    def change_cbar(self):
        """ Change the color bar """
        txt = str(self.cbox_cbar.currentText())
        self.mmc.cbar = plt.get_cmap(txt)
        self.mmc.update_graph()

    def change_dtype(self):
        """ Combo box to change display type """
        txt = self.cbox_dtype.currentText()
        self.mmc.gmode = txt
        self.cbox_band1.show()

        if txt == 'Single Color Map':
            self.slabel.hide()
            self.labelc.show()
            self.cbox_hstype.hide()
            self.cbox_band2.hide()
            self.cbox_band3.hide()
            self.cbox_cbar.show()
            self.mmc.argb[0].set_visible(True)
            self.mmc.argb[1].set_visible(False)
            self.mmc.argb[2].set_visible(False)
            self.sslider.hide()
            self.aslider.hide()
            self.msc.hide()
            self.labela.hide()
            self.labels.hide()
            self.mmc.init_graph()

        if txt == 'Contour':
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
            self.msc.hide()
            self.labela.hide()
            self.labels.hide()
            self.mmc.init_graph()

        if txt == 'RGB Ternary':
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
            self.msc.hide()
            self.labela.hide()
            self.labels.hide()
            self.mmc.init_graph()

        if txt == 'Sunshade':
            self.labelc.show()
            self.msc.show()
            self.sslider.show()
            self.aslider.show()
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
            QtGui.QApplication.processEvents()
            self.msc.init_graph()
            self.mmc.init_graph()

    def change_green(self):
        """ Combo box to change display bands """
        txt = str(self.cbox_band2.currentText())
        self.mmc.hband[1] = txt
        self.mmc.init_graph()

    def change_hstype(self):
        """ Change HStype """
        txt = str(self.cbox_hstype.currentText())
        self.mmc.hstype = txt
        self.mmc.init_graph()

    def change_htype(self):
        """ Change Htype """
        txt = str(self.cbox_htype.currentText())
        self.mmc.htype = txt
        self.mmc.init_graph()

    def change_red(self):
        """ Combo box to change display bands """
        txt = str(self.cbox_band1.currentText())
        self.mmc.hband[0] = txt
        self.mmc.init_graph()

    def data_init(self):
        """ data init - entry point into routine """
        if 'Raster' not in self.indata:
            return

        if 'Cluster' in self.indata:
            self.indata = copy.deepcopy(self.indata)
            self.indata = dataprep.cluster_to_raster(self.indata)
        self.indata['Raster'] = dataprep.merge(self.indata['Raster'])

        data = self.indata['Raster']
        sdata = self.indata['Raster']

        self.mmc.data = data
        self.mmc.sdata = sdata
        self.mmc.hband[0] = data[0].bandid
        self.mmc.hband[1] = data[0].bandid
        self.mmc.hband[2] = data[0].bandid

        blist = []
        for i in data:
            blist.append(i.bandid)

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
        """ Move event is used to track changes to the sunshading """
        if event.inaxes == self.msc.axes:
            self.msc.sun.set_xdata(event.xdata)
            self.msc.sun.set_ydata(event.ydata)
            self.msc.figure.canvas.draw()

            phi = -event.xdata
            theta = np.pi/2. - np.arccos(event.ydata)
            self.mmc.phi = phi
            self.mmc.theta = theta
            self.mmc.update_graph()

    def settings(self):
        """ run """
        self.show()
        QtGui.QApplication.processEvents()

        self.mmc.init_graph()
        self.msc.init_graph()
        return True


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
        self.smallres = None
        self._sx, self._sy = None, None
        self._bounds = (None, None, None, None)
        super(ModestImage, self).__init__(*args, **kwargs)
        self.cbar = cm.jet
        self.htype = 'Linear'
        self.hstype = 'Linear'
        self.dtype = 'Single Color Map'
        self.cell = 100.
        self.phi = -np.pi/4.
        self.theta = np.pi/4.
        self.alpha = .0

    def set_data(self, A):
        """
        Set the image array

        ACCEPTS: numpy/PIL Image A
        """
        self._full_res = A
        self._A = A
        self.smallres = A

#        if self._A.dtype != np.uint8 and not np.can_cast(self._A.dtype,
#                                                         np.float):
#            raise TypeError("Image data can not convert to float")
#
#        if (self._A.ndim not in (2, 3) or
#            (self._A.ndim == 3 and self._A.shape[-1] not in (3, 4))):
#            raise TypeError("Invalid dimensions for image data")

        self._imcache = None
        self._rgbacache = None
        self._oldxslice = None
        self._oldyslice = None
        self._sx, self._sy = None, None
        if self.axes.dataLim.x0 != np.inf:
            self._scale_to_res()

#    def get_array(self):
#        """Override to return the full-resolution array"""
#        return self._full_res

    def _scale_to_res(self):
        """ Change self._A and _extent to render an image whose
        resolution is matched to the eventual rendering."""

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

        sy = int(np.ceil(dy/(ddy*ext[1]))/divtmp)
        sx = int(np.ceil(dx/(ddx*ext[0]))/divtmp)

        if self._sx is None:
            pass
        elif (sx >= self._sx and sy >= self._sy and
              x0 >= self._bounds[0] and x1 <= self._bounds[1] and
              y0 >= self._bounds[2] and y1 <= self._bounds[3]):
            return

        if self.dtype == 'Single Color Map':
            pseudo = self._full_res[(rows-y1):(rows-y0):sy, x0:x1:sx]
            mask = pseudo.mask

            if self.htype == '95% Linear, 5% Compact':
                pseudo = histcomp(pseudo)

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

            if self.htype == 'Histogram Equalization':
                pseudo = histeq(pseudo)

            if self.hstype == '95% Linear, 5% Compact':
                sun = histcomp(sun)

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

        elif self.dtype == 'RGB Ternary':
            red = self._full_res[0][(rows-y1):(rows-y0):sy, x0:x1:sx]
            green = self._full_res[1][(rows-y1):(rows-y0):sy, x0:x1:sx]
            blue = self._full_res[2][(rows-y1):(rows-y0):sy, x0:x1:sx]
            mask = np.logical_or(red.mask, green.mask)
            mask = np.logical_or(mask, blue.mask)

            if self.htype == '95% Linear, 5% Compact':
                red = histcomp(red)
                green = histcomp(green)
                blue = histcomp(blue)

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

            self._A = colormap

        y0 = ylim[0]
        y1 = ylim[1]
        x0 = xlim[0]
        x1 = xlim[1]

#        self.set_extent([x0 - .5, x1 - .5, y0 - .5, y1 - .5])
        self.set_extent([x0, x1, y0, y1])
        self._sx = sx
        self._sy = sy
        self._bounds = (x0, x1, y0, y1)
        self.changed()

    def draw(self, renderer, *args, **kwargs):
        """ Draw """

# This loop forces the histograms to remain static
        for argb in self.figure.axes[1:]:
            argb.set_xlim(argb.dataLim.x0, argb.dataLim.x1)
            argb.set_ylim(argb.dataLim.y0, argb.dataLim.y1*1.2)

        self._scale_to_res()
        # The next command runs the original draw for this class.
        super().draw(renderer, *args, **kwargs)


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


def aspect2(data):
    """ Aspect of a dataset"""
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
    """ Blinn shader
    alpha: how much incident light is reflected
    n: how compact teh bright patch is
    phi: azimuth
    theta: sun elevation (also called g in code below)
    """
    asp = aspect2(data)
    pinit = asp[1]
    qinit = asp[2]

# Update cell
    p = ne.evaluate('pinit/cell')
    q = ne.evaluate('qinit/cell')
    sqrt_1p2q2 = ne.evaluate('sqrt(1+p**2+q**2)')

# Update angle
    cosg2 = cos(theta/2)
    p0 = -cos(phi)*tan(theta)
    q0 = -sin(phi)*tan(theta)
    sqrttmp = (1+np.sqrt(1+p0**2+q0**2))
    p1 = p0 / sqrttmp
    q1 = q0 / sqrttmp

    n = 2.0

    cosi = ne.evaluate('((1+p0*p+q0*q)/(sqrt_1p2q2*sqrt(1+p0**2+q0**2)))')
    coss = ne.evaluate('((1+p1*p+q1*q)/(sqrt_1p2q2*sqrt(1+p1**2+q1**2)))')
    Ps = ne.evaluate('coss**n')
    R = np.ma.masked_invalid(ne.evaluate('((1-alpha)+alpha*Ps)*cosi/cosg2'))

    return R


def norm2(dat):
    """ Normalise vector """
    datmin = float(dat.min())
    datptp = float(dat.ptp())
    out = np.ma.array(ne.evaluate('(dat-datmin)/datptp'))
    out.mask = dat.mask
    return out


def norm255(dat):
    """ Normalise vector between 1 and 255"""
    datmin = float(dat.min())
    datptp = float(dat.ptp())
    out = ne.evaluate('255*(dat-datmin)/datptp+1')
    out = out.astype(np.uint8)
    return out


def img2rgb(img, cbar, imgold):
    """ convert img to color img """
    im2 = img.copy()
    im2[im2 < imgold.min()] = imgold.min()
    im2[im2 > imgold.max()] = imgold.max()
    im2 = norm255(im2)
    cbartmp = cbar(range(256))
    cbartmp[:, :-1] *= 255
    cbartmp = cbartmp.astype(np.uint8)
    im2 = cbartmp[im2]
    im2[:, :, 3] = np.logical_not(img.mask)*254+1

    return im2


def histeq(img, nbr_bins=2048):
    """ Histogram Equalization """
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


def histcomp(img, nbr_bins=256):
    """ Histogram Compaction """
# get image histogram
    tmp = img.compressed()
#        if tmp.mask.size > 1:
#            tmp = tmp.data[tmp.mask == 0]
    imhist, bins = np.histogram(tmp, nbr_bins)

    cdf = imhist.cumsum()  # cumulative distribution function
    cdf = cdf / float(cdf[-1])  # normalize

    perc = 5
    perc = perc/100.

    sindx = np.arange(nbr_bins)[cdf > perc][0]
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

# use linear interpolation of cdf to find new pixel values
    return img2
    