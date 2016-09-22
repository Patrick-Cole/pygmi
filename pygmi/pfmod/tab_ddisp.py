# -----------------------------------------------------------------------------
# Name:        tab_ddisp.py (part of PyGMI)
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
""" Data Display Routines found on Data Display Tab"""

from PyQt4 import QtGui, QtCore
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as clrs
from matplotlib.backends.backend_qt4agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT


class DataDisplay(object):
    """ Widget class to call the main interface """
    def __init__(self, parent):
        self.parent = parent
        self.lmod1 = parent.lmod1
        self.grid_stretch = 'linear'
        self.grid1 = self.lmod1.griddata['Calculated Magnetics']
        self.grid2 = self.lmod1.griddata['Calculated Gravity']
        self.grid1txt = 'Calculated Magnetics'
        self.grid2txt = 'Calculated Gravity'

        self.userint = QtGui.QWidget()
        self.mmc = MyMplCanvas(len(self.lmod1.custprofx))
        self.mpl_toolbar = NavigationToolbar2QT(self.mmc, self.userint)

        self.ddisp_plot = self.mmc
        self.sb_profnum = QtGui.QSpinBox()
        self.label_profile_xy = QtGui.QLabel()
        self.hslider_profile = QtGui.QSlider()
        self.hslider_grid = QtGui.QSlider()
        self.combo_grid1 = QtGui.QComboBox()
        self.combo_grid2 = QtGui.QComboBox()
        self.rb_ew = QtGui.QRadioButton()
        self.rb_ns = QtGui.QRadioButton()

        self.setupui()

    def setupui(self):
        """ Setup UI """
        gtmp = ['Calculated Magnetics', 'Calculated Gravity']
        gridlayout = QtGui.QGridLayout(self.userint)
        groupbox = QtGui.QGroupBox()
        verticallayout = QtGui.QVBoxLayout(groupbox)

        sizepolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred,
                                       QtGui.QSizePolicy.Fixed)

        self.label_profile_xy.setSizePolicy(sizepolicy)
        self.hslider_profile.setSizePolicy(sizepolicy)
        self.hslider_profile.setOrientation(QtCore.Qt.Horizontal)
        self.hslider_grid.setOrientation(QtCore.Qt.Horizontal)
        self.rb_ew.setChecked(True)
        self.sb_profnum.setFixedWidth(220)
        self.sb_profnum.setMaximum(self.lmod1.numy-1)
        self.combo_grid1.addItems(gtmp)
        self.combo_grid1.setCurrentIndex(0)  # set to mag
        self.combo_grid2.addItems(gtmp)
        self.combo_grid2.setCurrentIndex(1)  # set to grav

        self.sb_profnum.setPrefix("Profile: ")
        self.label_profile_xy.setText("Easting:")
        groupbox.setTitle("Profile Orientation")
        self.rb_ew.setText("Profile is W-E")
        self.rb_ns.setText("Profile is S-N")

        verticallayout.addWidget(self.rb_ew)
        verticallayout.addWidget(self.rb_ns)

        gridlayout.addWidget(self.mpl_toolbar, 0, 0, 1, 1)
        gridlayout.addWidget(self.ddisp_plot, 1, 0, 8, 1)
        gridlayout.addWidget(self.hslider_grid, 9, 0, 1, 1)
        gridlayout.addWidget(self.sb_profnum, 0, 1, 1, 1)
        gridlayout.addWidget(self.label_profile_xy, 1, 1, 1, 1)
        gridlayout.addWidget(self.hslider_profile, 2, 1, 1, 1)
        gridlayout.addWidget(groupbox, 3, 1, 1, 1)
        gridlayout.addWidget(self.combo_grid2, 4, 1, 1, 1)
        gridlayout.addWidget(self.combo_grid1, 5, 1, 1, 1)

    # Buttons
        self.combo_grid1.currentIndexChanged.connect(self.combo1)
        self.combo_grid2.currentIndexChanged.connect(self.combo2)
        self.hslider_grid.sliderMoved.connect(self.hs_grid)
        self.hslider_profile.sliderMoved.connect(self.profnum)
        self.sb_profnum.valueChanged.connect(self.profnum)
        self.rb_ew.clicked.connect(self.prof_dir)
        self.rb_ns.clicked.connect(self.prof_dir)

    def combo1(self):
        """ Combo box to choose grid 1 """
        self.mpl_toolbar.home()
        ctxt = str(self.combo_grid1.currentText())

        reg = 0
        if ctxt == 'Calculated Gravity':
            reg = self.lmod1.gregional

        self.grid1txt = ctxt
        self.grid1 = self.lmod1.griddata[ctxt]

        self.mmc.init_grid1(self.grid1, reg, ctxt)

    def combo2(self):
        """ Combo box to choose grid 2 """
        self.mpl_toolbar.home()
        ctxt = str(self.combo_grid2.currentText())

        reg = 0
        if ctxt == 'Calculated Gravity':
            reg = self.lmod1.gregional

        self.grid2txt = ctxt
        self.grid2 = self.lmod1.griddata[ctxt]
        self.mmc.init_grid2(self.grid2, reg, ctxt)
    # Needed to force the drawing of colorbar
        QtGui.QApplication.processEvents()
        self.hs_grid()

    def hs_grid(self):
        """ Horizontal slider used to show grid1 and grid2 """
        ctxt = str(self.combo_grid1.currentText())

        self.grid1 = self.lmod1.griddata[ctxt]
        hsgval = self.hslider_grid.sliderPosition()
        perc = hsgval / float(self.hslider_grid.maximum())
        self.mmc.slide_grid1(perc)

    def prof_dir(self):
        """ Radio button for EW or NS profiles """
        self.sb_profnum.setValue(0.0)

        if self.rb_ew.isChecked():
            self.sb_profnum.setMaximum(self.lmod1.numy-1)
            self.hslider_profile.setMaximum(self.lmod1.numy-1)
            self.lmod1.is_ew = True
        elif self.rb_ns.isChecked():
            self.sb_profnum.setMaximum(self.lmod1.numx-1)
            self.hslider_profile.setMaximum(self.lmod1.numx-1)
            self.lmod1.is_ew = False

    def profnum(self):
        """ Routine to change a profile from spinbox"""

        if self.hslider_profile.isSliderDown():
            self.sb_profnum.setValue(self.hslider_profile.sliderPosition())
        else:
            self.hslider_profile.setValue(self.sb_profnum.value())

        self.lmod1.curprof = self.sb_profnum.value()

        xrng = np.array(self.lmod1.xrange)
        yrng = np.array(self.lmod1.yrange)

        ytmp = [self.lmod1.curprof*self.lmod1.dxy+self.lmod1.dxy/2+yrng[0]]
        xtmp = [self.lmod1.curprof*self.lmod1.dxy+self.lmod1.dxy/2+xrng[0]]

        xtmp = np.array([xtmp, xtmp])
        ytmp = np.array([ytmp, ytmp])

        if self.rb_ew.isChecked():
            xys = self.lmod1.yrange[0]+self.lmod1.curprof*self.lmod1.dxy
            self.label_profile_xy.setText('Northing: '+str(xys))
            self.mmc.init_line(self.lmod1.xrange, [xys, xys], self.lmod1)
        elif self.rb_ns.isChecked():
            xys = self.lmod1.xrange[0]+self.lmod1.curprof*self.lmod1.dxy
            self.label_profile_xy.setText('Easting: '+str(xys))
            self.mmc.init_line([xys, xys], self.lmod1.yrange, self.lmod1)

    def update_combos(self):
        """ Update the combos """
        tmp = list(self.lmod1.griddata.keys())

        self.combo_grid1.blockSignals(True)
        self.combo_grid2.blockSignals(True)

        self.combo_grid1.clear()
        self.combo_grid1.addItems(tmp)
        self.combo_grid1.setCurrentIndex(tmp.index(self.grid1txt))

        self.combo_grid2.clear()
        self.combo_grid2.addItems(tmp)
        self.combo_grid2.setCurrentIndex(tmp.index(self.grid2txt))

        self.combo_grid1.blockSignals(False)
        self.combo_grid2.blockSignals(False)

    def tab_activate(self):
        """ Runs when the tab is activated """
        self.lmod1 = self.parent.lmod1
        self.mmc.set_limits(self.lmod1)
        self.mmc.update_ncust(len(self.lmod1.custprofx))
        self.update_combos()
        self.combo2()
        self.combo1()

        if self.rb_ew.isChecked():
            self.sb_profnum.setMaximum(self.lmod1.numy-1)
            self.hslider_profile.setMaximum(self.lmod1.numy-1)
            self.lmod1.is_ew = True
        elif self.rb_ns.isChecked():
            self.sb_profnum.setMaximum(self.lmod1.numx-1)
            self.hslider_profile.setMaximum(self.lmod1.numx-1)
            self.lmod1.is_ew = False
        self.sb_profnum.setValue(self.lmod1.curprof)
        self.profnum()


class MyMplCanvas(FigureCanvas):
    """
    Canvas for the actual plot

    Attributes
    ----------
    """
    def __init__(self, ncust=1):
        # figure stuff
        fig = Figure()

        self.cbar = cm.jet
        self.gmode = None
        self.xlims = None
        self.ylims = None

        self.axes = fig.add_subplot(111)
        self.axes.xaxis.set_label_text("Eastings (m)")
        self.axes.yaxis.set_label_text("Northings (m)")

        FigureCanvas.__init__(self, fig)

#        FigureCanvas.setSizePolicy(self,
#                                   QtGui.QSizePolicy.Expanding,
#                                   QtGui.QSizePolicy.Expanding)
#        FigureCanvas.updateGeometry(self)

        dat = np.zeros([100, 100])

        self.ims2 = self.axes.imshow(dat, cmap=self.cbar,
                                     interpolation='nearest')
        self.ims = self.axes.imshow(self.cbar(dat), interpolation='nearest')
        self.ims2.set_clim(0, 1)
        self.ibar = self.figure.colorbar(self.ims, fraction=0.025, pad=0.1)
        self.ibar2 = self.figure.colorbar(self.ims2, cax=self.ibar.ax.twinx())
        self.ibar.ax.set_aspect('auto')
        self.ibar.set_label('')
        self.ibar2.set_label('')
        self.ibar.ax.yaxis.set_label_position('left')

#        self.ibar2.ax.yaxis.set_ticks_position('left')

        self.figure.canvas.draw()
        self.bbox = self.figure.canvas.copy_from_bbox(self.axes.bbox)
        self.lbbox = self.figure.canvas.copy_from_bbox(self.axes.bbox)

        self.prf = self.axes.plot([0, 1], [0, 1])
        for _ in range(ncust):
            self.prf += self.axes.plot([0, 1], [0, 1])

        self.figure.canvas.draw()

        self.gcol1 = self.cbar(dat)

    def update_ncust(self, ncust):
        """ Updates mumber of custom profiles """
        diff = ncust-(len(self.prf)-1)
        if diff < 1:
            return
        for _ in range(diff):
            self.prf += self.axes.plot([0, 1], [0, 1])

    def init_grid1(self, dat1, reg=0, lbl=''):
        """ Updates the upper single color map """
        try:
            if dat1.units != '':
                lbl += ' ('+dat1.units+')'
        except AttributeError:
            pass
        self.ibar.set_label(lbl)

        dmin = dat1.data.min()+reg
        dmax = dat1.data.max()+reg
        if dmin == dmax:
            dmax = dmin+1
        self.ims.set_clim(dmin, dmax)

#        cnorm = clrs.Normalize()(dat1.data[::-1]-reg)
        cnorm = clrs.Normalize()(dat1.data+reg)
        if cnorm.mask.size == 1:
            cnorm.mask = (cnorm.mask*np.ones_like(cnorm.data)).astype(bool)
        cnorm.data[cnorm.mask] = 0.0
        tmp = self.cbar(cnorm)

        self.ims.set_data(tmp)
        self.ims.set_extent(self.dat_extent(dat1))
        if self.xlims is not None:
            self.axes.set_xlim(self.xlims)
            self.axes.set_ylim(self.ylims)

        self.figure.canvas.draw()

    def init_grid2(self, dat2, reg=0, lbl=''):
        """ Updates the lower single color map. It has to disable the upper map
        while doing so - in order to update the blit """
        try:
            if dat2.units != '':
                lbl += ' ('+dat2.units+')'
        except AttributeError:
            pass
        self.ibar2.set_label(lbl)

        for i in self.prf:
            i.set_visible(False)
        self.ims.set_visible(False)
#        self.ims2.set_data(dat2.data[::-1]+reg)
        self.ims2.set_data(dat2.data+reg)
        self.ims2.set_extent(self.dat_extent(dat2))
        if self.xlims is not None:
            self.axes.set_xlim(self.xlims)
            self.axes.set_ylim(self.ylims)

        dmin = dat2.data.min()+reg
        dmax = dat2.data.max()+reg
        if dmin == dmax:
            dmax = dmin+1
        self.ims2.set_clim(dmin, dmax)

        self.figure.canvas.draw()
        self.bbox = self.figure.canvas.copy_from_bbox(self.axes.bbox)
        self.ims.set_visible(True)
        self.lbbox = self.figure.canvas.copy_from_bbox(self.axes.bbox)
        for i in self.prf:
            i.set_visible(True)

    def slide_grid1(self, perc=0.0):
        """ Slider """
        self.ims.set_alpha(perc)
        self.figure.canvas.restore_region(self.bbox)
        self.axes.draw_artist(self.ims)
#        self.figure.canvas.blit(self.axes.bbox)
        self.figure.canvas.update()

        self.lbbox = self.figure.canvas.copy_from_bbox(self.axes.bbox)
        for i in self.prf:
            self.axes.draw_artist(i)
#        self.figure.canvas.blit(self.axes.bbox)
        self.figure.canvas.update()

    def init_line(self, xrng, yrng, lmod):
        """ Updates the line position """
        kmdiv = 1.
        if self.axes.xaxis.get_label_text().find('km') > -1:
            xrng = [xrng[0]/1000., xrng[1]/1000.]
            yrng = [yrng[0]/1000., yrng[1]/1000.]
            kmdiv = 1000.

        self.prf[0].set_data([xrng, yrng])
        for i in range(len(lmod.custprofx)):
            self.prf[i+1].set_data([[lmod.custprofx[i][0]/kmdiv,
                                     lmod.custprofx[i][1]/kmdiv],
                                    [lmod.custprofy[i][0]/kmdiv,
                                     lmod.custprofy[i][1]/kmdiv]])
        self.figure.canvas.restore_region(self.lbbox)
        self.axes.draw_artist(self.prf[0])
        for i in range(len(lmod.custprofx)):
            self.axes.draw_artist(self.prf[i+1])
        self.figure.canvas.update()
#        self.figure.canvas.blit(self.axes.bbox)

    def dat_extent(self, dat):
        """ Gets the extent of the dat variable """
        left = dat.tlx
        top = dat.tly
        right = left + dat.cols*dat.xdim
        bottom = top - dat.rows*dat.ydim

#        if (right-left) > 10000 or (top-bottom) > 10000:
#            self.axes.xaxis.set_label_text("Eastings (km)")
#            self.axes.yaxis.set_label_text("Northings (km)")
#            left /= 1000.
#            right /= 1000.
#            top /= 1000.
#            bottom /= 1000.
#        else:
#            self.axes.xaxis.set_label_text("Eastings (m)")
#            self.axes.yaxis.set_label_text("Northings (m)")

        self.axes.xaxis.set_label_text("Eastings (m)")
        self.axes.yaxis.set_label_text("Northings (m)")

        return (left, right, bottom, top)

    def set_limits(self, lmod):
        """ Sets limits for the axes """
        left, right = lmod.xrange
        bottom, top = lmod.yrange
#        if (right-left) > 10000 or (top-bottom) > 10000:
#            left /= 1000.
#            right /= 1000.
#            top /= 1000.
#            bottom /= 1000.

        self.xlims = (left, right)
        self.ylims = (bottom, top)
        self.axes.set_xlim(self.xlims)
        self.axes.set_ylim(self.ylims)
