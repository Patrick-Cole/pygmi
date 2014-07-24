# -----------------------------------------------------------------------------
# Name:        tab_prof.py (part of PyGMI)
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
""" Profile Display Tab Routines """

# pylint: disable=E1101
from PySide import QtGui, QtCore
import numpy as np
import scipy.interpolate as si
import pygmi.pfmod.misc as misc
import pygmi.raster.iodefs as ir
import os

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as \
    FigureCanvas
from matplotlib.backends.backend_qt4agg import \
    NavigationToolbar2QTAgg as NavigationToolbar


class ProfileDisplay(object):
    """ Widget class to call the main interface """
    def __init__(self, parent):
        self.parent = parent
        self.lmod1 = parent.lmod1
        self.lmod2 = parent.lmod2
        self.showtext = parent.showtext
        self.pbar = self.parent.pbar_sub

        mainwindow = QtGui.QWidget()

        self.mmc = MyMplCanvas(self, self.lmod1)
        self.mpl_toolbar = NavigationToolbar(self.mmc, mainwindow)

        self.userint = mainwindow
        self.toolboxpage1 = QtGui.QWidget()
        self.groupbox = QtGui.QGroupBox(self.toolboxpage1)

        self.gridlayout = QtGui.QGridLayout(mainwindow)
        self.sb_profnum2 = QtGui.QSpinBox(mainwindow)
        self.hslider_profile2 = QtGui.QSlider(mainwindow)
        self.combo_profpic = QtGui.QComboBox(mainwindow)
        self.hs_ppic_opacity = QtGui.QSlider(mainwindow)
        self.toolbox = QtGui.QToolBox(mainwindow)

        self.verticallayout = QtGui.QVBoxLayout(self.groupbox)
        self.rb_axis_datamax = QtGui.QRadioButton(self.groupbox)
        self.rb_axis_profmax = QtGui.QRadioButton(self.groupbox)
        self.rb_axis_calcmax = QtGui.QRadioButton(self.groupbox)

        self.groupbox3 = QtGui.QGroupBox(self.toolboxpage1)
        self.sb_profile_linethick = QtGui.QSpinBox(self.toolboxpage1)
        self.gridlayout_20 = QtGui.QGridLayout(self.toolboxpage1)
        self.lw_prof_defs = QtGui.QListWidget(self.toolboxpage1)

        self.pb_prof_rcopy = QtGui.QPushButton(self.toolboxpage1)
        self.pb_lbound = QtGui.QPushButton(self.toolboxpage1)
        self.pb_export_csv = QtGui.QPushButton(self.toolboxpage1)

        self.gridlayout4 = QtGui.QGridLayout(self.groupbox3)
        self.rb_magnetic = QtGui.QRadioButton(self.groupbox3)
        self.rb_gravity = QtGui.QRadioButton(self.groupbox3)

        self.setupui()

    def setupui(self):
        """ Setup UI """
        self.sb_profnum2.setWrapping(True)
        self.sb_profnum2.setMaximum(999999999)
        self.gridlayout.addWidget(self.sb_profnum2, 0, 1, 1, 1)
        sizepolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred,
                                       QtGui.QSizePolicy.Fixed)
        self.hslider_profile2.setSizePolicy(sizepolicy)
        self.hslider_profile2.setOrientation(QtCore.Qt.Horizontal)
        self.gridlayout.addWidget(self.hslider_profile2, 1, 1, 1, 1)
        self.gridlayout.addWidget(self.combo_profpic, 2, 1, 1, 1)
#        self.combo_profpic.hide()
        self.hs_ppic_opacity.setSizePolicy(sizepolicy)
        self.hs_ppic_opacity.setMaximum(255)
        self.hs_ppic_opacity.setProperty("value", 255)
        self.hs_ppic_opacity.setOrientation(QtCore.Qt.Horizontal)
        self.hs_ppic_opacity.setTickPosition(QtGui.QSlider.TicksAbove)
        self.gridlayout.addWidget(self.hs_ppic_opacity, 3, 1, 1, 1)
#        self.hs_ppic_opacity.hide()
        self.toolbox.setMaximumSize(QtCore.QSize(220, 16777215))
#        sizepolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum,
#                                       QtGui.QSizePolicy.Preferred)
#        self.toolbox.setSizePolicy(sizepolicy)
#        self.toolboxpage1.setGeometry(QtCore.QRect(0, 0, 200, 646))
        self.gridlayout4.addWidget(self.rb_magnetic, 1, 0, 1, 1)
        self.gridlayout4.addWidget(self.rb_gravity, 4, 0, 1, 1)
        self.verticallayout.addWidget(self.rb_axis_datamax)
        self.verticallayout.addWidget(self.rb_axis_profmax)
        self.verticallayout.addWidget(self.rb_axis_calcmax)
        self.sb_profile_linethick.setMinimum(1)
        self.sb_profile_linethick.setMaximum(1000)

        self.gridlayout_20.addWidget(self.lw_prof_defs, 3, 0, 1, 1)
        self.gridlayout_20.addWidget(self.sb_profile_linethick, 4, 0, 1, 1)
        self.gridlayout_20.addWidget(self.groupbox3, 5, 0, 1, 1)
        self.gridlayout_20.addWidget(self.groupbox, 6, 0, 1, 1)
        self.gridlayout_20.addWidget(self.pb_prof_rcopy, 7, 0, 1, 1)
        self.gridlayout_20.addWidget(self.pb_lbound, 8, 0, 1, 1)
        self.gridlayout_20.addWidget(self.pb_export_csv, 9, 0, 1, 1)
        self.toolbox.addItem(self.toolboxpage1, "")
        self.gridlayout.addWidget(self.toolbox, 5, 1, 1, 1)
        self.gridlayout.addWidget(self.mpl_toolbar, 0, 0, 1, 1)
        self.gridlayout.addWidget(self.mmc, 1, 0, 5, 1)

        self.sb_profnum2.setPrefix("Profile: ")
        self.groupbox3.setTitle("Profile Data")
        self.rb_magnetic.setText("Magnetic Data")
        self.rb_gravity.setText("Gravity Data")
        self.groupbox.setTitle("Profile Y-Axis Scale")
        self.rb_axis_datamax.setText("Scale to dataset maximum")
        self.rb_axis_profmax.setText("Scale to profile maximum")
        self.rb_axis_calcmax.setText("Scale to calculated maximum")
        self.sb_profile_linethick.setPrefix("Line Thickness: ")
        self.toolbox.setItemText(self.toolbox.indexOf(self.toolboxpage1),
                                 "General")
        self.pb_prof_rcopy.setText("Ranged Copy")
        self.pb_lbound.setText("Add Lithological Boundary")
        self.pb_export_csv.setText("Export Profile")

    # Buttons etc
        self.rb_axis_datamax.setChecked(True)
        self.rb_gravity.setChecked(True)
        self.sb_profile_linethick.valueChanged.connect(self.width)
        self.lw_prof_defs.currentItemChanged.connect(self.change_defs)
        self.pb_prof_rcopy.clicked.connect(self.rcopy)
        self.rb_gravity.clicked.connect(self.rb_mag_grav)
        self.rb_magnetic.clicked.connect(self.rb_mag_grav)
        self.pb_lbound.clicked.connect(self.lbound)
        self.hslider_profile2.valueChanged.connect(self.hprofnum)
        self.sb_profnum2.valueChanged.connect(self.sprofnum)
        self.rb_axis_calcmax.clicked.connect(self.rb_plot_scale)
        self.rb_axis_profmax.clicked.connect(self.rb_plot_scale)
        self.rb_axis_datamax.clicked.connect(self.rb_plot_scale)
        self.pb_export_csv.clicked.connect(self.export_csv)
        self.hs_ppic_opacity.valueChanged.connect(self.profpic_hs)
        self.combo_profpic.currentIndexChanged.connect(self.profpic_hs)

    def change_defs(self):
        """ List box in profile tab for definitions """

        i = self.lw_prof_defs.currentRow()
        if i == -1:
            misc.update_lith_lw(self.lmod1, self.lw_prof_defs)
            i = 0
        itxt = self.lw_prof_defs.item(i).text()

        if itxt not in self.lmod1.lith_list.keys():
            return

        lith = self.lmod1.lith_list[itxt]
        self.mmc.curmodel = lith.lith_index

    def change_model(self, slide=False):
        """ Change Model """

        bottom = self.lmod1.zrange[0]
        top = self.lmod1.zrange[1]

    # First we plot the model stuff
        newprof = self.lmod1.curprof
        if self.lmod1.is_ew:
#            gtmp = self.lmod1.lith_index[:, newprof,:].T.copy()
            gtmp = self.lmod1.lith_index[:, newprof, ::-1].T.copy()
            left = self.lmod1.xrange[0]
            right = self.lmod1.xrange[1]
        else:
#            gtmp = self.lmod1.lith_index[newprof,:,:].T.copy()
            gtmp = self.lmod1.lith_index[newprof, :, ::-1].T.copy()
            left = self.lmod1.yrange[0]
            right = self.lmod1.yrange[1]

        extent = (left, right, bottom, top)

        ctxt = self.combo_profpic.currentText()
        if len(self.lmod1.profpics) > 0 and ctxt != u'':
            gtmpl = self.lmod1.profpics[ctxt]
            opac = self.hs_ppic_opacity.value()/self.hs_ppic_opacity.maximum()
        else:
            gtmpl = None
            opac = 1.0

        if slide is True:
            self.mmc.slide_grid(gtmp, gtmpl, opac)
        else:
            self.mmc.init_grid(gtmp, extent, gtmpl, opac)

        alt = self.lmod1.zrange[1]-self.lmod1.curlayer*self.lmod1.d_z

        if self.lmod1.is_ew:
            self.mmc.update_line(self.lmod1.xrange, [alt, alt])
        else:
            self.mmc.update_line(self.lmod1.yrange, [alt, alt])

    def export_csv(self):
        """ Export Profile to csv """
        self.parent.pbars.resetall()
        filename = QtGui.QFileDialog.getSaveFileName(
            self.parent, 'Save File', '.', 'Comma separated values (*.csv)')[0]
        if filename == '':
            return
        os.chdir(filename.rpartition('/')[0])

        maggrid = self.lmod1.griddata['Calculated Magnetics'].data
        grvgrid = self.lmod1.griddata['Calculated Gravity'].data
        curprof = self.lmod1.curprof

        if self.lmod1.is_ew:
            cmag = maggrid[-curprof-1]
            cgrv = grvgrid[-curprof-1]
            xrng = (np.arange(cmag.shape[0])*self.lmod1.dxy +
                    self.lmod1.xrange[0]+self.lmod1.dxy/2.)
            yrng = np.zeros_like(xrng)+self.lmod1.dxy/2.

        else:
            cmag = maggrid[::-1, curprof]
            cgrv = grvgrid[::-1, curprof]
            yrng = (np.arange(cmag.shape[0])*self.lmod1.dxy +
                    self.lmod1.yrange[0]+self.lmod1.dxy/2.)
            xrng = np.zeros_like(yrng)+self.lmod1.dxy/2.

        newdata = np.transpose([xrng, yrng, cgrv, cmag])

        fno = open(filename, 'wb')
        fno.write(b'"x","y","Gravity","Magnetics"\n')
        np.savetxt(fno, newdata, delimiter=',')
        fno.close()
        self.parent.pbars.incr()
        self.showtext('Profile save complete')

    def lbound_dialog(self):
        """ Main routine to perform actual ranged copy """
        lmod1 = self.lmod1
        lbnd = LithBound()
        lw_lithupper = lbnd.lw_lithupper
        lw_lithlower = lbnd.lw_lithlower

        lw_lithupper.addItem(r'Do Not Change')
        lw_lithlower.addItem(r'Do Not Change')
        for i in lmod1.lith_list.keys():
            lw_lithupper.addItem(i)
            lw_lithlower.addItem(i)

        lw_lithupper.setCurrentItem(lw_lithupper.item(0))
        lw_lithlower.setCurrentItem(lw_lithlower.item(1))

        tmp = lbnd.exec_()
        if tmp == 0:
            return

        tupper = lw_lithupper.selectedItems()
        tlower = lw_lithlower.selectedItems()

        if tupper[0].text() == r'Do Not Change':
            lithupper = -999
        else:
            lithupper = lmod1.lith_list[tupper[0].text()].lith_index

        if tlower[0].text() == r'Do Not Change':
            lithlower = -999
        else:
            lithlower = lmod1.lith_list[tlower[0].text()].lith_index

        return lithlower, lithupper

    def lbound(self):
        """ Insert a lithological boundary """
        curgrid = self.load_data()
        if curgrid is None:
            return

        self.update_model()
        lowerb, upperb = self.lbound_dialog()

        if lowerb == -999 and upperb == -999:
            return

        cols = curgrid.cols
        rows = curgrid.rows
        tlx = curgrid.tlx
        tly = curgrid.tly
        d_x = curgrid.xdim
        d_y = curgrid.ydim
        regz = self.lmod1.zrange[1]
        d_z = self.lmod1.d_z

        gxrng = np.array([tlx+i*d_x for i in range(cols)])
        gyrng = np.array([(tly-(rows-1)*d_y)+i*d_y for i in range(rows)])

# This section gets rid of null values quickly
        xt, yt = np.meshgrid(gxrng, gyrng)
        zt = curgrid.data.data
        msk = np.logical_not(np.isnan(zt))
        zt = zt[msk]
        xy = np.transpose([xt[msk], yt[msk]])
        xy2 = np.transpose([xt, yt])
        newgrid = np.transpose(si.griddata(xy, zt, xy2, 'nearest'))

# Back to splines
        fgrid = si.RectBivariateSpline(gyrng, gxrng, newgrid)

        for i in range(self.lmod1.numx):
            for j in range(self.lmod1.numy):
                imod = i*self.lmod1.dxy+self.lmod1.xrange[0]
                jmod = j*self.lmod1.dxy+self.lmod1.yrange[0]

                igrd = int((imod-tlx)/d_x)
                jgrd = int((tly-jmod)/d_y)

                if igrd >= 0 and jgrd >= 0 and igrd < cols and jgrd < rows:
#                    k_2 = int((regz-fgrid.ev(jmod, imod))/d_z)
                    k_2 = int((regz-fgrid(jmod, imod))/d_z)
                    if k_2 < 0:
                        k_2 = 0
                    lfilt = self.lmod1.lith_index[i, j, k_2:] != -1
                    ufilt = self.lmod1.lith_index[i, j, :k_2] != -1
                    if lowerb != -999:
                        self.lmod1.lith_index[i, j, k_2:][lfilt] = lowerb
                    if upperb != -999:
                        self.lmod1.lith_index[i, j, :k_2][ufilt] = upperb

        self.change_model()
        self.update_plot()

    def load_data(self):
        """ Used to load Layer data"""
        self.pbar.setMaximum(100)
        self.pbar.setValue(0)

        dtmp = ir.ImportData()
        tmp = dtmp.settings()
        if tmp is False:
            return None
        data = dtmp.outdata['Raster']

        data[0].data = data[0].data
        data[0].data = data[0].data[::-1]  # need this reversed

        self.pbar.setValue(100)
        return data[0]

    def profpic_hs(self):
        """ Horizontal slider to change the profile """
        self.update_model()
        self.change_model(slide=True)
        self.update_plot(slide=True)

    def sprofnum(self):
        """ Routine to change a profile from spinbox"""
        self.hslider_profile2.setValue(self.sb_profnum2.value())
        self.profnum()

    def hprofnum(self):
        """ Routine to change a profile from spinbox"""
        self.sb_profnum2.setValue(self.hslider_profile2.sliderPosition())
        self.profnum()

    def profnum(self):
        """ Routine to change a profile from spinbox"""
        self.update_model()

        self.lmod1.curprof = self.sb_profnum2.value()
        self.change_model(slide=True)
        self.update_plot(slide=True)

    def rcopy(self):
        """ Do a ranged copy on a profile """
        self.update_model()
        misc.rcopy_dialog(self.lmod1, islayer=False,
                          is_ew=self.lmod1.is_ew)

    def update_model(self):
        """ Update model itself """
        if self.lmod1.is_ew:
            tmp = self.lmod1.lith_index[:, self.lmod1.curprof, ::-1]
        else:
            tmp = self.lmod1.lith_index[self.lmod1.curprof, :, ::-1]

        if tmp.shape != self.mmc.mdata.T.shape:
            return

        if self.lmod1.is_ew:
            self.lmod1.lith_index[:, self.lmod1.curprof, ::-1] = (
                self.mmc.mdata.T.copy())
        else:
            self.lmod1.lith_index[self.lmod1.curprof, :, ::-1] = (
                self.mmc.mdata.T.copy())

    def rb_plot_scale(self):
        """ plot scale """
        self.change_model()
        self.update_plot()
        self.mpl_toolbar.update()

    def rb_mag_grav(self):
        """ Used to change the magnetic and gravity radiobutton """
        self.update_model()
        self.change_model()
        self.update_plot()

    def update_plot(self, slide=False):
        """ Update the profile on the model view """

# Display the calculated profile
        data = None
        if self.rb_magnetic.isChecked():
            if 'Calculated Magnetics' in self.lmod1.griddata.keys():
                data = self.lmod1.griddata['Calculated Magnetics'].data
#            data2 = self.lmod2.griddata['Calculated Magnetics'].data
            self.mmc.ptitle = 'Magnetic Intensity: '
            self.mmc.punit = 'nT'
        else:
            if 'Calculated Gravity' in self.lmod1.griddata.keys():
                data = self.lmod1.griddata['Calculated Gravity'].data
#            data2 = self.lmod2.griddata['Calculated Gravity'].data
            self.mmc.ptitle = 'Gravity: '
            self.mmc.punit = 'mGal'

#        if data2.shape != data.shape:
#            data2 = np.zeros_like(data)
#        alt = self.lmod1.zrange[1]-self.lmod1.curlayer*self.lmod1.d_z

        if self.lmod1.is_ew and data is not None:
            self.mmc.ptitle += 'West to East'
            self.mmc.xlabel = "Eastings (m)"

            tmpprof = data[self.lmod1.numy-1-self.lmod1.curprof, :]
#            tmpprof2 = data2[self.lmod1.curprof, :]
            tmprng = (np.arange(tmpprof.shape[0])*self.lmod1.dxy +
                      self.lmod1.xrange[0]+self.lmod1.dxy/2.)
            self.sb_profnum2.setMaximum(self.lmod1.numy-1)
#            self.mmc.update_line(self.lmod1.xrange, [alt, alt])
            extent = self.lmod1.xrange

        elif not(self.lmod1.is_ew) and data is not None:
            self.mmc.ptitle += 'South to North'
            self.mmc.xlabel = "Northings (m)"

            tmpprof = data[::-1, self.lmod1.curprof]
#            tmpprof2 = data2[:, self.lmod1.curprof]
            tmprng = (np.arange(tmpprof.shape[0])*self.lmod1.dxy +
                      self.lmod1.yrange[0]+self.lmod1.dxy/2.)
            self.sb_profnum2.setMaximum(self.lmod1.numx-1)
#            self.mmc.update_line(self.lmod1.yrange, [alt, alt])
            extent = self.lmod1.yrange

        if self.rb_axis_datamax.isChecked():
            extent = list(extent)+[data.min(), data.max()]
        else:
            extent = list(extent)+[tmpprof.min(), tmpprof.max()]

# Load in observed data - if there is any
        data2 = None
        tmprng2 = None
        tmpprof2 = None
        if ('Magnetic Dataset' in self.lmod1.griddata.keys() and
                self.rb_magnetic.isChecked()):
            data2 = self.lmod1.griddata['Magnetic Dataset']
            regtmp = 0.0
        elif ('Gravity Dataset' in self.lmod1.griddata.keys() and
                self.rb_gravity.isChecked()):
            data2 = self.lmod1.griddata['Gravity Dataset']
            regtmp = self.lmod1.gregional

        if data2 is not None:
            if self.lmod1.is_ew:
                ycrdl = self.lmod1.yrange[0]+self.lmod1.curprof*self.lmod1.dxy
#               ycrdl = self.lmod1.yrange[1]-self.lmod1.curprof*self.lmod1.dxy
                ycrd = int((data2.tly - ycrdl)/data2.ydim)
#               ycrd = data2.rows-int((data2.tly - ycrdl)/data2.ydim)

                if ycrd < 0 or ycrd >= data2.rows:
                    if slide is False:
                        self.mmc.init_plot(tmprng, tmpprof, extent, tmprng2,
                                           tmpprof2)
                    else:
                        self.mmc.slide_plot(tmprng, tmpprof, extent, tmprng2,
                                            tmpprof2)
                    return
                else:
                    tmpprof2 = data2.data[ycrd, :] - regtmp
                    tmprng2 = (data2.tlx + np.arange(data2.cols)*data2.xdim +
                               data2.xdim/2)
            else:
                xcrdl = self.lmod1.xrange[0]+self.lmod1.curprof*self.lmod1.dxy
                xcrd = int((xcrdl-data2.tlx)/data2.xdim)

                if xcrd < 0 or xcrd >= data2.cols:
                    if slide is False:
                        self.mmc.init_plot(tmprng, tmpprof, extent, tmprng2,
                                           tmpprof2)
                    else:
                        self.mmc.slide_plot(tmprng, tmpprof, extent, tmprng2,
                                            tmpprof2)
                    return
                else:
                    tmpprof2 = data2.data[::-1, xcrd] - regtmp
                    tmprng2 = (data2.tly-data2.rows*data2.ydim +
                               np.arange(data2.rows)*data2.ydim + data2.ydim/2)

            tmpprof2 = tmpprof2[tmprng2 >= tmprng[0]]  # must be before tmprng2
            tmprng2 = tmprng2[tmprng2 >= tmprng[0]]
            tmpprof2 = tmpprof2[tmprng2 <= tmprng[-1]]  # must bebefore tmprng2
            tmprng2 = tmprng2[tmprng2 <= tmprng[-1]]

            nomask = np.logical_not(tmpprof2.mask)
            tmprng2 = tmprng2[nomask]
            tmpprof2 = tmpprof2[nomask]

            if self.rb_axis_datamax.isChecked():
                extent[2:] = [data2.data.min()-regtmp, data2.data.max()-regtmp]
            elif self.rb_axis_profmax.isChecked():
                extent[2:] = [tmpprof2.min(), tmpprof2.max()]

        if slide is True:
            self.mmc.slide_plot(tmprng, tmpprof, extent, tmprng2, tmpprof2)
        else:
            self.mmc.init_plot(tmprng, tmpprof, extent, tmprng2, tmpprof2)

    def width(self, width):
        """ Sets the width of the edits on the profile view """

        self.mmc.width = width

    def tab_activate(self):
        """ Runs when the tab is activated """
        self.lmod1 = self.parent.lmod1
        self.mmc.lmod = self.lmod1

        misc.update_lith_lw(self.lmod1, self.lw_prof_defs)

        self.hslider_profile2.valueChanged.disconnect()
        self.combo_profpic.currentIndexChanged.disconnect()
        self.sb_profnum2.valueChanged.disconnect()

        self.hslider_profile2.setMinimum(0)
        if self.lmod1.is_ew:
            self.hslider_profile2.setMaximum(self.lmod1.numy-1)
        else:
            self.hslider_profile2.setMaximum(self.lmod1.numx-1)

        if len(self.lmod1.profpics) > 0:
            self.combo_profpic.clear()
            self.combo_profpic.addItems(list(self.lmod1.profpics.keys()))
            self.combo_profpic.setCurrentIndex(0)

        self.change_model()  # needs to happen before profnum set value
        self.sb_profnum2.setValue(self.lmod1.curprof)
        self.hslider_profile2.setValue(self.sb_profnum2.value())
        self.update_plot()
#        self.change_model() # needs to happen before profnum set value
        self.sb_profnum2.valueChanged.connect(self.sprofnum)
        self.hslider_profile2.valueChanged.connect(self.hprofnum)
        self.combo_profpic.currentIndexChanged.connect(self.profpic_hs)


class LithBound(QtGui.QDialog):
    """ Class to call up a dialog for lithological boundary"""
    def __init__(self, parent=None):
        QtGui.QDialog.__init__(self, parent)

        self.gridlayout = QtGui.QGridLayout(self)
        self.buttonbox = QtGui.QDialogButtonBox(self)
        self.lw_lithupper = QtGui.QListWidget(self)
        self.lw_lithlower = QtGui.QListWidget(self)
        self.label_3 = QtGui.QLabel(self)
        self.label_4 = QtGui.QLabel(self)
        self.setupui()

    def setupui(self):
        """ Setup UI """
        self.gridlayout.addWidget(self.label_3, 0, 0, 1, 1)
        self.lw_lithupper.setSelectionMode(
            QtGui.QAbstractItemView.SingleSelection)
        self.gridlayout.addWidget(self.lw_lithupper, 0, 1, 1, 1)
        self.gridlayout.addWidget(self.label_4, 1, 0, 1, 1)
        self.lw_lithlower.setSelectionMode(
            QtGui.QAbstractItemView.SingleSelection)
        self.gridlayout.addWidget(self.lw_lithlower, 1, 1, 1, 1)
        self.buttonbox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonbox.setStandardButtons(
            QtGui.QDialogButtonBox.Cancel | QtGui.QDialogButtonBox.Ok)
        self.gridlayout.addWidget(self.buttonbox, 2, 0, 1, 2)

        self.setWindowTitle("Add Lithological Boundary")
        self.label_3.setText("Lithologies Above Layer")
        self.label_4.setText("Lithologies Below Layer")

        QtCore.QObject.connect(self.buttonbox, QtCore.SIGNAL("rejected()"),
                               self.reject)
        QtCore.QObject.connect(self.buttonbox, QtCore.SIGNAL("accepted()"),
                               self.accept)
#        QtCore.QMetaObject.connectSlotsByName(Dialog)


class MyMplCanvas(FigureCanvas):
    """This is a QWidget"""
    def __init__(self, parent, lmod):
        #fig = Figure()
        fig = plt.figure()
        FigureCanvas.__init__(self, fig)

        self.parent = parent
        self.lmod = lmod
        self.cbar = plt.cm.jet
        self.curmodel = 0
        self.width = 1
        self.xold = None
        self.yold = None
        self.press = False
        self.newline = False
        self.mdata = np.zeros([10, 100])
        self.ptitle = ''
        self.punit = ''
        self.xlabel = 'Eastings (m)'
        self.plotisinit = False

# Events
        self.figure.canvas.mpl_connect('motion_notify_event', self.move)
        self.figure.canvas.mpl_connect('button_press_event', self.button_press)
        self.figure.canvas.mpl_connect('button_release_event',
                                       self.button_release)

# Initial Images
        self.paxes = fig.add_subplot(211)
        self.paxes.yaxis.set_label_text("mGal")
#        x = [1, 2, 3]
#        y = [1, 2, 3]
#        self.cal = self.paxes.plot(x, y)
#        self.obs = plt.plot(x, y, 'o')
        self.cal = self.paxes.plot([], [])
        self.obs = plt.plot([], [], 'o')

        self.axes = fig.add_subplot(212)
        self.axes.xaxis.set_label_text(self.xlabel)
        self.axes.yaxis.set_label_text("Altitude (m)")

        tmp = self.cbar(self.mdata)
        tmp[:, :, 3] = 0

        self.ims2 = plt.imshow(tmp.copy(), interpolation='nearest',
                               aspect='auto')
        self.ims = plt.imshow(tmp.copy(), interpolation='nearest',
                              aspect='auto')
        self.figure.canvas.draw()

        self.bbox = self.figure.canvas.copy_from_bbox(self.axes.bbox)
        self.pbbox = self.figure.canvas.copy_from_bbox(self.paxes.bbox)
        self.prf = self.axes.plot([0, 0])
        self.figure.canvas.draw()
        self.lbbox = self.figure.canvas.copy_from_bbox(self.axes.bbox)

    def button_press(self, event):
        """ Button press """
        nmode = self.axes.get_navigate_mode()
        if event.button == 1 and nmode is None:
            self.press = True
            self.newline = True

    def button_release(self, event):
        """ Button press """
        nmode = self.axes.get_navigate_mode()
        if event.button == 1:
            self.press = False
            if nmode == 'ZOOM':
                extent = self.axes.get_xbound()
                self.paxes.set_xbound(extent[0], extent[1])
                self.figure.canvas.draw()
            else:
                self.parent.update_model()

    def move(self, event):
        """ Mouse is moving """
        if event.inaxes == self.axes and self.press is True:
            if self.lmod.is_ew:
                col = int((event.xdata - self.lmod.xrange[0])/self.lmod.dxy)
            else:
                col = int((event.xdata - self.lmod.yrange[0])/self.lmod.dxy)
            row = int((event.ydata - self.lmod.zrange[0])/self.lmod.d_z)

            xdata = col
            ydata = row

            if self.newline is True:
                self.newline = False
                self.set_mdata(xdata, ydata)
            elif xdata != self.xold:
                mmm = float(ydata-self.yold)/(xdata-self.xold)
                ccc = ydata - mmm * xdata
                x_1 = min([self.xold, xdata])
                x_2 = max([self.xold, xdata])
                for i in range(x_1+1, x_2+1):
                    jold = int(mmm*(i-1)+ccc)
                    jnew = int(mmm*i+ccc)
                    if jold > jnew:
                        jold, jnew = jnew, jold
                    for j in range(jold, jnew+1):
                        self.set_mdata(i, j)

            elif ydata != self.yold:
                y_1 = min([self.yold, ydata])
                y_2 = max([self.yold, ydata])
                for j in range(y_1, y_2+1):
                    self.set_mdata(xdata, j)

            self.xold = xdata
            self.yold = ydata

            self.slide_grid(self.mdata)

    def set_mdata(self, xdata, ydata):
        """ Routine to 'draw' the line on mdata """
        gheight = self.mdata.shape[0]
        gwidth = self.mdata.shape[1]

        width = self.width-1  # 'pen' width
        xstart = xdata-width-1
        xend = xdata+width
        ystart = ydata-width-1
        yend = ydata+width
        if xstart < 0:
            xstart = 0
        if xend > gwidth:
            xend = gwidth
        if ystart < 0:
            ystart = 0
        if yend > gheight:
            yend = gheight

        if xstart < xend and ystart < yend:
            mtmp = self.mdata[ystart:yend, xstart:xend]
            mtmp[mtmp != -1] = self.curmodel

    def luttodat(self, dat):
        """ lut to dat grid """
        mlut = self.lmod.mlut
        tmp = np.zeros([dat.shape[0], dat.shape[1], 4])

        for i in np.unique(dat):
            if i == -1:
                ctmp = [0, 0, 0, 0]
            else:
                ctmp = np.array(mlut[i]+[255])/255.

            tmp[dat[::-1] == i] = ctmp
#            tmp[dat[:] == i] = ctmp

        return tmp

    def init_grid(self, dat, extent, dat2, opac):
        """ Updates the single color map """
        # Note that because we clear the current axes, we must put the objects
        # back into it are the graph will go blank on a draw(). This is the
        # reason for the ims and prf commands below.

#        self.axes.cla()
        self.paxes.set_xbound(extent[0], extent[1])
#        self.figure.canvas.draw()
#        QtGui.QApplication.processEvents()
#
#        tmp = self.luttodat(dat)
#        if dat2 is not None:
#            self.ims2 = plt.imshow(dat2.data, interpolation='nearest',
#                                   aspect='auto')
#            self.ims2.set_data(dat2.data)
#            self.ims2.set_extent(self.dat_extent(dat2))
#            self.ims2.set_clim(dat2.data.min(), dat2.data.max())
#
#        self.figure.canvas.draw()
#        QtGui.QApplication.processEvents()
#        self.bbox = self.figure.canvas.copy_from_bbox(self.axes.bbox)
#
#        self.ims = plt.imshow(tmp, interpolation='nearest', aspect='auto')
#        self.ims.set_extent(extent)
#        self.ims.set_alpha(opac)
#        self.figure.canvas.draw()
#        self.lbbox = self.figure.canvas.copy_from_bbox(self.axes.bbox)
#        self.axes.add_line(self.prf[0])
#
#        self.mdata = dat

        self.ims.set_visible(False)
        self.axes.set_xlim(extent[0], extent[1])
        self.axes.set_ylim(extent[2], extent[3])

        if dat2 is not None:
            self.ims2.set_data(dat2.data)
            self.ims2.set_extent(self.dat_extent(dat2))
            self.ims2.set_clim(dat2.data.min(), dat2.data.max())

        self.figure.canvas.draw()
        QtGui.QApplication.processEvents()
        self.bbox = self.figure.canvas.copy_from_bbox(self.axes.bbox)

        self.ims.set_visible(True)
        self.ims.set_extent(extent)
        tmp = self.luttodat(dat)
        self.ims.set_data(tmp)
        self.ims.set_alpha(opac)

        self.lbbox = self.figure.canvas.copy_from_bbox(self.axes.bbox)
        self.figure.canvas.draw()
        QtGui.QApplication.processEvents()

        self.mdata = dat

    def slide_grid(self, dat, dat2=None, opac=1.0):
        """ Slider """
#        self.mdata = dat
#        tmp = self.luttodat(dat)
#        self.ims.set_data(tmp)
#
#        self.figure.canvas.restore_region(self.bbox)
#        self.axes.draw_artist(self.ims)
#        self.figure.canvas.blit(self.axes.bbox)
#        self.lbbox = self.figure.canvas.copy_from_bbox(self.axes.bbox)
#        self.axes.draw_artist(self.prf[0])
#        self.figure.canvas.blit(self.axes.bbox)
        self.mdata = dat
        tmp = self.luttodat(dat)
        self.ims.set_data(tmp)
        self.ims.set_alpha(opac)
#        if dat2 is not None:
# #            self.ims2.set_data(dat2.data[::-1])
#            self.ims2.set_data(dat2.data)

        self.figure.canvas.restore_region(self.bbox)
#        self.axes.draw_artist(self.ims2)
        self.axes.draw_artist(self.ims)
#        self.figure.canvas.blit(self.axes.bbox)
        self.figure.canvas.update()

        self.lbbox = self.figure.canvas.copy_from_bbox(self.axes.bbox)
        self.axes.draw_artist(self.prf[0])
#        self.figure.canvas.blit(self.axes.bbox)
        self.figure.canvas.update()

    def update_line(self, xrng, yrng):
        """ Updates the line position """
        self.prf[0].set_data([xrng, yrng])
        self.figure.canvas.restore_region(self.lbbox)
        self.axes.draw_artist(self.prf[0])
#        self.figure.canvas.blit(self.axes.bbox)
        self.figure.canvas.update()

    def dat_extent(self, dat):
        """ Gets the extend of the dat variable """
        left = dat.tlx
        top = dat.tly
        right = left + dat.cols*dat.xdim
        bottom = top - dat.rows*dat.ydim
        return (left, right, bottom, top)

# This section is just for the profile line plot

    def extentchk(self, extent):
        """ Checks extent """
        dmin = extent[2]
        dmax = extent[3]
        if dmin == dmax:
            dmax = dmin+1
        return dmin, dmax

    def init_plot(self, xdat, dat, extent, xdat2, dat2):
        """ Updates the single color map """
        self.paxes.autoscale(False)
        dmin, dmax = self.extentchk(extent)
        self.paxes.cla()
        self.paxes.set_title(self.ptitle)
        self.axes.xaxis.set_label_text(self.xlabel)
        self.paxes.yaxis.set_label_text(self.punit)
        self.paxes.set_ylim(dmin, dmax)
        self.paxes.set_xlim(extent[0], extent[1])
        self.figure.canvas.draw()
        QtGui.QApplication.processEvents()
        self.pbbox = self.figure.canvas.copy_from_bbox(self.paxes.bbox)

        self.paxes.set_autoscalex_on(False)
        self.cal = self.paxes.plot(xdat, dat)
        if xdat2 is not None:
            self.obs = self.paxes.plot(xdat2, dat2, 'o')
        else:
            self.obs = self.paxes.plot([], [], 'o')
        self.figure.canvas.draw()
        QtGui.QApplication.processEvents()
        self.plotisinit = True

    def slide_plot(self, xdat, dat, extent, xdat2, dat2):
        """ Slider """
        dmin, dmax = self.extentchk(extent)

        self.figure.canvas.restore_region(self.pbbox)
        self.cal[0].set_data([xdat, dat])
        if xdat2 is not None:
            self.obs[0].set_data([xdat2, dat2])
        else:
            self.obs[0].set_data([[], []])
        self.paxes.set_ylim(dmin, dmax)
        self.paxes.set_xlim(extent[0], extent[1])

        self.paxes.draw_artist(self.cal[0])
        if xdat2 is not None:
            self.paxes.draw_artist(self.obs[0])
#        self.figure.canvas.blit(self.paxes.bbox)
        self.figure.canvas.update()

        QtGui.QApplication.processEvents()
