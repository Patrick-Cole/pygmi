# -----------------------------------------------------------------------------
# Name:        tab_prof.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2018 Council for Geoscience
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

import os
from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
import scipy.ndimage as ndimage
#import scipy.interpolate as si
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib import cm
from matplotlib import rcParams
from osgeo import gdal
import pandas as pd
import pygmi.raster.iodefs as ir
from pygmi.pfmod import misc
from pygmi.pfmod.grvmag3d import gridmatch
import pygmi.menu_default as menu_default
from pygmi.raster.dataprep import gdal_to_dat
from pygmi.raster.dataprep import data_to_gdal_mem

rcParams['savefig.dpi'] = 600.


class ProfileDisplay(QtWidgets.QWidget):
    """ Widget class to call the main interface """
    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent)

        self.parent = parent
        self.lmod1 = parent.lmod1
        self.showtext = parent.showtext
        self.pbar = self.parent.pbar_sub
        self.viewmagnetics = True
        self.plot_custmin = 0.
        self.plot_custmax = 50.
        self.pscale_type = 'allmax'
        self.pcntmax = len(self.lmod1.custprofx)-1
        self.lmod1.custprofx['adhoc'] = [0., 1.]
        self.lmod1.custprofy['adhoc'] = [0., 1.]
        self.extent_side = None
        self.pdxy = None
        self.ipdx1 = None
        self.ipdx2 = None
        self.xxx = None
        self.yyy = None
        self.rxxx = None
        self.ryyy = None

        self.mmc = MyMplCanvas(self)
        self.mpl_toolbar = MyToolbar(self)

        self.hs_overview = MySlider()
        self.hs_sideview = MySlider()
        self.combo_overview = QtWidgets.QComboBox()
        self.label_sideview = QtWidgets.QLabel('None')

        self.sb_layer = QtWidgets.QSpinBox()
        self.hs_layer = MySlider()
        self.sb_profnum = QtWidgets.QSpinBox()
        self.hs_profnum = MySlider()

        self.sb_profile_linethick = QtWidgets.QSpinBox()
        self.lw_prof_defs = QtWidgets.QListWidget()

        self.dial_prof_dir = GaugeWidget()
        self.sb_prof_dir = QtWidgets.QSpinBox()
        self.setupui()

    def setupui(self):
        """ Setup UI """
        sizepolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                           QtWidgets.QSizePolicy.Fixed)

        self.lw_prof_defs.setFixedWidth(220)

        pb_rcopy = QtWidgets.QPushButton('Ranged Copy')
        pb_lbound = QtWidgets.QPushButton('Add Lithological Boundary')
        pb_export_csv = QtWidgets.QPushButton('Export All Profiles (In current direction)')

        lbl_dial_value = QtWidgets.QLabel('Profile Direction: ')

        self.dial_prof_dir.setMaximum(359)
        self.sb_prof_dir.setMaximum(359)
        self.hs_sideview.setEnabled(False)

        self.sb_layer.setMaximum(self.lmod1.numz-1)
        self.sb_layer.setPrefix('Layer:')
        self.sb_layer.setWrapping(True)

        self.hs_overview.setMaximum(100)
        self.hs_overview.setProperty('value', 0)
        self.hs_overview.setOrientation(QtCore.Qt.Horizontal)

        self.hs_sideview.setMaximum(100)
        self.hs_sideview.setProperty('value', 0)
        self.hs_sideview.setOrientation(QtCore.Qt.Horizontal)

        self.hs_layer.setSizePolicy(sizepolicy)
        self.hs_layer.setOrientation(QtCore.Qt.Horizontal)

        self.hs_profnum.setSizePolicy(sizepolicy)
        self.hs_profnum.setOrientation(QtCore.Qt.Horizontal)

        self.sb_profnum.setPrefix('Profile:')
        self.sb_profnum.setWrapping(True)

        self.sb_profile_linethick.setMinimum(1)
        self.sb_profile_linethick.setMaximum(1000)
        self.sb_profile_linethick.setPrefix('Line Thickness: ')

# Set groupboxes and layouts
        gridlayout = QtWidgets.QGridLayout(self)

        hl_profnum = QtWidgets.QHBoxLayout()
        hl_profnum.addWidget(self.sb_profnum)
        hl_profnum.addWidget(self.hs_profnum)

        hl_layer = QtWidgets.QHBoxLayout()
        hl_layer.addWidget(self.sb_layer)
        hl_layer.addWidget(self.hs_layer)

        hl_pics = QtWidgets.QHBoxLayout()
        hl_pics.addWidget(self.combo_overview)
        hl_pics.addWidget(self.hs_overview)
        hl_pics.addWidget(self.label_sideview)
        hl_pics.addWidget(self.hs_sideview)

        gb_dir = QtWidgets.QGroupBox('Profile Orientation')
        hl_dir = QtWidgets.QHBoxLayout(gb_dir)
        hl_dir.addWidget(self.dial_prof_dir)
        hl_dir.addWidget(lbl_dial_value)
        hl_dir.addWidget(self.sb_prof_dir)

        vl_plots = QtWidgets.QVBoxLayout()
        vl_plots.addWidget(self.mpl_toolbar)
        vl_plots.addWidget(self.mmc)
        vl_plots.addLayout(hl_pics)

        vl_tools = QtWidgets.QVBoxLayout()
        vl_tools.addWidget(gb_dir)
        vl_tools.addLayout(hl_profnum)
        vl_tools.addLayout(hl_layer)
        vl_tools.addWidget(self.lw_prof_defs)
        vl_tools.addWidget(self.sb_profile_linethick)
        vl_tools.addWidget(pb_rcopy)
        vl_tools.addWidget(pb_lbound)
        vl_tools.addWidget(pb_export_csv)

        gridlayout.addLayout(vl_plots, 0, 0, 8, 1)
        gridlayout.addLayout(vl_tools, 0, 1, 8, 1)

    # Buttons etc
        self.sb_profile_linethick.valueChanged.connect(self.setwidth)
        self.lw_prof_defs.currentItemChanged.connect(self.change_defs)
        self.hs_profnum.valueChanged.connect(self.hprofnum)
        self.sb_profnum.valueChanged.connect(self.sprofnum)
        self.hs_layer.valueChanged.connect(self.hlayer)
        self.sb_layer.valueChanged.connect(self.slayer)

        self.hs_sideview.valueChanged.connect(self.pic_sideview)
        self.dial_prof_dir.sliderReleased.connect(self.prof_dir)
        self.sb_prof_dir.valueChanged.connect(self.sprofdir)

#        self.combo_sideview.currentIndexChanged.connect(self.pic_sideview)

        self.hs_overview.valueChanged.connect(self.pic_overview2)
        self.combo_overview.currentIndexChanged.connect(self.pic_overview)

        pb_export_csv.clicked.connect(self.export_csv)
        pb_rcopy.clicked.connect(self.rcopy)
        pb_lbound.clicked.connect(self.lbound)

    ### Misc
    def export_csv_old(self):
        """ Export Profile to csv """
        self.parent.pbars.resetall()
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.parent, 'Save File', '.', 'Comma separated values (*.csv)')
        if filename == '':
            return
        os.chdir(os.path.dirname(filename))

        xrng = (np.arange(self.lmod1.numx)*self.lmod1.dxy +
                self.lmod1.xrange[0]+self.lmod1.dxy/2.)
        yrng = (np.arange(self.lmod1.numy)*self.lmod1.dxy +
                self.lmod1.yrange[0]+self.lmod1.dxy/2.)
        yrng = yrng[::-1]
        xxx, yyy = np.meshgrid(xrng, yrng)
        xlines = np.arange(self.lmod1.numx)
        ylines = np.arange(self.lmod1.numy, 0, -1)-1
        _, lines = np.meshgrid(xlines, ylines)

        # This is only for ew
        xxx = xxx.T
        yyy = yyy.T
        lines, _ = np.meshgrid(xlines, ylines)
        lines = lines.T

        header = '"Line","x","y"'
        newdata = [lines.flatten(), xxx.flatten(), yyy.flatten()]

        for i in self.lmod1.griddata:
            if 'Calculated' not in i:
                data = gridmatch(self.lmod1, 'Calculated Magnetics', i)
            else:
                data = self.lmod1.griddata[i].data
                if 'Gravity' in i:
                    data = data + self.lmod1.gregional

            newdata.append(np.array(data.flatten()))
            header = header+',"'+i+'"'
        newdata = np.transpose(newdata)
        header = header+'\n'

        with open(filename, 'wb') as fno:
            fno.write(bytes(header, 'utf-8'))
            np.savetxt(fno, newdata, delimiter=',')

        self.parent.pbars.incr()
        self.showtext('Profile save complete')

    def export_csv(self):
        """ Export Profile to csv """
        self.parent.pbars.resetall()
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.parent, 'Save File', '.', 'Comma separated values (*.csv)')
        if filename == '':
            return
        os.chdir(os.path.dirname(filename))

        bly = self.lmod1.yrange[0]
        tlx = self.lmod1.xrange[0]
        dxy = self.lmod1.dxy

        dfall = None
        for line in range(self.sb_profnum.maximum()+1):
            self.calc_prof_limits(line)
            self.get_model()

#            px1, px2 = self.lmod1.custprofx['rotate']
            data2 = {}
            data2['LINE'] = np.zeros(self.rxxx.size, dtype=int)+line

            data2['X'] = self.rxxx*dxy+tlx
            data2['Y'] = self.ryyy*dxy+bly

            for i in self.lmod1.griddata:
                data = self.lmod1.griddata[i].data
                if 'Gravity' in i:
                    data = data + self.lmod1.gregional

                tmpprof = ndimage.map_coordinates(data.data[::-1],
                                                  [self.ryyy-0.5,
                                                   self.rxxx-0.5],
                                                  order=1, cval=np.nan)
                data2[i] = tmpprof
#                tmpprof = tmpprof[np.logical_not(np.isnan(tmpprof))]

            if dfall is None:
                dfall = pd.DataFrame(data2)
            else:
                dtmp = pd.DataFrame(data2)
                dfall = dfall.append(dtmp)

        dfall = dfall.dropna(thresh=4)
        dfall.to_csv(filename, index=False)

        self.parent.pbars.incr()
        self.showtext('Profile save complete')

    def lbound(self):
        """ Insert a lithological boundary """
        self.pbar.setMaximum(100)
        self.pbar.setValue(0)

        dtmp = ir.ImportData()
        tmp = dtmp.settings()
        if tmp is False:
            return
#        curgrid = dtmp.outdata['Raster'][0].data[::-1]
        curgrid = dtmp.outdata['Raster'][0]

        self.pbar.setValue(100)

        if curgrid is None:
            return

        lbnd = LithBound(self.lmod1)
        tmp = lbnd.exec_()
        if tmp == 0:
            return

        isdepths = lbnd.rb_depth.isChecked()

        lowerb, upperb = lbnd.get_lith()
        if lowerb == -999 and upperb == -999:
            return

        rows, cols = curgrid.data.shape
        tlx = curgrid.extent[0]
        tly = curgrid.extent[-1]
        d_x = curgrid.xdim
        d_y = curgrid.ydim
        regz = self.lmod1.zrange[1]
        d_z = self.lmod1.d_z

#        gxrng = np.array([tlx+i*d_x for i in range(cols)])
#        gyrng = np.array([(tly-(rows-1)*d_y)+i*d_y for i in range(rows)])

# This section gets rid of null values quickly
#        xt, yt = np.meshgrid(gxrng, gyrng)
        zt = curgrid.data.data

        if isdepths is True:
            zt2 = 0
            if 'DTM Dataset' in self.lmod1.griddata:
                zt2 = gridmatch2(curgrid, self.lmod1.griddata['DTM Dataset'])
            zt = zt2 - curgrid.data
            zt = zt.data

        msk = np.logical_not(np.logical_or(curgrid.data.mask, np.isnan(zt)))

#        zt = zt[msk]
#        xy1 = np.transpose([xt[msk], yt[msk]])
#        xy2 = np.transpose([xt, yt])
#        newgrid = np.transpose(si.griddata(xy1, zt, xy2, 'nearest'))

# Back to splines
#        fgrid = si.RectBivariateSpline(gyrng, gxrng, newgrid)

        for i in range(self.lmod1.numx):
            for j in range(self.lmod1.numy):
                imod = i*self.lmod1.dxy+self.lmod1.xrange[0]
                jmod = j*self.lmod1.dxy+self.lmod1.yrange[0]

                igrd = int((imod-tlx)/d_x)
                jgrd = int((tly-jmod)/d_y)

#                if igrd >= 0 and jgrd >= 0 and igrd < cols and jgrd < rows:
                if 0 <= igrd < cols and 0 <= jgrd < rows:
                    if not msk[jgrd, igrd]:
                        continue

                    k_2 = int((regz - zt[jgrd, igrd])/d_z)

#                    k_2 = int((regz-fgrid(jmod, imod))/d_z)
                    if k_2 < 0:
                        k_2 = 0
                    lfilt = self.lmod1.lith_index[i, j, k_2:] != -1
                    ufilt = self.lmod1.lith_index[i, j, :k_2] != -1
                    if lowerb != -999:
                        self.lmod1.lith_index[i, j, k_2:][lfilt] = lowerb
                    if upperb != -999:
                        self.lmod1.lith_index[i, j, :k_2][ufilt] = upperb

        gtmp = self.get_model()
        self.mmc.init_grid(gtmp)
        self.mmc.init_grid_top()
#        self.mmc.figure.canvas.draw()

    def rcopy(self):
        """ Do a ranged copy on a profile """

        rcopy = RangedCopy(self)

        tmp = rcopy.exec_()
        if tmp == 0:
            return

        if rcopy.rb_overview.isChecked():
            self.rcopy_layer(rcopy)
        else:
            self.rcopy_prof(rcopy)

        self.slayer()
        self.sprofnum()

    def rcopy_layer(self, rcopy):
        """ ranged copy on a layer """

        lithcopy = rcopy.lw_lithcopy.selectedItems()
        lithdel = rcopy.lw_lithdel.selectedItems()
        lstart = rcopy.sb_start.value()
        lend = rcopy.sb_end.value()
        lmaster = rcopy.sb_master.value()

        if lstart > lend:
            lstart, lend = lend, lstart

        viewlim = self.mmc.laxes.viewLim
        x0 = int(round((viewlim.x0 - self.lmod1.xrange[0])/self.lmod1.dxy))
        x1 = int(round((viewlim.x1 - self.lmod1.xrange[0])/self.lmod1.dxy))
        y0 = int(round((viewlim.y0 - self.lmod1.yrange[0])/self.lmod1.dxy))
        y1 = int(round((viewlim.y1 - self.lmod1.yrange[0])/self.lmod1.dxy))
        if x0 == 0:
            x0 = None
        if y0 == 0:
            y0 = None

        mtmp = self.lmod1.lith_index[x0:x1, y0:y1, lmaster]
        mslice = np.zeros_like(mtmp)

        for i in lithcopy:
            mslice[mtmp == self.lmod1.lith_list[i.text()].lith_index] = 1

        for i in range(lstart, lend+1):
            ltmp = self.lmod1.lith_index[x0:x1, y0:y1, i]

            lslice = np.zeros_like(ltmp)
            for j in lithdel:
                lslice[ltmp == self.lmod1.lith_list[j.text()].lith_index] = 1

            mlslice = np.logical_and(mslice, lslice)
            ltmp[mlslice] = mtmp[mlslice]

    def rcopy_prof(self, rcopy):
        """ ranged copy on a profile """
        lithcopy = rcopy.lw_lithcopy.selectedItems()
        lithdel = rcopy.lw_lithdel.selectedItems()
        lstart = rcopy.sb_start.value()
        lend = rcopy.sb_end.value()
        lmaster = rcopy.sb_master.value()
        if lstart > lend:
            lstart, lend = lend, lstart

        self.calc_prof_limits(lmaster)
        mtmp = self.get_model()

        # TODO: Ranged copy zoom
#        viewlim = self.mmc.axes.viewLim
#        z0 = int(round((self.lmod1.zrange[1] - viewlim.y1)/self.lmod1.d_z))
#        z1 = int(round((self.lmod1.zrange[1] - viewlim.y0)/self.lmod1.d_z))
#        if z0 == 0:
#            z0 = None
#
#        x0 = int(round((viewlim.x0 - self.extent_side[0]) /
#                       self.pdxy))
#        x1 = int(round((viewlim.x1 - self.extent_side[0]) /
#                       self.pdxy))
#        if x0 == 0:
#            x0 = None
#
#        mtmp = self.lmod1.lith_index[x0:x1, lmaster, z0:z1]
#        if lend > self.lmod1.numy:
#            lend = self.lmod1.numy

        # Create a filter for all lithologies to be copied
        mslice = np.zeros_like(mtmp)
#        lcopynums = []
        for i in lithcopy:
            mslice[mtmp == self.lmod1.lith_list[i.text()].lith_index] = 1
#            lcopynums.append(self.lmod1.lith_list[i.text()].lith_index)

        for i in range(lstart, lend+1):
            self.calc_prof_limits(i)
            ltmp = self.get_model()

            lslice = np.zeros_like(ltmp)
            for j in lithdel:
                lslice[ltmp == self.lmod1.lith_list[j.text()].lith_index] = 1

            mlslice = np.logical_and(mslice, lslice)
            ltmp[mlslice] = mtmp[mlslice]
            ltmp = ltmp[::-1]

            udatad = {}

            for ixy, xy in enumerate(np.transpose([self.xxx, self.yyy])):
                for zz in range(self.lmod1.numz):
                    ref = (xy[0], xy[1], zz)
                    if ref not in udatad:
                        udatad[ref] = []
                    udatad[ref].append(ltmp[zz, self.ipdx1+ixy])

            for i2 in udatad:
                if 0 in udatad[i2]:
                    zcnt = udatad[i2].count(0)
                    if (zcnt/len(udatad[i2])) <= 0.8:
                        udatad[i2] = [j for j in udatad[i2] if j != 0]

                udatadmode = max(set(udatad[i2]), key=udatad[i2].count)  # mode
                xx, yy, zz = i2
                self.lmod1.lith_index[xx, yy, zz] = udatadmode

        # Reset the profile back to the current profile
        self.calc_prof_limits()

    ### Profiles
    def addprof(self):
        """ add another profile """
        xnodes = self.lmod1.custprofx
        ynodes = self.lmod1.custprofy

        self.update_model()
        (tx0, okay) = QtWidgets.QInputDialog.getDouble(
            self.parent, 'Add Custom Profile',
            'Please enter first x coordinate', self.lmod1.xrange[0])
        if not okay:
            return
        (ty0, okay) = QtWidgets.QInputDialog.getDouble(
            self.parent, 'Add Custom Profile',
            'Please enter first y coordinate', self.lmod1.yrange[0])
        if not okay:
            return
        (tx1, okay) = QtWidgets.QInputDialog.getDouble(
            self.parent, 'Add Custom Profile',
            'Please enter last x coordinate', self.lmod1.xrange[-1])
        if not okay:
            return
        (ty1, okay) = QtWidgets.QInputDialog.getDouble(
            self.parent, 'Add Custom Profile',
            'Please enter last y coordinate', self.lmod1.yrange[-1])
        if not okay:
            return

        self.pcntmax += 1
        xnodes[self.pcntmax] = [float(tx0), float(tx1)]
        ynodes[self.pcntmax] = [float(ty0), float(ty1)]

        self.hs_profnum.valueChanged.disconnect()
        self.sb_profnum.valueChanged.disconnect()
        self.combo_sideview.currentIndexChanged.disconnect()

        self.hs_profnum.setMaximum(self.pcntmax)
        self.sb_profnum.setMaximum(self.pcntmax)

        self.hs_profnum.valueChanged.connect(self.hprofnum)
        self.sb_profnum.valueChanged.connect(self.sprofnum)
#        self.combo_sideview.currentIndexChanged.connect(self.profpic_hs)

    def change_defs(self):
        """ List box in profile tab for definitions """

        i = self.lw_prof_defs.currentRow()
        if i == -1:
            misc.update_lith_lw(self.lmod1, self.lw_prof_defs)
            i = 0
        itxt = str(self.lw_prof_defs.item(i).text())

        if itxt not in self.lmod1.lith_list:
            return

        lith = self.lmod1.lith_list[itxt]
        self.mmc.curmodel = lith.lith_index

    def get_model(self):
        """ gets model slice """
        x1, x2 = self.lmod1.custprofx['adhoc']
        y1, y2 = self.lmod1.custprofy['adhoc']
        px1, px2 = self.lmod1.custprofx['rotate']

        # convert units to cells
        bly = self.lmod1.yrange[0]
        tlx = self.lmod1.xrange[0]
        dxy = self.lmod1.dxy

        x1 = (x1-tlx)/dxy
        x2 = (x2-tlx)/dxy
        y1 = (y1-bly)/dxy
        y2 = (y2-bly)/dxy

        # this is number of samples times 10
        self.pdxy = dxy/10  # ten times the cells
        rcell = int((px2-px1)/self.pdxy)
        rrcell = int((px2-px1)/dxy)*2+1

        if rcell == 0:
            rcell = 1

        self.xxx = np.linspace(x1, x2, rcell, False, dtype=int)
        self.yyy = np.linspace(y1, y2, rcell, False, dtype=int)

        self.rxxx = np.linspace(x1, x2, rrcell, True)
        self.ryyy = np.linspace(y1, y2, rrcell, True)

        if x1 > x2:
            self.xxx -= 1
        if y1 > y2:
            self.yyy -= 1

        # some indices are -1 which is where the error lies

        self.ryyy = self.ryyy[self.rxxx >= 0]
        self.rxxx = self.rxxx[self.rxxx >= 0]
        self.rxxx = self.rxxx[self.ryyy >= 0]
        self.ryyy = self.ryyy[self.ryyy >= 0]

#        tmp = np.transpose([self.xxx, self.yyy])
#        udata = [list(x) for x in set(tuple(x) for x in tmp)]
#        udata = np.sort(udata, 0)

        # get model now
        self.ipdx1 = int(px1/self.pdxy)
        self.ipdx2 = self.ipdx1+self.xxx.shape[0]

        gtmp = []
        for i in range(self.lmod1.numz):
            tmp = np.zeros(int(self.extent_side[1]/self.pdxy))-1
            tmp[self.ipdx1:self.ipdx2] = self.lmod1.lith_index[self.xxx,
                                                               self.yyy, i]

            gtmp.append(tmp)

    ### debug
        gtmp = np.array(gtmp[::-1])

        return gtmp

    def hprofnum(self):
        """ Routine to change a profile from spinbox"""
        self.sb_profnum.setValue(self.hs_profnum.sliderPosition())

    def pic_sideview(self):
        """
        Horizontal slider to change the opacity of profile and overlain
        picture
        """
        # This is used for custom profiles with pictures. I think that below
        # should be slide_grid

        gtmp = self.get_model()
        self.mmc.init_grid(gtmp)
        self.mmc.figure.canvas.draw()

    def plot_scale(self):
        """ plot scale """
        pscale = PlotScale(self, self.lmod1)
        tmp = pscale.exec_()
        if tmp == 0:
            return

        self.plot_custmin = pscale.dsb_axis_custmin.value()
        self.plot_custmax = pscale.dsb_axis_custmax.value()

        if pscale.rb_axis_calcmax.isChecked():
            self.pscale_type = 'calcmax'
        elif pscale.rb_axis_custmax.isChecked():
            self.pscale_type = 'custmax'
        elif pscale.rb_axis_datamax.isChecked():
            self.pscale_type = 'datamax'
        elif pscale.rb_axis_allmax.isChecked():
            self.pscale_type = 'allmax'
        else:
            self.pscale_type = 'profmax'

        self.update_plot()
        self.mpl_toolbar.update()

    def setwidth(self, width):
        """ Sets the width of the edits on the profile view """
        self.mmc.mywidth = width

    def sprofnum(self):
        """ Routine to change a profile from spinbox"""
        self.hs_profnum.valueChanged.disconnect()
        self.hs_profnum.setValue(self.sb_profnum.value())
        self.hs_profnum.valueChanged.connect(self.hprofnum)

    # First we plot the model stuff
        ctxt = str(self.label_sideview.text)
        if self.lmod1.profpics and ctxt != u'':
            gtmpl = self.lmod1.profpics[ctxt]
        else:
            gtmpl = None

        self.calc_prof_limits()

        gtmp = self.get_model()

        self.mmc.slide_grid(gtmp, gtmpl)
        self.mmc.update_line()
        self.mmc.update_line_top()

        self.update_plot(slide=True)

    ### Layers
    def hlayer(self):
        """ Horizontal slider to change the layer """
        self.sb_layer.setValue(self.hs_layer.sliderPosition())

    def pic_overview(self):
        """
        Horizontal slider to change the opacity of profile and overlain
        picture
        """
        self.mmc.init_grid_top(self.combo_overview.currentText(),
                               self.hs_overview.value())
#        self.mmc.slide_grid_top()
        self.mmc.figure.canvas.draw()

    def pic_overview2(self):
        """
        Horizontal slider to change the opacity of profile and overlain
        picture
        """
#        self.mmc.init_grid_top(self.combo_overview.currentText(),
#                               self.hs_overview.value())
        self.mmc.slide_grid_top(self.hs_overview.value())
        self.mmc.figure.canvas.draw()

    def slayer(self):
        """ This increases or decreases the model layer """
        self.hs_layer.valueChanged.disconnect()
        self.hs_layer.setValue(self.sb_layer.value())
        self.hs_layer.valueChanged.connect(self.hlayer)

        self.mmc.slide_grid_top()
        self.mmc.update_line()
        self.mmc.figure.canvas.draw()

    ### Profile Direction
    def calc_prof_limits(self, curprof=None):
        """ Calculate profile limits """

        dirval = self.dial_prof_dir.value()
        pdirval = 360-dirval
        if pdirval == 360:
            pdirval = 0.

        m = np.tan(np.deg2rad(pdirval))

        xrng = self.lmod1.xrange
        yrng = self.lmod1.yrange
        dxy = self.lmod1.dxy

        if curprof is None:
            curprof = self.sb_profnum.value()

        if pdirval == 0:
            x1 = np.array([xrng[0]])
            x2 = np.array([xrng[1]])
            y1 = np.array([yrng[0]+dxy/2+curprof*dxy])
            y2 = y1

        elif pdirval == 180:
            x1 = np.array([xrng[1]])
            x2 = np.array([xrng[0]])
            y1 = np.array([yrng[1]-dxy/2-curprof*dxy])
            y2 = y1

        elif pdirval == 90:
            y1 = np.array([yrng[0]])
            y2 = np.array([yrng[1]])
            x1 = np.array([xrng[1]-dxy/2-curprof*dxy])
            x2 = x1

        elif pdirval == 270:
            y1 = np.array([yrng[1]])
            y2 = np.array([yrng[0]])
            x1 = np.array([xrng[0]+dxy/2+curprof*dxy])
            x2 = x1

        elif 0 < pdirval < 90:
            x1 = np.arange(xrng[1]-dxy/2, xrng[0], -dxy)
            y1 = np.ones_like(x1)*yrng[0]
            y1a = np.arange(yrng[0]+dxy/2, yrng[1], dxy)
            x1a = np.ones_like(y1a)*xrng[0]
            y1 = np.append(y1, y1a)
            x1 = np.append(x1, x1a)

            c = y1-m*x1

            x2 = np.ones_like(x1)*xrng[1]
            y2 = x2*m+c

            filt = (y2 > yrng[1])

            y2[filt] = yrng[1]
            x2[filt] = (yrng[1]-c[filt])/m

        elif 90 < pdirval < 180:
            x1 = np.arange(xrng[1]-dxy/2, xrng[0], -dxy)
            y1 = np.ones_like(x1)*yrng[1]
            y1a = np.arange(yrng[1]-dxy/2, yrng[0], -dxy)
            x1a = np.ones_like(y1a)*xrng[0]
            y1 = np.append(y1, y1a)
            x1 = np.append(x1, x1a)

            c = y1-m*x1

            x2 = np.ones_like(x1)*xrng[1]
            y2 = x2*m+c

            filt = (y2 < yrng[0])

            y2[filt] = yrng[0]
            x2[filt] = (yrng[0]-c[filt])/m

            x1, x2 = x2, x1
            y1, y2 = y2, y1

        elif 180 < pdirval < 270:
            x1 = np.arange(xrng[0]+dxy/2, xrng[1], dxy)
            y1 = np.ones_like(x1)*yrng[1]
            y1a = np.arange(yrng[1]-dxy/2, yrng[0], -dxy)
            x1a = np.ones_like(y1a)*xrng[1]
            y1 = np.append(y1, y1a)
            x1 = np.append(x1, x1a)

            c = y1-m*x1

            x2 = np.ones_like(x1)*xrng[0]
            y2 = x2*m+c

            filt = (y2 < yrng[0])

            y2[filt] = yrng[0]
            x2[filt] = (yrng[0]-c[filt])/m

        elif pdirval > 270:
            x1 = np.arange(xrng[0]+dxy/2, xrng[1], dxy)
            y1 = np.ones_like(x1)*yrng[0]
            y1a = np.arange(yrng[0]+dxy/2, yrng[1], dxy)
            x1a = np.ones_like(y1a)*xrng[1]
            y1 = np.append(y1, y1a)
            x1 = np.append(x1, x1a)

            c = y1-m*x1

            x2 = np.ones_like(x1)*xrng[0]
            y2 = x2*m+c

            filt = (y2 > yrng[1])

            y2[filt] = yrng[1]
            x2[filt] = (yrng[1]-c[filt])/m

            x1, x2 = x2, x1
            y1, y2 = y2, y1

        if len(x1) == 1:
            curprof = 0

        ang = -np.deg2rad(pdirval)
        cntr = np.array([x1[0], y1[0]])

        pts1 = np.transpose([x1, y1])
        pts1r = rotate2d(pts1, cntr, ang)

        pts2 = np.transpose([x2, y2])
        pts2r = rotate2d(pts2, cntr, ang)

        px1 = pts1r[:, 0]
        px2 = pts2r[:, 0]

        minx = px1.min()
        px1 = px1 - minx
        px2 = px2 - minx

        bottom, top = self.lmod1.zrange
        right = px2.max()

        self.extent_side = [0., right, bottom, top]

        # TODO: 2nd and 4th quadrant offset

        self.lmod1.custprofx['rotate'] = [px1[curprof], px2[curprof]]
        self.lmod1.custprofx['adhoc'] = [x1[curprof], x2[curprof]]
        self.lmod1.custprofy['adhoc'] = [y1[curprof], y2[curprof]]

    def prof_dir(self, slide=True):
        """ Radio button profile direction """
        dirval = self.dial_prof_dir.value()
        pdirval = 360-dirval
        if pdirval == 360:
            pdirval = 0.

        self.sb_prof_dir.valueChanged.disconnect()
        self.sb_prof_dir.setValue(pdirval)
        self.sb_prof_dir.valueChanged.connect(self.sprofdir)

        self.sb_profnum.setValue(0.0)
        self.hs_profnum.setValue(0.0)
        self.hs_sideview.setEnabled(False)

        self.sb_layer.setMaximum(self.lmod1.numz-1)
        if pdirval in (0., 180):
            self.sb_profnum.setMaximum(self.lmod1.numy-1)
            self.hs_profnum.setMaximum(self.lmod1.numy-1)
        elif pdirval in (90., 270):
            self.sb_profnum.setMaximum(self.lmod1.numx-1)
            self.hs_profnum.setMaximum(self.lmod1.numx-1)
        else:
            self.sb_profnum.setMaximum(self.lmod1.numx+self.lmod1.numy-1)
            self.hs_profnum.setMaximum(self.lmod1.numx+self.lmod1.numy-1)

        self.calc_prof_limits()
        gtmp = self.get_model()

        self.mmc.init_grid(gtmp)
        self.mmc.init_grid_top(self.combo_overview.currentText(),
                               self.hs_overview.value())
        self.mmc.update_line_top()

        self.update_plot(slide=slide)

    def sprofdir(self):
        """ spinbox prof dir """
        pdirval = self.sb_prof_dir.value()
        dirval = 360-pdirval
        if dirval == 360:
            dirval = 0.

        self.dial_prof_dir.setValue(dirval)
        self.prof_dir()

    def update_plot(self, slide=False):
        """ Update the profile on the model view """

# Display the calculated profile
        if self.viewmagnetics:
            data = self.lmod1.griddata['Calculated Magnetics']
            self.mmc.ptitle = 'Magnetic Intensity: '
            self.mmc.punit = 'nT'
            regtmp = 0.0
        else:
            data = self.lmod1.griddata['Calculated Gravity']
            self.mmc.ptitle = 'Gravity: '
            self.mmc.punit = 'mGal'
            regtmp = self.lmod1.gregional

        self.mmc.xlabel = 'Distance (m)'

        px1, px2 = self.lmod1.custprofx['rotate']

        tmprng = np.linspace(px1, px2, len(self.rxxx))
        # 0.5 offset below is because map_coordinates uses centre of a cell
        # as 0, whereas normal coordinates has that as the edge of the cell.
        tmpprof = ndimage.map_coordinates(data.data[::-1],
                                          [self.ryyy-0.5,
                                           self.rxxx-0.5],
                                          order=1, cval=np.nan)
        tmprng = tmprng[np.logical_not(np.isnan(tmpprof))]
        tmpprof = tmpprof[np.logical_not(np.isnan(tmpprof))]+regtmp

        if self.pscale_type == 'custmax':
            extent = [self.plot_custmin, self.plot_custmax]
        elif self.pscale_type == 'calcmax' or self.pscale_type == 'allmax':
            extent = [data.data.min()+regtmp, data.data.max()+regtmp]
        elif tmpprof.size > 0:
            extent = [tmpprof.min(), tmpprof.max()]
        else:
            extent = [data.data.min()+regtmp, data.data.max()+regtmp]

# Load in observed data - if there is any
        data2 = None
        tmprng2 = None
        tmpprof2 = None
        if 'Magnetic Dataset' in self.lmod1.griddata and self.viewmagnetics:
            data2 = self.lmod1.griddata['Magnetic Dataset']
        elif ('Gravity Dataset' in self.lmod1.griddata and
              not self.viewmagnetics):
            data2 = self.lmod1.griddata['Gravity Dataset']

        if data2 is not None:
            xratio = data.xdim/data2.xdim
            yratio = data.ydim/data2.ydim

            dtlx = data.extent[0]
            d2tlx = data2.extent[0]
            dtly = data.extent[-1]
            d2tly = data2.extent[-1]

            rxxx2 = (dtlx-d2tlx)/data2.xdim+self.rxxx*xratio
            ryyy2 = (d2tly-dtly)/data2.ydim+self.ryyy*yratio

            tmprng2 = np.linspace(px1, px2, len(rxxx2))
            tmpprof2 = ndimage.map_coordinates(data2.data[::-1],
                                               [ryyy2-0.5,
                                                rxxx2-0.5],
                                               order=1, cval=np.nan)

            tmprng2 = tmprng2[np.logical_not(np.isnan(tmpprof2))]
            tmpprof2 = tmpprof2[np.logical_not(np.isnan(tmpprof2))]

            if self.pscale_type == 'datamax':
                extent = [data2.data.min(), data2.data.max()]
            elif self.pscale_type == 'allmax':
                extent = [min(extent[0], data2.data.min()),
                          max(extent[1], data2.data.max())]
            elif self.pscale_type == 'profmax':
                if tmpprof2.size > 0:
                    extent = [tmpprof2.min(), tmpprof2.max()]

        if slide is True:
#            xlim = self.mmc.axes.get_xlim()
#            extent = [xlim[0], xlim[1]] + extent
            self.mmc.slide_plot(tmprng, tmpprof, tmprng2, tmpprof2)
            self.mmc.figure.canvas.draw()
        else:
            extent = [self.extent_side[0], self.extent_side[1]] + extent
            self.mmc.init_plot(tmprng, tmpprof, extent, tmprng2, tmpprof2)

            gtmp = self.get_model()  # This may be slow
            self.mmc.init_grid(gtmp)
            self.mmc.figure.canvas.draw()
            self.mpl_toolbar.update()  # used to set original view limits.
            self.pic_overview()

    def tab_activate(self):
        """ Runs when the tab is activated """
        self.sb_profnum.valueChanged.disconnect()
        self.hs_profnum.valueChanged.disconnect()
        self.combo_overview.currentIndexChanged.disconnect()

        self.lmod1 = self.parent.lmod1
        self.mmc.lmod1 = self.lmod1

        citems = list(self.lmod1.griddata.keys())
        self.combo_overview.clear()
        self.combo_overview.addItems(citems)

        curtext = self.combo_overview.currentText()
        cindex = self.combo_overview.findText(curtext,
                                              QtCore.Qt.MatchFixedString)
        if cindex == -1:
            cindex = 0
        self.combo_overview.setCurrentIndex(cindex)

        txtmsg = ('Note: The display of gravity or magnetic data is '
                  'triggered off their respective calculations. Press '
                  '"Calculate Gravity" to see the gravity plot and '
                  '"Calculate Magnetics" to see the magnetic plot')

        if txtmsg not in self.parent.txtmsg.split('\n'):
            self.showtext(txtmsg)

        misc.update_lith_lw(self.lmod1, self.lw_prof_defs)

        self.hs_layer.setMaximum(self.lmod1.numz-1)
        self.hs_profnum.setMinimum(0)

        self.prof_dir(slide=False)
        self.calc_prof_limits()

        self.sb_profnum.setValue(0)
        self.hs_profnum.setValue(0)

        self.sb_profnum.valueChanged.connect(self.sprofnum)
        self.hs_profnum.valueChanged.connect(self.hprofnum)
        self.combo_overview.currentIndexChanged.connect(self.pic_overview)


class MyMplCanvas(FigureCanvas):
    """This is a QWidget"""
    def __init__(self, parent):
        fig = Figure()
        FigureCanvas.__init__(self, fig)

        self.lmod1 = parent.lmod1
        self.cbar = cm.jet
        self.curmodel = 0
        self.mywidth = 1
        self.xold = None
        self.yold = None
        self.press = False
        self.newline = False
        self.mdata = np.zeros([10, 100])
        self.lmdata = np.zeros([100, 100])
        self.ptitle = ''
        self.punit = ''
        self.xlabel = 'Eastings (m)'
        self.plotisinit = False
        self.opac = 1.0
        self.lopac = 1.0
        self.xlims = None
        self.ylims = None
        self.crd = None
        self.myparent = parent

# Events
        self.figure.canvas.mpl_connect('motion_notify_event', self.move)
        self.figure.canvas.mpl_connect('button_press_event', self.button_press)
        self.figure.canvas.mpl_connect('button_release_event',
                                       self.button_release)
        self.figure.canvas.mpl_connect('resize_event', self.on_resize)

# Initial Images
        self.paxes = fig.add_subplot(222)
        self.paxes.yaxis.set_label_text('mGal')
        self.paxes.ticklabel_format(useOffset=False)

        self.cal = self.paxes.plot([], [], zorder=10, color='blue')
        self.obs = self.paxes.plot([], [], '.', zorder=1, color='orange')

        self.axes = fig.add_subplot(224, sharex=self.paxes)
        self.axes.xaxis.set_label_text(self.xlabel)
        self.axes.yaxis.set_label_text('Altitude (m)')
#        self.axes.callbacks.connect('xlim_changed', self.paxes_lim_update)

        self.laxes = fig.add_subplot(121)

        self.laxes.set_title('Layer View')
        self.laxes.xaxis.set_label_text('Eastings (m)')
        self.laxes.yaxis.set_label_text('Northings (m)')

        self.lims2 = self.laxes.imshow(self.lmdata, cmap=self.cbar,
                                       aspect='equal')

        self.colbar = fig.colorbar(self.lims2)

        self.lims = self.laxes.imshow(self.cbar(self.lmdata), aspect='equal')
        self.lprf = self.laxes.plot([0, 1], [0, 1])

        self.ims2 = self.axes.imshow(self.cbar(self.mdata), aspect='auto')
        self.ims = self.axes.imshow(self.cbar(self.mdata), aspect='auto')

        self.ims.format_cursor_data = lambda x: ''
        self.ims2.format_cursor_data = lambda x: ''
        self.lims.format_cursor_data = lambda x: ''
        self.lims2.format_cursor_data = lambda x: ''

        self.prf = self.axes.plot([0, 0])

#        self.figure.tight_layout()
#        self.figure.canvas.draw()

    def button_press(self, event):
        """ Button press """
        if event.inaxes is None:
            return

        nmode = event.inaxes.get_navigate_mode()
#        nmode = self.figure.canvas.toolbar._active
        if event.button == 1 and nmode is None:
            self.press = True
            self.newline = True
            self.move(event)

    def button_release(self, event):
        """ Button release """
        self.press = False

#        nmode = self.axes.get_navigate_mode()
#        if event.button == 1:
#            if nmode == 'ZOOM':
#                extent = self.axes.get_xbound()
#                self.paxes.set_xbound(extent[0], extent[1])
#                self.myparent.mpl_toolbar.zoom()
#        if nmode == 'PAN':
#            self.myparent.mpl_toolbar.pan()

#    def paxes_lim_update(self, event):
#        """ plot axes limit update """
#        extent = self.axes.get_xbound()
#        self.paxes.set_xbound(extent[0], extent[1])
#        self.figure.canvas.draw()

    def move(self, event):
        """ Mouse is moving """
        curaxes = event.inaxes
        if curaxes not in (self.axes, self.laxes):
            self.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
            return

        if curaxes == self.axes:
            mdata = self.mdata
            xptp = self.lmod1.xrange[1]-self.lmod1.xrange[0]
#            xmin = self.lmod1.xrange[0]
            xmin = 0

            dx = self.myparent.pdxy
            dy = self.lmod1.d_z
            yptp = self.lmod1.zrange[1]-self.lmod1.zrange[0]
            ymin = self.lmod1.zrange[0]
        else:
            mdata = self.lmdata
            xptp = self.lmod1.xrange[1]-self.lmod1.xrange[0]
            yptp = self.lmod1.yrange[1]-self.lmod1.yrange[0]
            dx = self.lmod1.dxy
            dy = self.lmod1.dxy
            xmin = self.lmod1.xrange[0]
            ymin = self.lmod1.yrange[0]

        if self.figure.canvas.toolbar._active is None:
            vlim = curaxes.viewLim
            tmp0 = curaxes.transData.transform((vlim.x0, vlim.y0))
            tmp1 = curaxes.transData.transform((vlim.x1, vlim.y1))
            width, height = tmp1-tmp0
            width /= mdata.shape[1]
            height /= mdata.shape[0]
            width *= xptp/vlim.width
            height *= yptp/vlim.height
            width *= (2*self.mywidth-1)
            height *= (2*self.mywidth-1)

            width = np.ceil(width)  # *10
            height = np.ceil(height)

            cbit = QtGui.QBitmap(width, height)
            cbit.fill(QtCore.Qt.color1)
            self.setCursor(QtGui.QCursor(cbit))

        if self.press is True:
            xdata = int((event.xdata - xmin)/dx)+1
            ydata = int((event.ydata - ymin)/dy)+1

            if self.newline is True:
                self.newline = False
                self.set_mdata(xdata, ydata, mdata)
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
                        self.set_mdata(i, j, mdata)

            elif ydata != self.yold:
                y_1 = min([self.yold, ydata])
                y_2 = max([self.yold, ydata])
                for j in range(y_1, y_2+1):
                    self.set_mdata(xdata, j, mdata)

            self.xold = xdata
            self.yold = ydata

            curlayer = self.myparent.sb_layer.value()

            xxx = self.myparent.xxx
            yyy = self.myparent.yyy
            ipdx1 = self.myparent.ipdx1
            ipdx2 = self.myparent.ipdx2

            if curaxes == self.axes:
                tmp = (mdata[:, ipdx1:ipdx2] == self.curmodel)
                for i, rpix in enumerate(tmp[::-1]):
                    xxx2 = xxx[rpix]
                    yyy2 = yyy[rpix]
                    self.lmod1.lith_index[xxx2, yyy2, i] = self.curmodel

#                self.lmod1.lith_index[xxx, yyy, ::-1] = mdata[:, ipdx1:ipdx2].T
#                self.mdata = mdata
                self.lmdata = self.lmod1.lith_index[:, :, curlayer].T
                self.mdata[:, ipdx1:ipdx2] = self.lmod1.lith_index[xxx, yyy, ::-1].T

            else:
                self.lmod1.lith_index[:, :, curlayer] = mdata.T
                self.lmdata = mdata
                self.mdata = self.lmod1.lith_index[xxx, yyy, ::-1].T

            self.slide_grid(self.mdata)
            self.slide_grid_top()
            self.figure.canvas.draw()

    def set_mdata(self, xdata, ydata, mdata):
        """ Routine to 'draw' the line on mdata """
        width = self.mywidth-1  # 'pen' width
        xstart = max(0, xdata-width-1)
        xend = min(mdata.shape[1], xdata+width)
        ystart = max(0, ydata-width-1)
        yend = min(mdata.shape[0], ydata+width)
        if xstart < xend and ystart < yend:
            mtmp = mdata[ystart:yend, xstart:xend]
            mtmp[np.logical_and(mtmp != -1, mtmp < 900)] = self.curmodel

    def luttodat(self, dat):
        """ lut to dat grid """
        mlut = self.lmod1.mlut
        tmp = np.zeros([dat.shape[0], dat.shape[1], 4])

        for i in np.unique(dat):
            if i == -1:
                ctmp = [0, 0, 0, 0]
            else:
                ctmp = np.array(mlut[i]+[255])/255.

            tmp[dat[::-1] == i] = ctmp

        return tmp

    def on_resize(self, event):
        """
        matplotlib resize event - used to make sure tight_layout happens on
        startup
        """
        self.figure.tight_layout()
        self.figure.canvas.draw()

    def init_grid(self, dat, dat2=None, opac=100.0):
        """ Updates the single color map """
        self.opac = 1.0 - float(opac) / 100.

        extent = self.myparent.extent_side

        self.paxes.set_xbound(extent[0], extent[1])

        self.ims.set_visible(False)
        self.ims2.set_visible(False)
        self.axes.set_xlim(extent[0], extent[1])
        self.axes.set_ylim(extent[2], extent[3])

        if dat2 is not None and self.opac < 1.0:
            self.ims2.set_visible(True)
            self.ims2.set_data(dat2.data)
            self.ims2.set_extent(dat2.extent)
            self.ims2.set_clim(dat2.data.min(), dat2.data.max())
            self.ims2.set_alpha(self.opac)

        self.ims.set_visible(True)
        self.ims.set_extent(extent)
        tmp = self.luttodat(dat)
        self.ims.set_data(tmp)

        self.figure.canvas.draw()
        self.myparent.mpl_toolbar.update()  # used to set original view limits.

        self.mdata = dat

    def init_grid_top(self, dat2=None, opac=100.0):
        """ init grid top """
        dat2 = self.myparent.combo_overview.currentText()

        extent = self.lmod1.xrange+self.lmod1.yrange
        curlayer = self.myparent.sb_layer.value()

        self.lopac = 1.0 - float(opac) / 100.
        dat = self.lmod1.lith_index[:, :, curlayer].T

        self.lmdata = dat
        tmp = self.luttodat(dat)

        self.lims.set_visible(False)
        self.lims2.set_visible(False)
        self.lims.set_data(tmp)
        self.lims.set_extent(extent)
        self.lims.set_alpha(self.lopac)

        if dat2 is not None and dat2 != '':
            self.lims2.set_visible(True)
            dat2 = self.lmod1.griddata[str(dat2)]
            self.lims2.set_data(dat2.data)
            self.lims2.set_extent(dat2.extent)
            self.lims2.set_clim(dat2.data.min(), dat2.data.max())
            self.colbar.ax.set_xlabel(dat2.units)

        left, right = self.lmod1.xrange
        bottom, top = self.lmod1.yrange

        self.xlims = (left, right)
        self.ylims = (bottom, top)
        self.laxes.set_xlim(self.xlims)
        self.laxes.set_ylim(self.ylims)
#
        self.lims.set_visible(True)

        self.figure.tight_layout()
        self.figure.canvas.draw()

    def slide_grid(self, dat, dat2=None):
        """ Slider """

        # There may be errors here in terms of dat and dat2

        self.mdata = dat

        tmp = self.luttodat(dat)
        self.ims.set_data(tmp)

        if dat2 is not None:
            self.ims2.set_visible(True)
            self.ims2.set_alpha(self.opac)

        self.axes.draw_artist(self.ims)
        self.axes.draw_artist(self.ims2)
        self.axes.draw_artist(self.prf[0])

    def slide_grid_top(self, opac=None):
        """ Slider """
        if opac is not None:
            self.lopac = 1.0 - float(opac) / 100.

        curlayer = self.myparent.sb_layer.value()

        dat = self.lmod1.lith_index[:, :, curlayer].T
        self.lmdata = dat

        tmp = self.luttodat(dat)
        self.lims.set_data(tmp)
        self.lims.set_alpha(self.lopac)

#        if dat2 is not None:
#            dat2 = self.lmod1.griddata[str(dat2)]
#            self.lims2.set_data(dat2.data)

        self.laxes.draw_artist(self.lims2)
        self.laxes.draw_artist(self.lims)
        self.laxes.draw_artist(self.lprf[0])

    def update_line(self):
        """ Updates the line position """

        curlayer = self.myparent.sb_layer.value()
        extent = self.myparent.extent_side
        xrng = [extent[0], extent[1]]

        alt = self.lmod1.zrange[1]-curlayer*self.lmod1.d_z
        yrng = [alt, alt]

        self.prf[0].set_data([xrng, yrng])
        self.axes.draw_artist(self.prf[0])

    def update_line_top(self):
        """ Updates the line position """
        xrng = self.lmod1.custprofx['adhoc']
        yrng = self.lmod1.custprofy['adhoc']

        self.lprf[0].set_data([xrng, yrng])
        self.laxes.draw_artist(self.lprf[0])

# This section is just for the profile line plot

    def init_plot(self, xdat, dat, extent, xdat2=None, dat2=None):
        """ Updates the single color map """
        self.paxes.autoscale(False)
        dmin, dmax = extent[2], extent[3]
        if dmin == dmax:
            dmax = dmin+1

        self.paxes.cla()
        self.paxes.ticklabel_format(useOffset=False)
        self.paxes.set_title(self.ptitle)
        self.axes.xaxis.set_label_text(self.xlabel)
        self.paxes.yaxis.set_label_text(self.punit)
        self.paxes.set_ylim(dmin, dmax)
        self.paxes.set_xlim(extent[0], extent[1])

        self.paxes.set_autoscalex_on(False)
        if xdat2 is not None:
            self.obs = self.paxes.plot(xdat2, dat2, '.', zorder=1,
                                       color='orange')
        else:
            self.obs = self.paxes.plot([], [], '.', zorder=1, color='orange')
        self.cal = self.paxes.plot(xdat, dat, zorder=10, color='blue')

        self.plotisinit = True

    def slide_plot(self, xdat, dat, xdat2=None, dat2=None):
        """ Slider """
        if xdat2 is not None:
            self.obs[0].set_data([xdat2, dat2])
        else:
            self.obs[0].set_data([[], []])

        self.cal[0].set_data([xdat, dat])

        if xdat2 is not None:
            self.paxes.draw_artist(self.obs[0])
        self.paxes.draw_artist(self.cal[0])


class MySlider(QtWidgets.QSlider):
    """
    My Slider

    Custom class which allows clicking on a horizontal slider bar with slider
    moving to click in a single step.
    """
    def __init__(self, parent=None):
        QtWidgets.QSlider.__init__(self, parent)

    def mousePressEvent(self, event):
        """ Mouse Press Event """
        self.setValue(QtWidgets.QStyle.sliderValueFromPosition(self.minimum(),
                                                               self.maximum(),
                                                               event.x(),
                                                               self.width()))

    def mouseMoveEvent(self, event):
        """ Jump to pointer position while moving """
        self.setValue(QtWidgets.QStyle.sliderValueFromPosition(self.minimum(),
                                                               self.maximum(),
                                                               event.x(),
                                                               self.width()))


class LithBound(QtWidgets.QDialog):
    """ Class to call up a dialog for lithological boundary"""
    def __init__(self, lmod):
        QtWidgets.QDialog.__init__(self, None)

        self.lmod1 = lmod
        self.buttonbox = QtWidgets.QDialogButtonBox(self)
        self.lw_lithupper = QtWidgets.QListWidget(self)
        self.lw_lithlower = QtWidgets.QListWidget(self)
        self.rb_depth = QtWidgets.QRadioButton('Z-coordinate is in units of '
                                               'depth(positive down)')
        self.rb_height = QtWidgets.QRadioButton('Z-coordinate is in units of '
                                                'height above sea level')
        self.setupui()

    def setupui(self):
        """ Setup UI """
        gridlayout = QtWidgets.QGridLayout(self)
        label_3 = QtWidgets.QLabel('Lithologies Above Layer')
        label_4 = QtWidgets.QLabel('Lithologies Below Layer')

        gridlayout.addWidget(label_3, 0, 0, 1, 1)
        self.lw_lithupper.setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection)
        gridlayout.addWidget(self.lw_lithupper, 0, 1, 1, 1)
        gridlayout.addWidget(label_4, 1, 0, 1, 1)
        self.lw_lithlower.setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection)
        gridlayout.addWidget(self.lw_lithlower, 1, 1, 1, 1)
        self.buttonbox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonbox.setStandardButtons(
            QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)

        self.rb_depth.setChecked(True)

        gridlayout.addWidget(self.rb_depth, 2, 0, 1, 2)
        gridlayout.addWidget(self.rb_height, 3, 0, 1, 2)
        gridlayout.addWidget(self.buttonbox, 4, 0, 1, 2)

        self.setWindowTitle('Add Lithological Boundary')

        self.buttonbox.accepted.connect(self.accept)
        self.buttonbox.rejected.connect(self.reject)

        self.lw_lithupper.addItem(r'Do Not Change')
        self.lw_lithlower.addItem(r'Do Not Change')
        for i in self.lmod1.lith_list:
            self.lw_lithupper.addItem(i)
            self.lw_lithlower.addItem(i)

        self.lw_lithupper.setCurrentItem(self.lw_lithupper.item(0))
        self.lw_lithlower.setCurrentItem(self.lw_lithlower.item(1))

    def get_lith(self):
        """ Get Lith """

        tupper = self.lw_lithupper.selectedItems()
        tlower = self.lw_lithlower.selectedItems()

        if tupper[0].text() == r'Do Not Change':
            lithupper = -999
        else:
            lithupper = self.lmod1.lith_list[tupper[0].text()].lith_index

        if tlower[0].text() == r'Do Not Change':
            lithlower = -999
        else:
            lithlower = self.lmod1.lith_list[tlower[0].text()].lith_index

        return lithlower, lithupper


class PlotScale(QtWidgets.QDialog):
    """ Class to call up a dialog for plot axis scale"""
    def __init__(self, parent, lmod):
        QtWidgets.QDialog.__init__(self, parent)

        self.lmod1 = lmod
        self.buttonbox = QtWidgets.QDialogButtonBox(self)
        self.rb_axis_allmax = QtWidgets.QRadioButton('Scale to all maximum')
        self.rb_axis_datamax = QtWidgets.QRadioButton('Scale to dataset '
                                                      'maximum')
        self.rb_axis_profmax = QtWidgets.QRadioButton('Scale to profile '
                                                      'maximum')
        self.rb_axis_calcmax = QtWidgets.QRadioButton('Scale to calculated '
                                                      'maximum')
        self.rb_axis_custmax = QtWidgets.QRadioButton('Scale to custom '
                                                      'maximum')
        self.dsb_axis_custmin = QtWidgets.QDoubleSpinBox()
        self.dsb_axis_custmax = QtWidgets.QDoubleSpinBox()

        self.setupui()

    def setupui(self):
        """ Setup UI """
        self.setWindowTitle('Field Display Limits')

        self.rb_axis_allmax.setChecked(True)
        self.dsb_axis_custmin.setValue(0.)
        self.dsb_axis_custmax.setValue(50.)
        self.dsb_axis_custmin.setMinimum(-1000000.)
        self.dsb_axis_custmin.setMaximum(1000000.)
        self.dsb_axis_custmax.setMinimum(-1000000.)
        self.dsb_axis_custmax.setMaximum(1000000.)

        self.buttonbox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonbox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel |
                                          QtWidgets.QDialogButtonBox.Ok)

        vl_scale = QtWidgets.QVBoxLayout(self)
        vl_scale.addWidget(self.rb_axis_allmax)
        vl_scale.addWidget(self.rb_axis_datamax)
        vl_scale.addWidget(self.rb_axis_profmax)
        vl_scale.addWidget(self.rb_axis_calcmax)
        vl_scale.addWidget(self.rb_axis_custmax)
        vl_scale.addWidget(self.dsb_axis_custmin)
        vl_scale.addWidget(self.dsb_axis_custmax)
        vl_scale.addWidget(self.buttonbox)

#        self.dsb_axis_custmin.valueChanged.connect(self.rb_plot_scale)
#        self.dsb_axis_custmax.valueChanged.connect(self.rb_plot_scale)
#        self.rb_axis_calcmax.clicked.connect(self.rb_plot_scale)
#        self.rb_axis_profmax.clicked.connect(self.rb_plot_scale)
#        self.rb_axis_datamax.clicked.connect(self.rb_plot_scale)
#        self.rb_axis_custmax.clicked.connect(self.rb_plot_scale)
        self.buttonbox.accepted.connect(self.accept)
        self.buttonbox.rejected.connect(self.reject)


class RangedCopy(QtWidgets.QDialog):
    """ Class to call up a dialog for ranged copying """
    def __init__(self, parent):
        QtWidgets.QDialog.__init__(self, parent)

#        if parent.rb_cust.isChecked():
#            parent.showtext('You cannot perform a ranged copy '
#                            'on a custom profile')
#            self.reject()

        self.parent = parent
        self.lmod1 = self.parent.lmod1

        self.sb_master = QtWidgets.QSpinBox()
        self.sb_start = QtWidgets.QSpinBox()
        self.lw_lithdel = QtWidgets.QListWidget()
        self.lw_lithcopy = QtWidgets.QListWidget()
        self.sb_end = QtWidgets.QSpinBox()
        self.rb_sideview = QtWidgets.QRadioButton('Side View')
        self.rb_overview = QtWidgets.QRadioButton('Top View')

        self.setupui()

    def setupui(self):
        """ Setup UI """

        self.setWindowTitle('Ranged Copy')

        gridlayout = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.pfmod.misc.rangedcopy')

        label = QtWidgets.QLabel('Range Start')
        label_2 = QtWidgets.QLabel('Master Profile')
        label_3 = QtWidgets.QLabel('Lithologies To Copy')
        label_4 = QtWidgets.QLabel('Lithologies To Overwrite')
        label_5 = QtWidgets.QLabel('Range End')

        self.sb_master.setMaximum(999999999)
        self.sb_start.setMaximum(999999999)
        self.lw_lithcopy.setSelectionMode(self.lw_lithcopy.MultiSelection)
        self.lw_lithdel.setSelectionMode(self.lw_lithdel.MultiSelection)
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)
        self.sb_end.setMaximum(999999999)

        self.rb_sideview.setChecked(True)

        for i in self.lmod1.lith_list:
            self.lw_lithcopy.addItem(i)
            self.lw_lithdel.addItem(i)

        rmax = self.parent.sb_profnum.maximum()
        rval = self.parent.sb_profnum.value()

        self.sb_start.setMaximum(rmax)
        self.sb_end.setMaximum(rmax)
        self.sb_end.setValue(rmax)
        self.sb_master.setValue(rval)

        gb_target = QtWidgets.QGroupBox('Target:')
        vl_dir = QtWidgets.QHBoxLayout(gb_target)
        vl_dir.addWidget(self.rb_sideview)
        vl_dir.addWidget(self.rb_overview)

        gridlayout.addWidget(gb_target, 0, 0, 1, 2)

        gridlayout.addWidget(label_2, 1, 0, 1, 1)
        gridlayout.addWidget(self.sb_master, 1, 1, 1, 1)

        gridlayout.addWidget(label, 2, 0, 1, 1)
        gridlayout.addWidget(self.sb_start, 2, 1, 1, 1)

        gridlayout.addWidget(label_5, 3, 0, 1, 1)
        gridlayout.addWidget(self.sb_end, 3, 1, 1, 1)

        gridlayout.addWidget(label_3, 4, 0, 1, 1)
        gridlayout.addWidget(self.lw_lithcopy, 4, 1, 1, 1)

        gridlayout.addWidget(label_4, 5, 0, 1, 1)
        gridlayout.addWidget(self.lw_lithdel, 5, 1, 1, 1)

        gridlayout.addWidget(helpdocs, 6, 0, 1, 1)
        gridlayout.addWidget(buttonbox, 6, 1, 1, 1)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        self.rb_sideview.clicked.connect(self.target_update)
        self.rb_overview.clicked.connect(self.target_update)

    def target_update(self):
        """ updates target """

        if self.rb_overview.isChecked():
            rmax = self.parent.sb_layer.maximum()
        else:
            rmax = self.parent.sb_profnum.maximum()

        self.sb_start.setMaximum(rmax)
        self.sb_end.setMaximum(rmax)
        self.sb_end.setValue(rmax)

        if self.rb_overview.isChecked():
            self.sb_master.setValue(self.parent.sb_layer.value())
        else:
            self.sb_master.setValue(self.parent.sb_profnum.value())


class MyToolbar(NavigationToolbar2QT):
    """ Custom Matplotlib toolbar """
    toolitems = NavigationToolbar2QT.toolitems
    toolitems += ((None, None, None, None),
                  ('Field\nDisplay\nLimits',
                   'Axis Scale', 'Axis Scale', 'axis_scale'),)
#                  ('Ad-Hoc\nCustom\nProfile',
#                   'Custom Profile', 'Custom Profile', 'custom_profile'),)

    def __init__(self, parent):
        NavigationToolbar2QT.__init__(self, parent.mmc, parent)
        self.plotscale = parent.plot_scale
        self.custom_prof = parent.addprof

    def axis_scale(self):
        """ Axis scale """
        self.plotscale()

    def custom_profile(self):
        """ Axis scale """
        self.custom_prof()


class GaugeWidget(QtWidgets.QDial):
    """ Gauge widget """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ipth = os.path.dirname(misc.__file__)+'//'

        self._bg = QtGui.QPixmap(ipth+'DirectionDial.png')
        self.setValue(0)
        self.setMaximum(359)
        self.setFixedWidth(60)
        self.setFixedHeight(60)

    def paintEvent(self, event):
        """ Paint event """
        painter = QtGui.QPainter(self)
        painter.setRenderHint(painter.Antialiasing)
        rect = event.rect()

        gauge_rect = QtCore.QRect(rect)
        size = gauge_rect.size()

        painter.translate(size.width()/2, size.height()/2)
        painter.rotate(self.value())
        painter.translate(-size.width()/2, -size.height()/2)
        painter.drawPixmap(rect, self._bg)
        painter.end()


def gridmatch2(cgrv, rgrv):
    """ Matches the rows and columns of the second grid to the first
    grid """
#    rgrv = lmod.griddata[rtxt]
#    cgrv = lmod.griddata[ctxt]

    data = rgrv
    data2 = cgrv
    orig_wkt = data.wkt
    orig_wkt2 = data2.wkt

    doffset = 0.0
    if data.data.min() <= 0:
        doffset = data.data.min()-1.
        data.data = data.data - doffset

    drows, dcols = data.data.shape
    d2rows, d2cols = data2.data.shape

    gtr0 = data.get_gtr()
    gtr = data2.get_gtr()
    src = data_to_gdal_mem(data, gtr0, orig_wkt, dcols, drows)
    dest = data_to_gdal_mem(data, gtr, orig_wkt2, d2cols, d2rows, True)

    gdal.ReprojectImage(src, dest, orig_wkt, orig_wkt2, gdal.GRA_Bilinear)

    dat = gdal_to_dat(dest, data.dataid)

    if doffset != 0.0:
        dat.data = dat.data + doffset
        data.data = data.data + doffset

    return dat.data


def rotate2d(pts, cntr, ang=np.pi/4):
    """ pts = {} Rotates points(nx2) about center cntr(2) by angle ang(1) in
    radian"""

    trans = np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])
    pts2 = np.dot(pts-cntr, trans) + cntr
    return pts2
