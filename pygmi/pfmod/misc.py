# -----------------------------------------------------------------------------
# Name:        misc.py (part of PyGMI)
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
"""These are miscellaneous functions for the program."""

import time
from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
from osgeo import gdal
import pygmi.menu_default as menu_default
from pygmi.raster.dataprep import data_to_gdal_mem
from pygmi.raster.dataprep import gdal_to_dat


def update_lith_lw(lmod, lwidget):
    """
    Update the lithology list widget.

    Parameters
    ----------
    lmod : LithModel
        3D model.
    lwidget : QListWidget
        List widget.

    Returns
    -------
    None.

    """
    lwidget.clear()
    for i in lmod.lith_list:
        lwidget.addItem(i)

    for i in range(lwidget.count()):
        tmp = lwidget.item(i)
        tindex = lmod.lith_list[str(tmp.text())].lith_index
        tcol = lmod.mlut[tindex]
        tmp.setBackground(QtGui.QColor(tcol[0], tcol[1], tcol[2], 255))

        L = (tcol[0]*299 + tcol[1]*587 + tcol[2]*114)/1000.
        if L > 128.:
            tmp.setForeground(QtGui.QColor('black'))
        else:
            tmp.setForeground(QtGui.QColor('white'))


class ProgressBar():
    """Wrapper for a progress bar. It consists of two progress bars."""

    def __init__(self, pbar, pbarmain):
        self.pbar = pbar
        self.pbarmain = pbarmain
        self.max = 1
        self.value = 0
        self.mmax = 1
        self.mvalue = 0
        self.otime = None
        self.mtime = None
        self.resetall()

    def incr(self):
        """
        Increase value by one.

        Returns
        -------
        None.

        """
        if self.value < self.max:
            self.value += 1
            if self.value == self.max and self.mvalue < self.mmax:
                self.mvalue += 1
                self.value = 0
                self.pbarmain.setValue(self.mvalue)
            if self.mvalue == self.mmax:
                self.value = self.max
            self.pbar.setValue(self.value)

        QtCore.QCoreApplication.processEvents()

    def iter(self, iterable):
        """
        Iterate routine.

        Parameters
        ----------
        iterable : iterable
            Iterable.

        Yields
        ------
        obj : object
            Object in iterable.

        """
        total = len(iterable)
        self.max = total
        self.pbar.setMaximum(total)
        self.pbar.setMinimum(0)
        self.pbar.setValue(0)

        self.otime = time.perf_counter()
        time1 = self.otime
        time2 = self.otime

        i = 0
        for obj in iterable:
            yield obj
            i += 1

            time2 = time.perf_counter()
            if time2-time1 > 1:
                self.pbar.setValue(i)
                tleft = (total-i)*(time2-self.otime)/i
                if tleft > 60:
                    tleft = int(tleft // 60)
                    self.pbar.setFormat('%p% '+str(tleft)+'min left')
                else:
                    tleft = int(tleft)
                    self.pbar.setFormat('%p% '+str(tleft)+'s left')
                QtWidgets.QApplication.processEvents()
                time1 = time2

        self.pbar.setFormat('%p%')
        self.pbar.setValue(total)

        self.incrmain()
        self.value = 0
        QtWidgets.QApplication.processEvents()

    def incrmain(self, i=1):
        """
        Increase value by i.

        Parameters
        ----------
        i : int, optional
            Iteration step. The default is 1.

        Returns
        -------
        None.

        """
        self.mvalue += i
        self.pbarmain.setValue(self.mvalue)

        n = self.mvalue
        total = self.mmax
        tleft = (total-n)*(time.perf_counter()-self.mtime)/n
        if tleft > 60:
            tleft = int(tleft // 60)
            self.pbarmain.setFormat('%p% '+str(tleft)+'min left')
        else:
            tleft = int(tleft)
            self.pbarmain.setFormat('%p% '+str(tleft)+'s left')
        QtWidgets.QApplication.processEvents()

    def maxall(self):
        """
        Set all progress bars to maximum value.

        Returns
        -------
        None.

        """
        self.mvalue = self.mmax
        self.value = self.max
        self.pbarmain.setValue(self.mvalue)
        self.pbar.setValue(self.value)
        self.pbar.setFormat('%p%')
        self.pbarmain.setFormat('%p%')

    def resetall(self, maximum=1, mmax=1):
        """
        Set min and max and resets all bars to 0.

        Parameters
        ----------
        maximum : int, optional
            Maximum value. The default is 1.
        mmax : int, optional
            Maximum value. The default is 1.

        Returns
        -------
        None.

        """
        self.pbar.setFormat('%p%')
        self.pbarmain.setFormat('%p%')
        self.mtime = time.perf_counter()

        self.max = maximum
        self.value = 0
        self.mmax = mmax
        self.mvalue = 0
        self.pbar.setMinimum(self.value)
        self.pbar.setMaximum(self.max)
        self.pbar.setValue(self.value)
        self.pbarmain.setMinimum(self.mvalue)
        self.pbarmain.setMaximum(self.mmax)
        self.pbarmain.setValue(self.mvalue)

    def resetsub(self, maximum=1):
        """
        Set min and max and resets sub bar to 0.

        Parameters
        ----------
        maximum : int, optional
            Maximum value. The default is 1.

        Returns
        -------
        None.

        """
        self.pbar.setFormat('%p%')
        self.max = maximum
        self.value = 0
        self.pbar.setMinimum(self.value)
        self.pbar.setMaximum(self.max)
        self.pbar.setValue(self.value)

    def busysub(self):
        """
        Busy.

        Returns
        -------
        None.

        """
        self.pbar.setMinimum(0)
        self.pbar.setMaximum(0)
        self.pbar.setValue(-1)
        QtCore.QCoreApplication.processEvents()


class MergeMod3D(QtWidgets.QDialog):
    """
    Perform Merge of two models.

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    outdata : dictionary
        dictionary of output datasets
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.indata = {}
        self.outdata = {}
        self.parent = parent
        self.pbar = self.parent.pbar

        self.master = QtWidgets.QComboBox()
        self.slave = QtWidgets.QComboBox()
        if parent is None:
            self.showprocesslog = print
        else:
            self.showprocesslog = parent.showprocesslog

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gridlayout_main = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.pfmod.misc.mergemod3d')
        label_master = QtWidgets.QLabel('Master Dataset:')
        label_slave = QtWidgets.QLabel('Slave Dataset:')

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('3D Model Merge')

        gridlayout_main.addWidget(label_master, 0, 0, 1, 1)
        gridlayout_main.addWidget(self.master, 0, 1, 1, 1)

        gridlayout_main.addWidget(label_slave, 1, 0, 1, 1)
        gridlayout_main.addWidget(self.slave, 1, 1, 1, 1)
        gridlayout_main.addWidget(helpdocs, 3, 0, 1, 1)
        gridlayout_main.addWidget(buttonbox, 3, 1, 1, 3)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)

    def settings(self, nodialog=False):
        """
        Entry point into item.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        tmp = []
        if 'Model3D' not in self.indata:
            return False
        if len(self.indata['Model3D']) != 2:
            self.showprocesslog('You need two datasets connected!')
            return False

        for i in self.indata['Model3D']:
            tmp.append(i.name)

        self.master.addItems(tmp)
        self.slave.addItems(tmp)

        self.master.setCurrentIndex(0)
        self.slave.setCurrentIndex(1)

        tmp = self.exec_()

        if tmp == 1:
            tmp = self.acceptall()

        return tmp

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

    def acceptall(self):
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        bool
            True if successful, False otherwise

        """
        if self.master.currentText() == self.slave.currentText():
            self.showprocesslog('Your master dataset must be different'
                                ' to the slave dataset!')
            return False

        for data in self.indata['Model3D']:
            if data.name == self.master.currentText():
                datmaster = data
            if data.name == self.slave.currentText():
                datslave = data

        xrange = list(datmaster.xrange) + list(datslave.xrange)
        xrange.sort()
        xrange = [xrange[0], xrange[-1]]
        yrange = list(datmaster.yrange) + list(datslave.yrange)
        yrange.sort()
        yrange = [yrange[0], yrange[-1]]
        zrange = list(datmaster.zrange) + list(datslave.zrange)
        zrange.sort()
        zrange = [zrange[0], zrange[-1]]

        dxy = datmaster.dxy
        d_z = datmaster.d_z

        utlx = xrange[0]
        utly = yrange[-1]
        utlz = zrange[-1]

        xextent = xrange[-1]-xrange[0]
        yextent = yrange[-1]-yrange[0]
        zextent = zrange[-1]-zrange[0]

        cols = int(xextent//dxy)
        rows = int(yextent//dxy)
        layers = int(zextent//d_z)

        self.outdata['Raster'] = []

        for i in datmaster.griddata:
            if (i in ('DTM Dataset', 'Magnetic Dataset',
                      'Gravity Dataset', 'Study Area Dataset',
                      'Gravity Regional')):
                if i in datslave.griddata:
                    datmaster.griddata[i] = gmerge(datmaster.griddata[i],
                                                   datslave.griddata[i],
                                                   xrange, yrange)
                self.outdata['Raster'].append(datmaster.griddata[i])

        datmaster.update(cols, rows, layers, utlx, utly, utlz, dxy, d_z,
                         usedtm=False)
        datslave.update(cols, rows, layers, utlx, utly, utlz, dxy, d_z,
                        usedtm=False)

        lithcnt = 0
        newmlut = {0: datmaster.mlut[0]}
        newslut = {0: datslave.mlut[0]}
        all_liths = list(set(datmaster.lith_list) | set(datslave.lith_list))

        for lith in all_liths:
            if lith == 'Background':
                continue
            lithcnt += 1
            if lith in datslave.lith_list:
                oldlithindex = datslave.lith_list[lith].lith_index
                newslut[lithcnt] = datslave.mlut[oldlithindex]
                tmp = (datslave.lith_index == oldlithindex)
                datslave.lith_index[tmp] = lithcnt
                datslave.lith_list[lith].lith_index = lithcnt

            if lith in datmaster.lith_list:
                oldlithindex = datmaster.lith_list[lith].lith_index
                newmlut[lithcnt] = datmaster.mlut[oldlithindex]
                tmp = (datmaster.lith_index == oldlithindex)
                datmaster.lith_index[tmp] = lithcnt
                datmaster.lith_list[lith].lith_index = lithcnt

        datslave.mlut = newslut
        datmaster.mlut = newmlut
        datmaster.lith_index[datmaster.lith_index == 0] = \
            datslave.lith_index[datmaster.lith_index == 0]
#        datmaster.lith_index[datmaster.lith_index > 9000] -= 9000

        for lith in datslave.lith_list:
            if lith not in datmaster.lith_list:
                datmaster.lith_list[lith] = datslave.lith_list[lith]
                lithnum = datmaster.lith_list[lith].lith_index
                datmaster.mlut[lithnum] = datslave.mlut[lithnum]

        self.outdata['Model3D'] = [datmaster]
        return True


def gmerge(master, slave, xrange=None, yrange=None):
    """
    Routine used to merge two grids.

    Parameters
    ----------
    master : PyGMI Data
        PyGMI raster dataset.
    slave : PyGMI Data
        PyGMI raster dataset.
    xrange : list, optional
        List containg range of minimum and maximum X. The default is None.
    yrange : list, optional
        List containg range of minimum and maximum Y. The default is None.

    Returns
    -------
    PyGMI Data
        PyGMI raster dataset.

    """
    if xrange is None or yrange is None:
        return master

    xdim = master.xdim
    ydim = master.ydim
    orig_wkt = master.wkt

    xmin = xrange[0]
    xmax = xrange[-1]
    ymin = yrange[0]
    ymax = yrange[-1]

    cols = int((xmax - xmin)//xdim)+1
    rows = int((ymax - ymin)//ydim)+1
    gtr = (xmin, xdim, 0.0, ymax, 0.0, -ydim)

    dat = []

    for data in [master, slave]:
        doffset = 0.
        data.data = data.data.astype(float)
        if data.data.min() <= 0.:
            doffset = data.data.min()-1.
            data.data = data.data - doffset
        data.data.set_fill_value(0.)
        tmp = data.data.filled()
        data.data = np.ma.masked_equal(tmp, 0.)
        data.nullvalue = 0

        drows, dcols = data.data.shape

        gtr0 = data.get_gtr()
        src = data_to_gdal_mem(data, gtr0, orig_wkt, dcols, drows)
        dest = data_to_gdal_mem(data, gtr, orig_wkt, cols, rows, True)

        gdal.ReprojectImage(src, dest, orig_wkt, orig_wkt, gdal.GRA_Bilinear)

        dat.append(gdal_to_dat(dest, data.dataid))
        dat[-1].data = np.ma.masked_outside(dat[-1].data, 0.1,
                                            data.data.max() + 1000)
        dat[-1].data = dat[-1].data + doffset
        dat[-1].data.set_fill_value(1e+20)
        tmp = dat[-1].data.filled()
        dat[-1].data = np.ma.masked_equal(tmp, 1e+20)
        dat[-1].nullvalue = 1e+20

    imask = np.logical_and(dat[0].data.mask, np.logical_not(dat[1].data.mask))
    if imask.size > 1:
        dat[0].data.data[imask] = dat[1].data.data[imask]
        dat[0].data.mask[imask] = False

    return dat[0]
