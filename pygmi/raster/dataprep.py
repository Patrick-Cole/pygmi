# -----------------------------------------------------------------------------
# Name:        dataprep.py (part of PyGMI)
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
""" This is a set of Raster Data Preparation routines """

from __future__ import print_function

import os
import copy
from collections import Counter
from PyQt4 import QtGui, QtCore
import numpy as np
from osgeo import gdal, osr, ogr
from PIL import Image, ImageDraw
import scipy.ndimage as ndimage
import pygmi.menu_default as menu_default
from pygmi.raster.datatypes import Data
from pygmi.vector.datatypes import PData

gdal.PushErrorHandler('CPLQuietErrorHandler')


class DataCut(object):
    """
    Cut Data using shapefiles

    This class cuts raster datasets using a boundary defined by a polygon
    shapefile.

    Attributes
    ----------
    ifile : str
        input file name.
    name : str
        item name
    ext : str
        file name extension.
    pbar : progressbar
        reference to a progress bar.
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    outdata : dictionary
        dictionary of output datasets
    """
    def __init__(self, parent):
        self.ifile = ""
        self.name = "Cut Data:"
        self.ext = ""
        self.pbar = parent.pbar
        self.parent = parent
        self.indata = {}
        self.outdata = {}
#        self.dirname = ""

    def settings(self):
        """ Show Info """
#        if 'Cluster' in self.indata:
#            data = self.indata['Cluster']
        if 'Raster' in self.indata:
            data = copy.deepcopy(self.indata['Raster'])
        else:
            self.parent.showprocesslog('No raster data')
            return

        ext = "Shape file (*.shp)"

        filename = QtGui.QFileDialog.getOpenFileName(
            self.parent, 'Open Shape File', '.', ext)
        if filename == '':
            return False
        os.chdir(filename.rpartition('/')[0])

        self.ifile = str(filename)
        self.ext = filename[-3:]
        data = cut_raster(data, self.ifile)

        if data is None:
            err = ('There was a problem importing the shapefile. Please make '
                   'sure you have at all the individual files which make up '
                   'the shapefile.')
            QtGui.QMessageBox.warning(self.parent, 'Error', err,
                                      QtGui.QMessageBox.Ok,
                                      QtGui.QMessageBox.Ok)
            return False


#        data = trim_raster(data)
        self.pbar.to_max()
        self.outdata['Raster'] = data

        return True


class DataGrid(QtGui.QDialog):
    """
    Grid Point Data

    This class grids point data using a nearest neighbourhood technique.

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
        QtGui.QDialog.__init__(self, parent)

        self.indata = {}
        self.outdata = {}
        self.parent = parent
        self.pbar = parent.pbar

        self.dsb_dxy = QtGui.QDoubleSpinBox()
        self.dataid = QtGui.QComboBox()
        self.label_rows = QtGui.QLabel()
        self.label_cols = QtGui.QLabel()

        self.setupui()

    def setupui(self):
        """ Setup UI """
        gridlayout_main = QtGui.QGridLayout(self)
        buttonbox = QtGui.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.raster.dataprep.datagrid')
        label_band = QtGui.QLabel()
        label_dxy = QtGui.QLabel()

        self.dsb_dxy.setMaximum(9999999999.0)
        self.dsb_dxy.setMinimum(0.00001)
        self.dsb_dxy.setDecimals(5)
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle("Dataset Gridding")
        self.label_rows.setText("Rows: 0")
        self.label_cols.setText("Columns: 0")
        label_dxy.setText("Cell Size:")
        label_band.setText("Column to Grid:")

        gridlayout_main.addWidget(label_dxy, 0, 0, 1, 1)
        gridlayout_main.addWidget(self.dsb_dxy, 0, 1, 1, 1)
        gridlayout_main.addWidget(self.label_rows, 1, 0, 1, 2)
        gridlayout_main.addWidget(self.label_cols, 2, 0, 1, 2)
        gridlayout_main.addWidget(label_band, 3, 0, 1, 1)
        gridlayout_main.addWidget(self.dataid, 3, 1, 1, 1)
        gridlayout_main.addWidget(helpdocs, 4, 0, 1, 1)
        gridlayout_main.addWidget(buttonbox, 4, 1, 1, 3)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        self.dsb_dxy.valueChanged.connect(self.dxy_change)

    def dxy_change(self):
        """ update dxy """
        dxy = self.dsb_dxy.value()
        data = self.indata['Point'][0]
        x = data.xdata
        y = data.ydata

        cols = int(x.ptp()/dxy)
        rows = int(y.ptp()/dxy)

        self.label_rows.setText("Rows: "+str(rows))
        self.label_cols.setText("Columns: "+str(cols))

    def settings(self):
        """ Settings """
        tmp = []
        if 'Point' not in self.indata:
            return False

        for i in self.indata['Point']:
            tmp.append(i.dataid)

        self.dataid.addItems(tmp)

        data = self.indata['Point'][0]
        x = data.xdata
        y = data.ydata

        dx = x.ptp()/np.sqrt(x.size)
        dy = y.ptp()/np.sqrt(y.size)
        dxy = max(dx, dy)

#        xy = np.transpose([x, y])
#        xy = xy.tolist()
#        xy = np.array(xy)
#        xy = xy[:-1]-xy[1:]
#        dxy = np.median(np.sqrt(np.sum(xy**2, 1)))/3

        self.dsb_dxy.setValue(dxy)
        self.dxy_change()
        tmp = self.exec_()

        if tmp == 1:
            self.acceptall()
            tmp = True

        return tmp

    def acceptall(self):
        """ accept """
        dxy = self.dsb_dxy.value()
        data = self.indata['Point'][0]

        newdat = []
        for data in self.pbar.iter(self.indata['Point']):
            if data.dataid != self.dataid.currentText():
                continue
            x = data.xdata
            y = data.ydata
            z = data.zdata

            for i in [x, y, z]:
                filt = np.logical_not(np.isnan(i))
                x = x[filt]
                y = y[filt]
                z = z[filt]

            tmp = quickgrid(x, y, z, dxy, showtext=self.parent.showprocesslog)
            mask = tmp.mask
            gdat = tmp.data

    # Create dataset
            dat = Data()
            dat.data = np.ma.masked_invalid(gdat[::-1])
            dat.data.mask = mask[::-1]
            dat.rows, dat.cols = gdat.shape
            dat.nullvalue = dat.data.fill_value
            dat.dataid = data.dataid
            dat.tlx = x.min()
            dat.tly = y.max()
            dat.xdim = dxy
            dat.ydim = dxy
            newdat.append(dat)

        self.outdata['Raster'] = newdat
        self.outdata['Point'] = self.indata['Point']


class DataMerge(QtGui.QDialog):
    """
    Data Merge

    This class merges datasets which have different rows and columns. It
    resamples them so that they have the same rows and columns.

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
        QtGui.QDialog.__init__(self, parent)

        self.indata = {}
        self.outdata = {}
        self.parent = parent
#        self.pbar = self.parent.pbar

        self.dsb_dxy = QtGui.QDoubleSpinBox()
        self.label_rows = QtGui.QLabel()
        self.label_cols = QtGui.QLabel()

        self.setupui()

    def setupui(self):
        """ Setup UI """

        gridlayout_main = QtGui.QGridLayout(self)
        buttonbox = QtGui.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.raster.dataprep.datamerge')
        label_dxy = QtGui.QLabel()

        self.dsb_dxy.setMaximum(9999999999.0)
        self.dsb_dxy.setMinimum(0.00001)
        self.dsb_dxy.setDecimals(5)
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle("Dataset Merge and Resample")
        self.label_rows.setText("Rows: 0")
        self.label_cols.setText("Columns: 0")
        label_dxy.setText("Cell Size:")

        gridlayout_main.addWidget(label_dxy, 0, 0, 1, 1)
        gridlayout_main.addWidget(self.dsb_dxy, 0, 1, 1, 1)
        gridlayout_main.addWidget(self.label_rows, 1, 0, 1, 2)
        gridlayout_main.addWidget(self.label_cols, 2, 0, 1, 2)
        gridlayout_main.addWidget(helpdocs, 3, 0, 1, 1)
        gridlayout_main.addWidget(buttonbox, 3, 1, 1, 1)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        self.dsb_dxy.valueChanged.connect(self.dxy_change)

    def dxy_change(self):
        """ Update dxy - which is the size of a grid cell in the x and y
        directions."""
        data = self.indata['Raster'][0]
        dxy = self.dsb_dxy.value()

        xmin0 = data.tlx
        xmax0 = data.tlx+data.xdim*data.cols
        ymax0 = data.tly
        ymin0 = data.tly-data.ydim*data.rows

        for data in self.indata['Raster']:
            xmin = min(data.tlx, xmin0)
            xmax = max(data.tlx+data.xdim*data.cols, xmax0)
            ymax = max(data.tly, ymax0)
            ymin = min(data.tly-data.ydim*data.rows, ymin0)

        cols = int((xmax - xmin)/dxy)
        rows = int((ymax - ymin)/dxy)

        self.label_rows.setText("Rows: "+str(rows))
        self.label_cols.setText("Columns: "+str(cols))

    def settings(self):
        """ Settings """
        data = self.indata['Raster'][0]
        dxy0 = min(data.xdim, data.ydim)
        for data in self.indata['Raster']:
            dxy = min(dxy0, data.xdim, data.ydim)

        self.dsb_dxy.setValue(dxy)
        tmp = self.exec_()

        if tmp == 1:
            self.acceptall()
            tmp = True

        return tmp

    def acceptall(self):
        """
        This routine is called by settings() if accept is pressed. It contains
        the main merge routine.
        """
        dxy = self.dsb_dxy.value()
        data = self.indata['Raster'][0]
        orig_wkt = data.wkt

        xmin = data.tlx
        xmax = data.tlx+data.xdim*data.cols
        ymax = data.tly
        ymin = data.tly-data.ydim*data.rows

        for data in self.indata['Raster']:
            xmin = min(data.tlx, xmin)
            xmax = max(data.tlx+data.xdim*data.cols, xmax)
            ymax = max(data.tly, ymax)
            ymin = min(data.tly-data.ydim*data.rows, ymin)

        cols = int((xmax - xmin)/dxy)
        rows = int((ymax - ymin)/dxy)
        gtr = (xmin, dxy, 0.0, ymax, 0.0, -dxy)

        if cols == 0 or rows == 0:
            self.parent.showprocesslog("Your rows or cols are zero. " +
                                       "Your input projection may be wrong")
            return

        dat = []
#        for data in self.pbar.iter(self.indata['Raster']):
        for data in self.indata['Raster']:
            doffset = 0.0
            if data.data.min() <= 0:
                doffset = data.data.min()-1.
                data.data -= doffset
            gtr0 = (data.tlx, data.xdim, 0.0, data.tly, 0.0, -data.ydim)
            src = data_to_gdal_mem(data, gtr0, orig_wkt, data.cols, data.rows)
            dest = data_to_gdal_mem(data, gtr, orig_wkt, cols, rows, True)

            gdal.ReprojectImage(src, dest, orig_wkt, orig_wkt,
                                gdal.GRA_Bilinear)

            dat.append(gdal_to_dat(dest, data.dataid))
            dat[-1].data += doffset

#        mask = dat[-1].data.mask
#        for i in dat:
#            mask += i.data.mask
#        mask[mask != 0] = 1
#        for i in dat:
#            i.data.mask = mask

        self.outdata['Raster'] = dat
#        self.accept()


class DataReproj(QtGui.QDialog):
    """
    Reprojections

    This class reprojects datasets using the GDAL routines.

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
        QtGui.QDialog.__init__(self, parent)

        self.indata = {}
        self.outdata = {}
        self.parent = parent
        self.pbar = self.parent.pbar

        self.groupboxb = QtGui.QGroupBox()
        self.combobox_inp_epsg = QtGui.QComboBox()
        self.inp_epsg_info = QtGui.QLabel()
        self.groupbox2b = QtGui.QGroupBox()
        self.combobox_out_epsg = QtGui.QComboBox()
        self.out_epsg_info = QtGui.QLabel()
        self.in_proj = GroupProj('Input Projection')
        self.out_proj = GroupProj('Output Projection')

        self.setupui()

    def setupui(self):
        """ Setup UI """
        gridlayout_main = QtGui.QGridLayout(self)
        buttonbox = QtGui.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.raster.dataprep.datareproj')

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle("Dataset Reprojection")

        gridlayout_main.addWidget(self.in_proj, 0, 0, 1, 1)
        gridlayout_main.addWidget(self.out_proj, 0, 1, 1, 1)
        gridlayout_main.addWidget(helpdocs, 1, 0, 1, 1)
        gridlayout_main.addWidget(buttonbox, 1, 1, 1, 1)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)

    def acceptall(self):
        """
        This routine is called by settings() if accept is pressed. It contains
        the main routine.
        """

# Input stuff
        orig_wkt = self.in_proj.wkt

        orig = osr.SpatialReference()
        orig.ImportFromWkt(orig_wkt)

# Output stuff
        targ_wkt = self.out_proj.wkt

        targ = osr.SpatialReference()
        targ.ImportFromWkt(targ_wkt)

# Set transformation
        ctrans = osr.CoordinateTransformation(orig, targ)

# Now create virtual dataset
        dat = []
        for data in self.pbar.iter(self.indata['Raster']):
            datamin = data.data.min()
            if datamin <= 0:
                data.data = data.data-(datamin-1)

# Work out the boundaries of the new dataset in the target projection
            u_l = ctrans.TransformPoint(data.tlx, data.tly)
            u_r = ctrans.TransformPoint(data.tlx+data.xdim*data.cols, data.tly)
            l_l = ctrans.TransformPoint(data.tlx, data.tly-data.ydim*data.rows)
            l_r = ctrans.TransformPoint(data.tlx+data.xdim*data.cols,
                                        data.tly - data.ydim*data.rows)

            lrx = l_r[0]
            llx = l_l[0]
            ulx = u_l[0]
            urx = u_r[0]
            lry = l_r[1]
            lly = l_l[1]
            uly = u_l[1]
            ury = u_r[1]

            minx = min(llx, ulx, urx, lrx)
            maxx = max(llx, ulx, urx, lrx)
            miny = min(lly, lry, ury, uly)
            maxy = max(lly, lry, ury, uly)
            newdimx = (maxx-minx)/data.cols
            newdimy = (maxy-miny)/data.rows
            newdim = min(newdimx, newdimy)
            cols = round((maxx - minx)/newdim)
            rows = round((maxy - miny)/newdim)

            if cols == 0 or rows == 0:
                self.parent.showprocesslog("Your rows or cols are zero. " +
                                           "Your input projection may be " +
                                           "wrong")
                return

# top left x, w-e pixel size, rotation, top left y, rotation, n-s pixel size
            old_geo = (data.tlx, data.xdim, 0, data.tly, 0, -data.ydim)
            src = data_to_gdal_mem(data, old_geo, orig_wkt, data.cols,
                                   data.rows)

            new_geo = (minx, newdim, 0, maxy, 0, -newdim)
            dest = data_to_gdal_mem(data, new_geo, targ_wkt, cols, rows, True)

            gdal.ReprojectImage(src, dest, orig_wkt, targ_wkt,
                                gdal.GRA_Bilinear)

            data2 = gdal_to_dat(dest, data.dataid)
            if datamin <= 0:
                data2.data = data2.data+(datamin-1)
                data.data = data.data+(datamin-1)
#            mask = data2.data.mask
            data2.data = np.ma.masked_equal(data2.data.filled(data.nullvalue),
                                            data.nullvalue)
#            data2.data.mask = mask
#            data2.data.set_fill_value(data.nullvalue)
            data2.nullvalue = data.nullvalue
            data2.data = np.ma.masked_invalid(data2.data)
            data2.data = np.ma.masked_less(data2.data, data.data.min())
            data2.data = np.ma.masked_greater(data2.data, data.data.max())

            dat.append(data2)

        self.outdata['Raster'] = dat

    def settings(self):
        """ Settings """

        self.in_proj.set_current(self.indata['Raster'][0].wkt)
        self.out_proj.set_current(self.indata['Raster'][0].wkt)

        tmp = self.exec_()
        if tmp == 1:
            self.acceptall()
            tmp = True

        return tmp


class GetProf(object):
    """
    Get a Profile

    This class extracts a profile from a raster dataset using a line shapefile.

    Attributes
    ----------
    ifile : str
        input file name.
    name : str
        item name
    ext : str
        file name extension.
    pbar : progressbar
        reference to a progress bar.
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    outdata : dictionary
        dictionary of output datasets
    """
    def __init__(self, parent):
        self.ifile = ""
        self.name = "Get Profile: "
        self.ext = ""
        self.pbar = parent.pbar
        self.parent = parent
        self.indata = {}
        self.outdata = {}
#        self.dirname = ""

    def settings(self):
        """ Show Info """
#        if 'Cluster' in self.indata:
#            data = self.indata['Cluster']
        if 'Raster' in self.indata:
            data = copy.deepcopy(self.indata['Raster'])
        else:
            self.parent.showprocesslog('No raster data')
            return

        ext = "Shape file (*.shp)"

        filename = QtGui.QFileDialog.getOpenFileName(
            self.parent, 'Open Shape File', '.', ext)
        if filename == '':
            return False
        os.chdir(filename.rpartition('/')[0])

        self.ifile = str(filename)
        self.ext = filename[-3:]

        shapef = ogr.Open(self.ifile)
        if shapef is None:
            err = ('There was a problem importing the shapefile. Please make '
                   'sure you have at all the individual files which make up '
                   'the shapefile.')
            QtGui.QMessageBox.warning(self.parent, 'Error', err,
                                      QtGui.QMessageBox.Ok,
                                      QtGui.QMessageBox.Ok)
            return False

        lyr = shapef.GetLayer()
        line = lyr.GetNextFeature()
        if lyr.GetGeomType() is not ogr.wkbLineString:
            self.parent.showprocesslog('You need lines in that shape file')
            return

        allpoints = []
        for idata in self.pbar.iter(data):
            tmp = line.GetGeometryRef()
            points = tmp.GetPoints()

            x_0, y_0 = points[0]
            x_1, y_1 = points[1]

            bly = idata.tly-idata.ydim*idata.rows
            x_0 = (x_0-idata.tlx)/idata.xdim
            x_1 = (x_1-idata.tlx)/idata.xdim
            y_0 = (y_0-bly)/idata.ydim
            y_1 = (y_1-bly)/idata.ydim
            rcell = int(np.sqrt((x_1-x_0)**2+(y_1-y_0)**2))
#            rdist = np.sqrt((idata.xdim*(x_1-x_0))**2 +
#                            (idata.ydim*(y_1-y_0))**2)

            xxx = np.linspace(x_0, x_1, rcell, False)
            yyy = np.linspace(y_0, y_1, rcell, False)

            tmpprof = ndimage.map_coordinates(idata.data[::-1], [yyy, xxx],
                                              order=1, cval=np.nan)
            xxx = xxx[np.logical_not(np.isnan(tmpprof))]
            yyy = yyy[np.logical_not(np.isnan(tmpprof))]
            tmpprof = tmpprof[np.logical_not(np.isnan(tmpprof))]
            xxx = xxx*idata.xdim+idata.tlx
            yyy = yyy*idata.ydim+bly
#            allpoints.append(np.array([xxx, yyy, tmpprof]))
            allpoints.append(PData())
            allpoints[-1].xdata = xxx
            allpoints[-1].ydata = yyy
            allpoints[-1].zdata = tmpprof
            allpoints[-1].dataid = idata.dataid

        self.outdata['Point'] = allpoints

        return True


class GroupProj(QtGui.QWidget):
    """
    Group Proj

    Custom widget
    """
    def __init__(self, title='Projection', parent=None):
        QtGui.QWidget.__init__(self, parent)

        self.wkt = ''

        self.gridlayout = QtGui.QGridLayout(self)
        self.groupbox = QtGui.QGroupBox()
        self.combobox = QtGui.QComboBox()
        self.label = QtGui.QLabel()

        self.gridlayout.addWidget(self.groupbox, 1, 0, 1, 2)

        self.groupbox.setTitle(title)
        gridlayout = QtGui.QGridLayout(self.groupbox)
        gridlayout.addWidget(self.combobox, 0, 0, 1, 1)
        gridlayout.addWidget(self.label, 1, 0, 1, 1)

        self.epsg_proj = getepsgcodes()
        self.epsg_proj['Current'] = self.wkt
        tmp = list(self.epsg_proj.keys())
        tmp.sort(key=lambda c: c.lower())
        tmp = ['Current']+tmp

        self.combobox.addItems(tmp)
        self.combobox.currentIndexChanged.connect(self.combo_change)

    def set_current(self, wkt):
        """ Sets new wkt for current """
        self.wkt = wkt
        self.epsg_proj['Current'] = self.wkt
        self.combo_change()

    def combo_change(self):
        """ Change Combo """
        indx = self.combobox.currentIndex()
        txt = self.combobox.itemText(indx)

        self.wkt = self.epsg_proj[txt]

        srs = osr.SpatialReference()
        srs.ImportFromWkt(self.wkt)
        self.label.setText(srs.ExportToPrettyWkt())


class Metadata(QtGui.QDialog):
    """
    Edit Metadata

    This class allows the editing of the metadata for a raster dataset using a
    GUI.

    Attributes
    ----------
    name : oldtxt
        old text
    banddata : dictionary
        band data
    bandid : dictionary
        dictionary of strings containing band names.
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    outdata : dictionary
        dictionary of output datasets
    """
    def __init__(self, parent=None):
        QtGui.QDialog.__init__(self, parent)

        self.indata = {}
        self.outdata = {}
        self.banddata = {}
        self.dataid = {}
        self.oldtxt = ''
        self.parent = parent

        self.groupbox = QtGui.QGroupBox()
        self.combobox_bandid = QtGui.QComboBox()
        self.pb_rename_id = QtGui.QPushButton()
        self.lbl_rows = QtGui.QLabel()
        self.lbl_cols = QtGui.QLabel()
        self.inp_epsg_info = QtGui.QLabel()
        self.txt_null = QtGui.QLineEdit()
        self.dsb_tlx = QtGui.QLineEdit()
        self.dsb_tly = QtGui.QLineEdit()
        self.dsb_xdim = QtGui.QLineEdit()
        self.dsb_ydim = QtGui.QLineEdit()
        self.led_units = QtGui.QLineEdit()
        self.lbl_min = QtGui.QLabel()
        self.lbl_max = QtGui.QLabel()
        self.lbl_mean = QtGui.QLabel()

        self.proj = GroupProj('Input Projection')

        self.setupui()

    def setupui(self):
        """ Setup UI """
        gridlayout_main = QtGui.QGridLayout(self)
        buttonbox = QtGui.QDialogButtonBox()

        gridlayout = QtGui.QGridLayout(self.groupbox)
        label_tlx = QtGui.QLabel()
        label_tly = QtGui.QLabel()
        label_xdim = QtGui.QLabel()
        label_ydim = QtGui.QLabel()
        label_null = QtGui.QLabel()
        label_rows = QtGui.QLabel()
        label_cols = QtGui.QLabel()
        label_min = QtGui.QLabel()
        label_max = QtGui.QLabel()
        label_mean = QtGui.QLabel()
        label_units = QtGui.QLabel()
        label_bandid = QtGui.QLabel()

        sizepolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred,
                                       QtGui.QSizePolicy.Expanding)
        self.groupbox.setSizePolicy(sizepolicy)
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle("Dataset Metadata")
        self.pb_rename_id.setText("Rename Band Name")
        self.groupbox.setTitle("Dataset")
        label_bandid.setText('Band Name:')
        label_tlx.setText("Top Left X Coordinate:")
        label_tly.setText("Top Left Y Coordinate:")
        label_xdim.setText("X Dimension:")
        label_ydim.setText("Y Dimension:")
        label_null.setText("Null/Nodata value:")
        label_rows.setText("Rows:")
        label_cols.setText("Columns:")
        label_min.setText("Dataset Minimum:")
        label_max.setText("Dataset Maximum:")
        label_mean.setText("Dataset Mean:")
        label_units.setText("Dataset Units:")

        gridlayout_main.addWidget(label_bandid, 0, 0, 1, 1)
        gridlayout_main.addWidget(self.combobox_bandid, 0, 1, 1, 3)
        gridlayout_main.addWidget(self.pb_rename_id, 1, 1, 1, 3)
        gridlayout_main.addWidget(self.groupbox, 2, 0, 1, 2)
        gridlayout_main.addWidget(self.proj, 2, 2, 1, 2)
        gridlayout_main.addWidget(buttonbox, 4, 0, 1, 4)

        gridlayout.addWidget(label_tlx, 0, 0, 1, 1)
        gridlayout.addWidget(self.dsb_tlx, 0, 1, 1, 1)
        gridlayout.addWidget(label_tly, 1, 0, 1, 1)
        gridlayout.addWidget(self.dsb_tly, 1, 1, 1, 1)
        gridlayout.addWidget(label_xdim, 2, 0, 1, 1)
        gridlayout.addWidget(self.dsb_xdim, 2, 1, 1, 1)
        gridlayout.addWidget(label_ydim, 3, 0, 1, 1)
        gridlayout.addWidget(self.dsb_ydim, 3, 1, 1, 1)
        gridlayout.addWidget(label_null, 4, 0, 1, 1)
        gridlayout.addWidget(self.txt_null, 4, 1, 1, 1)
        gridlayout.addWidget(label_rows, 5, 0, 1, 1)
        gridlayout.addWidget(self.lbl_rows, 5, 1, 1, 1)
        gridlayout.addWidget(label_cols, 6, 0, 1, 1)
        gridlayout.addWidget(self.lbl_cols, 6, 1, 1, 1)
        gridlayout.addWidget(label_min, 7, 0, 1, 1)
        gridlayout.addWidget(self.lbl_min, 7, 1, 1, 1)
        gridlayout.addWidget(label_max, 8, 0, 1, 1)
        gridlayout.addWidget(self.lbl_max, 8, 1, 1, 1)
        gridlayout.addWidget(label_mean, 9, 0, 1, 1)
        gridlayout.addWidget(self.lbl_mean, 9, 1, 1, 1)
        gridlayout.addWidget(label_units, 10, 0, 1, 1)
        gridlayout.addWidget(self.led_units, 10, 1, 1, 1)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)

        self.combobox_bandid.currentIndexChanged.connect(self.update_vals)
        self.pb_rename_id.clicked.connect(self.rename_id)

    def acceptall(self):
        """
        This routine is called by settings() if accept is pressed. It contains
        the main routine.
        """
        wkt = self.proj.wkt

        self.update_vals()
        for tmp in self.indata['Raster']:
            for j in self.dataid.items():
                if j[1] == tmp.dataid:
                    i = self.banddata[j[0]]
                    tmp.dataid = j[0]
                    tmp.tlx = i.tlx
                    tmp.tly = i.tly
                    tmp.xdim = i.xdim
                    tmp.ydim = i.ydim
                    tmp.rows = i.rows
                    tmp.cols = i.cols
                    tmp.nullvalue = i.nullvalue
                    tmp.wkt = wkt
                    tmp.units = i.units
                    if tmp.dataid[-1] == ')':
                        tmp.dataid = tmp.dataid[:tmp.dataid.rfind(' (')]
                    if i.units != '':
                        tmp.dataid += ' ('+i.units+')'
#                    tmp.data.data[tmp.data.data == i.nullvalue] = np.nan
                    tmp.data.mask = (tmp.data.data == i.nullvalue)

    def rename_id(self):
        """ Renames the band name """
        ctxt = str(self.combobox_bandid.currentText())
        (skey, isokay) = QtGui.QInputDialog.getText(
            self.parent, 'Rename Band Name',
            'Please type in the new name for the band',
            QtGui.QLineEdit.Normal, ctxt)

        if isokay:
            self.combobox_bandid.currentIndexChanged.disconnect()
            indx = self.combobox_bandid.currentIndex()
            txt = self.combobox_bandid.itemText(indx)
            self.banddata[skey] = self.banddata.pop(txt)
            self.dataid[skey] = self.dataid.pop(txt)
            self.oldtxt = skey
            self.combobox_bandid.setItemText(indx, skey)
            self.combobox_bandid.currentIndexChanged.connect(self.update_vals)

    def update_vals(self):
        """ Updates the values on the interface """
        odata = self.banddata[self.oldtxt]
        odata.units = self.led_units.text()

        try:
            odata.nullvalue = float(self.txt_null.text())
            odata.tlx = float(self.dsb_tlx.text())
            odata.tly = float(self.dsb_tly.text())
            odata.xdim = float(self.dsb_xdim.text())
            odata.ydim = float(self.dsb_ydim.text())
        except ValueError:
            self.parent.showprocesslog('Value error - abandoning changes')

        indx = self.combobox_bandid.currentIndex()
        txt = self.combobox_bandid.itemText(indx)
        self.oldtxt = txt
        idata = self.banddata[txt]

        self.lbl_cols.setText(str(idata.cols))
        self.lbl_rows.setText(str(idata.rows))
        self.txt_null.setText(str(idata.nullvalue))
        self.dsb_tlx.setText(str(idata.tlx))
        self.dsb_tly.setText(str(idata.tly))
        self.dsb_xdim.setText(str(idata.xdim))
        self.dsb_ydim.setText(str(idata.ydim))
        self.lbl_min.setText(str(idata.min))
        self.lbl_max.setText(str(idata.max))
        self.lbl_mean.setText(str(idata.mean))
        self.led_units.setText(str(idata.units))

    def run(self):
        """ Entrypoint to start this routine """

        bandid = []
        self.proj.set_current(self.indata['Raster'][0].wkt)

        for i in self.indata['Raster']:
            bandid.append(i.dataid)
            self.banddata[i.dataid] = Data()
            tmp = self.banddata[i.dataid]
            self.dataid[i.dataid] = i.dataid
            tmp.tlx = i.tlx
            tmp.tly = i.tly
            tmp.xdim = i.xdim
            tmp.ydim = i.ydim
            tmp.rows = i.rows
            tmp.cols = i.cols
            tmp.nullvalue = i.nullvalue
            tmp.wkt = i.wkt
            tmp.min = i.data.min()
            tmp.max = i.data.max()
            tmp.mean = i.data.mean()
            try:
                tmp.units = i.units
            except AttributeError:
                tmp.units = ''

        self.combobox_bandid.currentIndexChanged.disconnect()
        self.combobox_bandid.addItems(bandid)
        indx = self.combobox_bandid.currentIndex()
        self.oldtxt = self.combobox_bandid.itemText(indx)
        self.combobox_bandid.currentIndexChanged.connect(self.update_vals)

        idata = self.banddata[self.oldtxt]
        self.lbl_cols.setText(str(idata.cols))
        self.lbl_rows.setText(str(idata.rows))
        self.txt_null.setText(str(idata.nullvalue))
        self.dsb_tlx.setText(str(idata.tlx))
        self.dsb_tly.setText(str(idata.tly))
        self.dsb_xdim.setText(str(idata.xdim))
        self.dsb_ydim.setText(str(idata.ydim))
        self.lbl_min.setText(str(idata.min))
        self.lbl_max.setText(str(idata.max))
        self.lbl_mean.setText(str(idata.mean))
        self.led_units.setText(str(idata.units))

        self.update_vals()

        tmp = self.exec_()
        if tmp == 1:
            self.acceptall()
            tmp = True

        return tmp


class RTP(QtGui.QDialog):
    """
    Perform Reduction to the Pole on Magnetic data.

    This class grids point data using a nearest neighbourhood technique.

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
        QtGui.QDialog.__init__(self, parent)

        self.indata = {}
        self.outdata = {}
        self.parent = parent
        self.pbar = self.parent.pbar

        self.dataid = QtGui.QComboBox()
        self.dsb_inc = QtGui.QDoubleSpinBox()
        self.dsb_dec = QtGui.QDoubleSpinBox()

        self.setupui()

    def setupui(self):
        """ Setup UI """
        gridlayout_main = QtGui.QGridLayout(self)
        buttonbox = QtGui.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.raster.dataprep.rtp')
        label_band = QtGui.QLabel()
        label_inc = QtGui.QLabel()
        label_dec = QtGui.QLabel()

        self.dsb_inc.setMaximum(90.0)
        self.dsb_inc.setMinimum(-90.0)
        self.dsb_dec.setMaximum(360.0)
        self.dsb_dec.setMinimum(-360.0)
#        self.dsb_dxy.setDecimals(5)
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle("Reduction to the Pole")
        label_band.setText("Band to Reduce to the Pole:")
        label_inc.setText("Inclination of Magnetic Field:")
        label_dec.setText("Declination of Magnetic Field:")

        gridlayout_main.addWidget(label_band, 0, 0, 1, 1)
        gridlayout_main.addWidget(self.dataid, 0, 1, 1, 1)

        gridlayout_main.addWidget(label_inc, 1, 0, 1, 1)
        gridlayout_main.addWidget(self.dsb_inc, 1, 1, 1, 1)
        gridlayout_main.addWidget(label_dec, 2, 0, 1, 1)
        gridlayout_main.addWidget(self.dsb_dec, 2, 1, 1, 1)
        gridlayout_main.addWidget(helpdocs, 3, 0, 1, 1)
        gridlayout_main.addWidget(buttonbox, 3, 1, 1, 3)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)

    def settings(self):
        """ Settings """
        tmp = []
        if 'Raster' not in self.indata:
            return False

        for i in self.indata['Raster']:
            tmp.append(i.dataid)

        self.dataid.addItems(tmp)

        self.dsb_inc.setValue(-62.5)
        self.dsb_dec.setValue(-16.75)
        tmp = self.exec_()

        if tmp == 1:
            self.acceptall()
            tmp = True

        return tmp

    def acceptall(self):
        """ accept """
        I_deg = self.dsb_inc.value()
        D_deg = self.dsb_dec.value()

        newdat = []
        for data in self.pbar.iter(self.indata['Raster']):
            if data.dataid != self.dataid.currentText():
                continue
            dat = rtp(data, I_deg, D_deg)
            newdat.append(dat)

        self.outdata['Raster'] = newdat


def rtp(data, I_deg, D_deg):
    """ Reduction to the Pole """

    datamedian = np.ma.median(data.data)
    ndat = data.data - datamedian
    ndat[ndat.mask] = 0

    fftmod = np.fft.fft2(ndat)

    ny, nx = fftmod.shape
    nyqx = 1/(2*data.xdim)
    nyqy = 1/(2*data.ydim)

    kx = np.linspace(-nyqx, nyqx, nx)
    ky = np.linspace(-nyqy, nyqy, ny)

    KX, KY = np.meshgrid(kx, ky)

    I = np.deg2rad(I_deg)
    D = np.deg2rad(D_deg)
    alpha = np.arctan2(KY, KX)

    filt = 1/(np.sin(I)+1j*np.cos(I)*np.cos(D-alpha))**2

    zrtp = np.fft.ifft2(fftmod*filt)
    zrtp = zrtp.real + datamedian

# Create dataset
    dat = Data()
    dat.data = np.ma.masked_invalid(zrtp)
    dat.data.mask = data.data.mask
    dat.rows, dat.cols = zrtp.shape
    dat.nullvalue = data.data.fill_value
    dat.dataid = data.dataid
    dat.tlx = data.tlx
    dat.tly = data.tly
    dat.xdim = data.xdim
    dat.ydim = data.ydim

    return dat


def check_dataid(out):
    """ Checks dataid for duplicates and renames where necessary """
    tmplist = []
    for i in out:
        tmplist.append(i.dataid)

    tmpcnt = Counter(tmplist)
    for elt, count in tmpcnt.items():
        j = 1
        for i in out:
            if elt == i.dataid and count > 1:
                i.dataid += '('+str(j)+')'
                j += 1

    return out


def cluster_to_raster(indata):
    """ Converts cluster datasets to raster datasets

    Some routines will not understand the datasets produced by cluster
    analysis routines, since they are designated 'Cluster' and not 'Raster'.
    This provides a work-around for that.

    Parameters
    ----------
    indata : Data
        PyGMI raster dataset

    Returns
    -------
    Data
        PyGMI raster dataset

    """
    if 'Cluster' not in indata:
        return indata
    if 'Raster' not in indata:
        indata['Raster'] = []

    for i in indata['Cluster']:
        indata['Raster'].append(i)
        indata['Raster'][-1].data += 1

    return indata


def cut_raster(data, ifile):
    """Cuts a raster dataset

    Cut a raster dataset using a shapefile

    Parameters
    ----------
    data : Data
        PyGMI Dataset
    ifile : str
        shapefile used to cut data

    Returns
    -------
    Data
        PyGMI Dataset
    """
    shapef = ogr.Open(ifile)
    if shapef is None:
        return None
    lyr = shapef.GetLayer()
    poly = lyr.GetNextFeature()
    if lyr.GetGeomType() is not ogr.wkbPolygon or poly is None:
        # self.parent.showprocesslog('You need polygons in that shape file')
        return

    for idata in data:
        # Convert the layer extent to image pixel coordinates
        minX, maxX, minY, maxY = lyr.GetExtent()
        ulX = max(0, int((minX - idata.tlx) / idata.xdim))
        ulY = max(0, int((idata.tly - maxY) / idata.ydim))
        lrX = int((maxX - idata.tlx) / idata.xdim)
        lrY = int((idata.tly - minY) / idata.ydim)

        # Map points to pixels for drawing the
        # boundary on a mas image
        points = []
        pixels = []
        geom = poly.GetGeometryRef()

        ifin = 0
        imax = 0
        if geom.GetGeometryName() == 'MULTIPOLYGON':
            for i in range(geom.GetGeometryCount()):
                geom.GetGeometryRef(i)
                itmp = geom.GetGeometryRef(i)
                itmp = itmp.GetGeometryRef(0).GetPointCount()
                if itmp > imax:
                    imax = itmp
                    ifin = i
            geom = geom.GetGeometryRef(ifin)

        pts = geom.GetGeometryRef(0)
        for p in range(pts.GetPointCount()):
            points.append((pts.GetX(p), pts.GetY(p)))
        for p in points:
            tmpx = int((p[0] - idata.tlx) / idata.xdim)
            tmpy = int((idata.tly - p[1]) / idata.ydim)
            pixels.append((tmpx, tmpy))
        rasterPoly = Image.new("L", (idata.cols, idata.rows), 1)
        rasterize = ImageDraw.Draw(rasterPoly)
        rasterize.polygon(pixels, 0)
        mask = np.array(rasterPoly)

        idata.data.mask = mask
        idata.data = idata.data[ulY:lrY, ulX:lrX]
        idata.cols = idata.data.shape[1]
        idata.rows = idata.data.shape[0]
        idata.tlx = ulX*idata.xdim + idata.tlx  # minX
        idata.tly = idata.tly - ulY*idata.ydim  # maxY
    data = trim_raster(data)
    return data


def dat_extent(dat):
    """ Gets the extend of the dat variable """
    left = dat.tlx
    top = dat.tly
    right = left + dat.cols*dat.xdim
    bottom = top - dat.rows*dat.ydim
    return (left, right, bottom, top)


def data_to_gdal_mem(data, gtr, wkt, cols, rows, nodata=False):
    """
    Data to GDAL mem format

    Parameters
    ----------
    data : PyGMI Data
        PyGMI Dataset
    gtr : tuple
        Geotransform
    wkt : str
        Projection in wkt (well known text) format
    cols : int
        columns
    rows : int
        rows
    nodata : bool, optional
        no data

    Returns
    -------
    src - GDAL mem format
    """
    data.data = np.ma.array(data.data)
    dtype = data.data.dtype
# Get rid of array() which can break driver.create later
    cols = int(cols)
    rows = int(rows)

    if dtype == np.uint8:
        fmt = gdal.GDT_Byte
    elif dtype == np.int32:
        fmt = gdal.GDT_Int32
    elif dtype == np.float64:
        fmt = gdal.GDT_Float64
    else:
        fmt = gdal.GDT_Float32

    driver = gdal.GetDriverByName('MEM')
    src = driver.Create('', cols, rows, 1, fmt)

    src.SetGeoTransform(gtr)
    src.SetProjection(wkt)

    if nodata is False:
        if data.nullvalue is not None:
            src.GetRasterBand(1).SetNoDataValue(data.nullvalue)
        src.GetRasterBand(1).WriteArray(data.data)
    else:
        tmp = np.ma.masked_all((rows, cols))
        src.GetRasterBand(1).SetNoDataValue(0)  # Set to this because of Reproj
        src.GetRasterBand(1).WriteArray(tmp)

    return src


def epsgtowkt(epsg):
    """ Convenience routine to get a wkt from an epsg code """
    orig = osr.SpatialReference()
    err = orig.ImportFromEPSG(int(epsg))
    if err != 0:
        return ''
    out = orig.ExportToWkt()
    return out


def gdal_to_dat(dest, bandid='Data'):
    """
    GDAL to Data format

    Parameters
    ----------
    dest - GDAL format
        GDAL format
    bandid - str
        band identity
    """
    dat = Data()
    gtr = dest.GetGeoTransform()

    rtmp = dest.GetRasterBand(1)
    dat.data = rtmp.ReadAsArray()
    nval = rtmp.GetNoDataValue()

    dat.data[np.isnan(dat.data)] = nval
    dat.data[np.isinf(dat.data)] = nval
    dat.data = np.ma.masked_equal(dat.data, nval)

#    dtype = dat.data.dtype
#    nval = np.nan
#    if dtype == np.float32 or dtype == np.float64:
#        dat.data[dat.data == 0.] = np.nan
# #    dat.data[dat.data == rtmp.GetNoDataValue()] = np.nan
#        dat.data = np.ma.masked_invalid(nval)
#
#    if dtype == np.uint8:
#        dat.data = np.ma.masked_equal(dat.data, 0)
#        nval = 0

#    dat.data[dat.data.mask] = rtmp.GetNoDataValue()

    dat.nrofbands = dest.RasterCount
    dat.tlx = gtr[0]
    dat.tly = gtr[3]
    dat.dataid = bandid
    dat.nullvalue = nval
    dat.rows = dest.RasterYSize
    dat.cols = dest.RasterXSize
    dat.xdim = abs(gtr[1])
    dat.ydim = abs(gtr[5])
    dat.wkt = dest.GetProjection()
    dat.gtr = gtr

    return dat


def getepsgcodes():
    """
    Convenience function used to get a list of EPSG codes
    """

    dfile = open(os.environ['GDAL_DATA']+'\\gcs.csv')
    dlines = dfile.readlines()
    dfile.close()

    dlines = dlines[1:]
    dcodes = {}
    for i in dlines:
        tmp = i.split(',')
        if tmp[1][0] == '"':
            tmp[1] = tmp[1][1:-1]
        wkttmp = epsgtowkt(tmp[0])
        if wkttmp != '':
            dcodes[tmp[1]] = wkttmp

    pfile = open(os.environ['GDAL_DATA']+'\\pcs.csv')
    plines = pfile.readlines()
    pfile.close()

    orig = osr.SpatialReference()

    pcodes = {}
    for i in dcodes:
        pcodes[i+r' / Geodetic Geographic'] = dcodes[i]

    plines = plines[1:]
    for i in plines:
        tmp = i.split(',')
        if tmp[1][0] == '"':
            tmp[1] = tmp[1][1:-1]
#        wkttmp = epsgtowkt(tmp[0])
        err = orig.ImportFromEPSG(int(tmp[0]))
        if err == 0:
            pcodes[tmp[1]] = orig.ExportToWkt()
#        if wkttmp != '':
#            pcodes[tmp[1]] = wkttmp

    clat = 0.
    scale = 1.
    f_e = 0.
    f_n = 0.
    orig = osr.SpatialReference()

    for datum in ['Cape', 'Hartebeesthoek94']:
        orig.ImportFromWkt(dcodes[datum])
        for clong in range(15, 35, 2):
            orig.SetTM(clat, clong, scale, f_e, f_n)
            orig.SetProjCS(datum+r' / TM'+str(clong))
            pcodes[datum+r' / TM'+str(clong)] = orig.ExportToWkt()

    return pcodes


def merge(dat):
    """ Merges datasets found in a single PyGMI data object.

    The aim is to ensure that all datasets have the same number of rows and
    columns.

    Parameters
    ----------
    dat : Data
        data object which stores datasets

    Returns
    -------
    Data
        data object which stores datasets
    """
    needsmerge = False
    for i in dat:
        if i.rows != dat[0].rows or i.cols != dat[0].cols:
            needsmerge = True

    if needsmerge is False:
        dat = check_dataid(dat)
        return dat

    mrg = DataMerge()
    mrg.indata['Raster'] = dat
    data = dat[0]
    dxy0 = min(data.xdim, data.ydim)
    for data in dat:
        dxy = min(dxy0, data.xdim, data.ydim)

    mrg.dsb_dxy.setValue(dxy)
    mrg.acceptall()
    out = mrg.outdata['Raster']

    out = check_dataid(out)

    return out

#def taper(data):
#    nr, nc = data.shape
#    nmax = np.max([nr, nc])
#    npts = int(2**cooper.__nextpow2(nmax))
#    npts *= 2
#
#    cdiff = int(np.floor((npts-nc)/2))
#    rdiff = int(np.floor((npts-nr)/2))
#    data1 = cooper.__taper2d(data, npts, nc, nr, cdiff, rdiff)
##    data1 = np.pad(data-np.median(data), ((rdiff, cdiff), (rdiff,cdiff)), 'edge')

#    return data1


def trim_raster(olddata):
    """ Function to trim nulls from a raster dataset.

    This function trims entire rows or columns of data which have only nulls,
    and are on the edges of the dataset.

    Parameters
    ----------
    olddata : Data
        PyGMI dataset

    Returns
    -------
    Data
        PyGMI dataset
    """

    for data in olddata:
        mask = data.data.mask
        data.data[mask] = data.nullvalue

        rowstart = 0
        for i in range(mask.shape[0]):
            if bool(mask[i].min()) is False:
                break
            rowstart += 1

        rowend = mask.shape[0]
        for i in range(mask.shape[0]-1, -1, -1):
            if bool(mask[i].min()) is False:
                break
            rowend -= 1

        colstart = 0
        for i in range(mask.shape[1]):
            if bool(mask[:, i].min()) is False:
                break
            colstart += 1

        colend = mask.shape[1]
        for i in range(mask.shape[1]-1, -1, -1):
            if bool(mask[:, i].min()) is False:
                break
            colend -= 1

        data.data = data.data[rowstart:rowend, colstart:colend]
        data.data.mask = (data.data.data == data.nullvalue)
#        data.data = np.ma.masked_invalid(data.data[rowstart:rowend,
#                                                   colstart:colend])
        data.rows, data.cols = data.data.shape
        data.tlx = data.tlx + colstart*data.xdim
        data.tly = data.tly - rowstart*data.ydim

    return olddata


def quickgrid(x, y, z, dxy, showtext=None, numits=4):
    """
    Do a quick grid

    Parameters
    ----------
    x : numpy array
        array of x coordinates
    y : numpy array
        array of y coordinates
    z : numpy array
        array of z values - this is the column being gridded
    dxy : float
        cell size for the grid, in both the x and y direction.
    showtext : module, optional
        showtext provided an alternative to print
    numits : int
        number of iterations. By default its 4. If this is negative, a maximum
        numits will be calculated and used.

    Returns
    -------
    newz : numpy array
        M x N array of z values
    """

    if showtext is None:
        showtext = print

    showtext('Creating Grid')
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    newmask = np.array([1])
    j = -1
    rows = int((ymax-ymin)/dxy)+1
    cols = int((xmax-xmin)/dxy)+1

    if numits < 1:
        numits = int(max(np.log2(cols), np.log2(rows)))

    while np.max(newmask) > 0 and j < (numits-1):
        j += 1
        jj = 2**j
        dxy2 = dxy*jj
        rows = int((ymax-ymin)/dxy2)+1
        cols = int((xmax-xmin)/dxy2)+1

        newz = np.zeros([rows, cols])
        zdiv = np.zeros([rows, cols])

        xindex = ((x-xmin)/dxy2).astype(int)
        yindex = ((y-ymin)/dxy2).astype(int)

        for i in range(z.size):
            newz[yindex[i], xindex[i]] += z[i]
            zdiv[yindex[i], xindex[i]] += 1

        filt = zdiv > 0
        newz[filt] = newz[filt]/zdiv[filt]

        if j == 0:
            newmask = np.ones([rows, cols])
            for i in range(z.size):
                newmask[yindex[i], xindex[i]] = 0
            zfin = newz
        else:
            xx, yy = newmask.nonzero()
            xx2 = xx//jj
            yy2 = yy//jj
            zfin[xx, yy] = newz[xx2, yy2]
            newmask[xx, yy] = np.logical_not(zdiv[xx2, yy2])

        showtext('Iteration done: '+str(j+1)+' of '+str(numits))

    showtext('Finished!')

    newz = np.ma.array(zfin)
    newz.mask = newmask
    return newz


def tests_rtp():
    """ Tests to debug RTP """
    from pygmi.raster.iodefs import get_raster
    import matplotlib.pyplot as plt

    datrtp = get_raster(r'C:\Work\Programming\pygmi\data\RTP\South_Africa_EMAG2_diffRTP_surfer.grd')
    dat = get_raster(r'C:\Work\Programming\pygmi\data\RTP\South_Africa_EMAG2_TMI_surfer.grd')
    dat = dat[0]
    datrtp = datrtp[0]
    incl = -65.
    decl = -22.

    dat2 = rtp(dat, incl, decl)

    plt.subplot(2, 1, 1)
    plt.imshow(dat.data, vmin=-1200, vmax=1200)
    plt.colorbar()
    plt.subplot(2, 1, 2)
    plt.imshow(dat2.data, vmin=-1200, vmax=1200)
    plt.colorbar()
    plt.show()


def func(x, y):
    """ Function """
    return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2


def tests():
    """ Tests to debug """
    import matplotlib.pyplot as plt
    from pygmi.misc import PTime
    import pdb
    import sys

    app = QtGui.QApplication(sys.argv)

    ttt = PTime()
    aaa = GroupProj('Input Projection')

    ttt.since_last_call()
    pdb.set_trace()

    points = np.random.rand(1000000, 2)
    values = func(points[:, 0], points[:, 1])


    dat = quickgrid(points[:, 0], points[:, 1], values, .001, numits=-1)

#    ttt.since_last_call()

#    grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]
#    dat = griddata(points, values, (grid_x, grid_y), method='nearest')

    plt.imshow(dat)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    tests()
