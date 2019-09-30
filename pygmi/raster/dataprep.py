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
"""A set of Raster Data Preparation routines."""

from __future__ import print_function

import os
import copy
from collections import Counter
from PyQt5 import QtWidgets, QtCore
import numpy as np
from osgeo import gdal, osr, ogr
from PIL import Image, ImageDraw
import scipy.ndimage as ndimage
import pygmi.menu_default as menu_default
from pygmi.raster.datatypes import Data
from pygmi.vector.datatypes import PData

gdal.PushErrorHandler('CPLQuietErrorHandler')


class DataCut():
    """
    Cut Data using shapefiles.

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
        self.ifile = ''
        self.name = 'Cut Data:'
        self.ext = ''
        self.pbar = parent.pbar
        self.parent = parent
        self.indata = {}
        self.outdata = {}

    def settings(self):
        """Settings."""
        if 'Raster' in self.indata:
            data = copy.deepcopy(self.indata['Raster'])
        else:
            self.parent.showprocesslog('No raster data')
            return False

        ext = 'Shape file (*.shp)'

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.parent, 'Open Shape File', '.', ext)
        if filename == '':
            return False
        os.chdir(os.path.dirname(filename))

        self.ifile = str(filename)
        self.ext = filename[-3:]
        data = cut_raster(data, self.ifile)

        if data is None:
            err = ('There was a problem importing the shapefile. Please make '
                   'sure you have at all the individual files which make up '
                   'the shapefile.')
            QtWidgets.QMessageBox.warning(self.parent, 'Error', err,
                                          QtWidgets.QMessageBox.Ok)
            return False

        self.pbar.to_max()
        self.outdata['Raster'] = data

        return True


class DataGrid(QtWidgets.QDialog):
    """
    Grid Point Data.

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
        QtWidgets.QDialog.__init__(self, parent)

        self.indata = {}
        self.outdata = {}
        self.parent = parent
        self.pbar = parent.pbar

        self.dsb_dxy = QtWidgets.QDoubleSpinBox()
        self.dsb_null = QtWidgets.QDoubleSpinBox()
        self.dataid = QtWidgets.QComboBox()
        self.label_rows = QtWidgets.QLabel('Rows: 0')
        self.label_cols = QtWidgets.QLabel('Columns: 0')

        self.setupui()

    def setupui(self):
        """Set up UI."""
        gridlayout_main = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.raster.dataprep.datagrid')
        label_band = QtWidgets.QLabel('Column to Grid:')
        label_dxy = QtWidgets.QLabel('Cell Size:')
        label_null = QtWidgets.QLabel('Null Value:')

        self.dsb_null.setMaximum(np.finfo(np.double).max)
        self.dsb_null.setMinimum(np.finfo(np.double).min)
        self.dsb_dxy.setMaximum(9999999999.0)
        self.dsb_dxy.setMinimum(0.00001)
        self.dsb_dxy.setDecimals(5)
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Dataset Gridding')

        gridlayout_main.addWidget(label_dxy, 0, 0, 1, 1)
        gridlayout_main.addWidget(self.dsb_dxy, 0, 1, 1, 1)
        gridlayout_main.addWidget(self.label_rows, 1, 0, 1, 2)
        gridlayout_main.addWidget(self.label_cols, 2, 0, 1, 2)
        gridlayout_main.addWidget(label_band, 3, 0, 1, 1)
        gridlayout_main.addWidget(self.dataid, 3, 1, 1, 1)
        gridlayout_main.addWidget(label_null, 4, 0, 1, 1)
        gridlayout_main.addWidget(self.dsb_null, 4, 1, 1, 1)
        gridlayout_main.addWidget(helpdocs, 5, 0, 1, 1)
        gridlayout_main.addWidget(buttonbox, 5, 1, 1, 3)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        self.dsb_dxy.valueChanged.connect(self.dxy_change)

    def dxy_change(self):
        """Update dxy."""
        dxy = self.dsb_dxy.value()
        data = self.indata['Point'][0]
        x = data.xdata
        y = data.ydata

        cols = int(x.ptp()/dxy)
        rows = int(y.ptp()/dxy)

        self.label_rows.setText('Rows: '+str(rows))
        self.label_cols.setText('Columns: '+str(cols))

    def settings(self):
        """Settings."""
        tmp = []
        if 'Point' not in self.indata:
            self.parent.showprocesslog('No Point Data')
            return False

        for i in self.indata['Point']:
            tmp.append(i.dataid)

        self.dataid.clear()
        self.dataid.addItems(tmp)

        data = self.indata['Point'][0]
        x = data.xdata
        y = data.ydata

        dx = x.ptp()/np.sqrt(x.size)
        dy = y.ptp()/np.sqrt(y.size)
        dxy = max(dx, dy)

        self.dsb_null.setValue(data.zdata.min())
        self.dsb_dxy.setValue(dxy)
        self.dxy_change()

        tmp = self.exec_()

        if tmp == 1:
            self.acceptall()
            tmp = True

        return tmp

    def acceptall(self):
        """Accept."""
        dxy = self.dsb_dxy.value()
        nullvalue = self.dsb_null.value()
        data = self.indata['Point'][0]

        newdat = []
        for data in self.pbar.iter(self.indata['Point']):
            if data.dataid != self.dataid.currentText():
                continue

            filt = (data.zdata != nullvalue)
            x = data.xdata[filt]
            y = data.ydata[filt]
            z = data.zdata[filt]

            for i in [x, y, z]:
                filt = np.logical_not(np.isnan(i))
                x = x[filt]
                y = y[filt]
                z = z[filt]

            tmp = quickgrid(x, y, z, dxy, showtext=self.parent.showprocesslog)
            mask = np.ma.getmaskarray(tmp)
            gdat = tmp.data

    # Create dataset
            dat = Data()
            dat.data = np.ma.masked_invalid(gdat[::-1])
            dat.data.mask = mask[::-1]
            dat.nullvalue = nullvalue
            dat.dataid = data.dataid
            dat.xdim = dxy
            dat.ydim = dxy
            dat.extent = [x.min(), x.max(), y.min(), y.max()]
            newdat.append(dat)

        self.outdata['Raster'] = newdat
        self.outdata['Point'] = self.indata['Point']


class DataMerge(QtWidgets.QDialog):
    """
    Data Merge.

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
        QtWidgets.QDialog.__init__(self, parent)

        self.indata = {}
        self.outdata = {}
        self.parent = parent

        self.dsb_dxy = QtWidgets.QDoubleSpinBox()
        self.label_rows = QtWidgets.QLabel('Rows: 0')
        self.label_cols = QtWidgets.QLabel('Columns: 0')

        self.setupui()

    def setupui(self):
        """Set up UI."""
        gridlayout_main = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.raster.dataprep.datamerge')
        label_dxy = QtWidgets.QLabel('Cell Size:')

        self.dsb_dxy.setMaximum(9999999999.0)
        self.dsb_dxy.setMinimum(0.00001)
        self.dsb_dxy.setDecimals(5)
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Dataset Merge and Resample')

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
        """
        Update dxy.

        This is the size of a grid cell in the x and y directions.
        """
        data = self.indata['Raster'][0]
        dxy = self.dsb_dxy.value()

        xmin0, xmax0, ymin0, ymax0 = data.extent

        for data in self.indata['Raster']:
            xmin, xmax, ymin, ymax = data.extent
            xmin = min(xmin, xmin0)
            xmax = max(xmax, xmax0)
            ymin = min(ymin, ymin0)
            ymax = max(ymax, ymax0)

        cols = int((xmax - xmin)/dxy)
        rows = int((ymax - ymin)/dxy)

        self.label_rows.setText('Rows: '+str(rows))
        self.label_cols.setText('Columns: '+str(cols))

    def settings(self):
        """Settings."""
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
        Accept.

        This routine is called by settings() if accept is pressed. It contains
        the main merge routine.
        """
        dxy = self.dsb_dxy.value()
        data = self.indata['Raster'][0]
        orig_wkt = data.wkt

        xmin0, xmax0, ymin0, ymax0 = data.extent

        for data in self.indata['Raster']:
            xmin, xmax, ymin, ymax = data.extent
            xmin = min(xmin, xmin0)
            xmax = max(xmax, xmax0)
            ymin = min(ymin, ymin0)
            ymax = max(ymax, ymax0)

        cols = int((xmax - xmin)/dxy)
        rows = int((ymax - ymin)/dxy)
        gtr = (xmin, dxy, 0.0, ymax, 0.0, -dxy)

        if cols == 0 or rows == 0:
            self.parent.showprocesslog('Your rows or cols are zero. ' +
                                       'Your input projection may be wrong')
            return

        dat = []
        for data in self.indata['Raster']:
            doffset = 0.0
            if data.data.min() <= 0:
                doffset = data.data.min()-1.
                data.data = data.data - doffset
            gtr0 = data.get_gtr()

            drows, dcols = data.data.shape
            src = data_to_gdal_mem(data, gtr0, orig_wkt, dcols, drows)
            dest = data_to_gdal_mem(data, gtr, orig_wkt, cols, rows, True)

            gdal.ReprojectImage(src, dest, orig_wkt, orig_wkt,
                                gdal.GRA_Bilinear)

            dat.append(gdal_to_dat(dest, data.dataid))
            dat[-1].data = dat[-1].data + doffset
            data.data = data.data + doffset

        self.outdata['Raster'] = dat


class DataReproj(QtWidgets.QDialog):
    """
    Reprojections.

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
        QtWidgets.QDialog.__init__(self, parent)

        self.indata = {}
        self.outdata = {}
        self.parent = parent
        self.pbar = self.parent.pbar

        self.groupboxb = QtWidgets.QGroupBox()
        self.combobox_inp_epsg = QtWidgets.QComboBox()
        self.inp_epsg_info = QtWidgets.QLabel()
        self.groupbox2b = QtWidgets.QGroupBox()
        self.combobox_out_epsg = QtWidgets.QComboBox()
        self.out_epsg_info = QtWidgets.QLabel()
        self.in_proj = GroupProj('Input Projection')
        self.out_proj = GroupProj('Output Projection')

        self.setupui()

    def setupui(self):
        """Set up UI."""
        gridlayout_main = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.raster.dataprep.datareproj')

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Dataset Reprojection')

        gridlayout_main.addWidget(self.in_proj, 0, 0, 1, 1)
        gridlayout_main.addWidget(self.out_proj, 0, 1, 1, 1)
        gridlayout_main.addWidget(helpdocs, 1, 0, 1, 1)
        gridlayout_main.addWidget(buttonbox, 1, 1, 1, 1)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)

    def acceptall(self):
        """
        Accept.

        This routine is called by settings() if accept is pressed. It contains
        the main routine.
        """
        if self.in_proj.wkt == 'Unknown' or self.out_proj.wkt == 'Unknown':
            self.parent.showprocesslog('Could not reproject')
            return

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
            rows, cols = data.data.shape
            u_l = ctrans.TransformPoint(data.extent[0], data.extent[-1])
            u_r = ctrans.TransformPoint(data.extent[1], data.extent[-1])
            l_l = ctrans.TransformPoint(data.extent[0], data.extent[-2])
            l_r = ctrans.TransformPoint(data.extent[1], data.extent[-2])

            lrx = l_r[0]
            llx = l_l[0]
            ulx = u_l[0]
            urx = u_r[0]
            lry = l_r[1]
            lly = l_l[1]
            uly = u_l[1]
            ury = u_r[1]

            drows, dcols = data.data.shape
            minx = min(llx, ulx, urx, lrx)
            maxx = max(llx, ulx, urx, lrx)
            miny = min(lly, lry, ury, uly)
            maxy = max(lly, lry, ury, uly)
            newdimx = (maxx-minx)/dcols
            newdimy = (maxy-miny)/drows
            newdim = min(newdimx, newdimy)
            cols = round((maxx - minx)/newdim)
            rows = round((maxy - miny)/newdim)

            if cols == 0 or rows == 0:
                self.parent.showprocesslog('Your rows or cols are zero. ' +
                                           'Your input projection may be ' +
                                           'wrong')
                return

# top left x, w-e pixel size, rotation, top left y, rotation, n-s pixel size
            old_geo = data.get_gtr()
            drows, dcols = data.data.shape
            src = data_to_gdal_mem(data, old_geo, orig_wkt, dcols, drows)

            new_geo = (minx, newdim, 0, maxy, 0, -newdim)
            dest = data_to_gdal_mem(data, new_geo, targ_wkt, cols, rows, True)

            gdal.ReprojectImage(src, dest, orig_wkt, targ_wkt,
                                gdal.GRA_Bilinear)

            data2 = gdal_to_dat(dest, data.dataid)
            data2.data = data2.data.astype(data.data.dtype)

            if datamin <= 0:
                data2.data = data2.data+(datamin-1)
                data.data = data.data+(datamin-1)
            data2.data = np.ma.masked_equal(data2.data.filled(data.nullvalue),
                                            data.nullvalue)
            data2.nullvalue = data.nullvalue
            data2.data = np.ma.masked_invalid(data2.data)
            data2.data = np.ma.masked_less(data2.data, data.data.min())
            data2.data = np.ma.masked_greater(data2.data, data.data.max())

            dat.append(data2)

        self.outdata['Raster'] = dat

    def settings(self):
        """Settings."""
        self.in_proj.set_current(self.indata['Raster'][0].wkt)
        self.out_proj.set_current(self.indata['Raster'][0].wkt)

        tmp = self.exec_()
        if tmp == 1:
            self.acceptall()
            tmp = True

        return tmp


class GetProf():
    """
    Get a Profile.

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
        self.ifile = ''
        self.name = 'Get Profile: '
        self.ext = ''
        self.pbar = parent.pbar
        self.parent = parent
        self.indata = {}
        self.outdata = {}

    def settings(self):
        """Settings."""
        if 'Raster' in self.indata:
            data = copy.deepcopy(self.indata['Raster'])
        else:
            self.parent.showprocesslog('No raster data')
            return False

        ext = 'Shape file (*.shp)'

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.parent, 'Open Shape File', '.', ext)
        if filename == '':
            return False
        os.chdir(os.path.dirname(filename))

        self.ifile = str(filename)
        self.ext = filename[-3:]

        shapef = ogr.Open(self.ifile)
        if shapef is None:
            err = ('There was a problem importing the shapefile. Please make '
                   'sure you have at all the individual files which make up '
                   'the shapefile.')
            QtWidgets.QMessageBox.warning(self.parent, 'Error', err,
                                          QtWidgets.QMessageBox.Ok)
            return False

        lyr = shapef.GetLayer()
        line = lyr.GetNextFeature()
        if lyr.GetGeomType() is not ogr.wkbLineString:
            self.parent.showprocesslog('You need lines in that shape file')
            return False

        allpoints = []
        for idata in self.pbar.iter(data):
            tmp = line.GetGeometryRef()
            points = tmp.GetPoints()

            x_0, y_0 = points[0]
            x_1, y_1 = points[1]

            bly = idata.extent[-2]
            tlx = idata.extent[0]
            x_0 = (x_0-tlx)/idata.xdim
            x_1 = (x_1-tlx)/idata.xdim
            y_0 = (y_0-bly)/idata.ydim
            y_1 = (y_1-bly)/idata.ydim
            rcell = int(np.sqrt((x_1-x_0)**2+(y_1-y_0)**2))

            xxx = np.linspace(x_0, x_1, rcell, False)
            yyy = np.linspace(y_0, y_1, rcell, False)

            tmpprof = ndimage.map_coordinates(idata.data[::-1], [yyy, xxx],
                                              order=1, cval=np.nan)
            xxx = xxx[np.logical_not(np.isnan(tmpprof))]
            yyy = yyy[np.logical_not(np.isnan(tmpprof))]
            tmpprof = tmpprof[np.logical_not(np.isnan(tmpprof))]
            xxx = xxx*idata.xdim+tlx
            yyy = yyy*idata.ydim+bly
            allpoints.append(PData())
            allpoints[-1].xdata = xxx
            allpoints[-1].ydata = yyy
            allpoints[-1].zdata = tmpprof
            allpoints[-1].dataid = idata.dataid

        shapef = None
        self.outdata['Point'] = allpoints

        return True


class GroupProj(QtWidgets.QWidget):
    """
    Group Proj.

    Custom widget
    """

    def __init__(self, title='Projection', parent=None):
        QtWidgets.QWidget.__init__(self, parent)

        self.wkt = ''

        self.gridlayout = QtWidgets.QGridLayout(self)
        self.groupbox = QtWidgets.QGroupBox(title)
        self.combobox = QtWidgets.QComboBox()
        self.label = QtWidgets.QLabel()

        self.gridlayout.addWidget(self.groupbox, 1, 0, 1, 2)

        gridlayout = QtWidgets.QGridLayout(self.groupbox)
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
        """Set new wkt for current option."""
        self.wkt = wkt
        self.epsg_proj['Current'] = self.wkt
        self.combo_change()

    def combo_change(self):
        """Change Combo."""
        indx = self.combobox.currentIndex()
        txt = self.combobox.itemText(indx)

        self.wkt = self.epsg_proj[txt]

        if type(self.wkt) is not str:
            self.wkt = epsgtowkt(self.wkt)

        srs = osr.SpatialReference()
        srs.ImportFromWkt(self.wkt)

        self.label.setText(srs.ExportToPrettyWkt())


class Metadata(QtWidgets.QDialog):
    """
    Edit Metadata.

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
        QtWidgets.QDialog.__init__(self, parent)

        self.indata = {}
        self.outdata = {}
        self.banddata = {}
        self.dataid = {}
        self.oldtxt = ''
        self.parent = parent

        self.combobox_bandid = QtWidgets.QComboBox()
        self.pb_rename_id = QtWidgets.QPushButton('Rename Band Name')
        self.lbl_rows = QtWidgets.QLabel()
        self.lbl_cols = QtWidgets.QLabel()
        self.inp_epsg_info = QtWidgets.QLabel()
        self.txt_null = QtWidgets.QLineEdit()
        self.dsb_tlx = QtWidgets.QLineEdit()
        self.dsb_tly = QtWidgets.QLineEdit()
        self.dsb_xdim = QtWidgets.QLineEdit()
        self.dsb_ydim = QtWidgets.QLineEdit()
        self.led_units = QtWidgets.QLineEdit()
        self.lbl_min = QtWidgets.QLabel()
        self.lbl_max = QtWidgets.QLabel()
        self.lbl_mean = QtWidgets.QLabel()

        self.proj = GroupProj('Input Projection')

        self.setupui()

    def setupui(self):
        """Set up UI."""
        gridlayout_main = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        groupbox = QtWidgets.QGroupBox('Dataset')

        gridlayout = QtWidgets.QGridLayout(groupbox)
        label_tlx = QtWidgets.QLabel('Top Left X Coordinate:')
        label_tly = QtWidgets.QLabel('Top Left Y Coordinate:')
        label_xdim = QtWidgets.QLabel('X Dimension:')
        label_ydim = QtWidgets.QLabel('Y Dimension:')
        label_null = QtWidgets.QLabel('Null/Nodata value:')
        label_rows = QtWidgets.QLabel('Rows:')
        label_cols = QtWidgets.QLabel('Columns:')
        label_min = QtWidgets.QLabel('Dataset Minimum:')
        label_max = QtWidgets.QLabel('Dataset Maximum:')
        label_mean = QtWidgets.QLabel('Dataset Mean:')
        label_units = QtWidgets.QLabel('Dataset Units:')
        label_bandid = QtWidgets.QLabel('Band Name:')

        sizepolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                           QtWidgets.QSizePolicy.Expanding)
        groupbox.setSizePolicy(sizepolicy)
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Dataset Metadata')

        gridlayout_main.addWidget(label_bandid, 0, 0, 1, 1)
        gridlayout_main.addWidget(self.combobox_bandid, 0, 1, 1, 3)
        gridlayout_main.addWidget(self.pb_rename_id, 1, 1, 1, 3)
        gridlayout_main.addWidget(groupbox, 2, 0, 1, 2)
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
        Accept.

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
                    tmp.xdim = i.xdim
                    tmp.ydim = i.ydim
                    tmp.extent = i.extent
                    tmp.nullvalue = i.nullvalue
                    tmp.wkt = wkt
                    tmp.units = i.units
                    if tmp.dataid[-1] == ')':
                        tmp.dataid = tmp.dataid[:tmp.dataid.rfind(' (')]
                    if i.units != '':
                        tmp.dataid += ' ('+i.units+')'
                    tmp.data.mask = (tmp.data.data == i.nullvalue)

    def rename_id(self):
        """Rename the band name."""
        ctxt = str(self.combobox_bandid.currentText())
        (skey, isokay) = QtWidgets.QInputDialog.getText(
            self.parent, 'Rename Band Name',
            'Please type in the new name for the band',
            QtWidgets.QLineEdit.Normal, ctxt)

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
        """Update the values on the interface."""
        odata = self.banddata[self.oldtxt]
        odata.units = self.led_units.text()

        try:
            odata.nullvalue = float(self.txt_null.text())
            left = float(self.dsb_tlx.text())
            top = float(self.dsb_tly.text())
            odata.xdim = float(self.dsb_xdim.text())
            odata.ydim = float(self.dsb_ydim.text())

            rows, cols = odata.data.shape
            right = left + odata.xdim*cols
            bottom = top - odata.ydim*rows

            odata.extent = (left, right, bottom, top)

        except ValueError:
            self.parent.showprocesslog('Value error - abandoning changes')

        indx = self.combobox_bandid.currentIndex()
        txt = self.combobox_bandid.itemText(indx)
        self.oldtxt = txt
        idata = self.banddata[txt]

        irows = idata.data.shape[0]
        icols = idata.data.shape[1]

        self.lbl_cols.setText(str(icols))
        self.lbl_rows.setText(str(irows))
        self.txt_null.setText(str(idata.nullvalue))
        self.dsb_tlx.setText(str(idata.extent[0]))
        self.dsb_tly.setText(str(idata.extent[-1]))
        self.dsb_xdim.setText(str(idata.xdim))
        self.dsb_ydim.setText(str(idata.ydim))
        self.lbl_min.setText(str(idata.data.min()))
        self.lbl_max.setText(str(idata.data.max()))
        self.lbl_mean.setText(str(idata.data.mean()))
        self.led_units.setText(str(idata.units))

    def run(self):
        """Entrypoint to start this routine."""
        bandid = []
        self.proj.set_current(self.indata['Raster'][0].wkt)

        for i in self.indata['Raster']:
            bandid.append(i.dataid)
            self.banddata[i.dataid] = Data()
            tmp = self.banddata[i.dataid]
            self.dataid[i.dataid] = i.dataid
            tmp.xdim = i.xdim
            tmp.ydim = i.ydim
            tmp.nullvalue = i.nullvalue
            tmp.wkt = i.wkt
            tmp.extent = i.extent
            tmp.data = i.data
            tmp.units = i.units

        self.combobox_bandid.currentIndexChanged.disconnect()
        self.combobox_bandid.addItems(bandid)
        indx = self.combobox_bandid.currentIndex()
        self.oldtxt = self.combobox_bandid.itemText(indx)
        self.combobox_bandid.currentIndexChanged.connect(self.update_vals)

        idata = self.banddata[self.oldtxt]

        irows = idata.data.shape[0]
        icols = idata.data.shape[1]

        self.lbl_cols.setText(str(icols))
        self.lbl_rows.setText(str(irows))
        self.txt_null.setText(str(idata.nullvalue))
        self.dsb_tlx.setText(str(idata.extent[0]))
        self.dsb_tly.setText(str(idata.extent[-1]))
        self.dsb_xdim.setText(str(idata.xdim))
        self.dsb_ydim.setText(str(idata.ydim))
        self.lbl_min.setText(str(idata.data.min()))
        self.lbl_max.setText(str(idata.data.max()))
        self.lbl_mean.setText(str(idata.data.mean()))
        self.led_units.setText(str(idata.units))

        self.update_vals()

        tmp = self.exec_()
        if tmp == 1:
            self.acceptall()
            tmp = True

        return tmp


class RTP(QtWidgets.QDialog):
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
        QtWidgets.QDialog.__init__(self, parent)

        self.indata = {}
        self.outdata = {}
        self.parent = parent
        self.pbar = self.parent.pbar

        self.dataid = QtWidgets.QComboBox()
        self.dsb_inc = QtWidgets.QDoubleSpinBox()
        self.dsb_dec = QtWidgets.QDoubleSpinBox()

        self.setupui()

    def setupui(self):
        """Set up UI."""
        gridlayout_main = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.raster.dataprep.rtp')
        label_band = QtWidgets.QLabel('Band to Reduce to the Pole:')
        label_inc = QtWidgets.QLabel('Inclination of Magnetic Field:')
        label_dec = QtWidgets.QLabel('Declination of Magnetic Field:')

        self.dsb_inc.setMaximum(90.0)
        self.dsb_inc.setMinimum(-90.0)
        self.dsb_dec.setMaximum(360.0)
        self.dsb_dec.setMinimum(-360.0)
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Reduction to the Pole')

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
        """Settings."""
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
        """Accept."""
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
    """Reduction to the Pole."""
    datamedian = np.ma.median(data.data)
    ndat = data.data - datamedian
    ndat.data[ndat.mask] = 0

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
    zrtp[data.data.mask] = data.data.fill_value

# Create dataset
    dat = Data()
    dat.data = np.ma.masked_invalid(zrtp)
    dat.data.mask = np.ma.getmaskarray(data.data)
    dat.nullvalue = data.data.fill_value
    dat.dataid = 'RTP_'+data.dataid
    dat.extent = data.extent
    dat.xdim = data.xdim
    dat.ydim = data.ydim

    return dat


def check_dataid(out):
    """Check dataid for duplicates and renames where necessary."""
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
    """
    Convert cluster datasets to raster datasets.

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
    """Cuts a raster dataset.

    Cut a raster dataset using a shapefile.

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
        return None

    for idata in data:
        # Convert the layer extent to image pixel coordinates
        minX, maxX, minY, maxY = lyr.GetExtent()
        itlx = idata.extent[0]
        itly = idata.extent[-1]

        ulX = max(0, int((minX - itlx) / idata.xdim))
        ulY = max(0, int((itly - maxY) / idata.ydim))
        lrX = int((maxX - itlx) / idata.xdim)
        lrY = int((itly - minY) / idata.ydim)

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
            tmpx = int((p[0] - idata.extent[0]) / idata.xdim)
            tmpy = int((idata.extent[-1] - p[1]) / idata.ydim)
            pixels.append((tmpx, tmpy))
        irows, icols = idata.data.shape
        rasterPoly = Image.new('L', (icols, irows), 1)
        rasterize = ImageDraw.Draw(rasterPoly)
        rasterize.polygon(pixels, 0)
        mask = np.array(rasterPoly)

        idata.data.mask = mask
        idata.data = idata.data[ulY:lrY, ulX:lrX]
        ixmin = ulX*idata.xdim + idata.extent[0]  # minX
        iymax = idata.extent[-1] - ulY*idata.ydim  # maxY
        ixmax = ixmin + icols*idata.xdim
        iymin = iymax - irows*idata.ydim
        idata.extent = [ixmin, ixmax, iymin, iymax]

    shapef = None
    data = trim_raster(data)
    return data


def data_to_gdal_mem(data, gtr, wkt, cols, rows, nodata=False):
    """
    Input Data to GDAL mem format.

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

    if data.isrgb is True:
        nbands = data.data.shape[2]
    else:
        nbands = 1

    if dtype == np.uint8:
        fmt = gdal.GDT_Byte
    elif dtype == np.int32:
        fmt = gdal.GDT_Int32
    elif dtype == np.float64:
        fmt = gdal.GDT_Float64
    else:
        fmt = gdal.GDT_Float32

    driver = gdal.GetDriverByName('MEM')
    src = driver.Create('', cols, rows, nbands, fmt)

    src.SetGeoTransform(gtr)
    src.SetProjection(wkt)

    for i in range(nbands):
        if nodata is False:
            if data.nullvalue is not None:
                src.GetRasterBand(i+1).SetNoDataValue(data.nullvalue)
            if data.isrgb is True:
                src.GetRasterBand(i+1).WriteArray(data.data[:, :, i])
            else:
                src.GetRasterBand(i+1).WriteArray(data.data)
        else:
            tmp = np.zeros((rows, cols))
            tmp = np.ma.masked_equal(tmp, 0)
            src.GetRasterBand(i+1).SetNoDataValue(0)  # Set to this because of Reproj
            src.GetRasterBand(i+1).WriteArray(tmp)

    return src


def epsgtowkt(epsg):
    """Routine to get a wkt from an epsg code."""
    orig = osr.SpatialReference()
    err = orig.ImportFromEPSG(int(epsg))
    if err != 0:
        return 'Unknown'
    out = orig.ExportToWkt()
    return out


def gdal_to_dat(dest, bandid='Data'):
    """
    GDAL to Data format.

    Parameters
    ----------
    dest - GDAL format
        GDAL format
    bandid - str
        band identity
    """
    dat = Data()
    gtr = dest.GetGeoTransform()

    nbands = dest.RasterCount

    if nbands == 1:
        rtmp = dest.GetRasterBand(1)
        dat.data = rtmp.ReadAsArray()
    else:
        dat.data = []
        for i in range(nbands):
            rtmp = dest.GetRasterBand(i+1)
            dat.data.append(rtmp.ReadAsArray())
        dat.data = np.array(dat.data)
        dat.data = np.moveaxis(dat.data, 0, -1)

    nval = rtmp.GetNoDataValue()

    dat.data = np.ma.masked_equal(dat.data, nval)
    dat.data.set_fill_value(nval)
    dat.data = np.ma.fix_invalid(dat.data)

    dat.extent_from_gtr(gtr)
    dat.dataid = bandid
    dat.nullvalue = nval
    dat.wkt = dest.GetProjection()

    return dat


def getepsgcodes():
    """Routine used to get a list of EPSG codes."""
    with open(os.path.join(os.environ['GDAL_DATA'], 'gcs.csv')) as dfile:
        dlines = dfile.readlines()

    dlines = dlines[1:]
    dcodes = {}
    for i in dlines:
        tmp = i.split(',')
        if tmp[1][0] == '"':
            tmp[1] = tmp[1][1:-1]
#        wkttmp = epsgtowkt(tmp[0])
#        if wkttmp != '':
#            dcodes[tmp[1]] = wkttmp
        dcodes[tmp[1]] = int(tmp[0])

    with open(os.path.join(os.environ['GDAL_DATA'], 'pcs.csv')) as pfile:
        plines = pfile.readlines()

    orig = osr.SpatialReference()

    pcodes = {}
    for i in dcodes:
        pcodes[i+r' / Geodetic Geographic'] = dcodes[i]

    plines = plines[1:]
    for i in plines:
        tmp = i.split(',')
        if tmp[1][0] == '"':
            tmp[1] = tmp[1][1:-1]
#        err = orig.ImportFromEPSG(int(tmp[0]))
#        if err == 0:
#            pcodes[tmp[1]] = orig.ExportToWkt()
        pcodes[tmp[1]] = int(tmp[0])

    clat = 0.
    scale = 1.
    f_e = 0.
    f_n = 0.
    orig = osr.SpatialReference()

    for datum in ['Cape', 'Hartebeesthoek94']:
        orig.ImportFromEPSG(dcodes[datum])
#        orig.ImportFromWkt(dcodes[datum])
        for clong in range(15, 35, 2):
            orig.SetTM(clat, clong, scale, f_e, f_n)
            orig.SetProjCS(datum+r' / TM'+str(clong))
            pcodes[datum+r' / TM'+str(clong)] = orig.ExportToWkt()

    return pcodes


def merge(dat):
    """
    Merge datasets found in a single PyGMI data object.

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
    rows, cols = dat[0].data.shape
    for i in dat:
        irows, icols = i.data.shape
        if irows != rows or icols != cols:
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


def trim_raster(olddata):
    """
    Trim nulls from a raster dataset.

    This function trims entire rows or columns of data which are masked,
    and are on the edges of the dataset. Masked values are set to the null
    value.

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
        mask = np.ma.getmaskarray(data.data)
        data.data.data[mask] = data.nullvalue

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

        drows, dcols = data.data.shape
        data.data = data.data[rowstart:rowend, colstart:colend]
        data.data.mask = (data.data.data == data.nullvalue)
        xmin = data.extent[0] + colstart*data.xdim
        ymax = data.extent[-1] - rowstart*data.ydim
        xmax = xmin + data.xdim*dcols
        ymin = ymax - data.ydim*drows
        data.extent = [xmin, xmax, ymin, ymax]

    return olddata


def quickgrid(x, y, z, dxy, showtext=None, numits=4):
    """
    Do a quick grid.

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
#        print(newz)

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


def func(x, y):
    """Function."""
    return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2
