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
""" This is the main Data Preparation set of routines """

# pylint: disable=E1101, C0103
from PyQt4 import QtGui, QtCore
import os
import numpy as np
from osgeo import gdal, osr, ogr
from .datatypes import Data
from pygmi.point.datatypes import PData
from PIL import Image, ImageDraw
import copy
import scipy.ndimage as ndimage
import scipy.interpolate as si


def data_to_gdal_mem(data, gtr, wkt, cols, rows, nodata=False):
    """ data to gdal mem format """
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


def gdal_to_dat(dest, bandid='Data'):
    """ Gdal to dat format """
    dat = Data()
    gtr = dest.GetGeoTransform()

    rtmp = dest.GetRasterBand(1)
    dat.data = rtmp.ReadAsArray()
    nval = rtmp.GetNoDataValue()
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
    dat.bandid = bandid
    dat.nullvalue = nval
    dat.rows = dest.RasterYSize
    dat.cols = dest.RasterXSize
    dat.xdim = abs(gtr[1])
    dat.ydim = abs(gtr[5])
    dat.wkt = dest.GetProjection()
    dat.gtr = gtr

    return dat


def merge(dat):
    """ Merges datasets found in a single PyGMI data object.

    The aim is to ensure that all datasets have the same number of rows and
    columns.

    Args:
        dat (Data): data object which stores datasets

    Returns:
        Data: data object which stores datasets
    """
    needsmerge = False
    for i in dat:
        if i.rows != dat[0].rows or i.cols != dat[0].cols:
            needsmerge = True

    if needsmerge is False:
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
    return out


def cluster_to_raster(indata):
    """ Converts cluster datasets to raster datasets

    Some routines will not understand the datasets produced by cluster
    analysis routines, since they are designated 'Cluster' and not 'Raster'.
    This provides a work-around for that.

    Args:
        indata (Data): PyGMI raster dataset

    Return:
        Data: PyGMI raster dataset

    """
    if 'Cluster' not in indata:
        return indata
    if 'Raster' not in indata:
        indata['Raster'] = []

    for i in indata['Cluster']:
        indata['Raster'].append(i)
        indata['Raster'][-1].data += 1

    return indata


class DataMerge(QtGui.QDialog):
    """ Merge """
    def __init__(self, parent=None):
        QtGui.QDialog.__init__(self, parent)

        self.indata = {}
        self.outdata = {}
        self.parent = parent

        self.gridlayout_main = QtGui.QGridLayout(self)
        self.buttonbox = QtGui.QDialogButtonBox(self)
        self.dsb_dxy = QtGui.QDoubleSpinBox(self)
        self.label_rows = QtGui.QLabel(self)
        self.label_cols = QtGui.QLabel(self)

        self.setupui()

    def setupui(self):
        """ Setup UI """
        self.setWindowTitle("Dataset Merge and Resample")

        self.label_rows.setText("Rows: 0")
        self.gridlayout_main.addWidget(self.label_rows, 1, 0, 1, 2)

        self.label_cols.setText("Columns: 0")
        self.gridlayout_main.addWidget(self.label_cols, 2, 0, 1, 2)

        label_dxy = QtGui.QLabel(self)
        label_dxy.setText("Cell Size:")
        self.gridlayout_main.addWidget(label_dxy, 0, 0, 1, 1)

        self.dsb_dxy.setMaximum(9999999999.0)
        self.dsb_dxy.setMinimum(0.00001)
        self.dsb_dxy.setDecimals(5)
        self.gridlayout_main.addWidget(self.dsb_dxy, 0, 1, 1, 1)

        self.buttonbox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonbox.setCenterButtons(True)
        self.buttonbox.setStandardButtons(
            QtGui.QDialogButtonBox.Cancel | QtGui.QDialogButtonBox.Ok)

        self.gridlayout_main.addWidget(self.buttonbox, 3, 0, 1, 4)
        QtCore.QObject.connect(self.buttonbox, QtCore.SIGNAL("rejected()"),
                               self.reject)
        QtCore.QObject.connect(self.buttonbox, QtCore.SIGNAL("accepted()"),
                               self.accept)

#        self.buttonbox.accepted.connect(self.acceptall)
        self.dsb_dxy.valueChanged.connect(self.dxy_change)

    def dxy_change(self):
        """ update dxy """
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
        """ accept """
        dxy = self.dsb_dxy.value()
        data = self.indata['Raster'][0]
        orig_wkt = data.wkt

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
        gtr = (xmin, dxy, 0.0, ymax, 0.0, -dxy)

        if cols == 0 or rows == 0:
            self.parent.showprocesslog("Your rows or cols are zero. " +
                                       "Your input projection may be wrong")
            return

        dat = []
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

            dat.append(gdal_to_dat(dest, data.bandid))
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
    """ Reprojections """
    def __init__(self, parent=None):
        QtGui.QDialog.__init__(self, parent)

        self.indata = {}
        self.outdata = {}
        self.parent = parent

        self.gridlayout_main = QtGui.QGridLayout(self)
        self.buttonbox = QtGui.QDialogButtonBox(self)

        self.groupbox = QtGui.QGroupBox(self)
        self.groupboxb = QtGui.QGroupBox(self)
        self.combobox_inp_datum = QtGui.QComboBox(self.groupbox)
        self.combobox_inp_proj = QtGui.QComboBox(self.groupbox)
        self.dsb_inp_latorigin = QtGui.QDoubleSpinBox(self.groupbox)
        self.dsb_inp_cm = QtGui.QDoubleSpinBox(self.groupbox)
        self.dsb_inp_scalefactor = QtGui.QDoubleSpinBox(self.groupbox)
        self.dsb_inp_fnorthing = QtGui.QDoubleSpinBox(self.groupbox)
        self.dsb_inp_feasting = QtGui.QDoubleSpinBox(self.groupbox)
        self.sb_inp_zone = QtGui.QSpinBox(self.groupbox)
        self.cb_inp_epsg = QtGui.QCheckBox(self.groupbox)
        self.sb_inp_epsg = QtGui.QSpinBox(self.groupbox)
        self.inp_epsg_info = QtGui.QLabel(self.groupbox)

        self.groupbox2 = QtGui.QGroupBox(self)
        self.groupbox2b = QtGui.QGroupBox(self)
        self.combobox_out_datum = QtGui.QComboBox(self.groupbox2)
        self.combobox_out_proj = QtGui.QComboBox(self.groupbox2)
        self.dsb_out_latorigin = QtGui.QDoubleSpinBox(self.groupbox2)
        self.dsb_out_cm = QtGui.QDoubleSpinBox(self.groupbox2)
        self.dsb_out_scalefactor = QtGui.QDoubleSpinBox(self.groupbox2)
        self.dsb_out_fnorthing = QtGui.QDoubleSpinBox(self.groupbox2)
        self.dsb_out_feasting = QtGui.QDoubleSpinBox(self.groupbox2)
        self.sb_out_zone = QtGui.QSpinBox(self.groupbox2)
        self.cb_out_epsg = QtGui.QCheckBox(self.groupbox2)
        self.sb_out_epsg = QtGui.QSpinBox(self.groupbox2)
        self.out_epsg_info = QtGui.QLabel(self.groupbox)

        self.setupui()

        self.ctrans = None

    def setupui(self):
        """ Setup UI """
        sizepolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred,
                                       QtGui.QSizePolicy.Expanding)

        self.setWindowTitle("Dataset Reprojection")

        label_help = QtGui.QLabel(self.groupbox)
        label_help.setText(
            "EPSG codes are available online at either " +
            "http://www.epsg-registry.org/, " +
            "http://spatialreference.org/ref/epsg/ \n or in the gcs.csv " +
            "and pcs.csv files in your python GDAL\\data directory.")
        self.gridlayout_main.addWidget(label_help, 3, 0, 1, 4)

        label_input = QtGui.QLabel(self.groupbox)
        label_input.setText("Input Projection")
        self.gridlayout_main.addWidget(label_input, 0, 0, 1, 2)

        self.groupbox.setTitle("Manual Entry")
        self.groupbox.setSizePolicy(sizepolicy)
        self.gridlayout_main.addWidget(self.groupbox, 2, 0, 1, 2)

        self.gridlayout_main.addWidget(self.groupboxb, 2, 0, 1, 2)

        self.groupboxb.setTitle("EPSG Information")
        self.groupboxb.setSizePolicy(sizepolicy)
        gridlayoutb = QtGui.QGridLayout(self.groupboxb)
        gridlayoutb.addWidget(self.inp_epsg_info, 0, 0, 1, 1)

        gridlayout = QtGui.QGridLayout(self.groupbox)

        self.cb_inp_epsg.setText("Use EPSG Code:")
        self.gridlayout_main.addWidget(self.cb_inp_epsg, 1, 0, 1, 1)
        self.sb_inp_epsg.setMinimum(1000)
        self.sb_inp_epsg.setMaximum(40000)
        self.sb_inp_epsg.setValue(32735)
        self.gridlayout_main.addWidget(self.sb_inp_epsg, 1, 1, 1, 1)

        label_inp_proj = QtGui.QLabel(self.groupbox)
        label_inp_proj.setText("Projection")
        gridlayout.addWidget(label_inp_proj, 0, 0, 1, 1)
        gridlayout.addWidget(self.combobox_inp_proj, 0, 1, 1, 1)

        label_inp_datum = QtGui.QLabel(self.groupbox)
        label_inp_datum.setText("Datum")
        gridlayout.addWidget(label_inp_datum, 1, 0, 1, 1)
        gridlayout.addWidget(self.combobox_inp_datum, 1, 1, 1, 1)

        label_inp_latorigin = QtGui.QLabel(self.groupbox)
        label_inp_latorigin.setText("Latitude of Origin")
        gridlayout.addWidget(label_inp_latorigin, 2, 0, 1, 1)
        self.dsb_inp_latorigin.setMinimum(-90.0)
        self.dsb_inp_latorigin.setMaximum(90.0)
        gridlayout.addWidget(self.dsb_inp_latorigin, 2, 1, 1, 1)

        label_inp_cm = QtGui.QLabel(self.groupbox)
        label_inp_cm.setText("Central Meridian")
        gridlayout.addWidget(label_inp_cm, 3, 0, 1, 1)
        self.dsb_inp_cm.setMinimum(-180.0)
        self.dsb_inp_cm.setMaximum(180.0)
        gridlayout.addWidget(self.dsb_inp_cm, 3, 1, 1, 1)

        label_inp_scalefactor = QtGui.QLabel(self.groupbox)
        label_inp_scalefactor.setText("Scale Factor")
        gridlayout.addWidget(label_inp_scalefactor, 4, 0, 1, 1)
        self.dsb_inp_scalefactor.setDecimals(4)
        self.dsb_inp_scalefactor.setMinimum(-100.0)
        gridlayout.addWidget(self.dsb_inp_scalefactor, 4, 1, 1, 1)

        label_inp_feasting = QtGui.QLabel(self.groupbox)
        label_inp_feasting.setText("False Easting")
        gridlayout.addWidget(label_inp_feasting, 5, 0, 1, 1)
        self.dsb_inp_feasting.setMaximum(1000000000.0)
        gridlayout.addWidget(self.dsb_inp_feasting, 5, 1, 1, 1)

        label_inp_fnorthing = QtGui.QLabel(self.groupbox)
        label_inp_fnorthing.setText("False Northing")
        gridlayout.addWidget(label_inp_fnorthing, 6, 0, 1, 1)
        self.dsb_inp_fnorthing.setMaximum(1000000000.0)
        gridlayout.addWidget(self.dsb_inp_fnorthing, 6, 1, 1, 1)

        label_inp_zone = QtGui.QLabel(self.groupbox)
        label_inp_zone.setText("UTM Zone")
        gridlayout.addWidget(label_inp_zone, 7, 0, 1, 1)
        self.sb_inp_zone.setMaximum(60)
        gridlayout.addWidget(self.sb_inp_zone, 7, 1, 1, 1)

# ############################
        label_input2 = QtGui.QLabel(self.groupbox2)
        label_input2.setText("Output Projection")
        self.gridlayout_main.addWidget(label_input2, 0, 2, 1, 2)

        self.groupbox2.setSizePolicy(sizepolicy)
        self.groupbox2.setTitle("Manual Entry")
        self.gridlayout_main.addWidget(self.groupbox2, 2, 2, 1, 2)
        self.gridlayout_main.addWidget(self.groupbox2b, 2, 2, 1, 2)

        self.groupbox2b.setTitle("EPSG Information")
        self.groupbox2b.setSizePolicy(sizepolicy)
        gridlayout2b = QtGui.QGridLayout(self.groupbox2b)
        gridlayout2b.addWidget(self.out_epsg_info, 0, 0, 1, 1)

        gridlayoutb = QtGui.QGridLayout(self.groupbox2)

        self.cb_out_epsg.setText("Use EPSG Code:")
        self.gridlayout_main.addWidget(self.cb_out_epsg, 1, 2, 1, 1)
        self.sb_out_epsg.setMinimum(1000)
        self.sb_out_epsg.setMaximum(40000)
        self.sb_out_epsg.setValue(32735)
        self.gridlayout_main.addWidget(self.sb_out_epsg, 1, 3, 1, 1)

        label_out_proj = QtGui.QLabel(self.groupbox2)
        label_out_proj.setText("Projection")
        gridlayoutb.addWidget(label_out_proj, 0, 0, 1, 1)
        gridlayoutb.addWidget(self.combobox_out_proj, 0, 1, 1, 1)

        label_out_datum = QtGui.QLabel(self.groupbox2)
        label_out_datum.setText("Datum")
        gridlayoutb.addWidget(label_out_datum, 1, 0, 1, 1)
        gridlayoutb.addWidget(self.combobox_out_datum, 1, 1, 1, 1)

        label_out_latorigin = QtGui.QLabel(self.groupbox2)
        label_out_latorigin.setText("Latitude of Origin")
        gridlayoutb.addWidget(label_out_latorigin, 2, 0, 1, 1)
        self.dsb_out_latorigin.setMinimum(-90.0)
        self.dsb_out_latorigin.setMaximum(90.0)
        gridlayoutb.addWidget(self.dsb_out_latorigin, 2, 1, 1, 1)

        label_out_cm = QtGui.QLabel(self.groupbox2)
        label_out_cm.setText("Central Meridian")
        gridlayoutb.addWidget(label_out_cm, 3, 0, 1, 1)
        self.dsb_out_cm.setMinimum(-180.0)
        self.dsb_out_cm.setMaximum(180.0)
        gridlayoutb.addWidget(self.dsb_out_cm, 3, 1, 1, 1)

        label_out_scalefactor = QtGui.QLabel(self.groupbox2)
        label_out_scalefactor.setText("Scale Factor")
        gridlayoutb.addWidget(label_out_scalefactor, 4, 0, 1, 1)
        self.dsb_out_scalefactor.setDecimals(4)
        self.dsb_out_scalefactor.setMinimum(-100.0)
        gridlayoutb.addWidget(self.dsb_out_scalefactor, 4, 1, 1, 1)

        label_out_feasting = QtGui.QLabel(self.groupbox2)
        label_out_feasting.setText("False Easting")
        gridlayoutb.addWidget(label_out_feasting, 5, 0, 1, 1)
        self.dsb_out_feasting.setMaximum(1000000000.0)
        gridlayoutb.addWidget(self.dsb_out_feasting, 5, 1, 1, 1)

        label_out_fnorthing = QtGui.QLabel(self.groupbox2)
        label_out_fnorthing.setText("False Northing")
        gridlayoutb.addWidget(label_out_fnorthing, 6, 0, 1, 1)
        self.dsb_out_fnorthing.setMaximum(1000000000.0)
        gridlayoutb.addWidget(self.dsb_out_fnorthing, 6, 1, 1, 1)

        label_out_zone = QtGui.QLabel(self.groupbox2)
        label_out_zone.setText("UTM Zone")
        gridlayoutb.addWidget(label_out_zone, 7, 0, 1, 1)
        self.sb_out_zone.setMaximum(60)
        gridlayoutb.addWidget(self.sb_out_zone, 7, 1, 1, 1)

# #########################
        self.buttonbox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonbox.setCenterButtons(True)
        self.buttonbox.setStandardButtons(
            QtGui.QDialogButtonBox.Cancel | QtGui.QDialogButtonBox.Ok)

        self.gridlayout_main.addWidget(self.buttonbox, 4, 0, 1, 4)
        QtCore.QObject.connect(self.buttonbox, QtCore.SIGNAL("rejected()"),
                               self.reject)
        QtCore.QObject.connect(self.buttonbox, QtCore.SIGNAL("accepted()"),
                               self.accept)

        datums = ['WGS84', 'Cape (Clarke1880)']
        proj = ['Geodetic', 'UTM (South)', 'UTM (North)',
                'Transverse Mercator']

        self.combobox_inp_datum.addItems(datums)
        self.combobox_inp_proj.addItems(proj)
        self.combobox_out_datum.addItems(datums)
        self.combobox_out_proj.addItems(proj)
        self.zone_inp(35)
        self.zone_out(35)

        self.combobox_inp_proj.currentIndexChanged.connect(self.proj_inp)
        self.sb_inp_zone.valueChanged.connect(self.zone_inp)
        self.combobox_out_proj.currentIndexChanged.connect(self.proj_out)
        self.sb_out_zone.valueChanged.connect(self.zone_out)
#        self.buttonbox.accepted.connect(self.acceptall)
        self.cb_inp_epsg.clicked.connect(self.in_epsg)
        self.cb_out_epsg.clicked.connect(self.out_epsg)
        self.sb_inp_epsg.valueChanged.connect(self.change_inp_epsg_info)
        self.sb_out_epsg.valueChanged.connect(self.change_out_epsg_info)

        self.in_epsg()
        self.out_epsg()

    def acceptall(self):
        """ accept """

# Input stuff
        if self.cb_inp_epsg.isChecked():
            orig = self.get_epsg_inp_proj()
        else:
            orig = self.get_manual_inp_proj()

        if orig is False:
            return

# Output stuff
        if self.cb_out_epsg.isChecked():
            targ = self.get_epsg_out_proj()
        else:
            targ = self.get_manual_out_proj()

        if targ is False:
            return

# Set transformation
        orig_wkt = orig.ExportToWkt()
        targ_wkt = targ.ExportToWkt()
        ctrans = osr.CoordinateTransformation(orig, targ)

# Now create virtual dataset
        dat = []
        for data in self.indata['Raster']:
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
                                           "Your input projection may be wrong"
                                           )
                return

# top left x, w-e pixel size, rotation, top left y, rotation, n-s pixel size
            old_geo = (data.tlx, data.xdim, 0, data.tly, 0, -data.ydim)
            src = data_to_gdal_mem(data, old_geo, orig_wkt, data.cols,
                                   data.rows)

            new_geo = (minx, newdim, 0, maxy, 0, -newdim)
            dest = data_to_gdal_mem(data, new_geo, targ_wkt, cols, rows, True)

            gdal.ReprojectImage(src, dest, orig_wkt, targ_wkt,
                                gdal.GRA_Bilinear)

            data2 = gdal_to_dat(dest, data.bandid)
            if datamin <= 0:
                data2.data = data2.data+(datamin-1)
            mask = data2.data.mask
            data2.data = np.ma.array(data2.data.filled(data.nullvalue))
            data2.data.mask = mask
            data2.nullvalue = data.nullvalue

            dat.append(data2)

        self.outdata['Raster'] = dat

#        self.accept()

    def change_inp_epsg_info(self):
        """ Input epsg is checked """
        txt = 'Invalid Code'
        test = osr.SpatialReference()
        err = test.ImportFromEPSG(self.sb_inp_epsg.value())
        if err == 0:
            txt = test.ExportToPrettyWkt()

        self.inp_epsg_info.setText(txt)

    def change_out_epsg_info(self):
        """ Input epsg is checked """
        txt = 'Invalid Code'
        test = osr.SpatialReference()
        err = test.ImportFromEPSG(self.sb_out_epsg.value())
        if err == 0:
            txt = test.ExportToPrettyWkt()

        self.out_epsg_info.setText(txt)

    def get_manual_inp_proj(self):
        """ get manual input projection """
        orig = osr.SpatialReference()
        orig.SetWellKnownGeogCS('WGS84')

        indx = self.combobox_inp_datum.currentIndex()
        txt = self.combobox_inp_datum.itemText(indx)

        if 'Cape' in txt:
            orig.ImportFromEPSG(4222)

        indx = self.combobox_inp_proj.currentIndex()
        txt = self.combobox_inp_proj.itemText(indx)

        if 'UTM' in txt:
            utmzone = self.sb_inp_zone.value()
            if 'North' in txt:
                orig.SetUTM(utmzone, True)
            else:
                orig.SetUTM(utmzone, False)

        if 'Transverse Mercator' in txt:
            clat = self.dsb_inp_latorigin.value()
            clong = self.dsb_inp_cm.value()
            scale = self.dsb_inp_scalefactor.value()
            f_e = self.dsb_inp_feasting.value()
            f_n = self.dsb_inp_fnorthing.value()
            orig.SetTM(clat, clong, scale, f_e, f_n)

        return orig

    def get_manual_out_proj(self):
        """ get manual input projection """
        targ = osr.SpatialReference()
        targ.SetWellKnownGeogCS('WGS84')

        indx = self.combobox_out_datum.currentIndex()
        txt = self.combobox_out_datum.itemText(indx)

        if 'Cape' in txt:
            targ.ImportFromEPSG(4222)

        indx = self.combobox_out_proj.currentIndex()
        txt = self.combobox_out_proj.itemText(indx)

        if 'UTM' in txt:
            utmzone = self.sb_out_zone.value()
            if 'North' in txt:
                targ.SetUTM(utmzone, True)
            else:
                targ.SetUTM(utmzone, False)

        if 'Transverse Mercator' in txt:
            clat = self.dsb_out_latorigin.value()
            clong = self.dsb_out_cm.value()
            scale = self.dsb_out_scalefactor.value()
            f_e = self.dsb_out_feasting.value()
            f_n = self.dsb_out_fnorthing.value()
            targ.SetTM(clat, clong, scale, f_e, f_n)

        return targ

    def get_epsg_inp_proj(self):
        """ get epsg input projection """
        orig = osr.SpatialReference()
        orig.SetWellKnownGeogCS('WGS84')
        err = orig.ImportFromEPSG(self.sb_inp_epsg.value())
        if err != 0:
            self.parent.showprocesslog('Problem with EPSG projection code')
            return False
        else:
            return orig

    def get_epsg_out_proj(self):
        """ get epsg output input projection """
        targ = osr.SpatialReference()
        targ.SetWellKnownGeogCS('WGS84')
        err = targ.ImportFromEPSG(self.sb_out_epsg.value())
        if err != 0:
            self.parent.showprocesslog('Problem with EPSG projection code')
            return False
        else:
            return targ

    def in_epsg(self):
        """ Input epsg is checked """
        if self.cb_inp_epsg.isChecked():
            self.inp_epsg_info.show()
            self.sb_inp_epsg.setEnabled(True)
            self.groupbox.hide()
            self.groupboxb.show()
        else:
            self.sb_inp_epsg.setEnabled(False)
            self.inp_epsg_info.hide()
            self.groupbox.show()
            self.groupboxb.hide()
            self.proj_inp()

        self.change_inp_epsg_info()
        self.change_out_epsg_info()

    def out_epsg(self):
        """ Input epsg is checked """
        if self.cb_out_epsg.isChecked():
            self.out_epsg_info.show()
            self.sb_out_epsg.setEnabled(True)
            self.groupbox2.hide()
            self.groupbox2b.show()
        else:
            self.out_epsg_info.hide()
            self.sb_out_epsg.setEnabled(False)
            self.groupbox2.show()
            self.groupbox2b.hide()
            self.proj_out()

        self.change_out_epsg_info()

    def proj_inp(self):
        """ used for choosing the projection """

        indx = self.combobox_inp_proj.currentIndex()
        txt = self.combobox_inp_proj.itemText(indx)

        if 'Geodetic' in txt:
            self.sb_inp_zone.setEnabled(False)
            self.dsb_inp_cm.setEnabled(False)
            self.dsb_inp_feasting.setEnabled(False)
            self.dsb_inp_fnorthing.setEnabled(False)
            self.dsb_inp_latorigin.setEnabled(False)
            self.dsb_inp_scalefactor.setEnabled(False)
        else:
            self.sb_inp_zone.setEnabled(True)
            self.dsb_inp_cm.setEnabled(True)
            self.dsb_inp_feasting.setEnabled(True)
            self.dsb_inp_fnorthing.setEnabled(True)
            self.dsb_inp_latorigin.setEnabled(True)
            self.dsb_inp_scalefactor.setEnabled(True)

        if 'UTM' in txt:
            self.sb_inp_zone.setEnabled(True)
            self.zone_inp(self.sb_inp_zone.value())
        else:
            self.sb_inp_zone.setEnabled(False)

        if txt == 'Transverse Mercator':
            self.dsb_inp_feasting.setValue(0.)
            self.dsb_inp_fnorthing.setValue(0.)
            self.dsb_inp_scalefactor.setValue(1.0)

    def proj_out(self):
        """ used for choosing the projection """

        indx = self.combobox_out_proj.currentIndex()
        txt = self.combobox_out_proj.itemText(indx)

        if 'Geodetic' in txt:
            self.sb_out_zone.setEnabled(False)
            self.dsb_out_cm.setEnabled(False)
            self.dsb_out_feasting.setEnabled(False)
            self.dsb_out_fnorthing.setEnabled(False)
            self.dsb_out_latorigin.setEnabled(False)
            self.dsb_out_scalefactor.setEnabled(False)
        else:
            self.sb_out_zone.setEnabled(True)
            self.dsb_out_cm.setEnabled(True)
            self.dsb_out_feasting.setEnabled(True)
            self.dsb_out_fnorthing.setEnabled(True)
            self.dsb_out_latorigin.setEnabled(True)
            self.dsb_out_scalefactor.setEnabled(True)

        if 'UTM' in txt:
            self.sb_out_zone.setEnabled(True)
            self.zone_out(self.sb_out_zone.value())
        else:
            self.sb_out_zone.setEnabled(False)

        if txt == 'Transverse Mercator':
            self.dsb_out_feasting.setValue(0.)
            self.dsb_out_fnorthing.setValue(0.)
            self.dsb_out_scalefactor.setValue(1.0)

    def settings(self):
        """ Settings """

        srs = osr.SpatialReference()
        wkt = self.indata['Raster'][0].wkt
        srs.ImportFromWkt(wkt)
        epsg = srs.GetAttrValue('AUTHORITY', 1)
        if epsg is not None:
            epsg = int(epsg)
            self.sb_inp_epsg.setValue(epsg)
            self.cb_inp_epsg.setChecked(True)
            self.in_epsg()
        else:
            self.proj_inp()

        self.proj_out()

        tmp = self.exec_()
        if tmp == 1:
            self.acceptall()
            tmp = True

        return tmp

    def zone_inp(self, val):
        """ used for changing UTM zone """
        c_m = -180.+(val-1)*6+3
        self.dsb_inp_cm.setValue(c_m)
        self.dsb_inp_latorigin.setValue(0.)
        self.dsb_inp_feasting.setValue(500000.)
        self.dsb_inp_fnorthing.setValue(0.)
        self.dsb_inp_scalefactor.setValue(0.9996)
        self.sb_inp_zone.setValue(val)

        indx = self.combobox_inp_proj.currentIndex()
        txt = self.combobox_inp_proj.itemText(indx)

        if txt == 'UTM (South)':
            self.dsb_inp_fnorthing.setValue(10000000.)

    def zone_out(self, val):
        """ used for changing UTM zone """
        c_m = -180.+(val-1)*6+3
        self.dsb_out_cm.setValue(c_m)
        self.dsb_out_latorigin.setValue(0.)
        self.dsb_out_feasting.setValue(500000.)
        self.dsb_out_fnorthing.setValue(0.)
        self.dsb_out_scalefactor.setValue(0.9996)
        self.sb_out_zone.setValue(val)

        indx = self.combobox_out_proj.currentIndex()
        txt = self.combobox_out_proj.itemText(indx)

        if txt == 'UTM (South)':
            self.dsb_out_fnorthing.setValue(10000000.)


class DataCut(object):
    """ Cut Data using shapefiles """
    def __init__(self, parent):
        self.ifile = ""
        self.name = "Cut Data: "
        self.ext = ""
        self.pbar = None
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
#        data = trim_raster(data)
        self.outdata['Raster'] = data

        return True


def cut_raster(data, ifile):
    """Cuts a raster dataset

    Cut a raster dataset using a shapefile

    Parameters:
        data (Data): PyGMI Dataset
        ifile (str): shapefile used to cut data

    Return:
        Data: PyGMI Dataset
    """
    shapef = ogr.Open(ifile)
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


def trim_raster(olddata):
    """ Function to trim nulls from a raster dataset.

    This function trims entire rows or columns of data which have only nulls,
    and are on the edges of the dataset.

    Args:
        olddata (Data): PyGMI dataset
    Return:
        Data: PyGMI dataset
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


class GetProf(object):
    """ Get Prof """
    def __init__(self, parent):
        self.ifile = ""
        self.name = "Get Profile: "
        self.ext = ""
        self.pbar = None
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
        lyr = shapef.GetLayer()
        line = lyr.GetNextFeature()
        if lyr.GetGeomType() is not ogr.wkbLineString:
            self.parent.showprocesslog('You need lines in that shape file')
            return

        allpoints = []
        for idata in data:
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
            allpoints[-1].dataid = idata.bandid

        self.outdata['Point'] = allpoints

        return True


class Metadata(QtGui.QDialog):
    """ Metadata """
    def __init__(self, parent=None):
        QtGui.QDialog.__init__(self, parent)

        self.indata = {}
        self.outdata = {}
        self.banddata = {}
        self.bandid = {}
        self.oldtxt = ''
        self.parent = parent

        self.gridlayout_main = QtGui.QGridLayout(self)
        self.buttonbox = QtGui.QDialogButtonBox(self)

        self.groupbox = QtGui.QGroupBox(self)
        self.groupboxb = QtGui.QGroupBox(self)
        self.combobox_bandid = QtGui.QComboBox(self)
        self.pb_rename_id = QtGui.QPushButton(self)
        self.lbl_rows = QtGui.QLabel(self.groupbox)
        self.lbl_cols = QtGui.QLabel(self.groupbox)
        self.inp_epsg_info = QtGui.QLabel(self.groupbox)
        self.txt_null = QtGui.QLineEdit(self.groupbox)
        self.dsb_tlx = QtGui.QLineEdit(self.groupbox)
        self.dsb_tly = QtGui.QLineEdit(self.groupbox)
        self.dsb_xdim = QtGui.QLineEdit(self.groupbox)
        self.dsb_ydim = QtGui.QLineEdit(self.groupbox)
        self.led_units = QtGui.QLineEdit(self.groupbox)
        self.lbl_min = QtGui.QLabel(self.groupbox)
        self.lbl_max = QtGui.QLabel(self.groupbox)
        self.lbl_mean = QtGui.QLabel(self.groupbox)

        self.setupui()
        self.ctrans = None

    def setupui(self):
        """ Setup UI """
        sizepolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred,
                                       QtGui.QSizePolicy.Expanding)

        self.setWindowTitle("Dataset Metadata")
        label_bandid = QtGui.QLabel(self)
        label_bandid.setText('Band Name:')
        self.gridlayout_main.addWidget(label_bandid, 0, 0, 1, 1)
        self.gridlayout_main.addWidget(self.combobox_bandid, 0, 1, 1, 3)

        self.pb_rename_id.setText("Rename Band Name")
        self.gridlayout_main.addWidget(self.pb_rename_id, 1, 1, 1, 3)

        self.groupbox.setTitle("Dataset")
        self.groupbox.setSizePolicy(sizepolicy)

        self.groupboxb.setTitle("WKT Information")
        self.groupboxb.setSizePolicy(sizepolicy)

        self.gridlayout_main.addWidget(self.groupbox, 2, 0, 1, 2)
        self.gridlayout_main.addWidget(self.groupboxb, 2, 2, 1, 2)

        gridlayoutb = QtGui.QGridLayout(self.groupboxb)
        gridlayoutb.addWidget(self.inp_epsg_info, 0, 0, 1, 1)

        gridlayout = QtGui.QGridLayout(self.groupbox)

        label_tlx = QtGui.QLabel(self.groupbox)
        label_tlx.setText("Top Left X Coordinate:")
        gridlayout.addWidget(label_tlx, 0, 0, 1, 1)
        gridlayout.addWidget(self.dsb_tlx, 0, 1, 1, 1)

        label_tly = QtGui.QLabel(self.groupbox)
        label_tly.setText("Top Left Y Coordinate:")
        gridlayout.addWidget(label_tly, 1, 0, 1, 1)
        gridlayout.addWidget(self.dsb_tly, 1, 1, 1, 1)

        label_xdim = QtGui.QLabel(self.groupbox)
        label_xdim.setText("X Dimension:")
        gridlayout.addWidget(label_xdim, 2, 0, 1, 1)
        gridlayout.addWidget(self.dsb_xdim, 2, 1, 1, 1)

        label_ydim = QtGui.QLabel(self.groupbox)
        label_ydim.setText("Y Dimension:")
        gridlayout.addWidget(label_ydim, 3, 0, 1, 1)
        gridlayout.addWidget(self.dsb_ydim, 3, 1, 1, 1)

        label_null = QtGui.QLabel(self.groupbox)
        label_null.setText("Null/Nodata value:")
        gridlayout.addWidget(label_null, 4, 0, 1, 1)
        gridlayout.addWidget(self.txt_null, 4, 1, 1, 1)

        label_rows = QtGui.QLabel(self.groupbox)
        label_rows.setText("Rows:")
        gridlayout.addWidget(label_rows, 5, 0, 1, 1)
        gridlayout.addWidget(self.lbl_rows, 5, 1, 1, 1)

        label_cols = QtGui.QLabel(self.groupbox)
        label_cols.setText("Columns:")
        gridlayout.addWidget(label_cols, 6, 0, 1, 1)
        gridlayout.addWidget(self.lbl_cols, 6, 1, 1, 1)

        label_min = QtGui.QLabel(self.groupbox)
        label_min.setText("Dataset Minimum:")
        gridlayout.addWidget(label_min, 7, 0, 1, 1)
        gridlayout.addWidget(self.lbl_min, 7, 1, 1, 1)

        label_max = QtGui.QLabel(self.groupbox)
        label_max.setText("Dataset Maximum:")
        gridlayout.addWidget(label_max, 8, 0, 1, 1)
        gridlayout.addWidget(self.lbl_max, 8, 1, 1, 1)

        label_mean = QtGui.QLabel(self.groupbox)
        label_mean.setText("Dataset Mean:")
        gridlayout.addWidget(label_mean, 9, 0, 1, 1)
        gridlayout.addWidget(self.lbl_mean, 9, 1, 1, 1)

        label_units = QtGui.QLabel(self.groupbox)
        label_units.setText("Dataset Units:")
        gridlayout.addWidget(label_units, 10, 0, 1, 1)
        gridlayout.addWidget(self.led_units, 10, 1, 1, 1)

        self.buttonbox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonbox.setCenterButtons(True)
        self.buttonbox.setStandardButtons(
            QtGui.QDialogButtonBox.Cancel | QtGui.QDialogButtonBox.Ok)

        self.gridlayout_main.addWidget(self.buttonbox, 4, 0, 1, 4)
        QtCore.QObject.connect(self.buttonbox, QtCore.SIGNAL("rejected()"),
                               self.reject)
        QtCore.QObject.connect(self.buttonbox, QtCore.SIGNAL("accepted()"),
                               self.accept)

        self.combobox_bandid.currentIndexChanged.connect(self.update_vals)
        self.pb_rename_id.clicked.connect(self.rename_id)

    def acceptall(self):
        """ accept """
        self.update_vals()
        for tmp in self.indata['Raster']:
            for j in self.bandid.items():
                if j[1] == tmp.bandid:
                    i = self.banddata[j[0]]
                    tmp.bandid = j[0]
                    tmp.tlx = i.tlx
                    tmp.tly = i.tly
                    tmp.xdim = i.xdim
                    tmp.ydim = i.ydim
                    tmp.rows = i.rows
                    tmp.cols = i.cols
                    tmp.nullvalue = i.nullvalue
                    tmp.wkt = i.wkt
                    tmp.units = i.units
                    if tmp.bandid[-1] == ')':
                        tmp.bandid = tmp.bandid[:tmp.bandid.rfind(' (')]
                    if i.units != '':
                        tmp.bandid += ' ('+i.units+')'
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
            self.bandid[skey] = self.bandid.pop(txt)
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

        srs = osr.SpatialReference()
        wkt = idata.wkt
        srs.ImportFromWkt(wkt)
        self.inp_epsg_info.setText(srs.ExportToPrettyWkt())

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
        """ Settings """
        bandid = []
        for i in self.indata['Raster']:
            bandid.append(i.bandid)
            self.banddata[i.bandid] = Data()
            tmp = self.banddata[i.bandid]
            self.bandid[i.bandid] = i.bandid
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


class DataGrid(QtGui.QDialog):
    """ Grid Point Data """
    def __init__(self, parent=None):
        QtGui.QDialog.__init__(self, parent)

        self.indata = {}
        self.outdata = {}
        self.parent = parent

        self.gridlayout_main = QtGui.QGridLayout(self)
        self.buttonbox = QtGui.QDialogButtonBox(self)
        self.dsb_dxy = QtGui.QDoubleSpinBox(self)
        self.label_rows = QtGui.QLabel(self)
        self.label_cols = QtGui.QLabel(self)

        self.setupui()

    def setupui(self):
        """ Setup UI """
        self.setWindowTitle("Dataset Gridding")

        self.label_rows.setText("Rows: 0")
        self.gridlayout_main.addWidget(self.label_rows, 1, 0, 1, 2)

        self.label_cols.setText("Columns: 0")
        self.gridlayout_main.addWidget(self.label_cols, 2, 0, 1, 2)

        label_dxy = QtGui.QLabel(self)
        label_dxy.setText("Cell Size:")
        self.gridlayout_main.addWidget(label_dxy, 0, 0, 1, 1)

        self.dsb_dxy.setMaximum(9999999999.0)
        self.dsb_dxy.setMinimum(0.00001)
        self.dsb_dxy.setDecimals(5)
        self.gridlayout_main.addWidget(self.dsb_dxy, 0, 1, 1, 1)

        self.buttonbox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonbox.setCenterButtons(True)
        self.buttonbox.setStandardButtons(
            QtGui.QDialogButtonBox.Cancel | QtGui.QDialogButtonBox.Ok)

        self.gridlayout_main.addWidget(self.buttonbox, 3, 0, 1, 4)
        QtCore.QObject.connect(self.buttonbox, QtCore.SIGNAL("rejected()"),
                               self.reject)
        QtCore.QObject.connect(self.buttonbox, QtCore.SIGNAL("accepted()"),
                               self.accept)

#        self.buttonbox.accepted.connect(self.acceptall)
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
        data = self.indata['Point'][0]
        x = data.xdata
        y = data.ydata
        xy = np.transpose([x, y])
        xy = xy.tolist()
#        xy.sort()
        xy = np.array(xy)
        xy = xy[:-1]-xy[1:]
        dxy = np.median(np.sqrt(np.sum(xy**2, 1)))/3

#        pd = sd.pdist(xy)
#        dxy = np.median(pd)/3

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

        x = data.xdata
        y = data.ydata

        xy = np.transpose([x, y])
        x2 = np.arange(x.min(), x.max(), dxy)
        y2 = np.arange(y.min(), y.max(), dxy)

        xnew, ynew = np.meshgrid(x2, y2)

        newdat = []
        for data in self.indata['Point']:
            z = data.zdata
            tmp = si.griddata(xy, z, (xnew, ynew))
            tmp = np.ma.masked_invalid(tmp)
            mask = tmp.mask
            gdat = tmp.data
#            rbf = si.Rbf(x, y, z, epsilon=dxy)
#            gdat = rbf(xnew, ynew)

    # Create dataset
            dat = Data()
            dat.data = np.ma.masked_invalid(gdat[::-1])
            dat.data.mask = mask[::-1]
            dat.rows, dat.cols = gdat.shape
            dat.nullvalue = np.nan
            dat.bandid = data.dataid
            dat.tlx = x2[0]-dxy/2
            dat.tly = y2[-1]+dxy/2
            dat.xdim = dxy
            dat.ydim = dxy
            newdat.append(dat)

        self.outdata['Raster'] = newdat
        self.outdata['Point'] = self.indata['Point']
#        self.accept()
