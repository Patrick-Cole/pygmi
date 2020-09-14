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

import math
import os
import copy
from collections import Counter
from PyQt5 import QtWidgets, QtCore
import numpy as np
from osgeo import gdal, osr, ogr
import pandas as pd
from PIL import Image, ImageDraw
import scipy.ndimage as ndimage
from scipy.signal import tukey
import scipy.interpolate as si
import pygmi.menu_default as menu_default
from pygmi.raster.datatypes import Data

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

    def __init__(self, parent=None):
        self.ifile = ''
        self.pbar = parent.pbar
        self.parent = parent
        self.indata = {}
        self.outdata = {}
        if parent is None:
            self.showprocesslog = print
        else:
            self.showprocesslog = parent.showprocesslog

    def settings(self, nodialog=False):
        """
        Entry point into item.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        if 'Raster' in self.indata:
            data = copy.deepcopy(self.indata['Raster'])
        else:
            self.showprocesslog('No raster data')
            return False

        nodialog = False
        if not nodialog:
            self.ifile, _ = QtWidgets.QFileDialog.getOpenFileName(
                self.parent, 'Open Shape File', '.', 'Shape file (*.shp)')
            if self.ifile == '':
                return False

        os.chdir(os.path.dirname(self.ifile))
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
        self.ifile = projdata['shapefile']

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

        projdata['shapefile'] = self.ifile

        return projdata


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
        super().__init__(parent)
        if parent is None:
            self.showprocesslog = print
        else:
            self.showprocesslog = parent.showprocesslog

        self.indata = {}
        self.outdata = {}
        self.parent = parent
        self.dxy = None
        self.piter = parent.pbar.iter

        self.dsb_dxy = QtWidgets.QDoubleSpinBox()
        self.label_rows = QtWidgets.QLabel('Rows: 0')
        self.label_cols = QtWidgets.QLabel('Columns: 0')

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
        helpdocs = menu_default.HelpButton('pygmi.raster.dataprep.datamerge')
        label_dxy = QtWidgets.QLabel('Cell Size:')

        self.dsb_dxy.setMaximum(9999999999.0)
        self.dsb_dxy.setMinimum(0.00001)
        self.dsb_dxy.setDecimals(5)
        self.dsb_dxy.setValue(40.)
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

        Returns
        -------
        None.

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

    def settings(self, nodialog=False):
        """
        Entry point into item.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        if not nodialog:
            data = self.indata['Raster'][0]

            if self.dxy is None:
                dxy0 = min(data.xdim, data.ydim)
                for data in self.indata['Raster']:
                    self.dxy = min(dxy0, data.xdim, data.ydim)

            self.dsb_dxy.setValue(self.dxy)

            tmp = self.exec_()
            if tmp != 1:
                return False

        self.acceptall()

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
        self.dxy = projdata['dxy']

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

        projdata['dxy'] = self.dsb_dxy.value()

        return projdata

    def acceptall(self):
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        None.

        """
        dxy = self.dsb_dxy.value()
        self.dxy = dxy
        # data = self.indata['Raster'][0]

        dat = merge(self.indata['Raster'], self.piter, dxy)

        # orig_wkt = data.wkt

        # xmin0, xmax0, ymin0, ymax0 = data.extent

        # for data in self.indata['Raster']:
        #     xmin, xmax, ymin, ymax = data.extent
        #     xmin = min(xmin, xmin0)
        #     xmax = max(xmax, xmax0)
        #     ymin = min(ymin, ymin0)
        #     ymax = max(ymax, ymax0)

        # cols = int((xmax - xmin)/dxy)
        # rows = int((ymax - ymin)/dxy)
        # gtr = (xmin, dxy, 0.0, ymax, 0.0, -dxy)

        # if cols == 0 or rows == 0:
        #     self.showprocesslog('Your rows or cols are zero. Your input projection may be '
        #           'wrong')
        #     return

        # dat = []
        # for data in self.indata['Raster']:
        #     doffset = 0.0
        #     if data.data.min() <= 0:
        #         doffset = data.data.min()-1.
        #         data.data = data.data - doffset
        #     gtr0 = data.get_gtr()

        #     drows, dcols = data.data.shape
        #     src = data_to_gdal_mem(data, gtr0, orig_wkt, dcols, drows)
        #     dest = data_to_gdal_mem(data, gtr, orig_wkt, cols, rows, True)

        #     gdal.ReprojectImage(src, dest, orig_wkt, orig_wkt,
        #                         gdal.GRA_Bilinear)

        #     dat.append(gdal_to_dat(dest, data.dataid))
        #     dat[-1].data = dat[-1].data + doffset
        #     data.data = data.data + doffset

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
        super().__init__(parent)
        if parent is None:
            self.showprocesslog = print
        else:
            self.showprocesslog = parent.showprocesslog

        self.indata = {}
        self.outdata = {}
        self.parent = parent
        self.pbar = self.parent.pbar
        self.orig_wkt = None
        self.targ_wkt = None

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
        """
        Set up UI.

        Returns
        -------
        None.

        """
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
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        None.

        """
        if self.in_proj.wkt == 'Unknown' or self.out_proj.wkt == 'Unknown':
            self.showprocesslog('Could not reproject')
            return

# Input stuff
        orig_wkt = self.in_proj.wkt

        orig = osr.SpatialReference()
        orig.ImportFromWkt(orig_wkt)
        orig.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

# Output stuff
        targ_wkt = self.out_proj.wkt

        targ = osr.SpatialReference()
        targ.ImportFromWkt(targ_wkt)
        targ.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

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
                self.showprocesslog('Your rows or cols are zero. Your input projection may '
                      'be wrong')
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

    def settings(self, nodialog=False):
        """
        Entry point into item.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        if self.orig_wkt is None:
            self.orig_wkt = self.indata['Raster'][0].wkt
        if self.targ_wkt is None:
            self.targ_wkt = self.indata['Raster'][0].wkt

        self.in_proj.set_current(self.orig_wkt)
        self.out_proj.set_current(self.targ_wkt)

        if not nodialog:
            tmp = self.exec_()
            if tmp != 1:
                return False

        self.acceptall()

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
        self.orig_wkt = projdata['orig_wkt']
        self.targ_wkt = projdata['targ_wkt']

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

        projdata['orig_wkt'] = self.in_proj.wkt
        projdata['targ_wkt'] = self.out_proj.wkt

        return projdata


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

    def __init__(self, parent=None):
        self.ifile = ''
        self.pbar = parent.pbar
        self.parent = parent
        self.indata = {}
        self.outdata = {}
        if parent is None:
            self.showprocesslog = print
        else:
            self.showprocesslog = parent.showprocesslog

    def settings(self, nodialog=False):
        """
        Entry point into item.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        if 'Raster' in self.indata:
            data = copy.deepcopy(self.indata['Raster'])
        else:
            self.showprocesslog('No raster data')
            return False

        ext = 'Shape file (*.shp)'

        nodialog = False
        if not nodialog:
            self.ifile, _ = QtWidgets.QFileDialog.getOpenFileName(
                self.parent, 'Open Shape File', '.', ext)
            if self.ifile == '':
                return False

        os.chdir(os.path.dirname(self.ifile))

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
            self.showprocesslog('You need lines in that shape file')
            return False

        data = merge(data)
        gdf = None

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

            if gdf is None:
                gdf = pd.DataFrame(xxx, columns=['X'])
                gdf['Y'] = yyy
                gdf['pygmiX'] = gdf['X']
                gdf['pygmiY'] = gdf['Y']

            gdf[idata.dataid] = tmpprof

        shapef = None
        gdf['line'] = 'None'

        self.outdata['Line'] = {'profile': gdf}

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
        self.ifile = projdata['shapefile']

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

        projdata['shapefile'] = self.ifile

        return projdata


class GroupProj(QtWidgets.QWidget):
    """
    Group Proj.

    Custom widget
    """

    def __init__(self, title='Projection', parent=None):
        super().__init__(parent)

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
        """
        Set new WKT for current option.

        Parameters
        ----------
        wkt : str
            Well Known Text descriptions for coordfinates (WKT) .

        Returns
        -------
        None.

        """
        self.wkt = wkt
        self.epsg_proj['Current'] = self.wkt
        self.combo_change()

    def combo_change(self):
        """
        Change Combo.

        Returns
        -------
        None.

        """
        indx = self.combobox.currentIndex()
        txt = self.combobox.itemText(indx)

        self.wkt = self.epsg_proj[txt]

        if not isinstance(self.wkt, str):
            self.wkt = epsgtowkt(self.wkt)

        srs = osr.SpatialReference()
        srs.ImportFromWkt(self.wkt)
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

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
        super().__init__(parent)
        if parent is None:
            self.showprocesslog = print
        else:
            self.showprocesslog = parent.showprocesslog

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
        """
        Set up UI.

        Returns
        -------
        None.

        """
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
        Accept option.

        Returns
        -------
        None.

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
        """
        Rename the band name.

        Returns
        -------
        None.

        """
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
        """
        Update the values on the interface.

        Returns
        -------
        None.

        """
        odata = self.banddata[self.oldtxt]
        odata.units = self.led_units.text()

        try:
            odata.nullvalue = float(self.txt_null.text())
            left = float(self.dsb_tlx.text())
            top = float(self.dsb_tly.text())
            odata.xdim = float(self.dsb_xdim.text())
            odata.ydim = float(self.dsb_ydim.text())

            rows = odata.data.shape[0]
            cols = odata.data.shape[1]

            right = left + odata.xdim*cols
            bottom = top - odata.ydim*rows

            odata.extent = (left, right, bottom, top)

        except ValueError:
            self.showprocesslog('Value error - abandoning changes')

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
        """
        Entrypoint to start this routine.

        Returns
        -------
        tmp : bool
            True if successful, False otherwise.

        """
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

        if tmp != 1:
            return False

        self.acceptall()

        return True


class RTP(QtWidgets.QDialog):
    """
    Perform Reduction to the Pole on Magnetic data.

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

        self.dataid = QtWidgets.QComboBox()
        self.dsb_inc = QtWidgets.QDoubleSpinBox()
        self.dsb_dec = QtWidgets.QDoubleSpinBox()

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
        helpdocs = menu_default.HelpButton('pygmi.raster.dataprep.rtp')
        label_band = QtWidgets.QLabel('Band to Reduce to the Pole:')
        label_inc = QtWidgets.QLabel('Inclination of Magnetic Field:')
        label_dec = QtWidgets.QLabel('Declination of Magnetic Field:')

        self.dsb_inc.setMaximum(90.0)
        self.dsb_inc.setMinimum(-90.0)
        self.dsb_dec.setMaximum(360.0)
        self.dsb_dec.setMinimum(-360.0)
        self.dsb_inc.setValue(-62.5)
        self.dsb_dec.setValue(-16.75)

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

    def settings(self, nodialog=False):
        """
        Entry point into item.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        tmp = []
        if 'Raster' not in self.indata:
            return False

        for i in self.indata['Raster']:
            tmp.append(i.dataid)

        self.dataid.clear()
        self.dataid.addItems(tmp)

        if not nodialog:
            tmp = self.exec_()

            if tmp != 1:
                return False

        self.acceptall()

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
        self.dataid.setCurrentText(projdata['band'])
        self.dsb_inc.setValue(projdata['inc'])
        self.dsb_dec.setValue(projdata['dec'])

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

        projdata['band'] = self.dataid.currentText()
        projdata['inc'] = self.dsb_inc.value()
        projdata['dec'] = self.dsb_dec.value()

        return projdata

    def acceptall(self):
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        None.

        """
        I_deg = self.dsb_inc.value()
        D_deg = self.dsb_dec.value()

        newdat = []
        for data in self.pbar.iter(self.indata['Raster']):
            if data.dataid != self.dataid.currentText():
                continue
            dat = rtp(data, I_deg, D_deg)
            newdat.append(dat)

        self.outdata['Raster'] = newdat


class Continuation(QtWidgets.QDialog):
    """
    Perform upward and downward contiuation on potential field data.

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

        self.dataid = QtWidgets.QComboBox()
        self.continuation = QtWidgets.QComboBox()
        self.dsb_height = QtWidgets.QDoubleSpinBox()

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
        helpdocs = menu_default.HelpButton('pygmi.raster.dataprep.cont')
        label_band = QtWidgets.QLabel('Band to perform continuation:')
        label_cont = QtWidgets.QLabel('Continuation type:')
        label_height = QtWidgets.QLabel('Continuation distance:')

        self.dsb_height.setMaximum(1000000.0)
        self.dsb_height.setMinimum(0.0)
        self.dsb_height.setValue(0.0)
        self.continuation.clear()
        self.continuation.addItems(['Upward', 'Downward'])

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Continuation')

        gridlayout_main.addWidget(label_band, 0, 0, 1, 1)
        gridlayout_main.addWidget(self.dataid, 0, 1, 1, 1)

        gridlayout_main.addWidget(label_cont, 1, 0, 1, 1)
        gridlayout_main.addWidget(self.continuation, 1, 1, 1, 1)
        gridlayout_main.addWidget(label_height, 2, 0, 1, 1)
        gridlayout_main.addWidget(self.dsb_height, 2, 1, 1, 1)
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
        if 'Raster' not in self.indata:
            return False

        for i in self.indata['Raster']:
            tmp.append(i.dataid)

        self.dataid.clear()
        self.dataid.addItems(tmp)

        if not nodialog:
            tmp = self.exec_()

            if tmp != 1:
                return False

        self.acceptall()

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
        self.dataid.setCurrentText(projdata['band'])
        self.continuation.setCurrenText(projdata['ctype'])
        self.dsb_height.setValue(projdata['height'])

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

        projdata['band'] = self.dataid.currentText()
        projdata['ctype'] = self.continuation.currentText()
        projdata['height'] = self.dsb_height.value()

        return projdata

    def acceptall(self):
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        None.

        """
        h = self.dsb_height.value()
        ctype = self.continuation.currentText()

        # Get data
        for i in self.indata['Raster']:
            if i.dataid == self.dataid.currentText():
                data = i
                break

        if ctype == 'Downward':
            dat = taylorcont(data, h)
        else:
            dat = fftcont(data, h)

        self.outdata['Raster'] = [dat]


def fftprep(data):
    """
    FFT preparation.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    dxy : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    datamedian = np.ma.median(data.data)
    ndat = data.data - datamedian

    nr, nc = data.data.shape
    cdiff = nc//2
    rdiff = nr//2

    # # Section to pad data

    # nr, nc = data.data.shape

    # z1 = np.zeros((nr+2*rdiff, nc+2*cdiff))-999
    # x1, y1 = np.mgrid[0: nr+2*rdiff, 0: nc+2*cdiff]
    # z1[rdiff:-rdiff, cdiff:-cdiff] = ndat.filled(-999)

    # z1[0] = 0
    # z1[-1] = 0
    # z1[:, 0] = 0
    # z1[:, -1] = 0

    # x = x1.flatten()
    # y = y1.flatten()
    # z = z1.flatten()

    # x = x[z != -999]
    # y = y[z != -999]
    # z = z[z != -999]

    # points = np.transpose([x, y])

    # zfin = si.griddata(points, z, (x1, y1), method='nearest')

    z1 = np.zeros((nr+2*rdiff, nc+2*cdiff))+np.nan
    x1, y1 = np.mgrid[0: nr+2*rdiff, 0: nc+2*cdiff]
    z1[rdiff:-rdiff, cdiff:-cdiff] = ndat.filled(np.nan)

    for j in range(2):
        z1[0] = 0
        z1[-1] = 0
        z1[:, 0] = 0
        z1[:, -1] = 0

        vert = np.zeros_like(z1)
        hori = np.zeros_like(z1)

        for i in range(z1.shape[0]):
            mask = ~np.isnan(z1[i])
            y = y1[i][mask]
            z = z1[i][mask]
            hori[i] = np.interp(y1[i], y, z)

        for i in range(z1.shape[1]):
            mask = ~np.isnan(z1[:, i])
            x = x1[:, i][mask]
            z = z1[:, i][mask]

            vert[:, i] = np.interp(x1[:, i], x, z)

        hori[hori == 0] = np.nan
        vert[vert == 0] = np.nan

        hv = hori.copy()
        hv[np.isnan(hori)] = vert[np.isnan(hori)]
        hv[~np.isnan(hv)] = np.nanmean([hori[~np.isnan(hv)],
                                        vert[~np.isnan(hv)]], 0)

        z1[np.isnan(z1)] = hv[np.isnan(z1)]

    zfin = z1

    nr, nc = zfin.shape
    zfin *= tukey(nc)
    zfin *= tukey(nr)[:, np.newaxis]

    return zfin, rdiff, cdiff, datamedian


def fft_getkxy(fftmod, xdim, ydim):
    """
    Get KX and KY.

    Parameters
    ----------
    fftmod : TYPE
        DESCRIPTION.
    dxy : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    ny, nx = fftmod.shape
    kx = np.fft.fftfreq(nx, xdim)*2*np.pi
    ky = np.fft.fftfreq(ny, ydim)*2*np.pi

    KX, KY = np.meshgrid(kx, ky)
    KY = -KY
    return KX, KY


def verticalp(data, order=1):
    """
    Vertical derivative.

    Parameters
    ----------
    data : numpy array
        Input data.
    npts : int, optional
        Number of points. The default is None.
    xint : float, optional
        X interval. The default is 1.

    Returns
    -------
    dz : numpy array
        Output data

    """
    xdim = data.xdim
    ydim = data.ydim

    ndat, rdiff, cdiff, _ = fftprep(data)
    fftmod = np.fft.fft2(ndat)

    KX, KY = fft_getkxy(fftmod, xdim, ydim)

    k = np.sqrt(KX**2+KY**2)
    filt = k**order

    zout = np.real(np.fft.ifft2(fftmod*filt))
    zout = zout[rdiff:-rdiff, cdiff:-cdiff]

    return zout


def fftcont(data, h):
    """
    Continuation.

    Parameters
    ----------
    data : PyGMI Data
        PyGMI raster data.
    h : float
        Height.

    Returns
    -------
    dat : PyGMI Data
        PyGMI raster data.

    """
    xdim = data.xdim
    ydim = data.ydim

    # self.showprocesslog('Preparing for FFT...')
    ndat, rdiff, cdiff, datamedian = fftprep(data)

    # self.showprocesslog('Continuing data...')

    fftmod = np.fft.fft2(ndat)

    ny, nx = fftmod.shape

    KX, KY = fft_getkxy(fftmod, xdim, ydim)
    k = np.sqrt(KX**2+KY**2)

    filt = np.exp(-np.abs(k)*h)

    zout = np.real(np.fft.ifft2(fftmod*filt))
    zout = zout[rdiff:-rdiff, cdiff:-cdiff]

    zout = zout + datamedian

    zout[data.data.mask] = data.data.fill_value

    dat = Data()
    dat.data = np.ma.masked_invalid(zout)
    dat.data.mask = np.ma.getmaskarray(data.data)
    dat.nullvalue = data.data.fill_value
    dat.dataid = 'Upward_'+str(h)+'_'+data.dataid
    dat.extent = data.extent
    dat.xdim = data.xdim
    dat.ydim = data.ydim

    return dat


def taylorcont(data, h):
    """
    Continuation.

    Parameters
    ----------
    data : PyGMI Data
        PyGMI raster data.
    h : float
        Height.

    Returns
    -------
    dat : PyGMI Data
        PyGMI raster data.

    """
    dz = verticalp(data, order=1)
    dz2 = verticalp(data, order=2)
    dz3 = verticalp(data, order=3)
    zout = (data.data + h*dz + h**2*dz2/math.factorial(2) +
            h**3*dz3/math.factorial(3))

    dat = Data()
    dat.data = np.ma.masked_invalid(zout)
    dat.data.mask = np.ma.getmaskarray(data.data)
    dat.nullvalue = data.data.fill_value
    dat.dataid = 'Downward_'+str(h)+'_'+data.dataid
    dat.extent = data.extent
    dat.xdim = data.xdim
    dat.ydim = data.ydim

    return dat


def rtp(data, I_deg, D_deg):
    """
    Reduction to th epole.

    Parameters
    ----------
    data : PyGMI Data
        PyGMI raster data.
    I_deg : float
        Magnetic inclination.
    D_deg : float
        Magnetic declination.

    Returns
    -------
    dat : PyGMI Data
        PyGMI raster data.

    """
    xdim = data.xdim
    ydim = data.ydim

    ndat, rdiff, cdiff, datamedian = fftprep(data)
    fftmod = np.fft.fft2(ndat)

    ny, nx = fftmod.shape
    KX, KY = fft_getkxy(fftmod, xdim, ydim)

    I = np.deg2rad(I_deg)
    D = np.deg2rad(D_deg)
    alpha = np.arctan2(KY, KX)

    filt = 1/(np.sin(I)+1j*np.cos(I)*np.sin(D+alpha))**2

    zout = np.real(np.fft.ifft2(fftmod*filt))
    zout = zout[rdiff:-rdiff, cdiff:-cdiff]
    zout = zout + datamedian

    zout[data.data.mask] = data.data.fill_value

    dat = Data()
    dat.data = np.ma.masked_invalid(zout)
    dat.data.mask = np.ma.getmaskarray(data.data)
    dat.nullvalue = data.data.fill_value
    dat.dataid = 'RTP_'+data.dataid
    dat.extent = data.extent
    dat.xdim = data.xdim
    dat.ydim = data.ydim

    return dat


def check_dataid(out):
    """
    Check dataid for duplicates and renames where necessary.

    Parameters
    ----------
    out : PyGMI Data
        PyGMI raster data.

    Returns
    -------
    out : PyGMI Data
        PyGMI raster data.

    """
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
    indata : Data
        PyGMI raster dataset

    """
    if 'Cluster' not in indata:
        return indata
    if 'Raster' not in indata:
        indata['Raster'] = []

    for i in indata['Cluster']:
        indata['Raster'].append(i)
        indata['Raster'][-1].data = indata['Raster'][-1].data + 1

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
    data : Data
        PyGMI Dataset
    """
    shapef = ogr.Open(ifile)
    if shapef is None:
        return None
    lyr = shapef.GetLayer()
    poly = lyr.GetNextFeature()
    geom = poly.GetGeometryRef()

    if 'POLYGON' not in geom.GetGeometryName() or poly is None:
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
    src : GDAL mem format
        GDAL memory format data

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
    """
    Routine to get a WKT from an epsg code.

    Parameters
    ----------
    epsg : str or int
        EPSG code.

    Returns
    -------
    out : str
        WKT description.

    """
    orig = osr.SpatialReference()
    orig.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

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
    dest : GDAL format
        GDAL format
    bandid : str
        band identity

    Returns
    -------
    dat : PyGMI Data
        PyGMI raster dataset.

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
    """
    Routine used to get a list of EPSG codes.

    Returns
    -------
    pcodes : dictionary
        Dictionary of codes per projection in WKT format.

    """
    with open(os.path.join(os.path.dirname(__file__), 'gcs.csv')) as dfile:
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

    with open(os.path.join(os.path.dirname(__file__), 'pcs.csv')) as pfile:
        plines = pfile.readlines()

    orig = osr.SpatialReference()
    orig.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

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
    orig.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    for datum in ['Cape', 'Hartebeesthoek94']:
        orig.ImportFromEPSG(dcodes[datum])
#        orig.ImportFromWkt(dcodes[datum])
        for clong in range(15, 35, 2):
            orig.SetTM(clat, clong, scale, f_e, f_n)
            orig.SetProjCS(datum+r' / TM'+str(clong))
            pcodes[datum+r' / TM'+str(clong)] = orig.ExportToWkt()

    return pcodes


# def merge(dat, piter=None):
#     """
#     Merge datasets found in a single PyGMI data object.

#     The aim is to ensure that all datasets have the same number of rows and
#     columns.

#     Parameters
#     ----------
#     dat : PyGMI Data
#         data object which stores datasets

#     Returns
#     -------
#     out : PyGMI Data
#         data object which stores datasets
#     """
#     if piter is None:
#         piter = iter

#     if dat[0].isrgb:
#         return dat

#     needsmerge = False
#     rows, cols = dat[0].data.shape
#     for i in dat:
#         irows, icols = i.data.shape
#         if irows != rows or icols != cols:
#             needsmerge = True

#     if needsmerge is False:
#         dat = check_dataid(dat)
#         return dat

#     self.showprocesslog('Merging data...')
#     mrg = DataMerge()
#     mrg.indata['Raster'] = dat
#     data = dat[0]
#     dxy0 = min(data.xdim, data.ydim)
#     for data in dat:
#         dxy = min(dxy0, data.xdim, data.ydim)

#     mrg.dsb_dxy.setValue(dxy)
#     mrg.acceptall()
#     out = mrg.outdata['Raster']

#     out = check_dataid(out)

#     return out


def merge(dat, piter=None, dxy=None):
    """
    Merge datasets found in a single PyGMI data object.

    The aim is to ensure that all datasets have the same number of rows and
    columns.

    Parameters
    ----------
    dat : PyGMI Data
        data object which stores datasets

    Returns
    -------
    out : PyGMI Data
        data object which stores datasets
    """

    if piter is None:
        piter = iter

    if dat[0].isrgb:
        return dat

    needsmerge = False
    rows, cols = dat[0].data.shape
    for i in dat:
        irows, icols = i.data.shape
        if irows != rows or icols != cols:
            needsmerge = True

    if needsmerge is False:
        dat = check_dataid(dat)
        return dat

    # self.showprocesslog('Merging data...')

    data = dat[0]
    dxy0 = min(data.xdim, data.ydim)
    if dxy is None:
        for data in dat:
            dxy = min(dxy0, data.xdim, data.ydim)

    orig_wkt = data.wkt
    xmin0, xmax0, ymin0, ymax0 = data.extent

    for data in dat:
        xmin, xmax, ymin, ymax = data.extent
        xmin = min(xmin, xmin0)
        xmax = max(xmax, xmax0)
        ymin = min(ymin, ymin0)
        ymax = max(ymax, ymax0)

    cols = int((xmax - xmin)/dxy)
    rows = int((ymax - ymin)/dxy)
    gtr = (xmin, dxy, 0.0, ymax, 0.0, -dxy)

    if cols == 0 or rows == 0:
        print('Your rows or cols are zero. Your input projection may be wrong')
        return None

    dat2 = []
    for data in piter(dat):
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

        dat2.append(gdal_to_dat(dest, data.dataid))
        dat2[-1].data = dat2[-1].data + doffset
        data.data = data.data + doffset

    out = check_dataid(dat2)

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
    olddata : Data
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


def testrtp():
    """Main RTP testing routine."""
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from pygmi.pfmod.grvmag3d import quick_model, calc_field
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')

# quick model
    finc = -57
    fdec = 50

    lmod = quick_model(numx=300, numy=300, numz=30, finc=finc, fdec=fdec)
    lmod.lith_index[100:200, 100:200, 0:10] = 1
#    lmod.lith_index[:, :, 10] = 1
    lmod.mht = 100
    calc_field(lmod, magcalc=True)

# Calculate the field

    magval = lmod.griddata['Calculated Magnetics'].data
    plt.imshow(magval, cmap=cm.jet)
    plt.show()

    dat2 = rtp(lmod.griddata['Calculated Magnetics'], finc, fdec)
    plt.imshow(dat2.data, cmap=cm.jet)
    plt.show()


def testdown():
    """Main continuation testing routine."""
    import matplotlib.pyplot as plt
    from pygmi.pfmod.grvmag3d import quick_model, calc_field
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')

    h = 4
    dxy = 1
    magcalc = True

# quick model
    lmod = quick_model(numx=100, numy=100, numz=10, dxy=dxy, d_z=1)
    lmod.lith_index[45:55, :, 1] = 1
    lmod.lith_index[45:50, :, 0] = 1
    lmod.ght = 10
    lmod.mht = 10
    calc_field(lmod, magcalc=magcalc)
    if magcalc:
        z = lmod.griddata['Calculated Magnetics']
        z.data = z.data + 5
    else:
        z = lmod.griddata['Calculated Gravity']

# Calculate the field
    lmod = quick_model(numx=100, numy=100, numz=10, dxy=dxy, d_z=1)
    lmod.lith_index[45:55, :, 1] = 1
    lmod.lith_index[45:50, :, 0] = 1
    lmod.ght = 10 - h
    lmod.mht = 10 - h
    calc_field(lmod, magcalc=magcalc)
    if magcalc:
        downz0 = lmod.griddata['Calculated Magnetics']
        downz0.data = downz0.data + 5
    else:
        downz0 = lmod.griddata['Calculated Gravity']

    downz0, z = z, downz0
    # h = -h

    dz = verticalp(z, order=1)
    dz2 = verticalp(z, order=2)
    dz3 = verticalp(z, order=3)

# normal downward
    zdownn = fftcont(z, h)

# downward, taylor
    h = -h
    zdown = (z.data + h*dz + h**2*dz2/math.factorial(2) +
             h**3*dz3/math.factorial(3))

# Plotting
    # plt.plot(dz3[50])
    # plt.show()

#    plt.plot(z[50], 'r-.')
    plt.plot(downz0.data[50], 'r.')
    # plt.plot(zdown.data[50], 'b')
    plt.plot(zdownn.data[50], 'k')
    plt.show()


def testgrid():
    from pygmi.raster.iodefs import get_raster
    from pygmi.misc import PTime
    from pygmi.vector.dataprep import quickgrid
    import matplotlib.pyplot as plt

    ttt = PTime()

    ifile = r'C:\Work\Workdata\upward\EB_MTEF_Mag_IGRFrem.ers'
    dat = get_raster(ifile)[0]

    # z = dat.data
    nr, nc = dat.data.shape

    datamedian = np.ma.median(dat.data)
    ndat = dat.data - datamedian

    cdiff = nc//2
    rdiff = nr//2

    # Section to pad data

    z1 = np.zeros((nr+2*rdiff, nc+2*cdiff))+np.nan
    x1, y1 = np.mgrid[0: nr+2*rdiff, 0: nc+2*cdiff]
    z1[rdiff:-rdiff, cdiff:-cdiff] = ndat.filled(np.nan)

    ttt.since_last_call('Preparation')

    for j in range(2):
        z1[0] = 0
        z1[-1] = 0
        z1[:, 0] = 0
        z1[:, -1] = 0

        vert = np.zeros_like(z1)
        hori = np.zeros_like(z1)

        for i in range(z1.shape[0]):
            mask = ~np.isnan(z1[i])
            y = y1[i][mask]
            z = z1[i][mask]
            hori[i] = np.interp(y1[i], y, z)

        for i in range(z1.shape[1]):
            mask = ~np.isnan(z1[:, i])
            x = x1[:, i][mask]
            z = z1[:, i][mask]

            vert[:, i] = np.interp(x1[:, i], x, z)

        hori[hori == 0] = np.nan
        vert[vert == 0] = np.nan

        hv = hori.copy()
        hv[np.isnan(hori)] = vert[np.isnan(hori)]
        hv[~np.isnan(hv)] = np.nanmean([hori[~np.isnan(hv)],
                                        vert[~np.isnan(hv)]], 0)

        z1[np.isnan(z1)] = hv[np.isnan(z1)]

    plt.imshow(z1)
    plt.show()

    ttt.since_last_call('Griddata, nearest')


if __name__ == "__main__":
    testdown()
