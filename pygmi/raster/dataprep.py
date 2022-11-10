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

import tempfile
import math
import os
import glob
import copy
from collections import Counter
from PyQt5 import QtWidgets, QtCore
import numpy as np
import pandas as pd
from scipy.signal import tukey
import rasterio
import rasterio.merge
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject
from rasterio.mask import mask as riomask
import geopandas as gpd
from shapely.geometry import LineString, Polygon, box

from pygmi import menu_default
from pygmi.raster.datatypes import Data
from pygmi.misc import ProgressBarText
from pygmi.raster.datatypes import numpy_to_pygmi


class DataCut():
    """
    Cut Data using shapefiles.

    This class cuts raster datasets using a boundary defined by a polygon
    shapefile.

    Attributes
    ----------
    ifile : str
        input file name.
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    outdata : dictionary
        dictionary of output datasets
    """

    def __init__(self, parent=None):
        self.ifile = ''
        self.parent = parent
        self.indata = {}
        self.outdata = {}
        if parent is None:
            self.showprocesslog = print
            self.pbar = None
        else:
            self.showprocesslog = parent.showprocesslog
            self.pbar = parent.pbar

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
        if 'Raster' in self.indata:
            data = self.indata['Raster']
        else:
            self.showprocesslog('No raster data')
            return False

        if not nodialog:
            self.ifile, _ = QtWidgets.QFileDialog.getOpenFileName(
                self.parent, 'Open Shape File', '.', 'Shape file (*.shp)')
            if self.ifile == '':
                return False

        os.chdir(os.path.dirname(self.ifile))
        data = cut_raster(data, self.ifile, pprint=self.showprocesslog)

        if data is None:
            return False

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


class DataLayerStack(QtWidgets.QDialog):
    """
    Data Layer Stack.

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
        self.cmask = QtWidgets.QCheckBox('Common mask for all bands')

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
        helpdocs = menu_default.HelpButton('pygmi.raster.dataprep.'
                                           'datalayerstack')
        label_dxy = QtWidgets.QLabel('Cell Size:')

        self.dsb_dxy.setMaximum(9999999999.0)
        self.dsb_dxy.setMinimum(0.00001)
        self.dsb_dxy.setDecimals(5)
        self.dsb_dxy.setValue(40.)
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.cmask.setChecked(True)

        self.setWindowTitle('Dataset Layer Stack and Resample')

        gridlayout_main.addWidget(label_dxy, 0, 0, 1, 1)
        gridlayout_main.addWidget(self.dsb_dxy, 0, 1, 1, 1)
        gridlayout_main.addWidget(self.label_rows, 1, 0, 1, 2)
        gridlayout_main.addWidget(self.label_cols, 2, 0, 1, 2)
        gridlayout_main.addWidget(self.cmask, 3, 0, 1, 2)
        gridlayout_main.addWidget(helpdocs, 4, 0, 1, 1)
        gridlayout_main.addWidget(buttonbox, 4, 1, 1, 1)

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

        Parameters
        ----------
        nodialog : bool, optional
            Run settings without a dialog. The default is False.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        if 'Raster' not in self.indata:
            self.showprocesslog('No Raster Data.')
            return False

        if not nodialog:
            data = self.indata['Raster'][0]

            if self.dxy is None:
                self.dxy = min(data.xdim, data.ydim)
                for data in self.indata['Raster']:
                    self.dxy = min(self.dxy, data.xdim, data.ydim)

            self.dsb_dxy.setValue(self.dxy)
            self.dxy_change()

            tmp = self.exec_()
            if tmp != 1:
                return False

        self.acceptall()

        if self.outdata['Raster'] is None:
            self.outdata = {}
            return False

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
        self.cmask.setChecked(projdata['cmask'])

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
        projdata['cmask'] = self.cmask.isChecked()

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
        dat = lstack(self.indata['Raster'], self.piter, dxy,
                     pprint=self.showprocesslog,
                     commonmask=self.cmask.isChecked())
        self.outdata['Raster'] = dat


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
            self.piter = ProgressBarText().iter
        else:
            self.showprocesslog = parent.showprocesslog
            self.piter = parent.pbar.iter

        self.indata = {}
        self.outdata = {}
        self.parent = parent
        self.idir = None
        self.method = merge_median
        self.rb_first = QtWidgets.QRadioButton('First - copy first file over '
                                               'last file at overlap.')
        self.rb_last = QtWidgets.QRadioButton('Last - copy last file over '
                                              'first file at overlap.')
        self.rb_min = QtWidgets.QRadioButton('Min - copy pixel wise minimum '
                                             'at overlap')
        self.rb_max = QtWidgets.QRadioButton('Max - copy pixel wise maximum '
                                             'at overlap')
        self.rb_median = QtWidgets.QRadioButton('Median - shift last file to '
                                                'median '
                                                'overlap value and copy over '
                                                'first file at overlap.')

        self.idirlist = QtWidgets.QLineEdit('')
        self.sfile = QtWidgets.QLineEdit('')
        self.files_diff = QtWidgets.QCheckBox('Merge by band labels, '
                                              'since band order may differ, '
                                              'or input files have different '
                                              'numbers of bands or '
                                              'nodata values.')
        self.shift_to_median = QtWidgets.QCheckBox('Shift bands to median '
                                                   'value before merge. May '
                                                   'allow for cleaner merge '
                                                   'if datasets are offset.')

        self.bands_to_files = QtWidgets.QCheckBox('Save each band separately '
                                                  'in a "merge" subdirectory.')
        self.forcetype = None
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
        pb_idirlist = QtWidgets.QPushButton('Batch Directory')
        pb_sfile = QtWidgets.QPushButton('Shapefile for boundary (optional)')

        self.files_diff.setChecked(False)
        self.shift_to_median.setChecked(False)
        self.rb_median.setChecked(True)

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Dataset Merge')

        gb_merge_method = QtWidgets.QGroupBox('Merge method')
        gl_merge_method = QtWidgets.QVBoxLayout(gb_merge_method)

        gl_merge_method.addWidget(self.rb_median)
        gl_merge_method.addWidget(self.rb_first)
        gl_merge_method.addWidget(self.rb_last)
        gl_merge_method.addWidget(self.rb_min)
        gl_merge_method.addWidget(self.rb_max)

        gridlayout_main.addWidget(pb_idirlist, 1, 0, 1, 1)
        gridlayout_main.addWidget(self.idirlist, 1, 1, 1, 1)
        gridlayout_main.addWidget(pb_sfile, 2, 0, 1, 1)
        gridlayout_main.addWidget(self.sfile, 2, 1, 1, 1)
        gridlayout_main.addWidget(self.files_diff, 3, 0, 1, 2)
        gridlayout_main.addWidget(self.shift_to_median, 4, 0, 1, 2)
        gridlayout_main.addWidget(gb_merge_method, 5, 0, 1, 2)
        gridlayout_main.addWidget(self.bands_to_files, 6, 0, 1, 2)
        gridlayout_main.addWidget(helpdocs, 7, 0, 1, 1)
        gridlayout_main.addWidget(buttonbox, 7, 1, 1, 1)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        pb_idirlist.pressed.connect(self.get_idir)
        pb_sfile.pressed.connect(self.get_sfile)
        self.shift_to_median.stateChanged.connect(self.shiftchanged)
        self.files_diff.stateChanged.connect(self.filesdiffchanged)
        self.rb_first.clicked.connect(self.method_change)
        self.rb_last.clicked.connect(self.method_change)
        self.rb_min.clicked.connect(self.method_change)
        self.rb_max.clicked.connect(self.method_change)
        self.rb_median.clicked.connect(self.method_change)

    def method_change(self):
        """
        Change method.

        Returns
        -------
        None.

        """
        if self.rb_first.isChecked():
            self.method = 'first'
        if self.rb_last.isChecked():
            self.method = 'last'
        if self.rb_min.isChecked():
            self.method = merge_min
        if self.rb_max.isChecked():
            self.method = merge_max
        if self.rb_median.isChecked():
            self.method = merge_median

    def shiftchanged(self):
        """
        Shift mean clicked.

        Returns
        -------
        None.

        """
        if self.shift_to_median.isChecked():
            self.files_diff.setChecked(True)

    def filesdiffchanged(self):
        """
        Files different clicked.

        Returns
        -------
        None.

        """
        if not self.files_diff.isChecked():
            self.shift_to_median.setChecked(False)
            self.bands_to_files.hide()
        else:
            self.bands_to_files.show()

    def get_idir(self):
        """
        Get the input directory.

        Returns
        -------
        None.

        """
        self.idir = QtWidgets.QFileDialog.getExistingDirectory(
             self.parent, 'Select Directory')

        self.idirlist.setText(self.idir)

        if self.idir == '':
            self.idir = None

    def get_sfile(self):
        """
        Get the input shapefile.

        Returns
        -------
        None.

        """
        ext = ('ESRI Shapefile (*.shp);;')

        sfile, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.parent, 'Open File', '.', ext)

        if not sfile:
            return False

        self.sfile.setText(sfile)

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
        if not nodialog:
            tmp = self.exec_()
            if tmp != 1:
                return False

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
        self.idir = projdata['idir']
        self.idirlist.setText(self.idir)
        self.files_diff.setChecked(projdata['files_diff'])
        self.shift_to_median.setChecked(projdata['mean_shift'])

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

        projdata['idir'] = self.idir
        projdata['files_diff'] = self.files_diff.isChecked()
        projdata['mean_shift'] = self.shift_to_median.isChecked()

        return projdata

    def acceptall(self):
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        bool
            Success of routine.

        """
        if self.files_diff.isChecked():
            tmp = self.merge_different()
        else:
            tmp = self.merge_same()

        return tmp

    def merge_different(self):
        """
        Merge files with different numbers of bands and/or band order.

        This uses more memory, but is flexible.

        Returns
        -------
        bool
            Success of routine.

        """
        # The next line is only to avoid circular dependancies with merge
        # function.

        from pygmi.raster.iodefs import get_raster, export_raster

        indata = []
        if 'Raster' in self.indata:
            for i in self.indata['Raster']:
                indata.append(i)

        if self.idir is not None:
            ifiles = []
            for ftype in ['*.tif', '*.hdr', '*.img', '*.ers']:
                ifiles += glob.glob(os.path.join(self.idir, ftype))

            if not ifiles:
                self.showprocesslog('No input files in that directory')
                return False

            for ifile in self.piter(ifiles):
                indata += get_raster(ifile, piter=iter, metaonly=True)

        if indata is None:
            self.showprocesslog('No input datasets')
            return False

        # Get projection information
        wkt = []
        crs = []
        for i in indata:
            if i.crs is None:
                self.showprocesslog(f'{i.dataid} has no projection. '
                                    'Please assign one.')
                return False

            wkt.append(i.crs.to_wkt())
            crs.append(i.crs)
            nodata = i.nodata

        wkt, iwkt, numwkt = np.unique(wkt, return_index=True,
                                      return_counts=True)
        if len(wkt) > 1:
            self.showprocesslog('Error: Mismatched input projections. '
                                'Selecting most common projection')

            crs = crs[iwkt[numwkt == numwkt.max()][0]]
        else:
            crs = indata[0].crs

        bounds = get_shape_bounds(self.sfile.text(), crs, self.showprocesslog)

        # Start Merge
        bandlist = []
        for i in indata:
            bandlist.append(i.dataid)

        bandlist = list(set(bandlist))

        outdat = []
        for dataid in bandlist:
            # if 'B4divB2' not in dataid:
            #     continue
            self.showprocesslog('Extracting '+dataid+'...')

            if self.bands_to_files.isChecked():
                odir = os.path.join(self.idir, 'merge')
                os.makedirs(odir, exist_ok=True)
                ofile = dataid+'.tif'
                ofile = ofile.replace(' ', '_')
                ofile = ofile.replace(',', '_')
                ofile = ofile.replace('*', 'mult')
                ofile = os.path.join(odir, ofile)

                if os.path.exists(ofile):
                    self.showprocesslog('Output file exists, skipping.')
                    continue

            ifiles = []

            for i in self.piter(indata):
                if i.dataid != dataid:
                    continue

                i2 = get_raster(i.filename, piter=iter, dataid=i.dataid)

                if i2 is None:
                    continue
                else:
                    i2 = i2[0]

                if i2.crs != crs:
                    src_height, src_width = i2.data.shape

                    transform, width, height = calculate_default_transform(
                        i2.crs, crs, src_width, src_height, *i2.bounds)

                    i2 = data_reproject(i2, i2.crs, crs, transform, height,
                                        width)

                if self.forcetype is not None:
                    i2.data = i2.data.astype(self.forcetype)

                if self.shift_to_median.isChecked():
                    mval = np.ma.median(i.data)
                else:
                    mval = 0

                trans = rasterio.transform.from_origin(i2.extent[0],
                                                       i2.extent[3],
                                                       i2.xdim, i2.ydim)

                tmpfile = os.path.join(tempfile.gettempdir(),
                                       os.path.basename(i.filename))

                tmpid = i2.dataid
                tmpid = tmpid.replace(' ', '_')
                tmpid = tmpid.replace(',', '_')
                tmpid = tmpid.replace('*', 'mult')
                tmpid = tmpid.replace(r'/', 'div')

                tmpfile = tmpfile[:-4]+'_'+tmpid+'.tif'

                raster = rasterio.open(tmpfile, 'w', driver='GTiff',
                                       height=i2.data.shape[0],
                                       width=i2.data.shape[1], count=1,
                                       dtype=i2.data.dtype,
                                       transform=trans)

                if np.issubdtype(i2.data.dtype, np.floating):
                    nodata = 1.0e+20
                else:
                    nodata = -99999

                tmpdat = i2.data-mval
                tmpdat = tmpdat.filled(nodata)
                tmpdat = np.ma.masked_equal(tmpdat, nodata)

                raster.write(tmpdat, 1)
                raster.write_mask(~np.ma.getmaskarray(i2.data))

                raster.close()
                ifiles.append(tmpfile)
                del i2

            if len(ifiles) < 2:
                self.showprocesslog('Too few bands of name '+dataid)
                continue

            self.showprocesslog('Merging '+dataid+'...')

            with rasterio.Env(CPL_DEBUG=True):
                mosaic, otrans = rasterio.merge.merge(ifiles, nodata=nodata,
                                                      method=self.method,
                                                      bounds=bounds)

            for j in ifiles:
                if os.path.exists(j):
                    os.remove(j)
                if os.path.exists(j+'.msk'):
                    os.remove(j+'.msk')

            mosaic = mosaic.squeeze()
            mosaic = np.ma.masked_equal(mosaic, nodata)
            mosaic = mosaic + mval
            outdat.append(numpy_to_pygmi(mosaic, dataid=dataid))
            outdat[-1].set_transform(transform=otrans)
            outdat[-1].crs = crs
            outdat[-1].nodata = nodata

            if self.bands_to_files.isChecked():
                # odir = os.path.join(self.idir, 'merge')
                # os.makedirs(odir, exist_ok=True)
                # ofile = dataid+'.tif'
                # ofile = ofile.replace(' ', '_')
                # ofile = ofile.replace(',', '_')
                # ofile = ofile.replace('*', 'mult')
                # ofile = os.path.join(odir, ofile)
                export_raster(ofile, outdat, 'GTiff', compression='ZSTD')

                # import matplotlib.pyplot as plt
                # vmin = mosaic.mean()-2*mosaic.std()
                # vmax = mosaic.mean()+2*mosaic.std()
                # plt.figure(dpi=150)
                # plt.title(dataid)
                # plt.imshow(mosaic, vmin=vmin, vmax=vmax, extent=outdat[-1].extent)
                # plt.colorbar()
                # plt.show()

                del outdat
                del mosaic
                outdat = []

        self.outdata['Raster'] = outdat

        return True

    def merge_different_old(self):
        """
        Merge files with different numbers of bands and/or band order.

        This uses more memory, but is flexible.

        Returns
        -------
        bool
            Success of routine.

        """
        # The next line is only to avoid circular dependancies with merge
        # function.

        from pygmi.raster.iodefs import get_raster

        indata = []
        if 'Raster' in self.indata:
            for i in self.indata['Raster']:
                indata.append(i)

        if self.idir is not None:
            ifiles = []
            for ftype in ['*.tif', '*.hdr', '*.img', '*.ers']:
                ifiles += glob.glob(os.path.join(self.idir, ftype))

            for ifile in self.piter(ifiles):
                indata += get_raster(ifile, piter=iter)

        if indata is None:
            self.showprocesslog('No input datasets')
            return False

        # Get projection information
        wkt = []
        for i in indata:
            if i.crs is None:
                self.showprocesslog(f'{i.dataid} has no projection. '
                                    'Please assign one.')
                return False

            wkt.append(i.crs.to_wkt())
            nodata = i.nodata

        wkt = list(set(wkt))

        if len(wkt) > 1:
            self.showprocesslog('Error: Mismatched input projections')
            return False

        crs = indata[0].crs

        # Start Merge
        bandlist = []
        hasfloatdtype = False
        for i in indata:
            bandlist.append(i.dataid)
            if np.issubdtype(i.data.dtype, np.floating):
                hasfloatdtype = True
        bandlist = list(set(bandlist))

        outdat = []
        for dataid in bandlist:
            self.showprocesslog('Extracting '+dataid+'...')
            ifiles = []
            for i in self.piter(indata):
                if i.dataid != dataid:
                    continue

                if self.forcetype is not None:
                    i.data = i.data.astype(self.forcetype)

                if self.shift_to_median.isChecked():
                    mval = np.ma.median(i.data)
                else:
                    mval = 0

                trans = rasterio.transform.from_origin(i.extent[0],
                                                       i.extent[3],
                                                       i.xdim, i.ydim)

                tmpfile = os.path.join(tempfile.gettempdir(),
                                       os.path.basename(i.filename))
                tmpfile = tmpfile[:-4]+'_'+i.dataid+'.tif'
                tmpfile = tmpfile.replace('*', 'mult')
                tmpfile = tmpfile.replace(r'/', 'div')

                raster = rasterio.open(tmpfile, 'w', driver='GTiff',
                                       height=i.data.shape[0],
                                       width=i.data.shape[1], count=1,
                                       dtype=i.data.dtype,
                                       transform=trans)

                if hasfloatdtype:
                    nodata = 1.0e+20
                    tmpdat = i.data.astype(float)

                else:
                    nodata = -99999
                    tmpdat = i.data.astype(int)

                tmpdat = i.data-mval
                tmpdat = tmpdat.filled(nodata)
                tmpdat = np.ma.masked_equal(tmpdat, nodata)

                raster.write(tmpdat, 1)
                raster.write_mask(~i.data.mask)
                raster.close()
                ifiles.append(tmpfile)

            if len(ifiles) < 2:
                self.showprocesslog('Too few bands of name '+dataid)

            self.showprocesslog('Merging '+dataid+'...')
            mosaic, otrans = rasterio.merge.merge(ifiles, nodata=nodata,
                                                  method=self.method)
            for j in ifiles:
                os.remove(j)

            mosaic = mosaic.squeeze()
            mosaic = np.ma.masked_equal(mosaic, nodata)
            mosaic = mosaic + mval
            outdat.append(numpy_to_pygmi(mosaic, dataid=dataid))
            outdat[-1].set_transform(transform=otrans)
            outdat[-1].crs = crs
            outdat[-1].nodata = nodata

        self.outdata['Raster'] = outdat

        return True

    def merge_same(self):
        """
        Merge files with same numbers of bands and band order.

        This uses much less memory, but is less flexible.

        Returns
        -------
        bool
            Success of routine.

        """
        ifiles = []
        if 'Raster' in self.indata:
            for i in self.indata['Raster']:
                ifiles.append(i.filename)

        if self.idir is not None:
            for ftype in ['*.tif', '*.hdr', '*.img', '*.ers']:
                ifiles += glob.glob(os.path.join(self.idir, ftype))

        if not ifiles:
            self.showprocesslog('No input datasets')
            return False

        for i, ifile in enumerate(ifiles):
            if ifile[-3:] == 'hdr':
                ifile = ifile[:-4]
                if os.path.exists(ifile+'.dat'):
                    ifiles[i] = ifile+'.dat'
                elif os.path.exists(ifile+'.raw'):
                    ifiles[i] = ifile+'.raw'
                elif os.path.exists(ifile+'.img'):
                    ifiles[i] = ifile+'.img'
                elif not os.path.exists(ifile):
                    return False

        # Get projection information
        wkt = []
        nodata = []
        for ifile in ifiles:
            with rasterio.open(ifile) as dataset:
                if dataset.crs is None:
                    self.showprocesslog(f'{ifile} has no projection. '
                                        'Please assign one.')
                    return False
                wkt.append(dataset.crs.wkt)
                crs = dataset.crs
                nodata.append(dataset.nodata)

        wkt = list(set(wkt))
        if len(wkt) > 1:
            self.showprocesslog('Error: Mismatched input projections')
            return False

        nodata = list(set(nodata))
        if len(nodata) > 1:
            self.showprocesslog('Error: Mismatched nodata values. '
                                'Try using merge by band labels merge option. '
                                'Please confirm bands to be merged have the '
                                'same label.')
            return False

        # Get band names and nodata
        with rasterio.open(ifiles[0]) as dataset:
            bnames = dataset.descriptions
            if None in bnames:
                bnames = ['Band '+str(i) for i in dataset.indexes]
            nodata = dataset.nodata

        # Start Merge
        mosaic, otrans = rasterio.merge.merge(ifiles, nodata=nodata,
                                              method=self.method)
        mosaic = np.ma.masked_equal(mosaic, nodata)

        outdat = []
        for i, dataid in enumerate(bnames):
            outdat.append(numpy_to_pygmi(mosaic[i], dataid=dataid))
            outdat[-1].set_transform(transform=otrans)
            outdat[-1].crs = crs
            outdat[-1].nodata = nodata

        self.outdata['Raster'] = outdat

        return True


class DataReproj(QtWidgets.QDialog):
    """
    Reprojections.

    This class reprojects datasets using the rasterio routines.

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
            self.piter = ProgressBarText().iter
        else:
            self.showprocesslog = parent.showprocesslog
            self.piter = parent.pbar.iter

        self.indata = {}
        self.outdata = {}
        self.parent = parent
        self.orig_wkt = None
        self.targ_wkt = None

        self.groupboxb = QtWidgets.QGroupBox()
        self.combobox_inp_epsg = QtWidgets.QComboBox()
        self.inp_epsg_info = QtWidgets.QLabel(wordWrap=True)
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
            self.showprocesslog('Unknown Projection. Could not reproject')
            return

        if self.in_proj.wkt == '' or self.out_proj.wkt == '':
            self.showprocesslog('Unknown Projection. Could not reproject')
            return

        # Input stuff
        src_crs = CRS.from_wkt(self.in_proj.wkt)

        # Output stuff
        dst_crs = CRS.from_wkt(self.out_proj.wkt)

        # Now create virtual dataset
        dat = []
        for data in self.piter(self.indata['Raster']):
            src_height, src_width = data.data.shape

            transform, width, height = calculate_default_transform(
                src_crs, dst_crs, src_width, src_height, *data.bounds)

            # Work out the boundaries of the new dataset in target projection
            data2 = data_reproject(data, src_crs, dst_crs, transform,
                                   height, width)

            dat.append(data2)

        self.outdata['Raster'] = dat

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
        if 'Raster' not in self.indata:
            self.showprocesslog('No Raster Data.')
            return False

        if self.indata['Raster'][0].crs is None:
            self.showprocesslog('Your input data has no projection. '
                                'Please assign one in the metadata summary.')
            return False

        if self.orig_wkt is None:
            self.orig_wkt = self.indata['Raster'][0].crs.wkt
        if self.targ_wkt is None:
            self.targ_wkt = self.indata['Raster'][0].crs.wkt

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
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    outdata : dictionary
        dictionary of output datasets
    """

    def __init__(self, parent=None):
        self.ifile = ''
        self.parent = parent
        self.indata = {}
        self.outdata = {}
        if parent is None:
            self.showprocesslog = print
            self.piter = ProgressBarText().iter
        else:
            self.showprocesslog = parent.showprocesslog
            self.piter = parent.pbar.iter

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
        if 'Raster' in self.indata:
            data = copy.deepcopy(self.indata['Raster'])
        else:
            self.showprocesslog('No raster data')
            return False

        ext = 'Shape file (*.shp)'

        if not nodialog:
            self.ifile, _ = QtWidgets.QFileDialog.getOpenFileName(
                self.parent, 'Open Shape File', '.', ext)
            if self.ifile == '':
                return False

        os.chdir(os.path.dirname(self.ifile))

        try:
            gdf = gpd.read_file(self.ifile)
        except:
            self.showprocesslog('There was a problem importing the shapefile. '
                                'Please make sure you have at all the '
                                'individual files which make up the '
                                'shapefile.')
            return None

        gdf = gdf[gdf.geometry != None]

        if gdf.geom_type.iloc[0] != 'LineString':
            self.showprocesslog('You need lines in that shape file')
            return False

        data = lstack(data, self.piter, pprint=self.showprocesslog)
        dxy = min(data[0].xdim, data[0].ydim)
        ogdf2 = None

        icnt = 0
        for line in gdf.geometry:
            line2 = redistribute_vertices(line, dxy)
            x, y = line2.coords.xy
            xy = np.transpose([x, y])
            ogdf = None

            for idata in self.piter(data):
                mdata = idata.to_mem()
                z = []
                for pnt in xy:
                    z.append(idata.data[mdata.index(pnt[0], pnt[1])])

                if ogdf is None:
                    ogdf = pd.DataFrame(xy[:, 0], columns=['X'])
                    ogdf['Y'] = xy[:, 1]
                    ogdf['pygmiX'] = ogdf['X']
                    ogdf['pygmiY'] = ogdf['Y']

                ogdf[idata.dataid] = z

            icnt += 1
            ogdf['line'] = str(icnt)
            if ogdf2 is None:
                ogdf2 = ogdf
            else:
                ogdf2 = ogdf2.append(ogdf, ignore_index=True)

        self.outdata['Line'] = {'profile': ogdf2}

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
        self.combodatum = QtWidgets.QComboBox()
        self.comboproj = QtWidgets.QComboBox()
        self.label = QtWidgets.QLabel()
        self.label.setWordWrap(True)

        self.gridlayout.addWidget(self.groupbox, 1, 0, 1, 2)

        gridlayout = QtWidgets.QGridLayout(self.groupbox)
        gridlayout.addWidget(self.combodatum, 0, 0, 1, 1)
        gridlayout.addWidget(self.comboproj, 1, 0, 1, 1)
        gridlayout.addWidget(self.label, 2, 0, 1, 1)

        self.epsg_proj = getepsgcodes()
        self.epsg_proj[r'Current / Current'] = self.wkt
        tmp = list(self.epsg_proj.keys())
        tmp.sort(key=lambda c: c.lower())

        self.plist = {}
        for i in tmp:
            if r' / ' in i:
                datum, proj = i.split(r' / ')
            else:
                datum = i
                proj = i

            if datum not in self.plist:
                self.plist[datum] = []
            self.plist[datum].append(proj)

        tmp = list(set(self.plist.keys()))
        tmp.sort()
        tmp = ['Current', 'WGS 84']+tmp

        for i in tmp:
            j = self.plist[i]
            if r'Geodetic Geographic' in j and j[0] != r'Geodetic Geographic':
                self.plist[i] = [r'Geodetic Geographic']+self.plist[i]

        self.combodatum.addItems(tmp)
        self.comboproj.addItem('Current')
        self.combodatum.currentIndexChanged.connect(self.combo_datum_change)
        self.comboproj.currentIndexChanged.connect(self.combo_change)

    def set_current(self, wkt):
        """
        Set new WKT for current option.

        Parameters
        ----------
        wkt : str
            Well Known Text descriptions for coordinates (WKT) .

        Returns
        -------
        None.

        """
        self.wkt = wkt
        self.epsg_proj[r'Current / Current'] = self.wkt
        self.combo_change()

    def combo_datum_change(self):
        """
        Change Combo.

        Returns
        -------
        None.

        """
        indx = self.combodatum.currentIndex()
        txt = self.combodatum.itemText(indx)
        self.comboproj.currentIndexChanged.disconnect()

        self.comboproj.clear()
        self.comboproj.addItems(self.plist[txt])

        self.comboproj.currentIndexChanged.connect(self.combo_change)

        self.combo_change()

    def combo_change(self):
        """
        Change Combo.

        Returns
        -------
        None.

        """
        dtxt = self.combodatum.currentText()
        ptxt = self.comboproj.currentText()

        txt = dtxt + r' / '+ptxt

        self.wkt = self.epsg_proj[txt]

        if not isinstance(self.wkt, str):
            self.wkt = epsgtowkt(self.wkt)

        # The next two lines make sure we have spaces after ALL commas.
        wkttmp = self.wkt.replace(', ', ',')
        wkttmp = wkttmp.replace(',', ', ')

        self.label.setText(wkttmp)


class Metadata(QtWidgets.QDialog):
    """
    Edit Metadata.

    This class allows the editing of the metadata for a raster dataset using a
    GUI.

    Attributes
    ----------
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
                    tmp.set_transform(transform=i.transform)
                    tmp.nodata = i.nodata
                    if wkt == 'None':
                        tmp.crs = None
                    else:
                        tmp.crs = CRS.from_wkt(wkt)
                    tmp.units = i.units
                    tmp.data.mask = (tmp.data.data == i.nodata)

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
            odata.nodata = float(self.txt_null.text())
            left = float(self.dsb_tlx.text())
            top = float(self.dsb_tly.text())
            xdim = float(self.dsb_xdim.text())
            ydim = float(self.dsb_ydim.text())

            odata.set_transform(xdim, left, ydim, top)

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
        self.txt_null.setText(str(idata.nodata))
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
        Entry point to start this routine.

        Returns
        -------
        tmp : bool
            True if successful, False otherwise.

        """
        bandid = []
        if self.indata['Raster'][0].crs is None:
            self.proj.set_current('None')
        else:
            self.proj.set_current(self.indata['Raster'][0].crs.wkt)

        for i in self.indata['Raster']:
            bandid.append(i.dataid)
            self.banddata[i.dataid] = Data()
            tmp = self.banddata[i.dataid]
            self.dataid[i.dataid] = i.dataid
            tmp.data = i.data
            tmp.set_transform(transform=i.transform)
            tmp.nodata = i.nodata
            tmp.crs = i.crs
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
        self.txt_null.setText(str(idata.nodata))
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


class Continuation(QtWidgets.QDialog):
    """
    Perform upward and downward continuation on potential field data.

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

        Parameters
        ----------
        nodialog : bool, optional
            Run settings without a dialog. The default is False.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        tmp = []
        if 'Raster' not in self.indata:
            self.showprocesslog('No Raster Data.')
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


def redistribute_vertices(geom, distance):
    """
    Redistribute vertices in a geometry.

    From https://stackoverflow.com/questions/34906124/interpolating-every-x-distance-along-multiline-in-shapely,
    and by Mike-T.

    Parameters
    ----------
    geom : shapely geometry
        Geometry from geopandas.
    distance : float
        sampling distance.

    Raises
    ------
    ValueError
        Error when there is an unknown geometry.

    Returns
    -------
    shapely geometry
        New geometry.

    """
    if geom.geom_type == 'LineString':
        num_vert = int(round(geom.length / distance))
        if num_vert == 0:
            num_vert = 1
        return LineString(
            [geom.interpolate(float(n) / num_vert, normalized=True)
             for n in range(num_vert + 1)])
    if geom.geom_type == 'MultiLineString':
        parts = [redistribute_vertices(part, distance)
                 for part in geom]
        return type(geom)([p for p in parts if not p.is_empty])
    raise ValueError('unhandled geometry %s', (geom.geom_type,))


def data_reproject(data, icrs, ocrs, otransform, orows, ocolumns):
    """
    Reproject dataset.

    Parameters
    ----------
    data : PyGMI Data
        PyGMI dataset.
    icrs : CRS
        input crs.
    ocrs : CRS
        output crs.
    otransform : Affine
        Output affine transform.
    orows : int
        output rows.
    ocolumns : int
        output columns.

    Returns
    -------
    data2 : TYPE
        DESCRIPTION.

    """
    odata = np.zeros((orows, ocolumns), dtype=data.data.dtype)
    odata, _ = reproject(source=data.data,
                         destination=odata,
                         src_transform=data.transform,
                         src_crs=icrs,
                         dst_transform=otransform,
                         dst_crs=ocrs,
                         src_nodata=data.nodata)

    data2 = Data()
    data2.data = odata
    data2.crs = ocrs
    data2.set_transform(transform=otransform)
    data2.data = data2.data.astype(data.data.dtype)
    data2.dataid = data.dataid
    data2.wkt = CRS.to_wkt(ocrs)

    data2.data = np.ma.masked_equal(data2.data, data.nodata)
    data2.nodata = data.nodata

    return data2


def merge_median(merged_data, new_data, merged_mask, new_mask, index=None,
                 roff=None, coff=None):
    """
    Merge using median for rasterio, taking minimum value.

    Parameters
    ----------
    merged_data : numpy array
        Old data.
    new_data : numpy array
        New data to merge to old data.
    merged_mask : float
        Old mask.
    new_mask : float
        New mask.
    index : int, optional
        index of the current dataset within the merged dataset collection.
        The default is None.
    roff : int, optional
        row offset in base array. The default is None.
    coff : int, optional
        col offset in base array. The default is None.

    Returns
    -------
    None.

    """
    merged_data = np.ma.array(merged_data, mask=merged_mask)
    new_data = np.ma.array(new_data, mask=new_mask)

    mtmp1 = np.logical_and(~merged_mask, ~new_mask)
    mtmp2 = np.logical_and(~merged_mask, new_mask)

    tmp1 = new_data.copy()

    if True in mtmp1:
        tmp1 = tmp1 - np.ma.median(new_data[mtmp1])
        tmp1 = tmp1 + np.ma.median(merged_data[mtmp1])
        # tmp1 = tmp1 - new_data[mtmp1].mean()
        # tmp1 = tmp1 + merged_data[mtmp1].mean()

    tmp1[mtmp2] = merged_data[mtmp2]

    # import matplotlib.pyplot as plt

    # plt.figure(dpi=150)
    # plt.suptitle(f'{roff} {coff}')
    # plt.subplot(221)
    # plt.title('merged_data')
    # plt.imshow(merged_data[0], vmin=0.5, vmax=3.0, interpolation='nearest')
    # plt.colorbar()

    # plt.subplot(222)
    # plt.title('new_data')
    # plt.imshow(new_data[0], vmin=0.5, vmax=3.0, interpolation='nearest')
    # plt.colorbar()
    # plt.tight_layout()

    # plt.subplot(223)
    # plt.title('tmp1')
    # plt.imshow(tmp1[0], vmin=0.5, vmax=3.0, interpolation='nearest')
    # plt.colorbar()
    # plt.tight_layout()
    # plt.show()

    merged_data[:] = tmp1


def merge_min(merged_data, new_data, merged_mask, new_mask, index=None,
              roff=None, coff=None):
    """
    Merge using minimum for rasterio, taking minimum value.

    Parameters
    ----------
    merged_data : numpy array
        Old data.
    new_data : numpy array
        New data to merge to old data.
    merged_mask : float
        Old mask.
    new_mask : float
        New mask.
    index : int, optional
        index of the current dataset within the merged dataset collection.
        The default is None.
    roff : int, optional
        row offset in base array. The default is None.
    coff : int, optional
        col offset in base array. The default is None.

    Returns
    -------
    None.

    """
    tmp = np.logical_and(~merged_mask, ~new_mask)

    tmp1 = merged_data.copy()
    tmp1[~new_mask] = new_data[~new_mask]
    tmp1[tmp] = np.minimum(merged_data[tmp], new_data[tmp])

    merged_data[:] = tmp1


def merge_max(merged_data, new_data, merged_mask, new_mask, index=None,
              roff=None, coff=None):
    """
    Merge using maximum for rasterio, taking maximum value.

    Parameters
    ----------
    merged_data : numpy array
        Old data.
    new_data : numpy array
        New data to merge to old data.
    merged_mask : float
        Old mask.
    new_mask : float
        New mask.
    index : int, optional
        index of the current dataset within the merged dataset collection.
        The default is None.
    roff : int, optional
        row offset in base array. The default is None.
    coff : int, optional
        col offset in base array. The default is None.

    Returns
    -------
    None.

    """
    tmp = np.logical_and(~merged_mask, ~new_mask)

    tmp1 = merged_data.copy()
    tmp1[~new_mask] = new_data[~new_mask]
    tmp1[tmp] = np.maximum(merged_data[tmp], new_data[tmp])

    merged_data[:] = tmp1


def fftprep(data):
    """
    FFT preparation.

    Parameters
    ----------
    data : PyGMI Data type
        Input dataset.

    Returns
    -------
    zfin : numpy array.
        Output prepared data.
    rdiff : int
        rows divided by 2.
    cdiff : int
        columns divided by 2.
    datamedian : float
        Median of data.

    """
    datamedian = np.ma.median(data.data)
    ndat = data.data - datamedian

    nr, nc = data.data.shape
    cdiff = nc//2
    rdiff = nr//2

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
    fftmod : numpy array
        FFT data.
    xdim : float
        cell x dimension.
    ydim : float
        cell y dimension.

    Returns
    -------
    KX : numpy array
        x sample frequencies.
    KY : numpy array
        y sample frequencies.

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
    order : float, optional
        Order. The default is 1.

    Returns
    -------
    dout : numpy array
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

    ndat, rdiff, cdiff, datamedian = fftprep(data)

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
    dat.nodata = data.data.fill_value
    dat.dataid = 'Upward_'+str(h)+'_'+data.dataid
    dat.set_transform(transform=data.transform)
    dat.crs = data.crs

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
    dat.nodata = data.data.fill_value
    dat.dataid = 'Downward_'+str(h)+'_'+data.dataid
    dat.set_transform(transform=data.transform)
    dat.crs = data.crs
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


def cut_raster(data, ifile, pprint=print):
    """Cuts a raster dataset.

    Cut a raster dataset using a shapefile.

    Parameters
    ----------
    data : Data
        PyGMI Dataset
    ifile : str
        shapefile used to cut data
    pprint : function, optional
        Function for printing text. The default is print.

    Returns
    -------
    data : Data
        PyGMI Dataset
    """
    data = copy.deepcopy(data)

    try:
        gdf = gpd.read_file(ifile)
    except:
        pprint('There was a problem importing the shapefile. Please make '
               'sure you have at all the individual files which make up '
               'the shapefile.')
        return None

    gdf = gdf[gdf.geometry != None]

    if gdf.geom_type.iloc[0] == 'MultiPolygon':
        pprint('You have a MultiPolygon. Only the first overlapping Polygon '
               'of the MultiPolygon will be used.')
        poly = gdf['geometry'].iloc[0]
        tmp = poly.geoms[0]

        dext = list(data[0].bounds)
        dpoly = box(dext[0], dext[1], dext[2], dext[3])

        for i in list(poly.geoms):
            if i.overlaps(dpoly):
                tmp = i
                break

        gdf.geometry.iloc[0] = tmp

    if gdf.geom_type.iloc[0] != 'Polygon':
        pprint('You need a polygon in that shape file')
        return None

    for idata in data:
        # Convert the layer extent to image pixel coordinates
        poly = gdf['geometry'].iloc[0]
        dext = idata.bounds
        lext = poly.bounds

        if ((dext[0] > lext[2]) or (dext[2] < lext[0])
                or (dext[1] > lext[3]) or (dext[3] < lext[1])):

            pprint('The shapefile is not in the same area as the raster '
                   'dataset. Please check its coordinates and make sure its '
                   'projection is the same as the raster dataset')
            return None

        # This section converts PolygonZ to Polygon, and takes first polygon.
        coords = gdf['geometry'].loc[0].exterior.coords
        coords = [Polygon([[p[0], p[1]] for p in coords])]

        dat, trans = riomask(idata.to_mem(), coords, crop=True)

        idata.data = np.ma.masked_equal(dat.squeeze(), idata.nodata)

        idata.set_transform(transform=trans)

    return data


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
    out = CRS.from_epsg(int(epsg)).to_wkt()
    return out


def getepsgcodes():
    """
    Routine used to get a list of EPSG codes.

    Returns
    -------
    pcodes : dictionary
        Dictionary of codes per projection in WKT format.

    """
    with open(os.path.join(os.path.dirname(__file__), 'gcs.csv'),
              encoding='utf-8') as dfile:
        dlines = dfile.readlines()

    dlines = dlines[1:]
    dcodes = {}
    for i in dlines:
        tmp = i.split(',')
        if tmp[1][0] == '"':
            tmp[1] = tmp[1][1:-1]

        dcodes[tmp[1]] = int(tmp[0])

    with open(os.path.join(os.path.dirname(__file__), 'pcs.csv'),
              encoding='utf-8') as pfile:
        plines = pfile.readlines()

    pcodes = {}
    for i in dcodes:
        pcodes[i+r' / Geodetic Geographic'] = dcodes[i]

    plines = plines[1:]
    for i in plines:
        tmp = i.split(',')
        if tmp[1][0] == '"':
            tmp[1] = tmp[1][1:-1]

        pcodes[tmp[1]] = int(tmp[0])

    for datum in ['Cape', 'Hartebeesthoek94']:
        for clong in range(15, 35, 2):
            if 'Cape' in datum:
                wkt = ('PROJCS["Cape / TM'+str(clong)+'",'
                       'GEOGCS["Cape",'
                       'DATUM["Cape",'
                       'SPHEROID["Clarke 1880 (Arc)",'
                       '6378249.145,293.4663077,'
                       'AUTHORITY["EPSG","7013"]],'
                       'AUTHORITY["EPSG","6222"]],'
                       'PRIMEM["Greenwich",0,'
                       'AUTHORITY["EPSG","8901"]],'
                       'UNIT["degree",0.0174532925199433,'
                       'AUTHORITY["EPSG","9122"]],'
                       'AUTHORITY["EPSG","4222"]],'
                       'PROJECTION["Transverse_Mercator"],'
                       'PARAMETER["latitude_of_origin",0],'
                       'PARAMETER["central_meridian",'+str(clong)+'],'
                       'PARAMETER["scale_factor",1],'
                       'PARAMETER["false_easting",0],'
                       'PARAMETER["false_northing",0],'
                       'UNIT["metre",1,AUTHORITY["EPSG","9001"]],'
                       'AXIS["Easting",EAST],'
                       'AXIS["Northing",NORTH]]')

            elif 'Hartebeesthoek94' in datum:
                wkt = ('PROJCS["Hartebeesthoek94 / TM'+str(clong)+'",'
                       'GEOGCS["Hartebeesthoek94",'
                       'DATUM["Hartebeesthoek94",'
                       'SPHEROID["WGS 84",6378137,298.257223563,'
                       'AUTHORITY["EPSG","7030"]],'
                       'AUTHORITY["EPSG","6148"]],'
                       'PRIMEM["Greenwich",0,'
                       'AUTHORITY["EPSG","8901"]],'
                       'UNIT["degree",0.0174532925199433,'
                       'AUTHORITY["EPSG","9122"]],'
                       'AUTHORITY["EPSG","4148"]],'
                       'PROJECTION["Transverse_Mercator"],'
                       'PARAMETER["latitude_of_origin",0],'
                       'PARAMETER["central_meridian",'+str(clong)+'],'
                       'PARAMETER["scale_factor",1],'
                       'PARAMETER["false_easting",0],'
                       'PARAMETER["false_northing",0],'
                       'UNIT["metre",1,AUTHORITY["EPSG","9001"]],'
                       'AXIS["Easting",EAST],'
                       'AXIS["Northing",NORTH]]')

            pcodes[datum+r' / TM'+str(clong)] = wkt

    return pcodes


def lstack(dat, piter=None, dxy=None, pprint=print, commonmask=False,
           masterid=None, nodeepcopy=False):
    """
    Layer stack datasets found in a single PyGMI data object.

    The aim is to ensure that all datasets have the same number of rows and
    columns.

    Parameters
    ----------
    dat : PyGMI Data
        data object which stores datasets
    piter : iter, optional
        Progress bar iterator. The default is None.
    dxy : float, optional
        Cell size. The default is None.
    pprint : function, optional
        Print function. The default is print.
    commonmask : bool, optional
        Create a common mask for all bands. The default is False.
    masterid : int, optional
        ID of master dataset. The default is None.

    Returns
    -------
    out : PyGMI Data
        data object which stores datasets

    """
    if piter is None:
        piter = ProgressBarText().iter

    if dat[0].isrgb:
        return dat

    needsmerge = False
    rows, cols = dat[0].data.shape

    dtypes = []
    for i in dat:
        irows, icols = i.data.shape
        if irows != rows or icols != cols:
            needsmerge = True
        if dxy is not None and (i.xdim != dxy or i.ydim != dxy):
            needsmerge = True
        if commonmask is True:
            needsmerge = True
        if i.extent != dat[0].extent:
            needsmerge = True
        dtypes.append(i.data.dtype)

    dtypes = np.unique(dtypes)
    dtype = None
    nodata = None
    if len(dtypes) > 1:
        needsmerge = True
        for i in dtypes:
            if np.issubdtype(i, np.floating):
                dtype = np.float64
                nodata = 1e+20
            elif dtype is None:
                dtype = np.int32
                nodata = 999999

    if needsmerge is False:
        if not nodeepcopy:
            dat = copy.deepcopy(dat)
        dat = check_dataid(dat)
        return dat

    pprint('Merging data...')
    if masterid is not None:
        data = dat[masterid]
        xmin, xmax, ymin, ymax = data.extent
        if dxy is None:
            dxy = min(data.xdim, data.ydim)
    else:
        data = dat[0]

        if dxy is None:
            dxy = min(data.xdim, data.ydim)
            for data in dat:
                dxy = min(dxy, data.xdim, data.ydim)

        xmin0, xmax0, ymin0, ymax0 = data.extent
        for data in dat:
            xmin, xmax, ymin, ymax = data.extent
            xmin = min(xmin, xmin0)
            xmax = max(xmax, xmax0)
            ymin = min(ymin, ymin0)
            ymax = max(ymax, ymax0)

    cols = int((xmax - xmin)/dxy)
    rows = int((ymax - ymin)/dxy)
    trans = rasterio.Affine(dxy, 0, xmin, 0, -1*dxy, ymax)

    if cols == 0 or rows == 0:
        pprint('Your rows or cols are zero. '
               'Your input projection may be wrong')
        return None

    dat2 = []
    cmask = None
    for data in piter(dat):

        if dtype is not None:
            data.data = data.data.astype(dtype)
            data.nodata = nodata

        if data.crs is None:
            pprint(f'{data.dataid} has no defined projection. '
                   'Assigning local.')

            data.crs = CRS.from_string('LOCAL_CS["Arbitrary",UNIT["metre",1,'
                                       'AUTHORITY["EPSG","9001"]],'
                                       'AXIS["Easting",EAST],'
                                       'AXIS["Northing",NORTH]]')

        doffset = 0.0
        data.data.set_fill_value(data.nodata)
        data.data = np.ma.array(data.data.filled(), mask=data.data.mask)

        if data.data.min() <= 0:
            doffset = data.data.min()-1.
            data.data = data.data - doffset

        trans0 = data.transform

        height, width = data.data.shape

        odata = np.zeros((rows, cols), dtype=data.data.dtype)
        odata, _ = reproject(source=data.data,
                             destination=odata,
                             src_transform=trans0,
                             src_crs=data.crs,
                             src_nodata=data.nodata,
                             dst_transform=trans,
                             dst_crs=data.crs)

        data2 = Data()
        data2.data = np.ma.masked_equal(odata, data.nodata)
        data2.data.mask = np.ma.getmaskarray(data2.data)
        data2.nodata = data.nodata
        data2.crs = data.crs
        data2.set_transform(transform=trans)
        data2.data = data2.data.astype(data.data.dtype)
        data2.dataid = data.dataid
        data2.filename = data.filename

        dat2.append(data2)

        if cmask is None:
            cmask = dat2[-1].data.mask
        else:
            cmask = np.logical_or(cmask, dat2[-1].data.mask)

        dat2[-1].metadata = data.metadata
        dat2[-1].data = dat2[-1].data + doffset

        dat2[-1].nodata = data.nodata
        dat2[-1].data.set_fill_value(data.nodata)
        dat2[-1].data = np.ma.array(dat2[-1].data.filled(),
                                    mask=dat2[-1].data.mask)

        data.data = data.data + doffset
        # breakpoint()
    if commonmask is True:
        for idat in piter(dat2):
            idat.data.mask = cmask
            idat.data = np.ma.array(idat.data.filled(idat.nodata), mask=cmask)

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
        data.data.data[mask] = data.nodata

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
        data.data.mask = (data.data.data == data.nodata)
        xmin = data.extent[0] + colstart*data.xdim
        ymax = data.extent[-1] - rowstart*data.ydim

        data.set_transform(data.xdim, xmin, data.ydim, ymax)

    return olddata

def cut_raster_basic(data, ifile, pprint=print):
    """Cuts a raster dataset.

    Cut a raster dataset using a shapefile.

    Parameters
    ----------
    data : Data
        PyGMI Dataset
    ifile : str
        shapefile used to cut data
    pprint : function, optional
        Function for printing text. The default is print.

    Returns
    -------
    data : Data
        PyGMI Dataset
    """
    data = copy.deepcopy(data)

    try:
        gdf = gpd.read_file(ifile)
    except:
        pprint('There was a problem importing the shapefile. Please make '
               'sure you have at all the individual files which make up '
               'the shapefile.')
        return None

    gdf = gdf[gdf.geometry != None]

    if gdf.geom_type.iloc[0] == 'MultiPolygon':
        pprint('You have a MultiPolygon. Only the first overlapping Polygon '
               'of the MultiPolygon will be used.')
        poly = gdf['geometry'].iloc[0]
        tmp = poly.geoms[0]

        dext = list(data[0].bounds)
        dpoly = box(dext[0], dext[1], dext[2], dext[3])

        for i in list(poly.geoms):
            if i.overlaps(dpoly):
                tmp = i
                break

        gdf.geometry.iloc[0] = tmp

    if gdf.geom_type.iloc[0] != 'Polygon':
        pprint('You need a polygon in that shape file')
        return None

    for idata in data:
        # Convert the layer extent to image pixel coordinates
        poly = gdf['geometry'].iloc[0]
        dext = idata.bounds
        lext = poly.bounds

        if ((dext[0] > lext[2]) or (dext[2] < lext[0]) or
                (dext[1] > lext[3]) or (dext[3] < lext[1])):

            pprint('The shapefile is not in the same area as the raster '
                   'dataset. Please check its coordinates and make sure its '
                   'projection is the same as the raster dataset')
            return None

        # This section convers PolygonZ to Polygon, and takes first polygon.
        coords = gdf['geometry'].loc[0].exterior.coords
        coords = [Polygon([[p[0], p[1]] for p in coords])]

        dat, trans = riomask(idata.to_mem(), coords, crop=True)

        idata.data = np.ma.masked_equal(dat.squeeze(), idata.nodata)

        idata.set_transform(transform=trans)

    return data


def get_shape_bounds(sfile, crs=None, pprint=print):
    """
    Get bounds from a shape file.

    Parameters
    ----------
    sfile : str
        Filename for shapefile.
    crs : rasterio CRS
        target crs for shapefile
    pprint : TYPE, optional
        Print. The default is print.

    Returns
    -------
    bounds : list
        Rasterio bounds.

    """
    if sfile == '' or sfile is None:
        return None

    gdf = gpd.read_file(sfile)

    gdf = gdf[gdf.geometry != None]

    if crs is not None:
        gdf =  gdf.to_crs(crs)

    if gdf.geom_type.iloc[0] == 'MultiPolygon':
        pprint('You have a MultiPolygon. Only the first Polygon '
               'of the MultiPolygon will be used.')
        poly = gdf['geometry'].iloc[0]
        tmp = poly.geoms[0]

        gdf.geometry.iloc[0] = tmp

    if gdf.geom_type.iloc[0] != 'Polygon':
        pprint('You need a polygon in that shape file')
        return None

    bounds = gdf.geometry.iloc[0].bounds

    return bounds

def _testdown():
    """Continuation testing routine."""
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
    plt.plot(downz0.data[50], 'r.')
    # plt.plot(zdown.data[50], 'b')
    plt.plot(zdownn.data[50], 'k')
    plt.show()


def _testgrid():
    """
    Test routine.

    Returns
    -------
    None.

    """
    from pygmi.raster.iodefs import get_raster
    from pygmi.misc import PTime
    import matplotlib.pyplot as plt

    ttt = PTime()

    ifile = r'd:\Work\Workdata\upward\EB_MTEF_Mag_IGRFrem.ers'
    dat = get_raster(ifile)[0]

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


def _testfft():
    """Test FFT."""
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import scipy
    from IPython import get_ipython
    from pygmi.raster.iodefs import get_raster

    get_ipython().run_line_magic('matplotlib', 'inline')

    ifile = r'D:\Workdata\geothermal\bushveldrtp.hdr'
    data = get_raster(ifile)[0]

    # quick model
    plt.imshow(data.data, cmap=cm.get_cmap('jet'), vmin=-500, vmax=500)
    plt.colorbar()
    plt.show()

    # Start new stuff
    xdim = data.xdim
    ydim = data.ydim

    ndat, rdiff, cdiff, datamedian = fftprep(data)

    datamedian = np.ma.median(data.data)
    ndat = data.data - datamedian

    fftmod = np.fft.fft2(ndat)

    KX, KY = fft_getkxy(fftmod, xdim, ydim)

    vmin = fftmod.real.mean()-2*fftmod.real.std()
    vmax = fftmod.real.mean()+2*fftmod.real.std()
    plt.imshow(np.fft.fftshift(fftmod.real), vmin=vmin, vmax=vmax)
    plt.show()

    knrm = np.sqrt(KX**2+KY**2)

    plt.imshow(knrm)

    plt.show()

    knrm = knrm.flatten()
    fftamp = np.abs(fftmod)**2
    fftamp = fftamp.flatten()

    plt.plot(knrm, fftamp, '.')
    plt.yscale('log')
    plt.show()

    bins = max(fftmod.shape)//2

    abins, bedge, _ = scipy.stats.binned_statistic(knrm, fftamp,
                                                   statistic='mean',
                                                   bins=bins)

    bins = (bedge[:-1] + bedge[1:])/2
    plt.plot(bins, abins)
    plt.yscale('log')
    plt.show()


def _testmerge():
    """Test Merge."""
    import sys
    from pygmi.raster.iodefs import export_raster

    app = QtWidgets.QApplication(sys.argv)

    # idir = r"d:\WorkProjects\ST-2022-1355 Onshore Mapping\Niger\4_7_5"
    # idir = r"c:\WorkProjects\ST-2022-1355 Onshore Mapping\Niger\full"
    idir = r"e:\WorkProjects\ST-2022-1355 Onshore Mapping\Niger\ratios"
    # idir = r'E:\WorkProjects\ST-2022-1355 Onshore Mapping\Niger\PCA'
    sfile = r"e:\WorkProjects\ST-2022-1355 Onshore Mapping\Niger\shapefiles\Agadez_block.shp"

    DM = DataMerge()
    DM.idir = idir
    DM.idirlist.setText(idir)
    DM.sfile.setText(sfile)

    DM.files_diff.setChecked(True)
    # DM.shift_to_median.setChecked(True)
    DM.forcetype = np.float32
    # DM.method = 'max'  # first last min max
    DM.settings()

    dat = DM.outdata['Raster']

    del DM

    # ofile = idir+'.tif'

    # export_raster(ofile, dat, 'GTiff', compression='ZSTD')

def _testreproj():
    """Test Reprojection."""
    import sys
    from pygmi.raster.iodefs import get_raster
    import matplotlib.pyplot as plt

    ifile = r""

    piter = ProgressBarText().iter

    dat = get_raster(ifile, piter=piter)

    app = QtWidgets.QApplication(sys.argv)

    DM = DataReproj()
    DM.indata['Raster'] = dat
    DM.settings()

    plt.figure(dpi=150)
    plt.imshow(DM.indata['Raster'][0].data,
               extent=DM.indata['Raster'][0].extent)
    plt.colorbar()
    plt.show()

    plt.figure(dpi=150)
    plt.imshow(DM.outdata['Raster'][0].data,
               extent=DM.outdata['Raster'][0].extent)
    plt.colorbar()
    plt.show()


def _testcut():
    """Test Reprojection."""
    import sys
    from pygmi.raster.iodefs import get_raster
    import matplotlib.pyplot as plt

    sfile  = r"d:/Workdata/raster/polygon cut get profile/cut_polygon.shp"
    ifile = r"d:\Workdata\raster\polygon cut get profile\mag_IGRFcorrected.ers"

    sfile = r"D:\Workdata\Janine\rsa_outline_utm35s.shp"
    ifile = r"D:\Workdata\Janine\oneband.tif"

    dat = get_raster(ifile)

    app = QtWidgets.QApplication(sys.argv)

    DM = DataCut()
    DM.indata['Raster'] = dat
    DM.ifile = sfile
    DM.settings(nodialog=True)

    plt.figure(dpi=150)
    plt.imshow(DM.indata['Raster'][0].data,
               extent=DM.indata['Raster'][0].extent)
    plt.colorbar()
    plt.show()

    plt.figure(dpi=150)
    plt.imshow(DM.outdata['Raster'][0].data,
               extent=DM.outdata['Raster'][0].extent)
    plt.colorbar()
    plt.show()


def _testprof():
    """Test Reprojection."""
    import sys
    from pygmi.raster.iodefs import get_raster
    import matplotlib.pyplot as plt

    ifile = r"d:\Workdata\bugs\Au5_SRTM30_utm36s.tif"
    sfile = r"d:\Workdata\bugs\Profiles_utm36s.shp"

    piter = ProgressBarText().iter

    dat = get_raster(ifile, piter=piter)

    app = QtWidgets.QApplication(sys.argv)

    DM = GetProf()
    DM.indata['Raster'] = dat
    DM.ifile = sfile
    DM.settings(nodialog=True)

    plt.figure(dpi=150)
    plt.imshow(DM.indata['Raster'][0].data,
               extent=DM.indata['Raster'][0].extent)
    plt.colorbar()
    plt.show()

    plt.figure(dpi=150)
    plt.imshow(DM.outdata['Raster'][0].data,
               extent=DM.outdata['Raster'][0].extent)
    plt.colorbar()
    plt.show()


def _testlstack():
    from pygmi.raster.iodefs import get_raster, export_raster
    import matplotlib.pyplot as plt

    idir = r'd:\Workdata\LULC\stack'

    ifiles = glob.glob(os.path.join(idir, '*.tif'))

    dat = []
    for ifile in ifiles:
        dat += get_raster(ifile)

    for i in dat:
        plt.figure(dpi=150)
        plt.title(i.dataid)
        plt.imshow(i.data)
        plt.colorbar()
        plt.show()
        print(i.nodata)
        print(i.data.dtype)

    dat2 = lstack(dat, dxy=30)

    for i in dat2:
        plt.figure(dpi=150)
        plt.title(i.dataid)
        plt.imshow(i.data)
        plt.colorbar()
        plt.show()
        print(i.nodata)
        print(i.data.dtype)

    ofile = r'd:/Workdata/LULC/2001_stack_norm_pc.tif'
    export_raster(ofile, dat2, 'GTiff')


def _testcut2():
    """Test Reprojection."""
    import sys
    from pygmi.raster.iodefs import get_raster, export_raster
    import matplotlib.pyplot as plt

    sfile  = r"D:\hypercut\shape\Areas_utm33s_east.shp"
    ifilt = r"D:\hypercut\*.hdr"
    odir = r"D:\hypercut\cut"

    pprint = print

    ifiles = glob.glob(ifilt)

    for ifile in ifiles:
        print(ifile)
        gdf = gpd.read_file(sfile)

        gdf = gdf[gdf.geometry != None]

        if gdf.geom_type.iloc[0] == 'MultiPolygon':
            pprint('You have a MultiPolygon. Only the first Polygon '
                   'of the MultiPolygon will be used.')
            poly = gdf['geometry'].iloc[0]
            tmp = poly.geoms[0]

            gdf.geometry.iloc[0] = tmp

        if gdf.geom_type.iloc[0] != 'Polygon':
            pprint('You need a polygon in that shape file')
            return None

        bounds = gdf.geometry.iloc[0].bounds

        dat = get_raster(ifile, bounds=bounds)

        if dat is None:
            continue

        for idat in dat:
            idat.nodata = 0

        ofile = os.path.join(odir, os.path.basename(ifile))
        ofile = ofile[:-4]+'new.tif'
        export_raster(ofile, dat, 'GTiff', bandsort=False)


def _testnewnull():
    """Test New null data assignment."""
    from pygmi.raster.iodefs import get_raster, export_raster

    ifilt = r"D:\hypercut\*.hdr"
    odir = r"D:\hypercut\cut"

    ifiles = glob.glob(ifilt)

    for ifile in ifiles:
        print(ifile)
        dat = get_raster(ifile)
        for idat in dat:
            idat.nodata = 0
            # idat.data = np.ma.masked_equal(idat.data, 0)

        ofile = os.path.join(odir, os.path.basename(ifile))
        ofile = ofile[:-4]+'.tif'
        export_raster(ofile, dat, 'GTiff', bandsort=False)
        # break


def _testlower():
    """Plot out files in a directory."""
    import matplotlib.pyplot as plt
    from pygmi.raster.modest_image import imshow
    from pygmi.raster.iodefs import get_raster
    from pygmi.raster.iodefs import export_raster

    ifilt = r"e:\WorkProjects\ST-2022-1355 Onshore Mapping\Niger\ratios\*.tif"
    odir = r"e:\WorkProjects\ST-2022-1355 Onshore Mapping\Niger\ratios2"
    dataid = 'B4divB2 Iron Oxide'

    ifiles = glob.glob(ifilt)

    for ifile in ifiles:
        print(ifile)
        dat = get_raster(ifile, dataid=dataid)
        dat = lstack(dat, dxy=20)

        ofile = os.path.join(odir, os.path.basename(ifile))
        export_raster(ofile, dat, 'GTiff', compression='ZSTD')

        # for i in dat:
        #     plt.figure(dpi=150)
        #     plt.title(os.path.basename(i.filename)+": "+i.dataid)

        #     vstd = i.data.std()
        #     vmean = i.data.mean()
        #     vmin = vmean-2*vstd
        #     vmax = vmean+2*vstd
        #     imshow(plt.gca,i.data, vmin=vmin, vmax=vmax, interpolation='nearest')
        #     plt.colorbar()
        #     plt.show()


if __name__ == "__main__":
    _testmerge()
