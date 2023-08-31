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
from collections import Counter
from PyQt5 import QtWidgets, QtCore
import numpy as np
import pandas as pd
from scipy.signal import tukey
import rasterio
import rasterio.merge
import pyproj
from pyproj.crs import CRS, ProjectedCRS
from pyproj.crs.coordinate_operation import TransverseMercatorConversion
from rasterio.warp import calculate_default_transform, reproject
from rasterio.mask import mask as riomask
import geopandas as gpd
from shapely.geometry import LineString

from pygmi import menu_default
from pygmi.raster.datatypes import Data
from pygmi.misc import ProgressBarText, ContextModule, BasicModule
from pygmi.raster.datatypes import numpy_to_pygmi


class Continuation(BasicModule):
    """Perform upward and downward continuation on potential field data."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cmb_dataid = QtWidgets.QComboBox()
        self.cmb_cont = QtWidgets.QComboBox()
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
        lbl_band = QtWidgets.QLabel('Band to perform continuation:')
        lbl_cont = QtWidgets.QLabel('Continuation type:')
        lbl_height = QtWidgets.QLabel('Continuation distance:')

        self.dsb_height.setMaximum(1000000.0)
        self.dsb_height.setMinimum(0.0)
        self.dsb_height.setValue(0.0)
        self.cmb_cont.clear()
        self.cmb_cont.addItems(['Upward', 'Downward'])

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Continuation')

        gridlayout_main.addWidget(lbl_band, 0, 0, 1, 1)
        gridlayout_main.addWidget(self.cmb_dataid, 0, 1, 1, 1)

        gridlayout_main.addWidget(lbl_cont, 1, 0, 1, 1)
        gridlayout_main.addWidget(self.cmb_cont, 1, 1, 1, 1)
        gridlayout_main.addWidget(lbl_height, 2, 0, 1, 1)
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
            self.showlog('No Raster Data.')
            return False

        for i in self.indata['Raster']:
            tmp.append(i.dataid)

        self.cmb_dataid.clear()
        self.cmb_dataid.addItems(tmp)

        if not nodialog:
            tmp = self.exec_()

            if tmp != 1:
                return False

        self.acceptall()

        return True

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.cmb_dataid)
        self.saveobj(self.cmb_cont)
        self.saveobj(self.dsb_height)

    def acceptall(self):
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        None.

        """
        h = self.dsb_height.value()
        ctype = self.cmb_cont.currentText()

        # Get data
        for i in self.indata['Raster']:
            if i.dataid == self.cmb_dataid.currentText():
                data = i
                break

        if ctype == 'Downward':
            dat = taylorcont(data, h)
        else:
            dat = fftcont(data, h)

        self.outdata['Raster'] = [dat]


class DataCut(BasicModule):
    """
    Cut Data using shapefiles.

    This class cuts raster datasets using a boundary defined by a polygon
    shapefile.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

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
            self.showlog('No raster data')
            return False

        if not nodialog:
            self.ifile, _ = QtWidgets.QFileDialog.getOpenFileName(
                self.parent, 'Open Shape File', '.', 'Shape file (*.shp)')
            if self.ifile == '':
                return False

        os.chdir(os.path.dirname(self.ifile))
        data = cut_raster(data, self.ifile, showlog=self.showlog)

        if data is None:
            return False

        self.outdata['Raster'] = data

        return True

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.ifile)


class DataLayerStack(BasicModule):
    """
    Data Layer Stack.

    This class merges datasets which have different rows and columns. It
    resamples them so that they have the same rows and columns.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dxy = None
        self.cb_cmask = QtWidgets.QCheckBox('Common mask for all bands')

        self.dsb_dxy = QtWidgets.QDoubleSpinBox()
        self.lbl_rows = QtWidgets.QLabel('Rows: 0')
        self.lbl_cols = QtWidgets.QLabel('Columns: 0')

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
        lbl_dxy = QtWidgets.QLabel('Cell Size:')

        self.dsb_dxy.setMaximum(9999999999.0)
        self.dsb_dxy.setMinimum(0.00001)
        self.dsb_dxy.setDecimals(5)
        self.dsb_dxy.setValue(40.)
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.cb_cmask.setChecked(True)

        self.setWindowTitle('Dataset Layer Stack and Resample')

        gridlayout_main.addWidget(lbl_dxy, 0, 0, 1, 1)
        gridlayout_main.addWidget(self.dsb_dxy, 0, 1, 1, 1)
        gridlayout_main.addWidget(self.lbl_rows, 1, 0, 1, 2)
        gridlayout_main.addWidget(self.lbl_cols, 2, 0, 1, 2)
        gridlayout_main.addWidget(self.cb_cmask, 3, 0, 1, 2)
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

        self.lbl_rows.setText('Rows: '+str(rows))
        self.lbl_cols.setText('Columns: '+str(cols))

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
        if 'RasterFileList' in self.indata:
            from pygmi.rsense.iodefs import get_data

            ifiles = self.indata['RasterFileList']
            self.showlog('Warning: Layer stacking a file list assumes '
                         'all datasets overlap in the same area')
            self.indata['Raster'] = []
            for ifile in ifiles:
                self.showlog('Processing '+os.path.basename(ifile))
                dat = get_data(ifile, piter=self.piter,
                               showlog=self.showlog)
                for i in dat:
                    i.data = i.data.astype(np.float32)
                self.indata['Raster'] += dat

        if 'Raster' not in self.indata:
            self.showlog('No Raster Data.')
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

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.dxy)
        self.saveobj(self.dsb_dxy)
        self.saveobj(self.cb_cmask)

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
                     showlog=self.showlog,
                     commonmask=self.cb_cmask.isChecked())
        self.outdata['Raster'] = dat


class DataMerge(BasicModule):
    """
    Data Merge.

    This class merges datasets which have different rows and columns. It
    resamples them so that they have the same rows and columns.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.idir = None
        self.method = merge_median
        self.rb_first = QtWidgets.QRadioButton('First - copy first file over '
                                               'last file at overlap.')
        self.rb_last = QtWidgets.QRadioButton('Last - copy last file over '
                                              'first file at overlap.')
        self.rb_min = QtWidgets.QRadioButton('Min - copy pixel wise minimum '
                                             'at overlap.')
        self.rb_max = QtWidgets.QRadioButton('Max - copy pixel wise maximum '
                                             'at overlap.')
        self.rb_median = QtWidgets.QRadioButton('Median - shift last file to '
                                                'median '
                                                'overlap value and copy over '
                                                'first file at overlap.')

        self.idirlist = QtWidgets.QLineEdit('')
        self.sfile = QtWidgets.QLineEdit('')
        self.cb_files_diff = QtWidgets.QCheckBox(
            'Mosaic by band labels, '
            'since band order may differ, or input files have different '
            'numbers of bands or nodata values.')
        self.cb_shift_to_median = QtWidgets.QCheckBox(
            'Shift bands to median value before mosaic. May '
            'allow for cleaner mosaic if datasets are offset.')

        self.cb_bands_to_files = QtWidgets.QCheckBox(
            'Save each band separately in a "mosaic" subdirectory.')
        self.forcetype = None
        self.singleband = False
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

        self.cb_files_diff.setChecked(True)
        self.cb_shift_to_median.setChecked(False)
        self.rb_median.setChecked(True)

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Dataset Mosaic')

        gb_merge_method = QtWidgets.QGroupBox('Mosiac method')
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
        gridlayout_main.addWidget(self.cb_files_diff, 3, 0, 1, 2)
        gridlayout_main.addWidget(self.cb_shift_to_median, 4, 0, 1, 2)
        gridlayout_main.addWidget(gb_merge_method, 5, 0, 1, 2)
        gridlayout_main.addWidget(self.cb_bands_to_files, 6, 0, 1, 2)
        gridlayout_main.addWidget(helpdocs, 7, 0, 1, 1)
        gridlayout_main.addWidget(buttonbox, 7, 1, 1, 1)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        pb_idirlist.pressed.connect(self.get_idir)
        pb_sfile.pressed.connect(self.get_sfile)
        self.cb_shift_to_median.stateChanged.connect(self.shiftchanged)
        self.cb_files_diff.stateChanged.connect(self.filesdiffchanged)
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
        if self.cb_shift_to_median.isChecked():
            self.cb_files_diff.setChecked(True)

    def filesdiffchanged(self):
        """
        Files different clicked.

        Returns
        -------
        None.

        """
        if not self.cb_files_diff.isChecked():
            self.cb_shift_to_median.setChecked(False)
            self.cb_bands_to_files.hide()
        else:
            self.cb_bands_to_files.show()

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

        return True

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

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.idir)
        self.saveobj(self.idirlist)
        self.saveobj(self.cb_files_diff)
        self.saveobj(self.cb_shift_to_median)

        self.saveobj(self.rb_first)
        self.saveobj(self.rb_last)
        self.saveobj(self.rb_min)
        self.saveobj(self.rb_max)
        self.saveobj(self.rb_median)

        self.saveobj(self.sfile)
        self.saveobj(self.cb_bands_to_files)
        self.saveobj(self.forcetype)
        self.saveobj(self.singleband)

    def acceptall(self):
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        bool
            Success of routine.

        """
        if self.cb_files_diff.isChecked():
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
                self.showlog('No input files in that directory')
                return False

            for ifile in self.piter(ifiles):
                indata += get_raster(ifile, piter=iter, metaonly=True)

        if indata is None:
            self.showlog('No input datasets')
            return False

        # Get projection information
        wkt = []
        crs = []
        for i in indata:
            if i.crs is None:
                self.showlog(f'{i.dataid} has no projection. '
                             'Please assign one.')
                return False

            wkt.append(i.crs.to_wkt())
            crs.append(i.crs)
            nodata = i.nodata

        wkt, iwkt, numwkt = np.unique(wkt, return_index=True,
                                      return_counts=True)
        if len(wkt) > 1:
            self.showlog('Error: Mismatched input projections. '
                         'Selecting most common projection')

            crs = crs[iwkt[numwkt == numwkt.max()][0]]
        else:
            crs = indata[0].crs

        bounds = get_shape_bounds(self.sfile.text(), crs, self.showlog)

        # Start Merge
        bandlist = []
        for i in indata:
            bandlist.append(i.dataid)

        bandlist = list(set(bandlist))

        if self.singleband is True:
            bandlist = ['Band_1']

        outdat = []
        for dataid in bandlist:
            self.showlog('Extracting '+dataid+'...')

            if self.cb_bands_to_files.isChecked():
                odir = os.path.join(self.idir, 'mosaic')
                os.makedirs(odir, exist_ok=True)
                ofile = dataid+'.tif'
                ofile = ofile.replace(' ', '_')
                ofile = ofile.replace(',', '_')
                ofile = ofile.replace('*', 'mult')
                ofile = os.path.join(odir, ofile)

                if os.path.exists(ofile):
                    self.showlog('Output file exists, skipping.')
                    continue

            ifiles = []
            allmval = []
            for i in self.piter(indata):
                if i.dataid != dataid and self.singleband is False:
                    continue
                metadata = i.metadata
                datetime = i.datetime

                i2 = get_raster(i.filename, piter=iter, dataid=i.dataid)

                if i2 is None:
                    continue

                i2 = i2[0]

                if i2.crs != crs:
                    src_height, src_width = i2.data.shape

                    transform, width, height = calculate_default_transform(
                        i2.crs, crs, src_width, src_height, *i2.bounds)

                    i2 = data_reproject(i2, crs, transform, height, width)

                if self.forcetype is not None:
                    i2.data = i2.data.astype(self.forcetype)

                if self.cb_shift_to_median.isChecked():
                    mval = np.ma.median(i2.data)
                else:
                    mval = 0
                allmval.append(mval)

                if self.singleband is True:
                    i2.dataid = 'Band_1'

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
                self.showlog('Too few bands of name '+dataid)
                continue

            self.showlog('Mosaicing '+dataid+'...')

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
            mosaic = mosaic + np.median(allmval)
            outdat.append(numpy_to_pygmi(mosaic, dataid=dataid))
            outdat[-1].set_transform(transform=otrans)
            outdat[-1].crs = crs
            outdat[-1].nodata = nodata
            outdat[-1].metadata = metadata
            outdat[-1].datetime = datetime

            if self.cb_bands_to_files.isChecked():
                export_raster(ofile, outdat, 'GTiff', compression='ZSTD',
                              showlog=self.showlog)

                del outdat
                del mosaic
                outdat = []

        if bounds is not None:
            outdat = cut_raster(outdat, self.sfile.text(), deepcopy=False)

        self.outdata['Raster'] = outdat

        return True

    def merge_same(self):
        """
        Mosaic files with same numbers of bands and band order.

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
            self.showlog('No input datasets')
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
                    self.showlog(f'{ifile} has no projection. '
                                 'Please assign one.')
                    return False
                wkt.append(dataset.crs.to_wkt())
                crs = dataset.crs
                nodata.append(dataset.nodata)

        wkt = list(set(wkt))
        if len(wkt) > 1:
            self.showlog('Error: Mismatched input projections')
            return False

        nodata = list(set(nodata))
        if len(nodata) > 1:
            self.showlog('Error: Mismatched nodata values. Try using merge '
                         'by band labels merge option. Please confirm bands '
                         'to be merged have the same label.')
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


class DataReproj(BasicModule):
    """
    Reprojections.

    This class reprojects datasets using the rasterio routines.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.orig_wkt = None
        self.targ_wkt = None

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
            self.showlog('Unknown Projection. Could not reproject')
            return

        if self.in_proj.wkt == '' or self.out_proj.wkt == '':
            self.showlog('Unknown Projection. Could not reproject')
            return

        # Input stuff
        src_crs = CRS.from_wkt(self.in_proj.wkt)

        # Output stuff
        dst_crs = CRS.from_wkt(self.out_proj.wkt)

        # Now create virtual dataset
        dat = []
        for data in self.piter(self.indata['Raster']):
            data2 = data_reproject(data, dst_crs, icrs=src_crs)

            dat.append(data2)

        self.orig_wkt = self.in_proj.wkt
        self.targ_wkt = self.out_proj.wkt
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
            self.showlog('No Raster Data.')
            return False

        if self.indata['Raster'][0].crs is None:
            self.showlog('Your input data has no projection. '
                         'Please assign one in the metadata summary.')
            return False

        if self.orig_wkt is None:
            self.orig_wkt = self.indata['Raster'][0].crs.to_wkt()
        if self.targ_wkt is None:
            self.targ_wkt = self.indata['Raster'][0].crs.to_wkt()

        self.in_proj.set_current(self.orig_wkt)
        self.out_proj.set_current(self.targ_wkt)

        if not nodialog:
            tmp = self.exec_()
            if tmp != 1:
                return False

        self.acceptall()

        return True

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.orig_wkt)
        self.saveobj(self.targ_wkt)


class GetProf(BasicModule):
    """
    Get a Profile.

    This class extracts a profile from a raster dataset using a line shapefile.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

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
            data = [i.copy() for i in self.indata['Raster']]
        else:
            self.showlog('No raster data')
            return False

        ext = 'Shape file (*.shp)'

        if not nodialog:
            self.ifile, _ = QtWidgets.QFileDialog.getOpenFileName(
                self.parent, 'Open Shape File', '.', ext)
            if self.ifile == '':
                return False

        os.chdir(os.path.dirname(self.ifile))

        try:
            gdf = gpd.read_file(self.ifile, engine='pyogrio')
        except:
            self.showlog('There was a problem importing the shapefile. '
                         'Please make sure you have at all the '
                         'individual files which make up the shapefile.')
            return None

        gdf = gdf[gdf.geometry != None]

        if gdf.geom_type.iloc[0] != 'LineString':
            self.showlog('You need lines in that shape file')
            return False

        data = lstack(data, self.piter, showlog=self.showlog)
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

                    x = ogdf['X']
                    y = ogdf['Y']
                    ogdf = gpd.GeoDataFrame(ogdf,
                                            geometry=gpd.points_from_xy(x, y))

                ogdf[idata.dataid] = z

            icnt += 1
            ogdf['line'] = str(icnt)
            if ogdf2 is None:
                ogdf2 = ogdf
            else:
                ogdf2 = ogdf2.append(ogdf, ignore_index=True)

        self.outdata['Vector'] = [ogdf2]

        return True

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.ifile)


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
        self.cmb_datum = QtWidgets.QComboBox()
        self.cmb_proj = QtWidgets.QComboBox()

        self.lbl_wkt = QtWidgets.QTextBrowser()
        self.lbl_wkt.setWordWrapMode(0)

        self.gridlayout.addWidget(self.groupbox, 1, 0, 1, 2)

        gridlayout = QtWidgets.QGridLayout(self.groupbox)
        gridlayout.addWidget(self.cmb_datum, 0, 0, 1, 1)
        gridlayout.addWidget(self.cmb_proj, 1, 0, 1, 1)
        gridlayout.addWidget(self.lbl_wkt, 2, 0, 1, 1)

        self.epsg_proj = getepsgcodes()
        self.epsg_proj[r'Current / Current'] = self.wkt
        self.epsg_proj[r'None / None'] = ''
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

        self.cmb_datum.addItems(tmp)
        self.cmb_proj.addItem('Current')
        self.cmb_datum.currentIndexChanged.connect(self.combo_datum_change)
        self.cmb_proj.currentIndexChanged.connect(self.combo_change)

    def set_current(self, wkt):
        """
        Set new WKT for current option.

        Parameters
        ----------
        wkt : str
            Well Known Text descriptions for coordinates (WKT).

        Returns
        -------
        None.

        """
        if wkt in ['', 'None']:
            self.cmb_datum.setCurrentText('None')
            return

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
        indx = self.cmb_datum.currentIndex()
        txt = self.cmb_datum.itemText(indx)
        self.cmb_proj.currentIndexChanged.disconnect()

        self.cmb_proj.clear()
        self.cmb_proj.addItems(self.plist[txt])

        self.cmb_proj.currentIndexChanged.connect(self.combo_change)

        self.combo_change()

    def combo_change(self):
        """
        Change Combo.

        Returns
        -------
        None.

        """
        dtxt = self.cmb_datum.currentText()
        ptxt = self.cmb_proj.currentText()

        txt = dtxt + r' / '+ptxt

        self.wkt = self.epsg_proj[txt]

        # if self.wkt is not a string, it must be epsg code
        if not isinstance(self.wkt, str):
            self.wkt = CRS.from_epsg(self.wkt).to_wkt(pretty=True)
        elif self.wkt not in ['', 'None']:
            self.wkt = CRS.from_wkt(self.wkt).to_wkt(pretty=True)

        # The next two lines make sure we have spaces after ALL commas.
        wkttmp = self.wkt.replace(', ', ',')
        wkttmp = wkttmp.replace(',', ', ')

        self.lbl_wkt.setText(wkttmp)


class Metadata(ContextModule):
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
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.banddata = {}
        self.dataid = {}
        self.oldtxt = ''

        self.cmb_bandid = QtWidgets.QComboBox()
        self.pb_rename_id = QtWidgets.QPushButton('Rename Band Name')
        self.lbl_rows = QtWidgets.QLabel()
        self.lbl_cols = QtWidgets.QLabel()
        self.txt_null = QtWidgets.QLineEdit()
        self.dsb_tlx = QtWidgets.QLineEdit()
        self.dsb_tly = QtWidgets.QLineEdit()
        self.dsb_xdim = QtWidgets.QLineEdit()
        self.dsb_ydim = QtWidgets.QLineEdit()
        self.led_units = QtWidgets.QLineEdit()
        self.lbl_min = QtWidgets.QLabel()
        self.lbl_max = QtWidgets.QLabel()
        self.lbl_mean = QtWidgets.QLabel()
        self.lbl_dtype = QtWidgets.QLabel()
        self.date = QtWidgets.QDateEdit()

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
        lbl_tlx = QtWidgets.QLabel('Top Left X Coordinate:')
        lbl_tly = QtWidgets.QLabel('Top Left Y Coordinate:')
        lbl_xdim = QtWidgets.QLabel('X Dimension:')
        lbl_ydim = QtWidgets.QLabel('Y Dimension:')
        lbl_null = QtWidgets.QLabel('Null/Nodata value:')
        lbl_rows = QtWidgets.QLabel('Rows:')
        lbl_cols = QtWidgets.QLabel('Columns:')
        lbl_min = QtWidgets.QLabel('Dataset Minimum:')
        lbl_max = QtWidgets.QLabel('Dataset Maximum:')
        lbl_mean = QtWidgets.QLabel('Dataset Mean:')
        lbl_units = QtWidgets.QLabel('Dataset Units:')
        lbl_bandid = QtWidgets.QLabel('Band Name:')
        lbl_dtype = QtWidgets.QLabel('Data Type:')
        lbl_date = QtWidgets.QLabel('Acquisition Date:')

        sizepolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                           QtWidgets.QSizePolicy.Expanding)
        groupbox.setSizePolicy(sizepolicy)
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Dataset Metadata')
        self.date.setCalendarPopup(True)

        gridlayout_main.addWidget(lbl_bandid, 0, 0, 1, 1)
        gridlayout_main.addWidget(self.cmb_bandid, 0, 1, 1, 3)
        gridlayout_main.addWidget(self.pb_rename_id, 1, 1, 1, 3)
        gridlayout_main.addWidget(groupbox, 2, 0, 1, 2)
        gridlayout_main.addWidget(self.proj, 2, 2, 1, 2)
        gridlayout_main.addWidget(buttonbox, 4, 0, 1, 4)

        gridlayout.addWidget(lbl_tlx, 0, 0, 1, 1)
        gridlayout.addWidget(self.dsb_tlx, 0, 1, 1, 1)
        gridlayout.addWidget(lbl_tly, 1, 0, 1, 1)
        gridlayout.addWidget(self.dsb_tly, 1, 1, 1, 1)
        gridlayout.addWidget(lbl_xdim, 2, 0, 1, 1)
        gridlayout.addWidget(self.dsb_xdim, 2, 1, 1, 1)
        gridlayout.addWidget(lbl_ydim, 3, 0, 1, 1)
        gridlayout.addWidget(self.dsb_ydim, 3, 1, 1, 1)
        gridlayout.addWidget(lbl_null, 4, 0, 1, 1)
        gridlayout.addWidget(self.txt_null, 4, 1, 1, 1)
        gridlayout.addWidget(lbl_rows, 5, 0, 1, 1)
        gridlayout.addWidget(self.lbl_rows, 5, 1, 1, 1)
        gridlayout.addWidget(lbl_cols, 6, 0, 1, 1)
        gridlayout.addWidget(self.lbl_cols, 6, 1, 1, 1)
        gridlayout.addWidget(lbl_min, 7, 0, 1, 1)
        gridlayout.addWidget(self.lbl_min, 7, 1, 1, 1)
        gridlayout.addWidget(lbl_max, 8, 0, 1, 1)
        gridlayout.addWidget(self.lbl_max, 8, 1, 1, 1)
        gridlayout.addWidget(lbl_mean, 9, 0, 1, 1)
        gridlayout.addWidget(self.lbl_mean, 9, 1, 1, 1)
        gridlayout.addWidget(lbl_units, 10, 0, 1, 1)
        gridlayout.addWidget(self.led_units, 10, 1, 1, 1)
        gridlayout.addWidget(lbl_dtype, 11, 0, 1, 1)
        gridlayout.addWidget(self.lbl_dtype, 11, 1, 1, 1)
        gridlayout.addWidget(lbl_date, 12, 0, 1, 1)
        gridlayout.addWidget(self.date, 12, 1, 1, 1)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)

        self.cmb_bandid.currentIndexChanged.connect(self.update_vals)
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
                    tmp.datetime = i.datetime
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
        ctxt = str(self.cmb_bandid.currentText())
        (skey, isokay) = QtWidgets.QInputDialog.getText(
            self.parent, 'Rename Band Name',
            'Please type in the new name for the band',
            QtWidgets.QLineEdit.Normal, ctxt)

        if isokay:
            self.cmb_bandid.currentIndexChanged.disconnect()
            indx = self.cmb_bandid.currentIndex()
            txt = self.cmb_bandid.itemText(indx)
            self.banddata[skey] = self.banddata.pop(txt)
            self.dataid[skey] = self.dataid.pop(txt)
            self.oldtxt = skey
            self.cmb_bandid.setItemText(indx, skey)
            self.cmb_bandid.currentIndexChanged.connect(self.update_vals)

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
            if self.txt_null.text().lower() != 'none':
                odata.nodata = float(self.txt_null.text())
            left = float(self.dsb_tlx.text())
            top = float(self.dsb_tly.text())
            xdim = float(self.dsb_xdim.text())
            ydim = float(self.dsb_ydim.text())

            odata.set_transform(xdim, left, ydim, top)
            odata.datetime = self.date.date().toPyDate()
        except ValueError:
            self.showlog('Value error - abandoning changes')

        indx = self.cmb_bandid.currentIndex()
        txt = self.cmb_bandid.itemText(indx)
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
        self.lbl_dtype.setText(str(idata.data.dtype))
        self.date.setDate(idata.datetime)

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
            crs = CRS.from_user_input(self.indata['Raster'][0].crs)
            self.proj.set_current(crs.to_wkt(pretty=True))

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
            tmp.datetime = i.datetime

        self.cmb_bandid.currentIndexChanged.disconnect()
        self.cmb_bandid.addItems(bandid)
        indx = self.cmb_bandid.currentIndex()
        self.oldtxt = self.cmb_bandid.itemText(indx)
        self.cmb_bandid.currentIndexChanged.connect(self.update_vals)

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
        self.lbl_dtype.setText(str(idata.data.dtype))
        self.date.setDate(idata.datetime)

        self.update_vals()

        tmp = self.exec_()

        if tmp != 1:
            return False

        self.acceptall()

        return True


def check_dataid(out):
    """
    Check dataid for duplicates and renames where necessary.

    Parameters
    ----------
    out : list of PyGMI Data
        PyGMI raster data.

    Returns
    -------
    out : list of PyGMI Data
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


def cut_raster(data, ifile, showlog=print, deepcopy=True):
    """
    Cut a raster dataset.

    Cut a raster dataset using a shapefile.

    Parameters
    ----------
    data : list of PyGMI Data
        PyGMI Dataset
    ifile : str
        shapefile used to cut data
    showlog : function, optional
        Function for printing text. The default is print.

    Returns
    -------
    data : list of PyGMI Data
        PyGMI Dataset
    """
    if deepcopy is True:
        data = [i.copy() for i in data]

    try:
        gdf = gpd.read_file(ifile)
    except:
        showlog('There was a problem importing the shapefile. Please make '
                'sure you have at all the individual files which make up '
                'the shapefile.')
        return None

    gdf = gdf[gdf.geometry != None]

    if 'Polygon' not in gdf.geom_type.iloc[0]:
        showlog('You need a polygon in that shape file')
        return None

    for idata in data:
        # Convert the layer extent to image pixel coordinates
        # poly = gdf['geometry'].iloc[0]
        dext = idata.bounds
        lext = gdf['geometry'].total_bounds

        if ((dext[0] > lext[2]) or (dext[2] < lext[0])
                or (dext[1] > lext[3]) or (dext[3] < lext[1])):

            showlog('The shapefile is not in the same area as the raster '
                    'dataset. Please check its coordinates and make sure its '
                    'projection is the same as the raster dataset')
            return None

        # This section converts PolygonZ to Polygon, and takes first polygon.
        # coords = gdf['geometry'].loc[0].exterior.coords
        # coords = [Polygon([[p[0], p[1]] for p in coords])]
        coords = gdf['geometry']

        dat, trans = riomask(idata.to_mem(), coords, crop=True)

        idata.data = np.ma.masked_equal(dat.squeeze(), idata.nodata)

        idata.set_transform(transform=trans)

    data = trim_raster(data)

    return data


def data_reproject(data, ocrs, otransform=None, orows=None,
                   ocolumns=None, icrs=None):
    """
    Reproject dataset.

    Parameters
    ----------
    data : PyGMI Data
        PyGMI dataset.
    ocrs : CRS
        output crs.
    otransform : Affine, optional
        Output affine transform. The default is None.
    orows : int, optional
        output rows. The default is None.
    ocolumns : int, optional
        output columns. The default is None.
    icrs : CRS, optional
        input crs. The default is None.

    Returns
    -------
    data2 : PyGMI Data
        Reprojected dataset.

    """
    if icrs is None:
        icrs = data.crs

    if otransform is None:
        src_height, src_width = data.data.shape

        otransform, ocolumns, orows = calculate_default_transform(
            icrs, ocrs, src_width, src_height, *data.bounds)

    odata = np.zeros((orows, ocolumns), dtype=data.data.dtype)
    odata, _ = reproject(source=data.data,
                         destination=odata,
                         src_transform=data.transform,
                         src_crs=icrs,
                         dst_transform=otransform,
                         dst_crs=ocrs,
                         src_nodata=data.nodata,
                         resampling=rasterio.enums.Resampling['bilinear'])

    data2 = Data()
    data2.data = odata
    data2.crs = ocrs
    data2.set_transform(transform=otransform)
    data2.data = data2.data.astype(data.data.dtype)
    data2.dataid = data.dataid
    data2.wkt = CRS.to_wkt(ocrs)
    data2.filename = data.filename[:-4]+'_prj'+data.filename[-4:]

    data2.data = np.ma.masked_equal(data2.data, data.nodata)
    data2.nodata = data.nodata
    data2.metadata = data.metadata

    return data2


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


def get_shape_bounds(sfile, crs=None, showlog=print):
    """
    Get bounds from a shape file.

    Parameters
    ----------
    sfile : str
        Filename for shapefile.
    crs : rasterio CRS
        target crs for shapefile
    showlog : function, optional
        Display information. The default is print.

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
        gdf = gdf.to_crs(crs)

    if gdf.geom_type.iloc[0] == 'MultiPolygon':
        showlog('You have a MultiPolygon. Only the first Polygon '
                'of the MultiPolygon will be used.')
        poly = gdf['geometry'].iloc[0]
        tmp = poly.geoms[0]

        gdf.geometry.iloc[0] = tmp

    if gdf.geom_type.iloc[0] != 'Polygon':
        showlog('You need a polygon in that shape file')
        return None

    bounds = gdf.geometry.iloc[0].bounds

    return bounds


def getepsgcodes():
    """
    Routine used to get a list of EPSG codes.

    Returns
    -------
    pcodes : dictionary
        Dictionary of codes per projection in WKT format.

    """
    crs_list = pyproj.database.query_crs_info(auth_name='EPSG', pj_types=None)

    pcodes = {}
    for i in crs_list:
        if '/' in i.name:
            pcodes[i.name] = int(i.code)
        else:
            pcodes[i.name+r' / Geodetic Geographic'] = int(i.code)

    for datum in [4222, 4148]:
        for clong in range(15, 35, 2):
            geog_crs = CRS.from_epsg(datum)
            proj_crs = ProjectedCRS(name=f'{geog_crs.name} / TM{clong}',
                                    conversion=TransverseMercatorConversion(
                                        latitude_natural_origin=0,
                                        longitude_natural_origin=clong,
                                        false_easting=0,
                                        false_northing=0,
                                        scale_factor_natural_origin=1.0,),
                                    geodetic_crs=geog_crs)

            pcodes[f'{geog_crs.name} / TM{clong}'] = proj_crs.to_wkt(pretty=True)

            # if 'Cape' in datum:
            #     wkt = ('PROJCS["Cape / TM'+str(clong)+'",'
            #            'GEOGCS["Cape",'
            #            'DATUM["Cape",'
            #            'SPHEROID["Clarke 1880 (Arc)",'
            #            '6378249.145,293.4663077,'
            #            'AUTHORITY["EPSG","7013"]],'
            #            'AUTHORITY["EPSG","6222"]],'
            #            'PRIMEM["Greenwich",0,'
            #            'AUTHORITY["EPSG","8901"]],'
            #            'UNIT["degree",0.0174532925199433,'
            #            'AUTHORITY["EPSG","9122"]],'
            #            'AUTHORITY["EPSG","4222"]],'
            #            'PROJECTION["Transverse_Mercator"],'
            #            'PARAMETER["latitude_of_origin",0],'
            #            'PARAMETER["central_meridian",'+str(clong)+'],'
            #            'PARAMETER["scale_factor",1],'
            #            'PARAMETER["false_easting",0],'
            #            'PARAMETER["false_northing",0],'
            #            'UNIT["metre",1,AUTHORITY["EPSG","9001"]],'
            #            'AXIS["Easting",EAST],'
            #            'AXIS["Northing",NORTH]]')

            # elif 'Hartebeesthoek94' in datum:
            #     wkt = ('PROJCS["Hartebeesthoek94 / TM'+str(clong)+'",'
            #            'GEOGCS["Hartebeesthoek94",'
            #            'DATUM["Hartebeesthoek94",'
            #            'SPHEROID["WGS 84",6378137,298.257223563,'
            #            'AUTHORITY["EPSG","7030"]],'
            #            'AUTHORITY["EPSG","6148"]],'
            #            'PRIMEM["Greenwich",0,'
            #            'AUTHORITY["EPSG","8901"]],'
            #            'UNIT["degree",0.0174532925199433,'
            #            'AUTHORITY["EPSG","9122"]],'
            #            'AUTHORITY["EPSG","4148"]],'
            #            'PROJECTION["Transverse_Mercator"],'
            #            'PARAMETER["latitude_of_origin",0],'
            #            'PARAMETER["central_meridian",'+str(clong)+'],'
            #            'PARAMETER["scale_factor",1],'
            #            'PARAMETER["false_easting",0],'
            #            'PARAMETER["false_northing",0],'
            #            'UNIT["metre",1,AUTHORITY["EPSG","9001"]],'
            #            'AXIS["Easting",EAST],'
            #            'AXIS["Northing",NORTH]]')

            # pcodes[datum+r' / TM'+str(clong)] = wkt

    return pcodes


def lstack(dat, piter=None, dxy=None, showlog=print, commonmask=False,
           masterid=None, nodeepcopy=False, resampling='nearest',
           checkdataid=True):
    """
    Layer stack datasets found in a single PyGMI data object.

    The aim is to ensure that all datasets have the same number of rows and
    columns.

    Parameters
    ----------
    dat : list of PyGMI Data
        data object which stores datasets
    piter : function, optional
        Progress bar iterator. The default is None.
    dxy : float, optional
        Cell size. The default is None.
    showlog : function, optional
        Display information. The default is print.
    commonmask : bool, optional
        Create a common mask for all bands. The default is False.
    masterid : str, optional
        ID of master dataset. The default is None.

    Returns
    -------
    out : list of PyGMI Data
        data object which stores datasets

    """
    if piter is None:
        piter = ProgressBarText().iter

    if dat[0].isrgb:
        return dat

    resampling = rasterio.enums.Resampling[resampling]
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
            dat = [i.copy() for i in dat]
        if checkdataid is True:
            dat = check_dataid(dat)
        return dat

    showlog('Merging data...')
    if masterid is not None:
        for i in dat:
            if i.dataid == masterid:
                data = i
                break

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
        showlog('Your rows or cols are zero. '
                'Your input projection may be wrong')
        return None

    dat2 = []
    cmask = None
    for data in piter(dat):

        if dtype is not None:
            data.data = data.data.astype(dtype)
            data.nodata = nodata

        if data.crs is None:
            showlog(f'{data.dataid} has no defined projection. '
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
                             dst_crs=data.crs,
                             resampling=resampling)

        data2 = Data()
        data2.data = np.ma.masked_equal(odata, data.nodata)
        data2.data.mask = np.ma.getmaskarray(data2.data)
        data2.nodata = data.nodata
        data2.crs = data.crs
        data2.set_transform(transform=trans)
        data2.data = data2.data.astype(data.data.dtype)
        data2.dataid = data.dataid
        data2.filename = data.filename
        data2.datetime = data.datetime

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

    if commonmask is True:
        for idat in piter(dat2):
            idat.data.mask = cmask
            idat.data = np.ma.array(idat.data.filled(idat.nodata), mask=cmask)

    if checkdataid is True:
        out = check_dataid(dat2)
    else:
        out = dat2

    return out


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

    tmp1[mtmp2] = merged_data[mtmp2]

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


def taylorcont(data, h):
    """
    Taylor Continuation.

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


def trim_raster(olddata):
    """
    Trim nulls from a raster dataset.

    This function trims entire rows or columns of data which are masked,
    and are on the edges of the dataset. Masked values are set to the null
    value.

    Parameters
    ----------
    olddata : list of PyGMI Data
        PyGMI dataset

    Returns
    -------
    olddata : list of PyGMI Data
        PyGMI dataset
    """
    for data in olddata:
        mask = np.ma.getmaskarray(data.data)
        # data.data.data[mask] = data.nodata

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
        data.data.mask = mask[rowstart:rowend, colstart:colend]
        # data.data.mask = (data.data.data == data.nodata)
        xmin = data.extent[0] + colstart*data.xdim
        ymax = data.extent[-1] - rowstart*data.ydim

        data.set_transform(data.xdim, xmin, data.ydim, ymax)

    return olddata


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
    plt.plot(zdown.data[50], 'b')
    plt.plot(zdownn.data[50], 'k')
    plt.show()


def _testfft():
    """Test FFT."""
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    import scipy
    from IPython import get_ipython
    from pygmi.raster.iodefs import get_raster

    get_ipython().run_line_magic('matplotlib', 'inline')

    ifile = r'D:\Workdata\geothermal\bushveldrtp.hdr'
    data = get_raster(ifile)[0]

    # quick model
    plt.imshow(data.data, cmap=colormaps['jet'], vmin=-500, vmax=500)
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


def _testfn():
    """Test."""
    import sys
    from pygmi.raster.iodefs import get_raster

    ifile = r"D:\WC\ASTER\Original_data\AST_05_07XT_20060411_15908_stack.tif"

    dat = get_raster(ifile)

    app = QtWidgets.QApplication(sys.argv)
    tmp = Metadata()
    tmp.indata['Raster'] = dat
    tmp.run()


if __name__ == "__main__":
    _testfn()
