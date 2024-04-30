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
"""Data Preparation for Vector Data."""

import os
import copy
import glob
from functools import partial
from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt
import geopandas as gpd
from pyproj import CRS, Transformer

from pygmi import menu_default
from pygmi.raster.dataprep import GroupProj
from pygmi.raster.datatypes import Data
from pygmi.vector.minc import minc
from pygmi.misc import BasicModule, ContextModule, ProgressBarText


class PointCut(BasicModule):
    """
    Cut Data using shapefiles.

    This class cuts point datasets using a boundary defined by a polygon
    shapefile.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_import = True

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
        if 'Vector' in self.indata:
            data = copy.deepcopy(self.indata['Vector'][0])
        else:
            self.showlog('No point data')
            return False

        if not nodialog:
            ext = 'Shape file (*.shp)'
            self.ifile, _ = QtWidgets.QFileDialog.getOpenFileName(
                self.parent, 'Open Shape File', '.', ext)
            if self.ifile == '':
                return False

        os.chdir(os.path.dirname(self.ifile))
        data = cut_point(data, self.ifile, self.showlog)

        if data is None:
            return False

        if self.pbar is not None:
            self.pbar.to_max()
        self.outdata['Vector'] = [data]

        return True

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.ifile)


class DataGrid(BasicModule):
    """
    Grid Point Data.

    This class grids point data using a nearest neighbourhood technique.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dxy = None
        self.dataid_text = None

        self.le_dxy = QtWidgets.QLineEdit('1.0')
        self.le_null = QtWidgets.QLineEdit('0.0')
        self.le_bdist = QtWidgets.QLineEdit('4.0')

        self.cmb_dataid = QtWidgets.QComboBox()
        self.cmb_grid_method = QtWidgets.QComboBox()
        self.lbl_rows = QtWidgets.QLabel('Rows: 0')
        self.lbl_cols = QtWidgets.QLabel('Columns: 0')
        self.lbl_bdist = QtWidgets.QLabel('Blanking Distance:')

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gl_main = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.vector.dataprep.datagrid')
        lbl_band = QtWidgets.QLabel('Column to Grid:')
        lbl_dxy = QtWidgets.QLabel('Cell Size:')
        lbl_null = QtWidgets.QLabel('Null Value:')
        lbl_method = QtWidgets.QLabel('Gridding Method:')

        val = QtGui.QDoubleValidator(0.0000001, 9999999999.0, 9)
        val.setNotation(QtGui.QDoubleValidator.ScientificNotation)
        val.setLocale(QtCore.QLocale(QtCore.QLocale.C))

        self.le_dxy.setValidator(val)
        self.le_null.setValidator(val)

        self.cmb_grid_method.addItems(['Nearest Neighbour', 'Linear', 'Cubic',
                                       'Minimum Curvature'])

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Dataset Gridding')

        gl_main.addWidget(lbl_method, 0, 0, 1, 1)
        gl_main.addWidget(self.cmb_grid_method, 0, 1, 1, 1)
        gl_main.addWidget(lbl_dxy, 1, 0, 1, 1)
        gl_main.addWidget(self.le_dxy, 1, 1, 1, 1)
        gl_main.addWidget(self.lbl_rows, 2, 0, 1, 2)
        gl_main.addWidget(self.lbl_cols, 3, 0, 1, 2)
        gl_main.addWidget(lbl_band, 4, 0, 1, 1)
        gl_main.addWidget(self.cmb_dataid, 4, 1, 1, 1)
        gl_main.addWidget(lbl_null, 5, 0, 1, 1)
        gl_main.addWidget(self.le_null, 5, 1, 1, 1)
        gl_main.addWidget(self.lbl_bdist, 6, 0, 1, 1)
        gl_main.addWidget(self.le_bdist, 6, 1, 1, 1)
        gl_main.addWidget(helpdocs, 7, 0, 1, 1)
        gl_main.addWidget(buttonbox, 7, 1, 1, 3)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        self.le_dxy.textChanged.connect(self.dxy_change)

    def dxy_change(self):
        """
        When dxy is changed on the interface, this updates rows and columns.

        Returns
        -------
        None.

        """
        txt = str(self.le_dxy.text())
        if txt.replace('.', '', 1).isdigit():
            self.dxy = float(self.le_dxy.text())
        else:
            return

        data = self.indata['Vector'][0]

        x = data.geometry.x.values
        y = data.geometry.y.values

        cols = round(x.ptp()/self.dxy)
        rows = round(y.ptp()/self.dxy)

        self.lbl_rows.setText('Rows: '+str(rows))
        self.lbl_cols.setText('Columns: '+str(cols))

    def grid_method_change(self):
        """
        When grid method is changed, this updated hidden controls.

        Returns
        -------
        None.

        """
        if self.cmb_grid_method.currentText() == 'Minimum Curvature':
            self.lbl_bdist.show()
            self.le_bdist.show()
        else:
            self.lbl_bdist.hide()
            self.le_bdist.hide()

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
        if 'Vector' not in self.indata:
            self.showlog('No Point Data')
            return False

        data = self.indata['Vector'][0]

        if data.geom_type.iloc[0] != 'Point':
            self.showlog('No Point Data')
            return False

        self.cmb_dataid.clear()

        filt = ((data.columns != 'geometry') &
                (data.columns != 'line'))

        cols = list(data.columns[filt])
        self.cmb_dataid.addItems(cols)

        if self.dataid_text is None:
            self.dataid_text = self.cmb_dataid.currentText()
        if self.dataid_text in cols:
            self.cmb_dataid.setCurrentText(self.dataid_text)

        if self.dxy is None:
            x = data.geometry.x.values
            y = data.geometry.y.values

            dx = x.ptp()/np.sqrt(x.size)
            dy = y.ptp()/np.sqrt(y.size)
            self.dxy = max(dx, dy)
            self.dxy = min([x.ptp(), y.ptp(), self.dxy])

        self.le_dxy.setText(f'{self.dxy:.8f}')
        self.dxy_change()

        self.grid_method_change()
        if not nodialog:
            tmp = self.exec()
            if tmp != 1:
                return False

        try:
            float(self.le_dxy.text())
            float(self.le_null.text())
            float(self.le_bdist.text())
        except ValueError:
            self.showlog('Value Error')
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
        self.saveobj(self.le_dxy)
        self.saveobj(self.le_null)
        self.saveobj(self.le_bdist)
        self.saveobj(self.dataid_text)
        self.saveobj(self.cmb_dataid)
        self.saveobj(self.cmb_grid_method)

    def acceptall(self):
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        None.

        """
        dxy = float(self.le_dxy.text())
        method = self.cmb_grid_method.currentText()
        nullvalue = float(self.le_null.text())
        bdist = float(self.le_bdist.text())
        data = self.indata['Vector'][0]
        dataid = self.cmb_dataid.currentText()
        newdat = []

        if bdist < 1:
            bdist = None
            self.showlog('Blanking distance too small.')

        data2 = data[['geometry', dataid]]
        data2 = data2.dropna()

        filt = (data2[dataid] != nullvalue)
        x = data2.geometry.x.values[filt]
        y = data2.geometry.y.values[filt]
        z = data2[dataid].values[filt]

        dat = gridxyz(x, y, z, dxy, nullvalue, method, bdist, self.showlog)
        dat.dataid = dataid
        dat.crs = data2.crs

        newdat.append(dat)

        self.outdata['Raster'] = newdat
        self.outdata['Vector'] = self.indata['Vector']


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
        gl_main = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.raster.dataprep.datareproj')

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Dataset Reprojection')

        gl_main.addWidget(self.in_proj, 0, 0, 1, 1)
        gl_main.addWidget(self.out_proj, 0, 1, 1, 1)
        gl_main.addWidget(helpdocs, 1, 0, 1, 1)
        gl_main.addWidget(buttonbox, 1, 1, 1, 1)

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
            self.showlog('Could not reproject')
            return

        data = self.indata['Vector'][0]

        # Input stuff
        orig_wkt = self.in_proj.wkt

        # Output stuff
        targ_wkt = self.out_proj.wkt

        data.set_crs(CRS.from_wkt(orig_wkt), inplace=True)
        data.to_crs(CRS.from_wkt(targ_wkt), inplace=True)

        data = data.assign(Xnew=data.geometry.x.values)
        data = data.assign(Ynew=data.geometry.y.values)

        self.outdata['Vector'] = [data]
        self.orig_wkt = self.in_proj.wkt
        self.targ_wkt = self.out_proj.wkt

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
        if 'Vector' not in self.indata:
            self.showlog('No vector data.')
            return False

        if self.indata['Vector'][0].crs is not None:
            self.orig_wkt = self.indata['Vector'][0].crs.to_wkt()

        if self.orig_wkt is None:
            indx = self.in_proj.cmb_datum.findText(r'WGS 84')
            self.in_proj.cmb_datum.setCurrentIndex(indx)
            self.orig_wkt = self.in_proj.wkt
        else:
            self.in_proj.set_current(self.orig_wkt)

        if self.targ_wkt is None:
            indx = self.in_proj.cmb_datum.findText(r'WGS 84')
            self.out_proj.cmb_datum.setCurrentIndex(indx)
            self.targ_wkt = self.out_proj.wkt
        else:
            self.out_proj.set_current(self.targ_wkt)

        if not nodialog:
            tmp = self.exec()

            if tmp != 1:
                return False

        if 'Vector' in self.indata:
            self.outdata['Vector'] = []
            for ivec in self.indata['Vector']:
                ivec = ivec.set_crs(self.in_proj.wkt)
                self.outdata['Vector'].append(ivec.to_crs(self.out_proj.wkt))
        else:
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


class Metadata(ContextModule):
    """
    Edit Metadata.

    This class allows the editing of the metadata for a vector dataset using a
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

        self.cmb_bandid = QtWidgets.QComboBox()
        self.proj = GroupProj('Input Projection')

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gl_main = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()

        lbl_bandid = QtWidgets.QLabel('Source:')

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Vector Dataset Metadata')

        gl_main.addWidget(lbl_bandid, 0, 0, 1, 1)
        gl_main.addWidget(self.cmb_bandid, 0, 1, 1, 3)
        gl_main.addWidget(self.proj, 2, 0, 1, 4)
        gl_main.addWidget(buttonbox, 4, 0, 1, 4)

        buttonbox.accepted.connect(self.acceptall)
        buttonbox.rejected.connect(self.reject)

        self.resize(-1, 320)

    def acceptall(self):
        """
        Accept option.

        Returns
        -------
        None.

        """
        wkt = self.proj.wkt

        for tmp in self.indata['Vector']:
            if wkt == 'None':
                tmp.crs = None
            else:
                tmp.crs = CRS.from_wkt(wkt)

        self.accept()

    def run(self):
        """
        Entry point to start this routine.

        Returns
        -------
        tmp : bool
            True if successful, False otherwise.

        """
        bandid = []
        if self.indata['Vector'][0].crs is None:
            self.proj.set_current('None')
        else:
            self.proj.set_current(self.indata['Vector'][0].crs.to_wkt())

        for i in self.indata['Vector']:
            if 'source' in i.attrs:
                bandid.append(i.attrs['source'])
            else:
                bandid.append('Unknown')

        self.cmb_bandid.addItems(bandid)

        self.show()


class TextFileSplit(BasicModule):
    """Split a text file into smaller text files."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_import = True

        self.le_ifile = QtWidgets.QLineEdit('')
        self.le_files = QtWidgets.QLineEdit('1')
        self.le_lines = QtWidgets.QLineEdit('1')
        self.le_bytes = QtWidgets.QLineEdit('1')
        self.chk_allfiles = QtWidgets.QCheckBox('Split all text files with '
                                                'same extension in current '
                                                'directory')

        self.cmb_method = QtWidgets.QComboBox()
        self.lbl_totsize = QtWidgets.QLabel('0')
        self.lbl_totlines = QtWidgets.QLabel('0')

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        pb_ifile = QtWidgets.QPushButton(' Filename')
        gl_main = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.vector.dataprep.filesplit')
        lbl_files = QtWidgets.QLabel('Number of files:')
        lbl_lines = QtWidgets.QLabel('Max lines per file:')
        lbl_bytes = QtWidgets.QLabel('Max bytes per file:')
        lbl_method = QtWidgets.QLabel('Split Method:')
        self.lbl_totsize = QtWidgets.QLabel('0')
        self.lbl_totlines = QtWidgets.QLabel('0')

        val = QtGui.QIntValidator(1, 2147483647)

        self.le_files.setValidator(val)
        self.le_lines.setValidator(val)
        self.le_bytes.setValidator(val)
        self.le_files.setEnabled(True)
        self.le_lines.setDisabled(True)
        self.le_bytes.setDisabled(True)

        self.cmb_method.addItems(['Files', 'Bytes', 'Lines'])

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Text File Split')

        gl_main.addWidget(pb_ifile, 0, 0, 1, 1)
        gl_main.addWidget(self.le_ifile, 0, 1, 1, 1)
        gl_main.addWidget(lbl_method, 1, 0, 1, 1)
        gl_main.addWidget(self.cmb_method, 1, 1, 1, 1)
        gl_main.addWidget(QtWidgets.QLabel('Total File Size:'), 2, 0, 1, 1)
        gl_main.addWidget(self.lbl_totsize, 2, 1, 1, 1)
        gl_main.addWidget(QtWidgets.QLabel('Total Lines:'), 3, 0, 1, 1)
        gl_main.addWidget(self.lbl_totlines, 3, 1, 1, 1)
        gl_main.addWidget(lbl_files, 4, 0, 1, 1)
        gl_main.addWidget(self.le_files, 4, 1, 1, 1)
        gl_main.addWidget(lbl_lines, 5, 0, 1, 1)
        gl_main.addWidget(self.le_lines, 5, 1, 1, 1)
        gl_main.addWidget(lbl_bytes, 6, 0, 1, 1)
        gl_main.addWidget(self.le_bytes, 6, 1, 1, 1)
        gl_main.addWidget(self.chk_allfiles, 7, 0, 1, 2)
        gl_main.addWidget(helpdocs, 8, 0, 1, 1)
        gl_main.addWidget(buttonbox, 8, 1, 1, 3)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        pb_ifile.pressed.connect(self.get_ifile)

        self.le_files.textChanged.connect(self.change_method)
        self.le_lines.textChanged.connect(self.change_method)
        self.le_bytes.textChanged.connect(self.change_method)
        self.cmb_method.currentIndexChanged.connect(self.change_method)

    def change_method(self):
        """Update fields when method changes."""
        method = self.cmb_method.currentText()

        totlines = int(self.lbl_totlines.text().replace(',', ''))
        totbytes = int(self.lbl_totsize.text().replace(',', ''))

        try:
            numfiles = int(self.le_files.text().replace(',', ''))
            numlines = int(self.le_lines.text().replace(',', ''))
            numbytes = int(self.le_bytes.text().replace(',', ''))
        except ValueError:
            return

        if method == 'Files':
            numlines = totlines // numfiles + 1
            numbytes = totbytes // numfiles + 1
            self.le_files.setEnabled(True)
            self.le_lines.setDisabled(True)
            self.le_bytes.setDisabled(True)
        elif method == 'Lines':
            numfiles = totlines // numlines + 1
            numbytes = totbytes // numfiles + 1
            self.le_files.setDisabled(True)
            self.le_lines.setEnabled(True)
            self.le_bytes.setDisabled(True)

        elif method == 'Bytes':
            numfiles = totbytes // numbytes + 1
            numlines = totlines // numfiles + 1
            self.le_files.setDisabled(True)
            self.le_lines.setDisabled(True)
            self.le_bytes.setEnabled(True)

        self.le_files.blockSignals(True)
        self.le_lines.blockSignals(True)
        self.le_bytes.blockSignals(True)

        self.le_files.setText(f'{numfiles:,}')
        self.le_lines.setText(f'{numlines:,}')
        self.le_bytes.setText(f'{numbytes:,}')

        self.le_files.blockSignals(False)
        self.le_lines.blockSignals(False)
        self.le_bytes.blockSignals(False)

    def get_ifile(self):
        """
        Get input file information.

        Returns
        -------
        None.

        """
        ext = ('Common formats (*.txt *.xyz *.csv);;')

        self.ifile, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.parent, 'Open File', '.', ext)

        if not self.ifile:
            return False

        self.le_ifile.setText(self.ifile)
        fsize = os.path.getsize(self.ifile)
        tlines = txtlinecnt(self.ifile)

        self.lbl_totsize.setText(f'{fsize:,}')
        self.lbl_totlines.setText(f'{tlines:,}')

        self.le_files.setValidator(QtGui.QIntValidator(1, fsize))
        self.le_lines.setValidator(QtGui.QIntValidator(1, tlines))
        self.le_bytes.setValidator(QtGui.QIntValidator(1, fsize))

        self.change_method()

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
            tmp = self.exec()
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
        self.saveobj(self.le_ifile)
        self.saveobj(self.le_files)
        self.saveobj(self.le_lines)
        self.saveobj(self.le_bytes)
        self.saveobj(self.chk_allfiles)
        self.saveobj(self.cmb_method)
        self.saveobj(self.lbl_totsize)
        self.saveobj(self.lbl_totlines)

    def acceptall(self):
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        None.

        """
        method = self.cmb_method.currentText()

        totlines = int(self.lbl_totlines.text().replace(',', ''))
        totbytes = int(self.lbl_totsize.text().replace(',', ''))

        try:
            numfiles = int(self.le_files.text().replace(',', ''))
            numlines = int(self.le_lines.text().replace(',', ''))
            numbytes = int(self.le_bytes.text().replace(',', ''))
        except ValueError:
            return

        if method == 'Bytes':
            num = numbytes
        elif method == 'Lines':
            num = numlines
        else:
            num = numfiles

        if self.chk_allfiles.isChecked():
            _, fext = os.path.splitext(self.ifile)
            fdir = os.path.dirname(self.ifile)
            ifiles = glob.glob(os.path.join(fdir, f'*{fext}'))
        else:
            ifiles = [self.ifile]

        for ifile in ifiles:
            self.showlog(f'Splitting {os.path.basename(ifile)}...')
            filesplit(ifile, num, method.lower(), showlog=self.showlog,
                      piter=self.piter)


def blanking(gdat, x, y, bdist, extent, dxy, nullvalue):
    """
    Blanks area further than a defined number of cells from input data.

    Parameters
    ----------
    gdat : numpy array
        grid data to blank.
    x : numpy array
        x coordinates.
    y : numpy array
        y coordinates.
    bdist : int
        Blanking distance in units for cell.
    extent : list
        extent of grid.
    dxy : float
        Cell size.
    Nullvalue : float
        Null or nodata value.

    Returns
    -------
    mask : numpy array
        Mask to be used for blanking.

    """
    if bdist is None:
        return gdat

    mask = np.zeros_like(gdat)

    points = np.transpose([x, y])

    for xy in points:
        col = int((xy[0]-extent[0])/dxy)
        row = int((xy[1]-extent[2])/dxy)

        mask[row, col] = 1

    dist = distance_transform_edt(np.logical_not(mask))
    mask = (dist > bdist)

    gdat[mask] = nullvalue

    return gdat


def cut_point(data, ifile, showlog=print):
    """
    Cuts a point dataset.

    Cut a point dataset using a shapefile.

    Parameters
    ----------
    data : GeoDataFrame
        GeoPandas GeoDataFrame
    ifile : str
        shapefile used to cut data

    Returns
    -------
    data : GeoDataFrame
        GeoPandas GeoDataFrame
    """
    gdf = gpd.read_file(ifile)
    gdf = gdf[gdf.geometry != None]

    if 'Polygon' not in gdf.geom_type.iloc[0]:
        showlog('No polygons in shapefile.')
        return None

    if data.crs != gdf.crs:
        showlog('Your shapefile does not have the same projection as the data '
                'to be clipped.')
        return None

    data = gpd.clip(data, gdf)

    return data


def txtlinecnt(filename):
    """
    Count lines in text file.

    Parameters
    ----------
    filename : str
        filename of text file.

    Returns
    -------
    int
        Total number of lines in a file.

    """
    with open(filename, 'rb') as f:
        bufgen = iter(partial(f.raw.read, 1024*1024), b'')
        linecnt = sum(buf.count(b'\n') for buf in bufgen)
    return linecnt


def filesplit(ifile, num, mode='bytes', showlog=print, piter=None):
    """
    Split an input file into a number of output files.

    Parameters
    ----------
    ifile : str
        Input filename.
    num : int
        Number of bytes or lines to split by.
    mode : str, optional
        Can be 'bytes', 'files' or 'lines'. The default is 'bytes'.
    showlog : function, optional
        Display information. The default is print.
    piter : iter, optional
        Progress iterator. The default is None.

    Returns
    -------
    None.

    """
    if piter is None:
        piter = ProgressBarText().iter

    fsize = os.path.getsize(ifile)
    fname, fext = os.path.splitext(ifile)

    if mode == 'files':
        numfiles = num
        numcnt = fsize // num + 1
    elif mode == 'bytes':
        numcnt = num
        numfiles = fsize // num + 1
    elif mode == 'lines':
        totlines = txtlinecnt(ifile)
        numfiles = totlines // num + 1
        numcnt = num

    txt = None
    with open(ifile, encoding='utf-8') as reader:
        for i in piter(range(numfiles)):
            if txt == '':
                continue

            with open(f'{fname}_{i+1}{fext}', 'w') as writer:
                fread = 0
                while fread < numcnt:
                    txt = reader.readline()
                    if txt == '':
                        break
                    if mode == 'lines':
                        fread += 1
                    else:
                        fread += len(txt)

                    writer.write(txt)


def gridxyz(x, y, z, dxy, nullvalue=1e+20, method='Nearest Neighbour',
            bdist=4.0, showlog=print):
    """
    Grid xyz data.

    Parameters
    ----------
    x : numpy array
        X coordinate values.
    y : numpy array
        Y coordinate values.
    z : numpy array
        Z or data values.
    dxy : float
        Grid cell size, in distance units.
    nullvalue : float, optional
        null or nodata value. The default is 1e+20.
    method : str, optional
        Gridding method. The default is 'Nearest Neighbour'.
    bdist : float, optional
        Blanking distance. The default is 4.0.
    showlog : function, optional
        Display information. The default is print.

    Returns
    -------
    dat : PyGMI Data.
        Output raster dataset.

    """
    if bdist is not None and bdist < 1:
        bdist = None
        showlog('Blanking distance too small.')

    if method == 'Minimum Curvature':
        gdat = minc(x, y, z, dxy, showlog=showlog,
                    bdist=bdist)
        gdat = np.ma.filled(gdat, fill_value=nullvalue)
    else:
        extent = np.array([x.min(), x.max(), y.min(), y.max()])

        xxx = np.arange(extent[0], extent[1]+dxy/2, dxy)
        yyy = np.arange(extent[2], extent[3]+dxy/2, dxy)

        xxx, yyy = np.meshgrid(xxx, yyy)

        points = np.transpose([x.flatten(), y.flatten()])

        if method == 'Nearest Neighbour':
            gdat = griddata(points, z, (xxx, yyy), method='nearest')
        elif method == 'Linear':
            gdat = griddata(points, z, (xxx, yyy), method='linear',
                            fill_value=nullvalue)
        elif method == 'Cubic':
            gdat = griddata(points, z, (xxx, yyy), method='cubic',
                            fill_value=nullvalue)

        gdat = blanking(gdat, x, y, bdist, extent, dxy, nullvalue)
        gdat = gdat[::-1]
    gdat = np.ma.masked_equal(gdat, nullvalue)

    # Create dataset
    dat = Data()
    dat.data = gdat
    dat.nodata = nullvalue

    rows, _ = dat.data.shape

    left = x.min() - dxy/2
    top = y.min() + dxy*rows - dxy/2

    dat.set_transform(dxy, left, dxy, top)

    return dat


def lltomap(lat, lon):
    """
    Convert a latitude and longitude to a 1:50,000 sheet name.

    Parameters
    ----------
    lat : float
        Latitude.
    lon : float
        Longitude.

    Returns
    -------
    mapsheet : str
        Map sheet number.

    """
    cdict = {(0, 0): 'A',
             (0, 1): 'B',
             (1, 0): 'C',
             (1, 1): 'D'}

    latfrac = abs(lat) % 1
    lonfrac = lon % 1

    latf = latfrac // .5
    lonf = lonfrac // .5
    letter1 = cdict[(latf, lonf)]

    latf = latfrac % .5
    lonf = lonfrac % .5

    latf = latf // .25
    lonf = lonf // .25

    letter2 = cdict[(latf, lonf)]

    mapsheet = f'{int(abs(lat))}{int(lon)}{letter1}{letter2}'

    return mapsheet


def maptobounds(mapsheet, crs_to=None, showlog=print):
    """
    Convert a South African map sheet name to bounds.

    Parameters
    ----------
    mapsheet : str
        Map sheet number. Four numbers and up to two letters denoting NE corner
        in latitude and longitude and quadrants (A to D). Eg, 2928AB is
        latitude 29, longitude 28, quadrant B of quadrant A.
    crs_to : CRS, optional
        Destination projection. The default is None.
    showlog : function, optional
        Display information. The default is print.

    Returns
    -------
    bounds : list
        output bounds.

    """
    i = mapsheet
    try:
        lat = float(i[:2])
        lon = float(i[2:4])
    except ValueError:
        showlog('Invalid Map Sheet Number')
        return None

    q1 = 'A'
    q2 = 'A'
    latincr = 1
    lonincr = 2
    if len(i) > 4:
        q1 = i[4:5]
        lonincr = .5
        latincr = .5
    if len(i) > 5:
        q2 = i[5:6]
        lonincr = .25
        latincr = .25

    qlat1 = {'A': 0.,
             'B': 0.,
             'C': 0.5,
             'D': 0.5}

    qlon1 = {'A': 0.,
             'B': 0.5,
             'C': 0.,
             'D': 0.5}

    qlat2 = {'A': 0.,
             'B': 0.,
             'C': 0.25,
             'D': 0.25}

    qlon2 = {'A': 0.,
             'B': 0.25,
             'C': 0.,
             'D': 0.25}

    lat = -(lat + qlat1[q1] + qlat2[q2])
    lon = lon + qlon1[q1] + qlon2[q2]

    xmin = lon
    ymin = lat-latincr
    xmax = lon+lonincr
    ymax = lat

    if crs_to is not None:
        crs_from = CRS.from_epsg(4326)
        transformer = Transformer.from_crs(crs_from, crs_to, always_xy=True)
        xmin, ymin = transformer.transform(xmin, ymin)
        xmax, ymax = transformer.transform(xmax, ymax)

    bounds = (xmin, ymin, xmax, ymax)

    return bounds


def quickgrid(x, y, z, dxy, numits=4, showlog=print):
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
    numits : int
        number of iterations. By default its 4. If this is negative, a maximum
        will be calculated and used.
    showlog : function, optional
        Routine to show text messages. The default is print.

    Returns
    -------
    newz : numpy array
        M x N array of z values
    """
    showlog('Creating Grid')
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

        showlog('Iteration done: '+str(j+1)+' of '+str(numits))

    showlog('Finished!')

    newz = np.ma.array(zfin)
    newz.mask = newmask
    return newz


def reprojxy(x, y, iwkt, owkt, showlog=print):
    """
    Reproject x and y coordinates.

    Parameters
    ----------
    x : numpy array or float
        x coordinates
    y : numpy array or float
        y coordinates
    iwkt : str, int
        Input wkt description or EPSG code (int)
    owkt : str, int
        Output wkt description or EPSG code (int)

    Returns
    -------
    xout : numpy array
        x coordinates.
    yout : numpy array
        y coordinates.

    """
    if isinstance(iwkt, int):
        crs_from = CRS.from_epsg(iwkt)
    else:
        crs_from = CRS.from_wkt(iwkt)

    if isinstance(owkt, int):
        crs_to = CRS.from_epsg(owkt)
    else:
        crs_to = CRS.from_wkt(owkt)

    try:
        transformer = Transformer.from_crs(crs_from, crs_to, always_xy=True)
    except:
        showlog('Problem reprojecting. Aborting.')
        return None, None

    xout, yout = transformer.transform(x, y)

    return xout, yout


def _testfn():
    """Test routine."""
    import sys
    from pygmi.vector.iodefs import ImportXYZ

    app = QtWidgets.QApplication(sys.argv)

    ifile = r"D:\Workdata\PyGMI Test Data\Vector\Line Data\SPECARCHIVE.XYZ"

    IO = ImportXYZ()
    IO.ifile = ifile
    IO.filt = 'Geosoft XYZ (*.xyz)'
    IO.settings(True)

    MD = Metadata()
    MD.indata = IO.outdata
    MD.run()

    app.exec()


def _testfn_pointcut():
    """Test routine."""
    import sys
    from pygmi.vector.iodefs import ImportXYZ

    app = QtWidgets.QApplication(sys.argv)

    ifile = r"D:\Workdata\PyGMI Test Data\Vector\linecut\test2.csv"
    sfile = r"D:\Workdata\PyGMI Test Data\Vector\linecut\test2_cut_outline.shp"

    IO = ImportXYZ()
    IO.ifile = ifile
    IO.filt = 'Comma Delimited (*.csv)'
    IO.settings(True)

    DR = PointCut()
    DR.indata = IO.outdata
    DR.ifile = sfile
    DR.settings(True)


def _testfn_filesplit():
    """Test Routine."""
    import sys

    ifile = r"D:\fsplit\bushveld_magarchive.xyz"

    app = QtWidgets.QApplication(sys.argv)

    # num = os.path.getsize(ifile)//10
    # num = 10
    # num = 772740

    # filesplit(ifile, num, mode='lines')

    app = TextFileSplit()
    app.settings()


if __name__ == "__main__":
    _testfn()
