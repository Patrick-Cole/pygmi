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
from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
# import matplotlib.path as mplPath
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt
import geopandas as gpd
from pyproj import CRS, Transformer

from pygmi import menu_default
from pygmi.raster.dataprep import GroupProj
from pygmi.raster.datatypes import Data
from pygmi.vector.minc import minc
from pygmi.misc import BasicModule, ContextModule


class PointCut(BasicModule):
    """
    Cut Data using shapefiles.

    This class cuts point datasets using a boundary defined by a polygon
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
        if 'Vector' in self.indata:
            data = copy.deepcopy(self.indata['Vector'][0])
        else:
            self.showlog('No point data')
            return False

        # nodialog = False
        if not nodialog:
            ext = 'Shape file (*.shp)'
            self.ifile, _ = QtWidgets.QFileDialog.getOpenFileName(
                self.parent, 'Open Shape File', '.', ext)
            if self.ifile == '':
                return False

        os.chdir(os.path.dirname(self.ifile))
        data = cut_point(data, self.ifile)

        if data is None:
            err = ('There was a problem importing the shapefile. Please make '
                   'sure you have at all the individual files which make up '
                   'the shapefile.')
            QtWidgets.QMessageBox.warning(self.parent, 'Error', err,
                                          QtWidgets.QMessageBox.Ok)
            return False

        if self.pbar is not None:
            self.pbar.to_max()
        self.outdata['Vector'] = [data]

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
        self.ifile = projdata['ifile']

        chk = self.settings(True)
        return chk

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        projdata : dictionary
            Project data to be saved to JSON project file.

        """
        projdata = {}

        projdata['ifile'] = self.ifile

        return projdata


class DataGrid(BasicModule):
    """
    Grid Point Data.

    This class grids point data using a nearest neighbourhood technique.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dxy = None
        self.dataid_text = None

        self.dsb_dxy = QtWidgets.QLineEdit('1.0')
        self.dsb_null = QtWidgets.QLineEdit('0.0')
        self.bdist = QtWidgets.QLineEdit('4.0')

        self.dataid = QtWidgets.QComboBox()
        self.grid_method = QtWidgets.QComboBox()
        self.label_rows = QtWidgets.QLabel('Rows: 0')
        self.label_cols = QtWidgets.QLabel('Columns: 0')
        self.label_bdist = QtWidgets.QLabel('Blanking Distance:')

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
        helpdocs = menu_default.HelpButton('pygmi.vector.dataprep.datagrid')
        label_band = QtWidgets.QLabel('Column to Grid:')
        label_dxy = QtWidgets.QLabel('Cell Size:')
        label_null = QtWidgets.QLabel('Null Value:')
        label_method = QtWidgets.QLabel('Gridding Method:')

        val = QtGui.QDoubleValidator(0.0000001, 9999999999.0, 9)
        val.setNotation(QtGui.QDoubleValidator.ScientificNotation)
        val.setLocale(QtCore.QLocale(QtCore.QLocale.C))

        self.dsb_dxy.setValidator(val)
        self.dsb_null.setValidator(val)

        self.grid_method.addItems(['Nearest Neighbour', 'Linear', 'Cubic',
                                   'Minimum Curvature'])

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Dataset Gridding')

        gridlayout_main.addWidget(label_method, 0, 0, 1, 1)
        gridlayout_main.addWidget(self.grid_method, 0, 1, 1, 1)
        gridlayout_main.addWidget(label_dxy, 1, 0, 1, 1)
        gridlayout_main.addWidget(self.dsb_dxy, 1, 1, 1, 1)
        gridlayout_main.addWidget(self.label_rows, 2, 0, 1, 2)
        gridlayout_main.addWidget(self.label_cols, 3, 0, 1, 2)
        gridlayout_main.addWidget(label_band, 4, 0, 1, 1)
        gridlayout_main.addWidget(self.dataid, 4, 1, 1, 1)
        gridlayout_main.addWidget(label_null, 5, 0, 1, 1)
        gridlayout_main.addWidget(self.dsb_null, 5, 1, 1, 1)
        gridlayout_main.addWidget(self.label_bdist, 6, 0, 1, 1)
        gridlayout_main.addWidget(self.bdist, 6, 1, 1, 1)
        gridlayout_main.addWidget(helpdocs, 7, 0, 1, 1)
        gridlayout_main.addWidget(buttonbox, 7, 1, 1, 3)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        self.dsb_dxy.textChanged.connect(self.dxy_change)

    def dxy_change(self):
        """
        When dxy is changed on the interface, this updates rows and columns.

        Returns
        -------
        None.

        """
        txt = str(self.dsb_dxy.text())
        if txt.replace('.', '', 1).isdigit():
            self.dxy = float(self.dsb_dxy.text())
        else:
            return

        data = self.indata['Vector'][0]

        x = data.geometry.x.values
        y = data.geometry.y.values

        cols = round(x.ptp()/self.dxy)
        rows = round(y.ptp()/self.dxy)

        self.label_rows.setText('Rows: '+str(rows))
        self.label_cols.setText('Columns: '+str(cols))

    def grid_method_change(self):
        """
        When grid method is changed, this updated hidden controls.

        Returns
        -------
        None.

        """
        if self.grid_method.currentText() == 'Minimum Curvature':
            self.label_bdist.show()
            self.bdist.show()
        else:
            self.label_bdist.hide()
            self.bdist.hide()

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

        self.dataid.clear()

        filt = ((data.columns != 'geometry') &
                (data.columns != 'line'))

        cols = list(data.columns[filt])
        self.dataid.addItems(cols)

        if self.dataid_text is None:
            self.dataid_text = self.dataid.currentText()
        if self.dataid_text in cols:
            self.dataid.setCurrentText(self.dataid_text)

        if self.dxy is None:
            x = data.geometry.x.values
            y = data.geometry.y.values

            dx = x.ptp()/np.sqrt(x.size)
            dy = y.ptp()/np.sqrt(y.size)
            self.dxy = max(dx, dy)
            self.dxy = min([x.ptp(), y.ptp(), self.dxy])

        self.dsb_dxy.setText(f'{self.dxy:.8f}')
        self.dxy_change()

        if not nodialog:
            tmp = self.exec_()
            if tmp != 1:
                return False

        try:
            float(self.dsb_dxy.text())
            float(self.dsb_null.text())
            float(self.bdist.text())
        except ValueError:
            self.showlog('Value Error')
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
        self.dsb_dxy.textChanged.disconnect()
        self.dataid_text = projdata['band']
        self.dsb_dxy.setText(projdata['dxy'])
        self.dsb_null.setText(projdata['nullvalue'])
        self.bdist.setText(projdata['bdist'])

        self.dsb_dxy.textChanged.connect(self.dxy_change)

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

        projdata['dxy'] = self.dsb_dxy.text()
        projdata['nullvalue'] = self.dsb_null.text()
        projdata['bdist'] = self.bdist.text()
        projdata['band'] = self.dataid_text

        return projdata

    def acceptall(self):
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        None.

        """
        dxy = float(self.dsb_dxy.text())
        method = self.grid_method.currentText()
        nullvalue = float(self.dsb_null.text())
        bdist = float(self.bdist.text())
        data = self.indata['Vector'][0]
        dataid = self.dataid.currentText()
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
            indx = self.in_proj.combodatum.findText(r'WGS 84')
            self.in_proj.combodatum.setCurrentIndex(indx)
            self.orig_wkt = self.in_proj.wkt
        else:
            self.in_proj.set_current(self.orig_wkt)

        if self.targ_wkt is None:
            indx = self.in_proj.combodatum.findText(r'WGS 84')
            self.out_proj.combodatum.setCurrentIndex(indx)
            self.targ_wkt = self.out_proj.wkt
        else:
            self.out_proj.set_current(self.targ_wkt)

        if not nodialog:
            tmp = self.exec_()

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
        self.banddata = {}
        self.dataid = {}
        self.oldtxt = ''

        self.combobox_bandid = QtWidgets.QComboBox()
        self.pb_rename_id = QtWidgets.QPushButton('Rename Column Name')
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
        self.lbl_dtype = QtWidgets.QLabel()

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
        label_dtype = QtWidgets.QLabel('Data Type:')

        sizepolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                           QtWidgets.QSizePolicy.Expanding)
        groupbox.setSizePolicy(sizepolicy)
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Vector Dataset Metadata')

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
        gridlayout.addWidget(label_dtype, 11, 0, 1, 1)
        gridlayout.addWidget(self.lbl_dtype, 11, 1, 1, 1)

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
            if self.txt_null.text().lower() != 'none':
                odata.nodata = float(self.txt_null.text())
            left = float(self.dsb_tlx.text())
            top = float(self.dsb_tly.text())
            xdim = float(self.dsb_xdim.text())
            ydim = float(self.dsb_ydim.text())

            odata.set_transform(xdim, left, ydim, top)

        except ValueError:
            self.showlog('Value error - abandoning changes')

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
        self.lbl_dtype.setText(str(idata.data.dtype))

    def run(self):
        """
        Entry point to start this routine.

        Returns
        -------
        tmp : bool
            True if successful, False otherwise.

        """
        breakpoint()

        bandid = []
        if self.indata['Vector'][0].crs is None:
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
        self.lbl_dtype.setText(str(idata.data.dtype))

        self.update_vals()

        tmp = self.exec_()

        if tmp != 1:
            return False

        self.acceptall()

        return True




def cut_point(data, ifile):
    """
    Cuts a point dataset.

    Cut a point dataset using a shapefile.

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
    gdf = gpd.read_file(ifile)
    gdf = gdf[gdf.geometry != None]

    if 'Polygon' not in gdf.geom_type.iloc[0]:
        return None
    # if 'Polygon' in gdf.geom_type.iloc[0]:
    #     dat = gdf.geometry.iloc[0]

    data = gpd.clip(data, gdf)

    # points = list(dat.exterior.coords)

    # bbpath = mplPath.Path(points)

    # chk = bbpath.contains_points(np.transpose([data.geometry.x.values,
    #                                            data.geometry.y.values]))

    # data = data[chk]

    return data


def gridxyz(x, y, z, dxy, nullvalue=1e+20, method='Nearest Neighbour',
            bdist=4.0, showlog=print):
    """Grid data."""
    if bdist < 1:
        bdist = None
        showlog('Blanking distance too small.')

    if method == 'Minimum Curvature':
        gdat = minc(x, y, z, dxy, showlog=showlog,
                    bdist=bdist)
        gdat = np.ma.filled(gdat, fill_value=nullvalue)
    else:
        extent = np.array([x.min(), x.max(), y.min(), y.max()])

        rows = int((extent[3] - extent[2])/dxy+1)
        cols = int((extent[1] - extent[0])/dxy+1)

        xxx = np.linspace(extent[0], extent[1], cols)
        yyy = np.linspace(extent[2], extent[3], rows)
        xxx, yyy = np.meshgrid(xxx, yyy)

        points = np.transpose([x.flatten(), y.flatten()])

        if method == 'Nearest Neighbour':
            gdat = griddata(points, z, (xxx, yyy), method='nearest')
            gdat = blanking(gdat, x, y, bdist, extent, dxy, nullvalue)
        elif method == 'Linear':
            gdat = griddata(points, z, (xxx, yyy), method='linear',
                            fill_value=nullvalue)
            gdat = blanking(gdat, x, y, bdist, extent, dxy, nullvalue)
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
    # dat.dataid = self.dataid.currentText()

    rows, _ = dat.data.shape

    left = x.min()
    top = y.min() + dxy*rows

    dat.set_transform(dxy, left, dxy, top)

    return dat


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


def reprojxy(x, y, iwkt, owkt):
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

    transformer = Transformer.from_crs(crs_from, crs_to, always_xy=True)
    xout, yout = transformer.transform(x, y)

    return xout, yout


def _testfn():
    """Test routine."""
    import sys
    import matplotlib.pyplot as plt
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

    # DG = DataGrid()
    # DG.indata = IO.outdata
    # DG.settings()

    # dat = DG.outdata['Raster'][0].data

    # plt.imshow(dat)
    # plt.show()

    # DR = DataReproj()
    # DR.indata = IO.outdata
    # DR.settings()


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


if __name__ == "__main__":
    _testfn()
