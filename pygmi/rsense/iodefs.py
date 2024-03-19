# -----------------------------------------------------------------------------
# Name:        iodefs.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2020 Council for Geoscience
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
"""Import remote sensing data."""

import os
import sys
import math
import xml.etree.ElementTree as ET
import glob
import tarfile
import zipfile
import datetime
from collections import defaultdict
import warnings

from PyQt5 import QtWidgets, QtCore
import numpy as np
import numexpr as ne
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point
import rasterio
from rasterio import Affine
from natsort import natsorted
from pyproj.crs import CRS

from pygmi import menu_default
from pygmi.raster.datatypes import Data, RasterMeta
from pygmi.raster.iodefs import get_raster, export_raster
from pygmi.raster.ginterp import histcomp, norm255, norm2, currentshader
from pygmi.misc import ProgressBarText, ContextModule, BasicModule
from pygmi.raster.dataprep import lstack, data_reproject
from pygmi.vector.dataprep import reprojxy

warnings.filterwarnings("ignore",
                        category=rasterio.errors.NotGeoreferencedWarning)

EDIST = {1: 0.98331, 2: 0.9833, 3: 0.9833, 4: 0.9833,
         5: 0.9833, 6: 0.98332, 7: 0.98333, 8: 0.98335,
         9: 0.98338, 10: 0.98341, 11: 0.98345, 12: 0.98349,
         13: 0.98354, 14: 0.98359, 15: 0.98365, 16: 0.98371,
         17: 0.98378, 18: 0.98385, 19: 0.98393, 20: 0.98401,
         21: 0.9841, 22: 0.98419, 23: 0.98428, 24: 0.98439,
         25: 0.98449, 26: 0.9846, 27: 0.98472, 28: 0.98484,
         29: 0.98496, 30: 0.98509, 31: 0.98523, 32: 0.98536,
         33: 0.98551, 34: 0.98565, 35: 0.9858, 36: 0.98596,
         37: 0.98612, 38: 0.98628, 39: 0.98645, 40: 0.98662,
         41: 0.9868, 42: 0.98698, 43: 0.98717, 44: 0.98735,
         45: 0.98755, 46: 0.98774, 47: 0.98794, 48: 0.98814,
         49: 0.98835, 50: 0.98856, 51: 0.98877, 52: 0.98899,
         53: 0.98921, 54: 0.98944, 55: 0.98966, 56: 0.98989,
         57: 0.99012, 58: 0.99036, 59: 0.9906, 60: 0.99084,
         61: 0.99108, 62: 0.99133, 63: 0.99158, 64: 0.99183,
         65: 0.99208, 66: 0.99234, 67: 0.9926, 68: 0.99286,
         69: 0.99312, 70: 0.99339, 71: 0.99365, 72: 0.99392,
         73: 0.99419, 74: 0.99446, 75: 0.99474, 76: 0.99501,
         77: 0.99529, 78: 0.99556, 79: 0.99584, 80: 0.99612,
         81: 0.9964, 82: 0.99669, 83: 0.99697, 84: 0.99725,
         85: 0.99754, 86: 0.99782, 87: 0.99811, 88: 0.9984,
         89: 0.99868, 90: 0.99897, 91: 0.99926, 92: 0.99954,
         93: 0.99983, 94: 1.00012, 95: 1.00041, 96: 1.00069,
         97: 1.00098, 98: 1.00127, 99: 1.00155, 100: 1.00184,
         101: 1.00212, 102: 1.0024, 103: 1.00269, 104: 1.00297,
         105: 1.00325, 106: 1.00353, 107: 1.00381, 108: 1.00409,
         109: 1.00437, 110: 1.00464, 111: 1.00492, 112: 1.00519,
         113: 1.00546, 114: 1.00573, 115: 1.006, 116: 1.00626,
         117: 1.00653, 118: 1.00679, 119: 1.00705, 120: 1.00731,
         121: 1.00756, 122: 1.00781, 123: 1.00806, 124: 1.00831,
         125: 1.00856, 126: 1.0088, 127: 1.00904, 128: 1.00928,
         129: 1.00952, 130: 1.00975, 131: 1.00998, 132: 1.0102,
         133: 1.01043, 134: 1.01065, 135: 1.01087, 136: 1.01108,
         137: 1.01129, 138: 1.0115, 139: 1.0117, 140: 1.01191,
         141: 1.0121, 142: 1.0123, 143: 1.01249, 144: 1.01267,
         145: 1.01286, 146: 1.01304, 147: 1.01321, 148: 1.01338,
         149: 1.01355, 150: 1.01371, 151: 1.01387, 152: 1.01403,
         153: 1.01418, 154: 1.01433, 155: 1.01447, 156: 1.01461,
         157: 1.01475, 158: 1.01488, 159: 1.015, 160: 1.01513,
         161: 1.01524, 162: 1.01536, 163: 1.01547, 164: 1.01557,
         165: 1.01567, 166: 1.01577, 167: 1.01586, 168: 1.01595,
         169: 1.01603, 170: 1.0161, 171: 1.01618, 172: 1.01625,
         173: 1.01631, 174: 1.01637, 175: 1.01642, 176: 1.01647,
         177: 1.01652, 178: 1.01656, 179: 1.01659, 180: 1.01662,
         181: 1.01665, 182: 1.01667, 183: 1.01668, 184: 1.0167,
         185: 1.0167, 186: 1.0167, 187: 1.0167, 188: 1.01669,
         189: 1.01668, 190: 1.01666, 191: 1.01664, 192: 1.01661,
         193: 1.01658, 194: 1.01655, 195: 1.0165, 196: 1.01646,
         197: 1.01641, 198: 1.01635, 199: 1.01629, 200: 1.01623,
         201: 1.01616, 202: 1.01609, 203: 1.01601, 204: 1.01592,
         205: 1.01584, 206: 1.01575, 207: 1.01565, 208: 1.01555,
         209: 1.01544, 210: 1.01533, 211: 1.01522, 212: 1.0151,
         213: 1.01497, 214: 1.01485, 215: 1.01471, 216: 1.01458,
         217: 1.01444, 218: 1.01429, 219: 1.01414, 220: 1.01399,
         221: 1.01383, 222: 1.01367, 223: 1.01351, 224: 1.01334,
         225: 1.01317, 226: 1.01299, 227: 1.01281, 228: 1.01263,
         229: 1.01244, 230: 1.01225, 231: 1.01205, 232: 1.01186,
         233: 1.01165, 234: 1.01145, 235: 1.01124, 236: 1.01103,
         237: 1.01081, 238: 1.0106, 239: 1.01037, 240: 1.01015,
         241: 1.00992, 242: 1.00969, 243: 1.00946, 244: 1.00922,
         245: 1.00898, 246: 1.00874, 247: 1.0085, 248: 1.00825,
         249: 1.008, 250: 1.00775, 251: 1.0075, 252: 1.00724,
         253: 1.00698, 254: 1.00672, 255: 1.00646, 256: 1.0062,
         257: 1.00593, 258: 1.00566, 259: 1.00539, 260: 1.00512,
         261: 1.00485, 262: 1.00457, 263: 1.0043, 264: 1.00402,
         265: 1.00374, 266: 1.00346, 267: 1.00318, 268: 1.0029,
         269: 1.00262, 270: 1.00234, 271: 1.00205, 272: 1.00177,
         273: 1.00148, 274: 1.00119, 275: 1.00091, 276: 1.00062,
         277: 1.00033, 278: 1.00005, 279: 0.99976, 280: 0.99947,
         281: 0.99918, 282: 0.9989, 283: 0.99861, 284: 0.99832,
         285: 0.99804, 286: 0.99775, 287: 0.99747, 288: 0.99718,
         289: 0.9969, 290: 0.99662, 291: 0.99634, 292: 0.99605,
         293: 0.99577, 294: 0.9955, 295: 0.99522, 296: 0.99494,
         297: 0.99467, 298: 0.9944, 299: 0.99412, 300: 0.99385,
         301: 0.99359, 302: 0.99332, 303: 0.99306, 304: 0.99279,
         305: 0.99253, 306: 0.99228, 307: 0.99202, 308: 0.99177,
         309: 0.99152, 310: 0.99127, 311: 0.99102, 312: 0.99078,
         313: 0.99054, 314: 0.9903, 315: 0.99007, 316: 0.98983,
         317: 0.98961, 318: 0.98938, 319: 0.98916, 320: 0.98894,
         321: 0.98872, 322: 0.98851, 323: 0.9883, 324: 0.98809,
         325: 0.98789, 326: 0.98769, 327: 0.9875, 328: 0.98731,
         329: 0.98712, 330: 0.98694, 331: 0.98676, 332: 0.98658,
         333: 0.98641, 334: 0.98624, 335: 0.98608, 336: 0.98592,
         337: 0.98577, 338: 0.98562, 339: 0.98547, 340: 0.98533,
         341: 0.98519, 342: 0.98506, 343: 0.98493, 344: 0.98481,
         345: 0.98469, 346: 0.98457, 347: 0.98446, 348: 0.98436,
         349: 0.98426, 350: 0.98416, 351: 0.98407, 352: 0.98399,
         353: 0.98391, 354: 0.98383, 355: 0.98376, 356: 0.9837,
         357: 0.98363, 358: 0.98358, 359: 0.98353, 360: 0.98348,
         361: 0.98344, 362: 0.9834, 363: 0.98337, 364: 0.98335,
         365: 0.98333, 366: 0.98331}

K1 = [3040.136402, 2482.375199, 1935.060183, 866.468575, 641.326517]
K2 = [1735.337945, 1666.398761, 1585.420044, 1350.069147, 1271.221673]
ESUN = [1848, 1549, 1114, 225.4, 86.63, 81.85, 74.85, 66.49, 59.85]


class ImportData(BasicModule):
    """Import Data - Interfaces with rasterio routines."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.filt = ''
        self.is_import = True

        self.le_sfile = QtWidgets.QLineEdit('')
        self.lw_tnames = QtWidgets.QListWidget()
        self.lbl_ftype = QtWidgets.QLabel('File Type:')
        self.cb_ensuresutm = QtWidgets.QCheckBox('Ensure WGS84 UTM is for '
                                                 'southern hemisphere')

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        pb_sfile = QtWidgets.QPushButton(' Filename')

        pixmapi = QtWidgets.QStyle.SP_DialogOpenButton
        icon = self.style().standardIcon(pixmapi)
        pb_sfile.setIcon(icon)
        pb_sfile.setStyleSheet('text-align:left;')

        self.setWindowTitle('Import Satellite Data')
        self.cb_ensuresutm.setChecked(True)

        gl_1 = QtWidgets.QGridLayout(self)

        self.lw_tnames.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)

        gl_1.addWidget(pb_sfile, 1, 0, 1, 1)
        gl_1.addWidget(self.le_sfile, 1, 1, 1, 1)
        gl_1.addWidget(self.lbl_ftype, 2, 0, 1, 2)
        gl_1.addWidget(self.lw_tnames, 3, 0, 1, 2)
        gl_1.addWidget(self.cb_ensuresutm, 4, 0, 1, 2)

        buttonbox = QtWidgets.QDialogButtonBox()
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(
            QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)

        gl_1.addWidget(buttonbox, 9, 0, 1, 2)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        pb_sfile.pressed.connect(self.get_sfile)

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
        self.lw_tnames.clear()
        self.indata['Raster'] = []
        self.le_sfile.setText('')
        self.lbl_ftype.setText('File Type:')

        if not nodialog:
            tmp = self.exec()

            if tmp != 1:
                return tmp

        tnames = []
        for i in range(self.lw_tnames.count()):
            item = self.lw_tnames.item(i)
            if item.isSelected():
                tnames.append(str(item.text()))

        if not tnames:
            return False

        os.chdir(os.path.dirname(self.ifile))

        dat = get_data(self.ifile, self.piter, self.showlog, tnames)

        if dat is None:
            QtWidgets.QMessageBox.warning(self.parent, 'Error',
                                          'Could not import the data.',
                                          QtWidgets.QMessageBox.Ok)
            return False

        if self.cb_ensuresutm.isChecked() is True:
            dat = utm_to_south(dat)

        self.outdata['Raster'] = dat

        return True

    def get_sfile(self):
        """Get the satellite filename."""
        self.lw_tnames.clear()
        self.indata['Raster'] = []
        self.le_sfile.setText('')
        self.lbl_ftype.setText('File Type:')

        ext = ('Common formats (*.zip *.hdf *.tar *.tar.gz *.xml *.h5);;')

        self.ifile, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.parent, 'Open File', '.', ext)

        if not self.ifile:
            return False

        self.le_sfile.setText(self.ifile)

        self.indata['Raster'] = get_data(self.ifile, self.piter,
                                         self.showlog, metaonly=True)

        if self.indata['Raster'] is None:
            self.showlog('Error: could not import data.')
            self.le_sfile.setText('')
            self.lbl_ftype.setText('File Type: Unknown')
            return False

        tmp = []
        for i in self.indata['Raster']:
            tmp.append(i.dataid)

        self.lw_tnames.clear()
        self.lw_tnames.addItems(tmp)

        for i in range(self.lw_tnames.count()):
            item = self.lw_tnames.item(i)

            if item.text()[0] == 'B':
                item.setSelected(True)

        # If nothing is selected, then select everything.
        if not self.lw_tnames.selectedItems():
            for i in range(self.lw_tnames.count()):
                self.lw_tnames.item(i).setSelected(True)

        instr = self.indata['Raster'][0].metadata['Raster']['Sensor']

        self.lbl_ftype.setText(f' File Type: {instr}')

        return True

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.ifile)
        self.saveobj(self.filt)
        self.saveobj(self.le_sfile)
        self.saveobj(self.lbl_ftype)
        self.saveobj(self.cb_ensuresutm)
        self.saveobj(self.lw_tnames)


class ImportBatch(BasicModule):
    """
    Batch Import Data Interface.

    This does not actually import data, but rather defines a list of datasets
    to be used by other routines.

    Attributes
    ----------
    idir : str
        Input directory.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.idir = ''
        self.tnames = None
        self.filelist = []
        self.bands = {}
        self.tnames = {}
        self.oldsensor = None
        self.is_import = True

        self.cmb_sensor = QtWidgets.QComboBox()
        self.le_sfile = QtWidgets.QLineEdit('')
        self.lw_tnames = QtWidgets.QListWidget()
        self.lbl_ftype = QtWidgets.QLabel('File Type:')
        self.cb_ensuresutm = QtWidgets.QCheckBox('Ensure WGS84 UTM is for '
                                                 'southern hemisphere')

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        pb_sfile = QtWidgets.QPushButton(' Directory')

        pixmapi = QtWidgets.QStyle.SP_DialogOpenButton
        icon = self.style().standardIcon(pixmapi)
        pb_sfile.setIcon(icon)
        pb_sfile.setStyleSheet('text-align:left;')

        self.setWindowTitle('Import Batch Data')
        self.cb_ensuresutm.setChecked(True)

        gl_1 = QtWidgets.QGridLayout(self)

        self.lw_tnames.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)

        gl_1.addWidget(pb_sfile, 1, 0, 1, 1)
        gl_1.addWidget(self.le_sfile, 1, 1, 1, 1)
        gl_1.addWidget(self.lbl_ftype, 2, 0, 1, 1)
        gl_1.addWidget(self.cmb_sensor, 2, 1, 1, 1)
        gl_1.addWidget(self.lw_tnames, 3, 0, 1, 2)
        gl_1.addWidget(self.cb_ensuresutm, 4, 0, 1, 2)

        buttonbox = QtWidgets.QDialogButtonBox()
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(
            QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)

        gl_1.addWidget(buttonbox, 9, 0, 1, 2)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        pb_sfile.pressed.connect(self.get_sfile)
        self.cmb_sensor.currentIndexChanged.connect(self.setsensor)

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
        if self.idir != '':
            self.get_sfile(True)

        if not nodialog or self.idir == '':
            tmp = self.exec()

            if tmp != 1:
                return tmp

        if not self.filelist:
            QtWidgets.QMessageBox.warning(self.parent, 'Error',
                                          'No valid files in the directory.',
                                          QtWidgets.QMessageBox.Ok)
            return False

        self.setsensor()

        for i in self.filelist:
            i.to_sutm = self.cb_ensuresutm.isChecked()

        self.outdata['RasterFileList'] = self.filelist

        return True

    def get_sfile(self, nodialog=False):
        """Get the satellite filenames."""
        if not nodialog:
            self.idir = QtWidgets.QFileDialog.getExistingDirectory(
                self.parent, 'Select Directory')

            if not self.idir:
                return False

        self.le_sfile.setText(self.idir)

        types = ['*.tif', '*.hdr', '*.hdf', '*.zip', '*.tar', '*.tar.gz',
                 '*[!aux].xml', '*.h5', '*.SAFE/MTD_MSIL2A.xml']

        allfiles = []
        for i in types:
            allfiles += glob.glob(os.path.join(self.idir, i))

        allfiles = consolidate_aster_list(allfiles)

        self.bands = {}
        self.tnames = {}
        self.filelist = []
        for ifile in self.piter(allfiles):
            dat = get_data(ifile, showlog=self.showlog,
                           metaonly=True, piter=iter)
            if dat is None:
                continue
            datm = RasterMeta()
            datm.fromData(dat)

            if datm.sensor == 'Generic':
                if 'Generic' not in self.tnames:
                    self.bands['Generic'] = []
                self.bands['Generic'] = list(set(self.bands['Generic'] +
                                                 datm.bands))
            else:
                self.bands[datm.sensor] = datm.bands

            self.tnames[datm.sensor] = self.bands[datm.sensor].copy()
            self.filelist.append(datm)

        self.cmb_sensor.disconnect()
        self.cmb_sensor.clear()
        self.cmb_sensor.addItems(self.bands.keys())
        self.cmb_sensor.currentIndexChanged.connect(self.setsensor)

        if not self.filelist:
            self.showlog('No valid files in the directory.')
        else:
            self.setsensor()

        return True

    def setsensor(self):
        """
        Set the sensor band data.

        Returns
        -------
        None.

        """
        if self.lw_tnames.count() > 0:
            self.tnames[self.oldsensor] = []
            for i in range(self.lw_tnames.count()):
                item = self.lw_tnames.item(i)
                if item.isSelected():
                    self.tnames[self.oldsensor].append(str(item.text()))

            for i in self.filelist:
                if i.sensor == self.oldsensor:
                    i.tnames = self.tnames[self.oldsensor]

        sensor = self.cmb_sensor.currentText()
        tmp = self.bands[sensor]

        self.lw_tnames.clear()
        self.lw_tnames.addItems(tmp)

        for i in range(self.lw_tnames.count()):
            item = self.lw_tnames.item(i)
            if sensor == 'Generic':
                item.setSelected(True)
            elif item.text() in self.tnames[sensor]:
                item.setSelected(True)

        self.oldsensor = sensor

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.idir)
        self.saveobj(self.tnames)
        self.saveobj(self.filelist)
        self.saveobj(self.bands)
        self.saveobj(self.tnames)
        self.saveobj(self.oldsensor)

        self.saveobj(self.cmb_sensor)
        self.saveobj(self.le_sfile)
        self.saveobj(self.lw_tnames)
        self.saveobj(self.lbl_ftype)
        self.saveobj(self.cb_ensuresutm)


class ImportSentinel5P(BasicModule):
    """Import Sentinel 5P data to shapefile."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.sfile = ''
        self.filt = ''
        self.indx = 0
        self.is_import = True

        self.cmb_subdata = QtWidgets.QComboBox()
        self.le_lonmin = QtWidgets.QLineEdit('16')
        self.le_lonmax = QtWidgets.QLineEdit('34')
        self.le_latmin = QtWidgets.QLineEdit('-35')
        self.le_latmax = QtWidgets.QLineEdit('-21')
        self.le_qathres = QtWidgets.QLineEdit('50')
        self.rb_cclip = QtWidgets.QRadioButton('Clip using coordinates')
        self.rb_sclip = QtWidgets.QRadioButton('Clip using shapefile')
        self.le_shpfile = QtWidgets.QLineEdit(self.sfile)
        self.lbl_sfile = QtWidgets.QPushButton('Load shapefile')
        self.lbl_lonmin = QtWidgets.QLabel('Minimum Longitude:')
        self.lbl_lonmax = QtWidgets.QLabel('Maximum Longitude:')
        self.lbl_latmin = QtWidgets.QLabel('Minimum Latitude:')
        self.lbl_latmax = QtWidgets.QLabel('Maximum Latitude:')

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
        helpdocs = menu_default.HelpButton('pygmi.rsense.iodefs.'
                                           'importsentinel5p')
        lbl_subdata = QtWidgets.QLabel('Product:')
        lbl_qathres = QtWidgets.QLabel('QA Threshold (0-100):')

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)
        self.rb_cclip.setChecked(True)
        self.lbl_sfile.hide()
        self.le_shpfile.hide()

        self.setWindowTitle(r'Import Sentinel-5P Data')

        gl_main.addWidget(lbl_subdata, 0, 0, 1, 1)
        gl_main.addWidget(self.cmb_subdata, 0, 1, 1, 1)

        gl_main.addWidget(self.rb_cclip, 1, 0, 1, 2)
        gl_main.addWidget(self.rb_sclip, 2, 0, 1, 2)

        gl_main.addWidget(self.lbl_lonmin, 3, 0, 1, 1)
        gl_main.addWidget(self.le_lonmin, 3, 1, 1, 1)

        gl_main.addWidget(self.lbl_lonmax, 4, 0, 1, 1)
        gl_main.addWidget(self.le_lonmax, 4, 1, 1, 1)

        gl_main.addWidget(self.lbl_latmin, 5, 0, 1, 1)
        gl_main.addWidget(self.le_latmin, 5, 1, 1, 1)

        gl_main.addWidget(self.lbl_latmax, 6, 0, 1, 1)
        gl_main.addWidget(self.le_latmax, 6, 1, 1, 1)

        gl_main.addWidget(self.lbl_sfile, 7, 0, 1, 1)
        gl_main.addWidget(self.le_shpfile, 7, 1, 1, 1)

        gl_main.addWidget(lbl_qathres, 8, 0, 1, 1)
        gl_main.addWidget(self.le_qathres, 8, 1, 1, 1)

        gl_main.addWidget(helpdocs, 10, 0, 1, 1)
        gl_main.addWidget(buttonbox, 10, 1, 1, 3)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        self.rb_cclip.clicked.connect(self.clipchoice)
        self.rb_sclip.clicked.connect(self.clipchoice)
        self.lbl_sfile.clicked.connect(self.loadshp)

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
            ext = 'Sentinel-5P (*.nc)'

            self.ifile, self.filt = QtWidgets.QFileDialog.getOpenFileName(
                self.parent, 'Open File', '.', ext)
            if self.ifile == '':
                return False

        os.chdir(os.path.dirname(self.ifile))

        meta = self.get_5P_meta()

        if meta is None:
            return False

        tmp = []
        for i in meta:
            if i in ['latitude', 'longitude', 'qa_value']:
                continue
            tmp.append(i)

        self.cmb_subdata.clear()
        self.cmb_subdata.addItems(tmp)
        self.cmb_subdata.setCurrentIndex(self.indx)

        if not nodialog:
            tmp = self.exec()

            if tmp != 1:
                return tmp

        try:
            _ = float(self.le_lonmin.text())
            _ = float(self.le_latmin.text())
            _ = float(self.le_lonmax.text())
            _ = float(self.le_latmax.text())
        except ValueError:
            self.showlog('Value error - abandoning import')
            return False

        gdf = self.get_5P_data(meta)

        if gdf is None:
            return False

        if gdf.geom_type.loc[0] == 'Point':
            if 'line' not in gdf.columns:
                gdf['line'] = 'None'
            else:
                gdf['line'] = gdf['line'].astype(str)

        gdf.attrs['source'] = os.path.basename(self.ifile)

        self.outdata['Vector'] = [gdf]

        return True

    def clipchoice(self):
        """
        Choose clip style.

        Returns
        -------
        None.

        """
        if self.rb_cclip.isChecked():
            self.lbl_sfile.hide()
            self.le_shpfile.hide()
            self.le_lonmin.show()
            self.le_lonmax.show()
            self.le_latmin.show()
            self.le_latmax.show()
            self.lbl_lonmin.show()
            self.lbl_lonmax.show()
            self.lbl_latmin.show()
            self.lbl_latmax.show()
        else:
            self.le_lonmin.hide()
            self.le_lonmax.hide()
            self.le_latmin.hide()
            self.le_latmax.hide()
            self.lbl_lonmin.hide()
            self.lbl_lonmax.hide()
            self.lbl_latmin.hide()
            self.lbl_latmax.hide()
            self.lbl_sfile.show()
            self.le_shpfile.show()

    def loadshp(self):
        """
        Load shapefile filename.

        Returns
        -------
        None.

        """
        ext = 'Shapefile (*.shp)'

        self.sfile, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.parent, 'Open File', '.', ext)

        self.le_shpfile.setText(self.sfile)

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.ifile)
        self.saveobj(self.filt)
        self.saveobj(self.sfile)
        self.saveobj(self.indx)

        self.saveobj(self.cmb_subdata)
        self.saveobj(self.le_lonmin)
        self.saveobj(self.le_lonmax)
        self.saveobj(self.le_latmin)
        self.saveobj(self.le_latmax)
        self.saveobj(self.le_qathres)
        self.saveobj(self.rb_cclip)
        self.saveobj(self.rb_sclip)

    def get_5P_meta(self):
        """
        Get 5P metadata.

        Returns
        -------
        meta : Dictionary
            Dictionary containing metadata.

        """
        with rasterio.open(self.ifile) as dataset:
            subdata = dataset.subdatasets

        meta = {}
        for i in subdata:
            tmp = i.split(':')
            if 'SUPPORT_DATA' in i:
                continue
            if 'METADATA' in i:
                continue
            if 'time_utc' in i:
                continue
            if 'delta_time' in i:
                continue
            if 'precision' in i:
                continue

            tmp = tmp[-1].replace('//PRODUCT/', '')
            tmp = tmp.replace('/PRODUCT/', '')
            tmp = tmp.replace('/', '')

            meta[tmp] = i

        dataset = None

        return meta

    def get_5P_data(self, meta):
        """
        Get 5P data.

        Parameters
        ----------
        meta : Dictionary
            Dictionary containing metadata.

        Returns
        -------
        gdf : DataFrame
            geopandas dataframe.

        """
        try:
            thres = int(self.le_qathres.text())
        except ValueError:
            self.showlog('Threshold text not an integer')
            return None

        with rasterio.open(meta['latitude']) as dataset:
            lats = dataset.read(1)

        with rasterio.open(meta['longitude']) as dataset:
            lons = dataset.read(1)

        with rasterio.open(meta['qa_value']) as dataset:
            qaval = dataset.read(1)

        with rasterio.open(meta['longitude']) as dataset:
            lons = dataset.read(1)

        del meta['latitude']
        del meta['longitude']

        if lats is None:
            self.showlog('No Latitudes in dataset')
            return None

        lats = lats.flatten()
        lons = lons.flatten()
        pnts = np.transpose([lons, lats])

        if self.rb_cclip.isChecked():
            lonmin = float(self.le_lonmin.text())
            latmin = float(self.le_latmin.text())
            lonmax = float(self.le_lonmax.text())
            latmax = float(self.le_latmax.text())
        else:
            shp = gpd.read_file(self.sfile)
            shp = shp.to_crs(4326)

            lonmin = float(shp.bounds.minx)
            lonmax = float(shp.bounds.maxx)
            latmin = float(shp.bounds.miny)
            latmax = float(shp.bounds.maxy)

        mask = ((lats > latmin) & (lats < latmax) & (lons < lonmax) &
                (lons > lonmin))

        idfile = self.cmb_subdata.currentText()

        dfile = meta[idfile]

        with rasterio.open(dfile) as dataset:
            dat = dataset.read(1)

        dat1 = dat.flatten()
        qaval1 = qaval.flatten()

        if mask.shape != dat1.shape:
            return None

        dat1 = dat1[mask]
        pnts1 = pnts[mask]
        qaval1 = qaval1[mask]

        qaval1 = qaval1[dat1 != 9.96921e+36]
        pnts1 = pnts1[dat1 != 9.96921e+36]
        dat1 = dat1[dat1 != 9.96921e+36]

        pnts1 = pnts1[qaval1 >= thres]
        dat1 = dat1[qaval1 >= thres]

        df = pd.DataFrame({'lon': pnts1[:, 0], 'lat': pnts1[:, 1]})
        df['data'] = dat1

        gdf = GeoDataFrame(df.drop(['lon', 'lat'], axis=1),
                           geometry=[Point(xy) for xy in zip(df.lon, df.lat)])

        gdf = gdf.set_crs("EPSG:4326")

        if self.rb_sclip.isChecked():
            gdf = gdf.clip(shp)

        if gdf.size == 0:
            self.showlog(idfile, 'is empty.')
            return None

        return gdf


class ExportBatch(ContextModule):
    """Export Raster File List."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.cmb_ofilt = QtWidgets.QComboBox()
        self.le_odir = QtWidgets.QLineEdit('')
        self.cmb_red = QtWidgets.QComboBox()
        self.cmb_green = QtWidgets.QComboBox()
        self.cmb_blue = QtWidgets.QComboBox()
        self.cmb_sunshade = QtWidgets.QComboBox()
        self.cmb_slvl = QtWidgets.QComboBox()
        self.cb_ternary = QtWidgets.QCheckBox('Ternary Export')

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
        helpdocs = menu_default.HelpButton('pygmi.rsense.iodefs.exportbatch')
        lbl_ofilt = QtWidgets.QLabel('Output Format:')
        lbl_red = QtWidgets.QLabel('Red Band:')
        lbl_green = QtWidgets.QLabel('Green Band:')
        lbl_blue = QtWidgets.QLabel('Blue Band:')
        lbl_shade = QtWidgets.QLabel('Sunshade Band:')
        lbl_slvl = QtWidgets.QLabel('Sunshade Level:')
        pb_odir = QtWidgets.QPushButton('Output Directory')

        ext = ('GeoTIFF', 'GeoTIFF compressed using DEFLATE',
               'GeoTIFF compressed using ZSTD', 'ENVI', 'ERMapper',
               'ERDAS Imagine')

        self.cmb_ofilt.addItems(ext)
        self.cmb_ofilt.setCurrentText('GeoTIFF compressed using DEFLATE')
        self.cmb_slvl.addItems(['Standard', 'Heavy'])
        self.cmb_slvl.setCurrentText('Standard')

        self.cb_ternary.setChecked(False)
        self.cmb_red.setEnabled(False)
        self.cmb_green.setEnabled(False)
        self.cmb_blue.setEnabled(False)
        self.cmb_sunshade.setEnabled(False)
        self.cmb_slvl.setEnabled(False)

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle(r'Export File List')

        gl_main.addWidget(self.le_odir, 0, 0, 1, 1)
        gl_main.addWidget(pb_odir, 0, 1, 1, 1)

        gl_main.addWidget(lbl_ofilt, 1, 0, 1, 1)
        gl_main.addWidget(self.cmb_ofilt, 1, 1, 1, 1)

        gl_main.addWidget(self.cb_ternary, 2, 0, 1, 2)

        gl_main.addWidget(lbl_red, 3, 0, 1, 1)
        gl_main.addWidget(self.cmb_red, 3, 1, 1, 1)

        gl_main.addWidget(lbl_green, 4, 0, 1, 1)
        gl_main.addWidget(self.cmb_green, 4, 1, 1, 1)

        gl_main.addWidget(lbl_blue, 5, 0, 1, 1)
        gl_main.addWidget(self.cmb_blue, 5, 1, 1, 1)

        gl_main.addWidget(lbl_shade, 6, 0, 1, 1)
        gl_main.addWidget(self.cmb_sunshade, 6, 1, 1, 1)

        gl_main.addWidget(lbl_slvl, 7, 0, 1, 1)
        gl_main.addWidget(self.cmb_slvl, 7, 1, 1, 1)

        gl_main.addWidget(helpdocs, 8, 0, 1, 1)
        gl_main.addWidget(buttonbox, 8, 1, 1, 3)

        buttonbox.accepted.connect(self.acceptall)
        buttonbox.rejected.connect(self.reject)
        pb_odir.pressed.connect(self.get_odir)
        self.cb_ternary.clicked.connect(self.click_ternary)

    def click_ternary(self):
        """
        Click ternary event.

        Returns
        -------
        None.

        """
        if self.cb_ternary.isChecked():
            self.cmb_red.setEnabled(True)
            self.cmb_green.setEnabled(True)
            self.cmb_blue.setEnabled(True)
            self.cmb_ofilt.setCurrentText('GeoTIFF')
            self.cmb_ofilt.setEnabled(False)
            self.cmb_sunshade.setEnabled(True)
            self.cmb_slvl.setEnabled(True)

        else:
            self.cmb_red.setEnabled(False)
            self.cmb_green.setEnabled(False)
            self.cmb_blue.setEnabled(False)
            self.cmb_ofilt.setEnabled(True)
            self.cmb_sunshade.setEnabled(False)
            self.cmb_slvl.setEnabled(False)

    def run(self):
        """
        Run.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        self.process_is_active(True)

        if 'RasterFileList' not in self.indata:
            self.showlog('No raster file list')
            self.process_is_active(False)
            return False

        dat = self.indata['RasterFileList'][0]
        bnames = dat.tnames

        bnames = natsorted(bnames)

        if 'Explained Variance Ratio' in bnames[0]:
            bnames = [i.split('Explained Variance Ratio')[0] for i in bnames]

        self.cmb_red.addItems(bnames)
        self.cmb_green.addItems(bnames)
        self.cmb_blue.addItems(bnames)
        self.cmb_sunshade.addItems(['None', 'External File (first band)'] +
                                   bnames)

        self.show()

    def acceptall(self):
        """Accept choice."""
        if self.le_odir.text() == '':
            self.showlog('No output directory')
            return

        filt = self.cmb_ofilt.currentText()
        odir = self.le_odir.text()
        sunfile = None

        if self.cb_ternary.isChecked():
            tnames = [self.cmb_red.currentText(),
                      self.cmb_green.currentText(),
                      self.cmb_blue.currentText()]
            otype = 'RGB'
            sunfile = self.cmb_sunshade.currentText()
            slevel = self.cmb_slvl.currentText()
            if slevel == 'Standard':
                cell = 25.
                alpha = .75
            else:
                cell = 1.
                alpha = .99

            if sunfile == 'External File (first band)':
                ext = ('Common formats (*.ers *.hdr *.tif *.tiff *.sdat *.img '
                       '*.pix *.bil);;')

                sunfile, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self.parent, 'Open File', '.', ext)
                if sunfile == '':
                    return False
        else:
            otype = None
            tnames = None
            slevel = None
            cell = 25.
            alpha = .75

        self.showlog('Export Data Busy...')

        export_batch(self.indata, odir, filt, tnames, piter=self.piter,
                     showlog=self.showlog, otype=otype, sunfile=sunfile,
                     cell=cell, alpha=alpha)

        self.showlog('Export Data Finished!')
        self.process_is_active(False)

        self.accept()

    def get_odir(self, odir=''):
        """
        Get output directory.

        Parameters
        ----------
        odir : str, optional
            Output directory submitted for testing. The default is ''.

        Returns
        -------
        None.

        """
        if odir == '':
            odir = QtWidgets.QFileDialog.getExistingDirectory(
                self.parent, 'Select Output Directory')

            if odir == '':
                return

        self.le_odir.setText(odir)


def calculate_toa(dat, showlog=print):
    """
    Top of atmosphere correction.

    Includes VNIR, SWIR and TIR bands.

    Parameters
    ----------
    dat : Data
        PyGMI raster dataset
    showlog : function, optional
        Routine to show text messages. The default is print.

    Returns
    -------
    out : Data
        PyGMI raster dataset
    """
    showlog('Calculating top of atmosphere...')

    datanew = {}
    for datai in dat:
        datanew[datai.dataid.split()[1]] = datai.copy()

    out = []
    for i in range(len(dat)):
        idtmp = 'ImageData'+str(i+1)
        if i+1 == 3:
            idtmp += 'N'
        datai = datanew[idtmp]

        gain = datai.metadata['Gain']
        sunelev = datai.metadata['SolarElev']
        jday = datai.metadata['JulianDay']

        lrad = (datai.data-1)*gain

        if i < 9:
            theta = np.deg2rad(90-sunelev)
            datai.data = np.pi*lrad*EDIST[jday]**2/(ESUN[i]*np.cos(theta))
        else:
            datai.data = K2[i-9]/np.log(K1[i-9]/lrad+1)
        datai.data.set_fill_value(datai.nodata)
        dmask = datai.data.mask
        datai.data = np.ma.array(datai.data.filled(), mask=dmask)
        out.append(datai)

    return out


def consolidate_aster_list(flist):
    """
    Consolidate ASTER files from a file list, getting rid of extra files.

    Parameters
    ----------
    flist : list
        List of filenames.

    Returns
    -------
    flist : list
        List of filenames.

    """
    asterhfiles = []
    asterzfiles = []
    otherfiles = []

    for ifile in flist:
        bfile = os.path.basename(ifile)
        ext = os.path.splitext(ifile)[1].lower()

        if 'AST_' in bfile and ext == '.hdf':
            asterhfiles.append(ifile)
        if 'AST_' in bfile and ext == '.zip':
            asterzfiles.append(ifile)
        else:
            otherfiles.append(ifile)

    asterfiles = []
    tmp = {}

    for ifile in asterzfiles:
        adate = os.path.basename(ifile).split('_')[2]
        tmp[adate] = ifile
    asterfiles += list(tmp.values())

    for ifile in asterhfiles:
        adate = os.path.basename(ifile).split('_')[2]
        tmp[adate] = ifile

    asterfiles += list(tmp.values())

    flist = asterfiles+otherfiles

    return flist


def convert_ll_to_utm(lon, lat):
    """
    Convert latitude and longitde to UTM.

    https://stackoverflow.com/a/40140326/4556479

    Parameters
    ----------
    lon : float
        DESCRIPTION.
    lat : float
        DESCRIPTION.

    Returns
    -------
    epsg_code : TYPE
        DESCRIPTION.

    """
    utm_band = str((math.floor((lon + 180) / 6) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0'+utm_band
    if lat >= 0:
        epsg_code = '326' + utm_band
        return epsg_code
    epsg_code = '327' + utm_band
    return epsg_code


def etree_to_dict(t):
    """
    Convert an ElementTree to dictionary.

    From K3--rnc: https://stackoverflow.com/questions/7684333/converting-xml-to-dictionary-using-elementtree

    Parameters
    ----------
    t : Elementtree
        Root.

    Returns
    -------
    d : dictionary
        Dictionary of ElementTree items.

    """
    d = {t.tag: {} if t.attrib else None}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(etree_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k: v[0] if len(v) == 1 else v
                     for k, v in dd.items()}}
    if t.attrib:
        d[t.tag].update(('@' + k, v)
                        for k, v in t.attrib.items())
    if t.text:
        text = t.text.strip()
        if children or t.attrib:
            if text:
                d[t.tag]['#text'] = text
        else:
            d[t.tag] = text
    return d


def export_batch(indata, odir, filt, tnames=None, piter=None,
                 showlog=print, otype=None, sunfile=None,
                 cell=25., alpha=.75):
    """
    Export a batch of files directly from satellite format to disk.

    Parameters
    ----------
    indata : dictionary
        Dictionary containing 'RasterFileList' as one of its keys.
    odir : str
        Output Directory.
    filt : str
        type of file to export.
    tnames : list, optional
        list of band names to import, in order. the default is None.
    piter : function, optional
        Progress bar iterable. Default is None.
    showlog : function, optional
        Routine to show text messages. The default is print.
    otype : str
        output type of file, regular or RGB ternary (with possible sunshading)
    sunfile : str
        either a filename of an external file to be used for sunshading, or an
        existing band name. the default is None.
    cell : float
        Between 1 and 100 - controls sunshade detail. The default is 25.
    alpha : float
        How much incident light is reflected (0 to 1). The default is .75.

    Returns
    -------
    None.

    """
    if 'RasterFileList' not in indata:
        showlog('You need a raster file list')
        return

    ifiles = indata['RasterFileList']

    if sunfile == 'None':
        sunfile = None
    elif tnames is not None and otype == 'RGB':
        if sunfile in ifiles[0].bands and sunfile not in tnames:
            tnames += [sunfile]

    filt2gdal = {'GeoTIFF compressed using ZSTD': 'GTiff',
                 'GeoTIFF compressed using DEFLATE': 'GTiff',
                 'GeoTIFF': 'GTiff',
                 'ENVI': 'ENVI',
                 'ERMapper': 'ERS',
                 'ERDAS Imagine': 'HFA'}

    compression = 'NONE'
    ofilt = filt2gdal[filt]
    if filt == 'GeoTIFF compressed using ZSTD':
        compression = 'ZSTD'
    elif filt == 'GeoTIFF compressed using DEFLATE':
        compression = 'DEFLATE'

    os.makedirs(odir, exist_ok=True)

    for ifile in ifiles:
        dat = get_from_rastermeta(ifile, piter=piter,
                                  showlog=showlog,
                                  tnames=tnames)

        odat = []
        if tnames is not None:
            for i in tnames:
                for j in dat:
                    if i == j.dataid:
                        odat.append(j)
                        break
        else:
            odat = dat

        ofile = set_export_filename(odat, odir, otype)
        showlog('Exporting '+os.path.basename(ofile))

        if otype == 'RGB':
            odat = get_ternary(odat, sunfile=sunfile, cell=cell, alpha=alpha,
                               piter=piter, showlog=showlog)
            export_raster(ofile, odat, 'GTiff', piter=piter,
                          bandsort=False, updatestats=False,
                          showlog=showlog)
        else:
            export_raster(ofile, odat, ofilt, piter=piter,
                          compression=compression, showlog=showlog)


def get_data(ifile, piter=None, showlog=print, tnames=None,
             metaonly=False):
    """
    Load a raster dataset off the disk using the rasterio libraries.

    It returns the data in a PyGMI data object.

    Parameters
    ----------
    ifile : str
        filename to import
    piter : function, optional
        Progress bar iterable. Default is None.
    showlog : function, optional
        Routine to show text messages. The default is print.
    tnames : list, optional
        list of band names to import, in order. The default is None.
    metaonly : bool, optional
        Retrieve only the metadata for the file. The default is False.

    Returns
    -------
    dat : list of PyGMI Data
        dataset imported
    """
    ifile = ifile[:]
    bfile = os.path.basename(ifile)
    ext = os.path.splitext(ifile)[1].lower()

    showlog('Importing', bfile)
    dtree = {}
    if '.xml' in bfile.lower():
        dtree = etree_to_dict(ET.parse(ifile).getroot())

    if 'AST_' in bfile and ext == '.hdf':
        idir = os.path.dirname(ifile)
        adate = os.path.basename(ifile).split('_')[2]
        ifiles = glob.glob(os.path.join(idir, '*'+adate+'*.hdf'))
        dat = []
        for afile in ifiles:
            dat += get_aster_hdf(afile, piter, showlog, tnames, metaonly)
    elif 'AST_' in bfile and ext == '.zip':
        idir = os.path.dirname(ifile)
        adate = os.path.basename(ifile).split('_')[2]
        ifiles = glob.glob(os.path.join(idir, '*'+adate+'*.zip'))
        dat = []
        for afile in ifiles:
            dat += get_aster_zip(afile, piter, showlog, tnames, metaonly)
    # if 'AST_' in bfile:
    #     idir = os.path.dirname(ifile)
    #     adate = os.path.basename(ifile).split('_')[2]
    #     ifiles = glob.glob(os.path.join(idir, '*'+adate+'*.hdf'))
    #     dat = []
    #     for afile in ifiles:
    #         dat += get_aster_hdf(afile, piter, showlog, tnames, metaonly)

    #     ifiles = glob.glob(os.path.join(idir, '*'+adate+'*.zip'))
    #     for afile in ifiles:
    #         dat += get_aster_zip(afile, piter, showlog, tnames, metaonly)
    elif (bfile[:4] in ['LT04', 'LT05', 'LE07', 'LC08', 'LM05', 'LC09'] and
          ('.tar' in bfile.lower() or '_MTL.txt' in bfile)):
        dat = get_landsat(ifile, piter, showlog, tnames, metaonly)
    elif ((ext == '.xml' and '.SAFE' in ifile) or
          ('S2A_' in bfile and ext == '.zip') or
          ('S2B_' in bfile and ext == '.zip')):
        dat = get_sentinel2(ifile, piter, showlog, tnames, metaonly)
    elif (ext == '.xml' and 'DIM' in ifile):
        dat = get_spot(ifile, piter, showlog, tnames, metaonly)
    elif (('MOD' in bfile or 'MCD' in bfile) and ext == '.hdf' and
          '.006.' in bfile):
        dat = get_modisv6(ifile, piter, showlog, tnames, metaonly)
    elif 'AG1' in bfile and ext == '.h5':
        dat = get_aster_ged(ifile, piter, showlog, tnames, metaonly)
    elif ext == '.zip' and 'EO1H' in bfile:
        dat = get_hyperion(ifile, piter, showlog, tnames, metaonly)
    elif ext == '.xml' and 'isd' in dtree:
        dat = get_worldview(ifile, piter, showlog, tnames, metaonly)
    else:
        dat = get_raster(ifile, piter=piter, showlog=showlog,
                         tnames=tnames, metaonly=metaonly)

    if dat is not None:
        for i in dat:
            if i.dataid is None:
                i.dataid = ''
            i.dataid = i.dataid.replace(',', ' ')

        # Sort in band order.
        dataid = [i.dataid for i in dat]
        dorder = [i for _, i in natsorted(zip(dataid, range(len(dataid))))]
        dat = [dat[i] for i in dorder]

    return dat


def get_from_rastermeta(ldata, piter=None, showlog=print, tnames=None):
    """
    Import data from a RasterMeta item.

    Parameters
    ----------
    ldata : RasterMeta or list
        List of RasterMeta or single item.
    piter : function, optional
        Progress bar iterable. Default is None.
    showlog : function, optional
        Routine to show text messages. The default is print.
    tnames : list, optional
        list of band names to import, in order. The default is None.

    Returns
    -------
    dat : list  of PyGMI Data
        List of data.

    """
    if tnames is None:
        tnames = ldata.tnames

    ifile = ldata.banddata[0].filename
    dat = get_data(ifile, piter=piter, showlog=showlog, tnames=tnames)

    if ldata.to_sutm is True:
        dat = utm_to_south(dat)

    return dat


def get_modisv6(ifile, piter=None, showlog=print, tnames=None,
                metaonly=False):
    """
    Get MODIS v006 data.

    Parameters
    ----------
    ifile : str
        filename to import
    piter : function, optional
        Progress bar iterable. Default is None.
    showlog : function, optional
        Routine to show text messages. The default is print.
    tnames : list, optional
        list of band names to import, in order. The default is None.
    metaonly : bool, optional
        Retrieve only the metadata for the file. The default is False.

    Returns
    -------
    dat : PyGMI raster Data
        dataset imported
    """
    if piter is None:
        piter = ProgressBarText().iter

    dat = []
    ifile = ifile[:]

    with rasterio.open(ifile) as dataset:
        if dataset is None:
            return None
        subdata = dataset.subdatasets

    dat = []
    lulc = None

    for ifile2 in subdata:
        dataset = rasterio.open(ifile2)

        wkt = dataset.crs.to_wkt()
        crs = dataset.crs
        if 'Sinusoidal' in wkt:
            wkt = wkt.replace('PROJCS["unnamed"', 'PROJCS["Sinusoidal"')
            wkt = wkt.replace('GEOGCS["Unknown datum based upon the custom '
                              'spheroid"',
                              'GEOGCS["GCS_Unknown"')
            wkt = wkt.replace('DATUM["Not specified '
                              '(based on custom spheroid)"',
                              'DATUM["D_Unknown"')
            wkt = wkt.replace('SPHEROID["Custom spheroid"',
                              'SPHEROID["S_Unknown"')
            crs = CRS.from_wkt(wkt)

        meta = dataset.tags()
        bandid = dataset.descriptions[0]
        nval = dataset.nodata
        datetxt = meta['RANGEBEGINNINGDATE']
        date = datetime.datetime.strptime(datetxt, '%Y-%m-%d')

        if bandid is None and ':' in ifile2:
            bandid = ifile2[ifile2.rindex(':')+1:]

        if tnames is not None and bandid not in tnames:
            continue

        if 'scale_factor' in meta:
            scale = float(meta['scale_factor'])
        else:
            scale = 1

        if 'MOD13' in ifile and scale > 1:
            scale = 1./scale

        if 'MOD44B' in ifile and '_SD' in bandid and scale == 1:
            scale = 0.01

        if 'add_offset' in meta:
            offset = float(meta['add_offset'])
        else:
            offset = 0

        dat.append(Data())
        if metaonly is False:
            rtmp2 = dataset.read(1)
            rtmp2 = rtmp2.astype(float)

            if nval == 32767:
                mask = (rtmp2 > 32760)
                lulc = np.zeros_like(rtmp2)
                lulc[mask] = rtmp2[mask]-32760
                lulc = np.ma.masked_equal(lulc, 0)
            else:
                mask = (rtmp2 == nval)

            rtmp2 = rtmp2*scale+offset
            rtmp2[mask] = 1e+20

            dat[-1].data = np.ma.array(rtmp2, mask=mask)

        dat[-1].dataid = bandid
        dat[-1].nodata = 1e+20
        dat[-1].meta_from_rasterio(dataset)
        dat[-1].crs = crs
        dat[-1].filename = ifile
        dat[-1].units = dataset.units[0]
        dat[-1].metadata['Raster']['Sensor'] = 'MODIS'
        dat[-1].datetime = date

        dataset.close()

    if lulc is not None:
        dat.append(dat[0].copy())
        dat[-1].data = lulc
        dat[-1].dataid = ('LULC: out of earth=7, water=6, barren=5, snow=4, '
                          'wetland=3, urban=2, unclassifed=1')
        dat[-1].nodata = 0

    showlog('Import complete')
    return dat


def get_landsat(ifilet, piter=None, showlog=print, tnames=None,
                metaonly=False):
    """
    Get Landsat Data.

    Parameters
    ----------
    ifilet : str
        filename to import
    piter : function, optional
        Progress bar iterable. Default is None.
    showlog : function, optional
        Routine to show text messages. The default is print.
    tnames : list, optional
        list of band names to import, in order. The default is None.
    metaonly : bool, optional
        Retrieve only the metadata for the file. The default is False.

    Returns
    -------
    out : Data
        PyGMI raster dataset
    """
    if piter is None:
        piter = ProgressBarText().iter

    platform = os.path.basename(ifilet)[2: 4]
    satbands = None
    lstband = None

    if platform in ('04', '05'):
        satbands = {'B1': [450, 520],
                    'B2': [520, 600],
                    'B3': [630, 690],
                    'B4': [760, 900],
                    'B5': [1550, 1750],
                    'B6': [10400, 12500],
                    'B7': [2080, 2350]}

    if platform == '07':
        satbands = {'B1': [450, 520],
                    'B2': [520, 600],
                    'B3': [630, 690],
                    'B4': [770, 900],
                    'B5': [1550, 1750],
                    'B6': [10400, 12500],
                    'B7': [2090, 2350],
                    'B8': [520, 900]}

    if platform in ('08', '09'):
        satbands = {'B1': [430, 450],
                    'B2': [450, 510],
                    'B3': [530, 590],
                    'B4': [640, 670],
                    'B5': [850, 880],
                    'B6': [1570, 1650],
                    'B7': [2110, 2290],
                    'B8': [500, 680],
                    'B9': [1360, 1380],
                    'B10': [1060, 11190],
                    'B11': [11500, 12510]}

    idir = os.path.dirname(ifilet)

    if '.tar' in ifilet:
        with tarfile.open(ifilet) as tar:
            tarnames = tar.getnames()
            ifile = next((i for i in tarnames if '_MTL.txt' in i), None)
            if ifile is None:
                showlog('Could not find MTL.txt file in tar archive')
                return None
            showlog('Extracting tar...')

            tar.extractall(idir)
            ifile = os.path.join(idir, ifile)
    elif '_MTL.txt' in ifilet:
        ifile = ifilet
    else:
        showlog('Input needs to be tar or _MTL.txt for Landsat. '
                'Trying regular import')
        return None

    with open(ifile, encoding='utf-8') as inp:
        headtxt = inp.readlines()

    refmult = {}
    refadd = {}
    radmult = {}
    radadd = {}
    K1 = {}
    K2 = {}
    for i in headtxt:
        if 'DATE_ACQUIRED' in i:
            datetxt = i.split('=')[1][:-1].strip()
            date = datetime.datetime.strptime(datetxt, '%Y-%m-%d')
        if 'REFLECTANCE_MULT_BAND' in i:
            tmp = i.split('=')
            tmp1 = 'B'+tmp[0].split('_')[-1].strip()
            tmp2 = float(tmp[1].strip())
            refmult[tmp1] = tmp2
        if 'REFLECTANCE_ADD_BAND' in i:
            tmp = i.split('=')
            tmp1 = 'B'+tmp[0].split('_')[-1].strip()
            tmp2 = float(tmp[1].strip())
            refadd[tmp1] = tmp2
        if 'RADIANCE_MULT_BAND' in i:
            tmp = i.split('=')
            tmp1 = 'B'+tmp[0].split('BAND_')[-1].strip()
            tmp2 = float(tmp[1].strip())
            radmult[tmp1] = tmp2
        if 'RADIANCE_ADD_BAND' in i:
            tmp = i.split('=')
            tmp1 = 'B'+tmp[0].split('BAND_')[-1].strip()
            tmp2 = float(tmp[1].strip())
            radadd[tmp1] = tmp2
        if 'K1_CONSTANT_BAND' in i:
            tmp = i.split('=')
            tmp1 = 'B'+tmp[0].split('BAND_')[-1].strip()
            tmp2 = float(tmp[1].strip())
            K1[tmp1] = tmp2
        if 'K2_CONSTANT_BAND' in i:
            tmp = i.split('=')
            tmp1 = 'B'+tmp[0].split('BAND_')[-1].strip()
            tmp2 = float(tmp[1].strip())
            K2[tmp1] = tmp2

    # Correct Landsat 7 thermal keys
    allkeys = list(K1.keys())
    for key in allkeys:
        if 'VCID' in key:
            key2 = key.replace('B6', 'ToABT')
            K1[key2] = K1.pop(key)
            K2[key2] = K2.pop(key)
            radadd[key2] = radadd.pop(key)
            radmult[key2] = radmult.pop(key)
            satbands[key2] = satbands['B6']

    files = glob.glob(ifile[:-7]+'*.tif')

    if 'LC08' in ifile or 'LC09' in ifile:
        lstband = 'B10'
    else:
        lstband = 'B6'

    if glob.glob(ifile[:-7]+'*ST_QA.tif'):
        satbands['LST'] = satbands[lstband]

    showlog('Importing Landsat data...')

    bnamelen = len(ifile[:-7])
    nval = 0
    dat = []
    for ifile2 in piter(files):
        fext = ifile2[bnamelen:-4]
        fext = fext.upper().replace('BAND', 'B')
        fext = fext.replace('SR_B', 'B')
        fext = fext.replace('ST_B', 'B')

        if lstband in fext and 'LST' in satbands:
            fext = 'LST'

        if tnames is not None and fext.replace(',', ' ') not in tnames:
            continue

        showlog('Importing Band '+fext)
        dataset = rasterio.open(ifile2)

        if dataset is None:
            showlog('Problem with band '+fext)
            continue

        dat.append(Data())

        if metaonly is False:
            dat[-1].data = dataset.read(1)
            dat[-1].data = np.ma.masked_invalid(dat[-1].data)

        nval = 0
        if fext in ['QA_PIXEL', 'SR_QA_AEROSOL']:
            nval = 1
        if fext in ['ST_CDIST', 'ST_QA']:
            nval = -9999
        if fext in ['ST_TRAD', 'ST_URAD', 'ST_DRAD']:
            nval = -9999
        if fext in ['ST_ATRAN', 'ST_EMIS', 'ST_EMSD']:
            nval = -9999

        dat[-1].data.mask = dat[-1].data.mask | (dat[-1].data == nval)
        if dat[-1].data.mask.size == 1:
            dat[-1].data.mask = (np.ma.make_mask_none(dat[-1].data.shape) +
                                 dat[-1].data.mask)

        dat[-1].units = 'DN'
        dat[-1].dataid = fext

        if 'L1TP' in ifile2:
            if fext in K1:
                showlog(f'Converting band {fext} to Kelvin.')
                L = dat[-1].data*radmult[fext]+radadd[fext]
                dat[-1].data = K2[fext]/np.log(K1[fext]/L+1)
                dat[-1].units = 'Kelvin'
                if not metaonly:
                    dat[-1].dataid = fext+' ToA Brightness Temperature'
            elif fext in satbands:
                showlog('Converting band '+fext+' to reflectance.')
                dat[-1].data = dat[-1].data*refmult[fext]+refadd[fext]
                dat[-1].units = 'Reflectance'
                if not metaonly:
                    dat[-1].dataid = fext+' ToA Reflectance'

        if 'L2SP' in ifile2:
            if fext == 'LST':
                showlog(f'Converting band {lstband} to Kelvin. '
                        f'Band renamed as {fext}')
                dat[-1].data = dat[-1].data*0.00341802 + 149.0
            elif fext in satbands:
                showlog('Converting band '+fext+' to reflectance.')
                dat[-1].data = dat[-1].data*0.0000275 - 0.2
            elif fext in ['ST_CDIST', 'ST_QA']:
                dat[-1].data = dat[-1].data*0.01
            elif fext in ['ST_TRAD', 'ST_URAD', 'ST_DRAD']:
                dat[-1].data = dat[-1].data*0.001
            elif fext in ['ST_ATRAN', 'ST_EMIS', 'ST_EMSD']:
                dat[-1].data = dat[-1].data*0.0001

            if fext in ['ST_QA', 'LST']:
                dat[-1].units = 'Kelvin'
            elif fext in ['QA_PIXEL', 'QA_AEROSOL', 'QA_RADSAT']:
                dat[-1].units = 'Bit Index'
            elif fext == 'ST_CDIST':
                dat[-1].units = 'km'
            elif fext in ['ST_TRAD', 'ST_URAD', 'ST_DRAD']:
                dat[-1].units = 'W/(m2.sr.μm)/DN'
            elif fext in satbands:
                dat[-1].units = 'Reflectance'
            else:
                dat[-1].units = ''

        dat[-1].nodata = nval
        dat[-1].meta_from_rasterio(dataset)
        dat[-1].filename = ifilet
        dat[-1].datetime = date

        bmeta = dat[-1].metadata['Raster']

        platform = os.path.basename(ifilet)[:4]
        bmeta['Sensor'] = f'Landsat {platform}'

        if satbands is not None and fext in satbands:
            bmeta['WavelengthMin'] = satbands[fext][0]
            bmeta['WavelengthMax'] = satbands[fext][1]
            bmeta['wavelength'] = (satbands[fext][0] + satbands[fext][1])/2

        dataset.close()

    if not dat:
        dat = None

    if '.tar' in ifilet:
        showlog('Cleaning Extracted tar files...')
        for tfile in piter(tarnames):
            os.remove(os.path.join(os.path.dirname(ifile), tfile))
    showlog('Import complete')
    return dat


def get_worldview(ifilet, piter=None, showlog=print, tnames=None,
                  metaonly=False):
    """
    Get WorldView Data.

    Parameters
    ----------
    ifilet : str
        filename to import
    piter : function, optional
        Progress bar iterable. Default is None.
    showlog : function, optional
        Routine to show text messages. The default is print.
    tnames : list, optional
        list of band names to import, in order. The default is None.
    metaonly : bool, optional
        Retrieve only the metadata for the file. The default is False.

    Returns
    -------
    out : Data
        PyGMI raster dataset
    """
    if piter is None:
        piter = ProgressBarText().iter

    dtree = etree_to_dict(ET.parse(ifilet).getroot())

    if 'isd' not in dtree:
        showlog('Wrong xml file. Please choose the xml file in the '
                'PAN or MUL directory')
        return None

    platform = dtree['isd']['TIL']['BANDID']
    satid = dtree['isd']['IMD']['IMAGE']['SATID']
    datetxt = dtree['isd']['IMD']['IMAGE']['FIRSTLINETIME']
    date = datetime.datetime.fromisoformat(datetxt)

    satbands = None

    if platform == 'P':
        satbands = {'1': [450, 800]}
        bnum2name = {0: 'BAND_P'}

    if platform == 'Multi' and 'WV' in satid:
        satbands = {'1': [400, 450],
                    '2': [450, 510],
                    '3': [510, 580],
                    '4': [585, 625],
                    '5': [630, 690],
                    '6': [705, 745],
                    '7': [770, 895],
                    '8': [860, 1040]}

        bnum2name = {0: 'BAND_C',
                     1: 'BAND_B',
                     2: 'BAND_G',
                     3: 'BAND_Y',
                     4: 'BAND_R',
                     5: 'BAND_RE',
                     6: 'BAND_N',
                     7: 'BAND_N2'}

    if platform == 'Multi' and 'GE' in satid:
        satbands = {'1': [450, 510],
                    '2': [510, 580],
                    '3': [655, 690],
                    '4': [780, 920]}

        bnum2name = {0: 'BAND_B',
                     1: 'BAND_G',
                     2: 'BAND_R',
                     3: 'BAND_N'}

    if satid == 'WV03':
        Esun = {'BAND_P': 1574.41,
                'BAND_C': 1757.89,
                'BAND_B': 2004.61,
                'BAND_G': 1830.18,
                'BAND_Y': 1712.07,
                'BAND_R': 1535.33,
                'BAND_RE': 1348.08,
                'BAND_N': 1055.94,
                'BAND_N2': 858.77}
    elif satid == 'WV02':
        Esun = {'BAND_P': 1580.8140,
                'BAND_C': 1758.2229,
                'BAND_B': 1974.2416,
                'BAND_G': 1856.4104,
                'BAND_Y': 1738.4791,
                'BAND_R': 1559.4555,
                'BAND_RE': 1342.0695,
                'BAND_N': 1069.7302,
                'BAND_N2': 861.2866}
    elif satid == 'GE01':
        Esun = {'BAND_B': 1960.0,
                'BAND_G': 1853.0,
                'BAND_R': 1505.0,
                'BAND_N': 1039.0}

    idir = os.path.dirname(ifilet)

    showlog('Importing WorldView tiles...')

    rmax = int(dtree['isd']['IMD']['NUMROWS'])
    cmax = int(dtree['isd']['IMD']['NUMCOLUMNS'])

    xmin = float(dtree['isd']['IMD']['MAP_PROJECTED_PRODUCT']['ORIGINX'])
    ymax = float(dtree['isd']['IMD']['MAP_PROJECTED_PRODUCT']['ORIGINY'])
    xdim = float(dtree['isd']['IMD']['MAP_PROJECTED_PRODUCT']['COLSPACING'])
    ydim = float(dtree['isd']['IMD']['MAP_PROJECTED_PRODUCT']['ROWSPACING'])

    dat = []
    nval = 0
    for i in range(len(satbands)):
        bname = f'Band {i+1}'
        if tnames is not None and bname not in tnames:
            continue

        fext = str(i+1)
        dat.append(Data())
        if metaonly is False:
            dat[-1].data = np.zeros((rmax, cmax))

        dat[-1].dataid = bname
        dat[-1].nodata = nval
        dat[-1].set_transform(xdim, xmin, ydim, ymax)
        dat[-1].datetime = date

        bmeta = dat[-1].metadata['Raster']

        bmeta['Sensor'] = f'WorldView {satid} {platform}'

        if satbands is not None and fext in satbands:
            bmeta['WavelengthMin'] = satbands[fext][0]
            bmeta['WavelengthMax'] = satbands[fext][1]
        bmeta['wavelength'] = (satbands[fext][0]+satbands[fext][1])/2

    for tile in dtree['isd']['TIL']['TILE']:
        ifile = os.path.join(idir, tile['FILENAME'])

        rmin = int(tile['ULROWOFFSET'])
        rmax = int(tile['LRROWOFFSET'])
        cmin = int(tile['ULCOLOFFSET'])
        cmax = int(tile['LRCOLOFFSET'])

        showlog('Importing '+tile['FILENAME'])
        dataset = rasterio.open(ifile)

        for i in piter(dataset.indexes):
            if metaonly is False:
                dat[i-1].data[rmin:rmax+1, cmin:cmax+1] = dataset.read(i)

            bmeta = dataset.tags(i)
            dat[i-1].crs = dataset.crs
            dat[i-1].filename = ifile
        dataset.close()

    showlog('Calculating radiance and reflectance...')
    indx = -1
    for i in piter(dat):
        if metaonly is True:
            continue
        indx += 1
        mask = (i.data == nval)

        scale = float(dtree['isd']['IMD'][bnum2name[indx]]['ABSCALFACTOR'])
        bwidth = float(dtree['isd']['IMD'][bnum2name[indx]]['EFFECTIVEBANDWIDTH'])

        i.data = i.data.astype(np.float32)

        date = dtree['isd']['IMD']['IMAGE']['FIRSTLINETIME']

        year = int(date[:4])
        month = int(date[5:7])
        day = int(date[8:10])
        hour = int(date[11:13])
        minute = int(date[14:16])
        sec = float(date[17:-1])

        UT = hour + minute/60. + sec/3600.

        # Julian day

        if month in (1, 2):
            year -= 1
            month += 12

        A = int(year/100.)
        B = 2 - A + int(A/4.)

        JD = int(365.25*(year+4716))+int(30.6001*(month+1))+day+UT/24.+B-1524.5

        D = JD - 2451545.0
        g = np.deg2rad(357.529 + 0.98560028 * D)

        dES = 1.00014 - 0.01671*np.cos(g) - 0.00014*np.cos(2*g)

        szenith = 90. - float(dtree['isd']['IMD']['IMAGE']['MEANSUNEL'])
        szenith = np.deg2rad(szenith)

        tmp = dES**2 * np.pi / (Esun[bnum2name[indx]] * np.cos(szenith))
        tmp = tmp * scale / bwidth

        idata = i.data
        i.data = ne.evaluate('idata * tmp')

        i.data = np.ma.array(i.data, mask=mask)

    if not dat:
        dat = None

    # Standardize labels for band ratioing.

    wvlabels = {'CoastalBlue': 'B1',
                'Blue': 'B2',
                'Green': 'B3',
                'Yellow': 'B4',
                'Red': 'B5',
                'RedEdge': 'B6',
                'NIR1': 'B7',
                'NIR2': 'B8'}
    for i in dat:
        if i.dataid.split()[0] in wvlabels:
            i.dataid = wvlabels[i.dataid.split()[0]]

    showlog('Import complete')
    return dat


def get_hyperion(ifile, piter=None, showlog=print, tnames=None,
                 metaonly=False):
    """
    Get Hyperion Data.

    Parameters
    ----------
    ifile : str
        filename to import
    piter : function, optional
        Progress bar iterable. Default is None.
    showlog : function, optional
        Routine to show text messages. The default is print.
    tnames : list, optional
        list of band names to import, in order. The default is None.
    metaonly : bool, optional
        Retrieve only the metadata for the file. The default is False.

    Returns
    -------
    out : Data
        PyGMI raster dataset
    """
    if piter is None:
        piter = ProgressBarText().iter

    wavelength = [
        355.59,  365.76,  375.94,  386.11,  396.29,  406.46,  416.64,  426.82,
        436.99,  447.17,  457.34,  467.52,  477.69,  487.87,  498.04,  508.22,
        518.39,  528.57,  538.74,  548.92,  559.09,  569.27,  579.45,  589.62,
        599.80,  609.97,  620.15,  630.32,  640.50,  650.67,  660.85,  671.02,
        681.20,  691.37,  701.55,  711.72,  721.90,  732.07,  742.25,  752.43,
        762.60,  772.78,  782.95,  793.13,  803.30,  813.48,  823.65,  833.83,
        844.00,  854.18,  864.35,  874.53,  884.70,  894.88,  905.05,  915.23,
        925.41,  935.58,  945.76,  955.93,  966.11,  976.28,  986.46,  996.63,
        1006.81, 1016.98, 1027.16, 1037.33, 1047.51, 1057.68,  851.92,  862.01,
        872.10,  882.19,  892.28,  902.36,  912.45,  922.54,  932.64,  942.73,
        952.82,  962.91,  972.99,  983.08,  993.17, 1003.30, 1013.30, 1023.40,
        1033.49, 1043.59, 1053.69, 1063.79, 1073.89, 1083.99, 1094.09, 1104.19,
        1114.19, 1124.28, 1134.38, 1144.48, 1154.58, 1164.68, 1174.77, 1184.87,
        1194.97, 1205.07, 1215.17, 1225.17, 1235.27, 1245.36, 1255.46, 1265.56,
        1275.66, 1285.76, 1295.86, 1305.96, 1316.05, 1326.05, 1336.15, 1346.25,
        1356.35, 1366.45, 1376.55, 1386.65, 1396.74, 1406.84, 1416.94, 1426.94,
        1437.04, 1447.14, 1457.23, 1467.33, 1477.43, 1487.53, 1497.63, 1507.73,
        1517.83, 1527.92, 1537.92, 1548.02, 1558.12, 1568.22, 1578.32, 1588.42,
        1598.51, 1608.61, 1618.71, 1628.81, 1638.81, 1648.90, 1659.00, 1669.10,
        1679.20, 1689.30, 1699.40, 1709.50, 1719.60, 1729.70, 1739.70, 1749.79,
        1759.89, 1769.99, 1780.09, 1790.19, 1800.29, 1810.38, 1820.48, 1830.58,
        1840.58, 1850.68, 1860.78, 1870.87, 1880.98, 1891.07, 1901.17, 1911.27,
        1921.37, 1931.47, 1941.57, 1951.57, 1961.66, 1971.76, 1981.86, 1991.96,
        2002.06, 2012.15, 2022.25, 2032.35, 2042.45, 2052.45, 2062.55, 2072.65,
        2082.75, 2092.84, 2102.94, 2113.04, 2123.14, 2133.24, 2143.34, 2153.34,
        2163.43, 2173.53, 2183.63, 2193.73, 2203.83, 2213.93, 2224.03, 2234.12,
        2244.22, 2254.22, 2264.32, 2274.42, 2284.52, 2294.61, 2304.71, 2314.81,
        2324.91, 2335.01, 2345.11, 2355.21, 2365.20, 2375.30, 2385.40, 2395.50,
        2405.60, 2415.70, 2425.80, 2435.89, 2445.99, 2456.09, 2466.09, 2476.19,
        2486.29, 2496.39, 2506.48, 2516.59, 2526.68, 2536.78, 2546.88, 2556.98,
        2566.98, 2577.08]

    fwhm = [
        11.3871, 11.3871, 11.3871, 11.3871, 11.3871, 11.3871, 11.3871, 11.3871,
        11.3871, 11.3871, 11.3871, 11.3871, 11.3871, 11.3784, 11.3538, 11.3133,
        11.2580, 11.1907, 11.1119, 11.0245, 10.9321, 10.8368, 10.7407, 10.6482,
        10.5607, 10.4823, 10.4147, 10.3595, 10.3188, 10.2942, 10.2856, 10.2980,
        10.3349, 10.3909, 10.4592, 10.5322, 10.6004, 10.6562, 10.6933, 10.7058,
        10.7276, 10.7907, 10.8833, 10.9938, 11.1044, 11.1980, 11.2600, 11.2824,
        11.2822, 11.2816, 11.2809, 11.2797, 11.2782, 11.2771, 11.2765, 11.2756,
        11.2754, 11.2754, 11.2754, 11.2754, 11.2754, 11.2754, 11.2754, 11.2754,
        11.2754, 11.2754, 11.2754, 11.2754, 11.2754, 11.2754, 11.0457, 11.0457,
        11.0457, 11.0457, 11.0457, 11.0457, 11.0457, 11.0457, 11.0457, 11.0457,
        11.0457, 11.0457, 11.0457, 11.0457, 11.0457, 11.0457, 11.0457, 11.0451,
        11.0423, 11.0372, 11.0302, 11.0218, 11.0122, 11.0013, 10.9871, 10.9732,
        10.9572, 10.9418, 10.9248, 10.9065, 10.8884, 10.8696, 10.8513, 10.8335,
        10.8154, 10.7979, 10.7822, 10.7663, 10.7520, 10.7385, 10.7270, 10.7174,
        10.7091, 10.7022, 10.6970, 10.6946, 10.6937, 10.6949, 10.6996, 10.7058,
        10.7163, 10.7283, 10.7437, 10.7612, 10.7807, 10.8034, 10.8267, 10.8534,
        10.8818, 10.9110, 10.9422, 10.9743, 11.0074, 11.0414, 11.0759, 11.1108,
        11.1461, 11.1811, 11.2156, 11.2496, 11.2826, 11.3146, 11.3460, 11.3753,
        11.4037, 11.4302, 11.4538, 11.4760, 11.4958, 11.5133, 11.5286, 11.5404,
        11.5505, 11.5580, 11.5621, 11.5634, 11.5617, 11.5563, 11.5477, 11.5346,
        11.5193, 11.5002, 11.4789, 11.4548, 11.4279, 11.3994, 11.3688, 11.3366,
        11.3036, 11.2696, 11.2363, 11.2007, 11.1666, 11.1333, 11.1018, 11.0714,
        11.0424, 11.0155, 10.9912, 10.9698, 10.9508, 10.9355, 10.9230, 10.9139,
        10.9083, 10.9069, 10.9057, 10.9013, 10.8951, 10.8854, 10.8740, 10.8591,
        10.8429, 10.8242, 10.8039, 10.7820, 10.7592, 10.7342, 10.7092, 10.6834,
        10.6572, 10.6312, 10.6052, 10.5803, 10.5560, 10.5328, 10.5101, 10.4904,
        10.4722, 10.4552, 10.4408, 10.4285, 10.4197, 10.4129, 10.4088, 10.4077,
        10.4077, 10.4077, 10.4077, 10.4077, 10.4077, 10.4077, 10.4077, 10.4077,
        10.4077, 10.4077, 10.4077, 10.4077, 10.4077, 10.4077, 10.4077, 10.4077,
        10.4077, 10.4077, 10.4077, 10.4077, 10.4077, 10.4077, 10.4077, 10.4077,
        10.4077, 10.4077]

    showlog('Extracting zip...')

    idir = os.path.dirname(ifile)
    with zipfile.ZipFile(ifile) as zfile:
        zipnames = zfile.namelist()
        zfile.extractall(idir)

    zipnames2 = [i for i in zipnames if i[-3:].lower() == 'tif']

    # This section is to import the correct scale factors
    # Pick first txt file since other should be readme.txt
    header = [i for i in zipnames if '.txt' in i.lower()]

    scale_vnir = 40.
    scale_swir = 80.

    if len(header) > 0:
        hfile = header[0]

        with open(os.path.join(idir, hfile), encoding='utf-8') as headerfile:
            txt = headerfile.read()

        txt = txt.split('\n')
        scale_vnir = [i for i in txt if 'SCALING_FACTOR_VNIR' in i][0]
        scale_swir = [i for i in txt if 'SCALING_FACTOR_SWIR' in i][0]
        datetxt = [i for i in txt if 'ACQUISITION_DATE' in i][0]

        scale_vnir = float(scale_vnir.split(' = ')[-1])
        scale_swir = float(scale_swir.split(' = ')[-1])
        datetxt = datetxt.split(' = ')[-1].strip()
        date = datetime.datetime.strptime(datetxt, '%Y-%m-%d')

    showlog('Importing Hyperion data...')

    nval = 0
    dat = []
    maskall = None
    for ifile2 in piter(zipnames2):
        fext = ifile2.split('_')[1]
        bandno = int(fext[1:])

        # Eliminate bands not illuminated
        if bandno <= 7 or bandno >= 225:
            continue

        # Overlap Region
        if 58 <= bandno <= 78:
            continue

        bname = f'Band {bandno}: {wavelength[bandno-1]} nm'
        if tnames is not None and bname not in tnames:
            continue

        showlog(f'Importing band {bandno}: {wavelength[bandno-1]} nm')
        dataset = rasterio.open(os.path.join(idir, ifile2))

        if dataset is None:
            showlog(f'Problem with band {bandno}')
            continue

        dat.append(Data())

        if metaonly is False:
            rtmp = dataset.read(1)

            if bandno <= 70:
                rtmp = rtmp / scale_vnir
            else:
                rtmp = rtmp / scale_swir

            dat[-1].data = rtmp

            dat[-1].data = np.ma.masked_invalid(dat[-1].data)
            dat[-1].data.mask = dat[-1].data.mask | (dat[-1].data == nval)
            if dat[-1].data.mask.size == 1:
                dat[-1].data.mask = (np.ma.make_mask_none(dat[-1].data.shape) +
                                     dat[-1].data.mask)

            if maskall is None:
                maskall = dat[-1].data.mask
            else:
                maskall = np.logical_and(maskall, dat[-1].data.mask)

        dat[-1].dataid = bname
        dat[-1].nodata = nval
        dat[-1].meta_from_rasterio(dataset)
        dat[-1].filename = ifile

        bmeta = {}
        bmeta['Sensor'] = 'Hyperion EO1H'
        # if satbands is not None and fext in satbands:

        dat[-1].metadata['Raster']['wavelength'] = wavelength[bandno-1]
        bmeta['WavelengthMin'] = wavelength[bandno-1]-fwhm[bandno-1]/2
        bmeta['WavelengthMax'] = wavelength[bandno-1]+fwhm[bandno-1]/2

        dat[-1].metadata['Raster'].update(bmeta)
        dat[-1].datetime = date

        dataset.close()

    if not dat:
        dat = None

    for i in dat:
        i.data.mask = maskall

    showlog('Cleaning Extracted zip files...')
    for zfile in zipnames:
        os.remove(os.path.join(idir, zfile))

    showlog('Import complete')
    return dat


def get_sentinel1(ifile, piter=None, showlog=print, tnames=None,
                  metaonly=False):
    """
    Get Sentinel-1 Data.

    Parameters
    ----------
    ifile : str
        filename to import
    piter : function, optional
        Progress bar iterable. Default is None.
    showlog : function, optional
        Routine to show text messages. The default is print.
    tnames : list, optional
        list of band names to import, in order. The default is None.
    metaonly : bool, optional
        Retrieve only the metadata for the file. The default is False.

    Returns
    -------
    dat : PyGMI raster Data
        dataset imported
    """
    if piter is None:
        piter = ProgressBarText().iter

    ifile = ifile[:]

    with rasterio.open(ifile) as dataset:
        if dataset is None:
            return None
        subdata = dataset.subdatasets

    nval = 0
    dat = []

    for bfile in subdata:
        tmp = bfile.split(':')
        bname = f'{tmp[0]}_{tmp[1]}_{tmp[-2]}_{tmp[-1]}'

        dataset1 = rasterio.open(bfile)
        showlog('Importing '+bname)
        if dataset1 is None:
            showlog('Problem with '+ifile)
            continue

        dataset = rasterio.vrt.WarpedVRT(dataset1)

        for i in piter(dataset.indexes):
            bmeta = dataset.tags(i)
            print(bmeta)

            if tnames is not None and bname not in tnames:
                continue

            dat.append(Data())

            if not metaonly:
                rtmp = dataset.read(i)

                dat[-1].data = rtmp
                dat[-1].data = np.ma.masked_invalid(dat[-1].data)
                dat[-1].data.mask = dat[-1].data.mask | (dat[-1].data == nval)
                if dat[-1].data.mask.size == 1:
                    dat[-1].mask = np.ma.getmaskarray(dat[-1].data)

            dat[-1].dataid = bname
            dat[-1].nodata = nval
            dat[-1].meta_from_rasterio(dataset)
            dat[-1].filename = ifile

            bmeta['Raster'] = dat[-1].metadata['Raster']
            bmeta['Raster']['Sensor'] = 'Sentinel-1'

            dat[-1].metadata.update(bmeta)

        dataset.close()

    if not dat:
        dat = None

    return dat


def get_sentinel2(ifile, piter=None, showlog=print, tnames=None,
                  metaonly=False):
    """
    Get Sentinel-2 Data.

    Parameters
    ----------
    ifile : str
        filename to import
    piter : function, optional
        Progress bar iterable. Default is None.
    showlog : function, optional
        Routine to show text messages. The default is print.
    tnames : list, optional
        list of band names to import, in order. The default is None.
    metaonly : bool, optional
        Retrieve only the metadata for the file. The default is False.

    Returns
    -------
    dat : PyGMI raster Data
        dataset imported
    """
    if piter is None:
        piter = ProgressBarText().iter

    ifile = ifile[:]

    with rasterio.open(ifile) as dataset:
        if dataset is None:
            return None
        subdata = dataset.subdatasets
        meta = dataset.tags()
        datetxt = meta['PRODUCT_STOP_TIME']
        date = datetime.datetime.fromisoformat(datetxt)

    subdata = [i for i in subdata if 'TCI' not in i]  # TCI is true color

    nval = 0
    dat = []
    for bfile in subdata:
        dataset = rasterio.open(bfile)
        showlog('Importing '+os.path.basename(bfile))
        if dataset is None:
            showlog('Problem with '+ifile)
            continue
        plvl = dataset.tags()['PROCESSING_LEVEL']

        for i in piter(dataset.indexes):
            bmeta = dataset.tags(i)

            bname = dataset.descriptions[i-1]+f' ({dataset.transform[0]}m)'
            bname = bname.replace(',', ' ')
            if tnames is not None and bname not in tnames:
                continue

            dat.append(Data())

            if not metaonly:
                rtmp = dataset.read(i)

                dat[-1].data = rtmp
                dat[-1].data = np.ma.masked_invalid(dat[-1].data)
                dat[-1].data.mask = dat[-1].data.mask | (dat[-1].data == nval)
                if dat[-1].data.mask.size == 1:
                    dat[-1].mask = np.ma.getmaskarray(dat[-1].data)

                if 'WAVELENGTH' in bmeta and plvl == 'Level-2A':
                    dat[-1].data = dat[-1].data.astype(np.float32)
                    dat[-1].data = dat[-1].data / 10000.

            dat[-1].dataid = bname
            dat[-1].nodata = nval
            dat[-1].meta_from_rasterio(dataset)
            dat[-1].filename = ifile
            if 'WAVELENGTH' in bmeta and plvl == 'Level-2A':
                dat[-1].units = 'Reflectance'
            dat[-1].datetime = date

            bmeta['Raster'] = dat[-1].metadata['Raster']
            bmeta['Raster']['Sensor'] = 'Sentinel-2'
            if 'WAVELENGTH' in bmeta and 'BANDWIDTH' in bmeta:
                wlen = float(bmeta['WAVELENGTH'])
                bwidth = float(bmeta['BANDWIDTH'])
                bmeta['Raster']['WavelengthMin'] = wlen - bwidth/2
                bmeta['Raster']['WavelengthMax'] = wlen + bwidth/2
                dat[-1].metadata['Raster']['wavelength'] = wlen

            dat[-1].metadata.update(bmeta)

        dataset.close()

    if not dat:
        dat = None

    return dat


def get_spot(ifile, piter=None, showlog=print, tnames=None, metaonly=False):
    """
    Get Spot DIMAP Data.

    Parameters
    ----------
    ifile : str
        filename to import
    piter : function, optional
        Progress bar iterable. Default is None.
    showlog : function, optional
        Routine to show text messages. The default is print.
    tnames : list, optional
        list of band names to import, in order. The default is None.
    metaonly : bool, optional
        Retrieve only the metadata for the file. The default is False.

    Returns
    -------
    dat : PyGMI raster Data
        dataset imported
    """
    if piter is None:
        piter = ProgressBarText().iter

    ifile = ifile[:]
    nval = 0
    dat = []

    with rasterio.open(ifile) as dataset:
        if dataset is None:
            return None
        # subdata = dataset.subdatasets
        meta = dataset.tags()
        datetxt = meta['DATASET_PRODUCTION_DATE']
        date = datetime.datetime.fromisoformat(datetxt)

        for i in piter(dataset.indexes):
            bmeta = dataset.tags(i)
            bname = f'Band {i} ({dataset.transform[0]}m)'
            bname = bname.replace(',', ' ')
            if tnames is not None and bname not in tnames:
                continue

            dat.append(Data())

            if not metaonly:
                rtmp = dataset.read(i)
                gain = float(bmeta['RADIANCE_GAIN'])
                bias = float(bmeta['RADIANCE_BIAS'])

                dat[-1].data = rtmp/gain+bias
                dat[-1].data = np.ma.masked_invalid(dat[-1].data)
                dat[-1].data.mask = dat[-1].data.mask | (dat[-1].data == nval)
                if dat[-1].data.mask.size == 1:
                    dat[-1].mask = np.ma.getmaskarray(dat[-1].data)

            dat[-1].dataid = bname
            dat[-1].nodata = nval
            dat[-1].meta_from_rasterio(dataset)
            dat[-1].filename = ifile
            dat[-1].datetime = date

            bmeta['Raster'] = dat[-1].metadata['Raster']
            bmeta['Raster']['Sensor'] = 'Spot'
            wmin = float(bmeta['SPECTRAL_RANGE_MIN'])
            wmax = float(bmeta['SPECTRAL_RANGE_MAX'])
            wlen = (wmin+wmax)/2
            bmeta['Raster']['WavelengthMin'] = wmin
            bmeta['Raster']['WavelengthMax'] = wmax
            bmeta['Raster']['wavelength'] = wlen

            dat[-1].metadata.update(bmeta)

            dat[-1].units = bmeta['RADIANCE_MEASURE_UNIT']

        dataset.close()

    if not dat:
        dat = None

    return dat


def get_aster_zip(ifile, piter=None, showlog=print, tnames=None,
                  metaonly=False):
    """
    Get ASTER zip Data.

    Parameters
    ----------
    ifile : str
        filename to import
    piter : function, optional
        Progress bar iterable. Default is None.
    showlog : function, optional
        Routine to show text messages. The default is print.
    tnames : list, optional
        list of band names to import, in order. The default is None.
    metaonly : bool, optional
        Retrieve only the metadata for the file. The default is False.

    Returns
    -------
    dat : PyGMI raster Data
        dataset imported
    """
    if piter is None:
        piter = ProgressBarText().iter

    satbands = {'1': [520, 600],
                '2': [630, 690],
                '3N': [780, 860],
                '3B': [780, 860],
                '4': [1600, 1700],
                '5': [2145, 2185],
                '6': [2185, 2225],
                '7': [2235, 2285],
                '8': [2295, 2365],
                '9': [2360, 2430],
                '10': [8125, 8475],
                '11': [8475, 8825],
                '12': [8925, 9275],
                '13': [10250, 10950],
                '14': [10950, 11650]}

    if 'AST_07' in ifile:
        scalefactor = 0.001
        units = 'Surface Reflectance'
    elif 'AST_05' in ifile:
        scalefactor = 0.001
        units = 'Surface Emissivity'
    elif 'AST_08' in ifile:
        scalefactor = 0.1
        units = 'Surface Kinetic Temperature'
    else:
        return None

    # Get Date
    datetxt = os.path.basename(ifile).split('_')[2][3:]
    date = datetime.datetime.strptime(datetxt, '%m%d%Y%H%M%S')

    showlog('Extracting zip...')

    idir = os.path.dirname(ifile)
    with zipfile.ZipFile(ifile) as zfile:
        zipnames = zfile.namelist()
        zfile.extractall(idir)

    platform = os.path.basename(ifile).split('_')[1]

    if 'VNIR' in zipnames[0]:
        platform += ' VNIR'
    elif 'SWIR' in zipnames[0]:
        platform += ' SWIR'

    wkt_lat = None
    wkt_lon = None
    for zfile in zipnames:
        if 'Latitude' in zfile:
            wkt_lat = np.loadtxt(os.path.join(idir, zfile)).mean()
        if 'Longitude' in zfile:
            wkt_lon = np.loadtxt(os.path.join(idir, zfile)).mean()

    dat = []
    nval = 0
    for zfile in piter(zipnames):
        if zfile.lower()[-4:] != '.tif':
            continue

        bname = zfile[zfile.index('Band'):zfile.index('.tif')]
        if tnames is not None and bname not in tnames:
            continue

        dataset1 = rasterio.open(os.path.join(idir, zfile))
        if dataset1 is None:
            showlog('Problem with '+zfile)
            continue

        wkt = dataset1.crs.to_wkt()

        dataset = rasterio.vrt.WarpedVRT(dataset1)
        dat.append(Data())

        if metaonly is False:
            dat[-1].data = dataset.read(1)
            dat[-1].data = np.ma.masked_invalid(dat[-1].data)*scalefactor
            dat[-1].data.mask = dat[-1].data.mask | (dat[-1].data == nval)
            if dat[-1].data.mask.size == 1:
                dat[-1].mask = np.ma.getmaskarray(dat[-1].data)

        dat[-1].dataid = bname
        dat[-1].nodata = nval
        dat[-1].meta_from_rasterio(dataset)
        if 'LOCAL_CS' in wkt and wkt_lat is not None and wkt_lon is not None:
            epsg = convert_ll_to_utm(wkt_lon, wkt_lat)
            dat[-1].crs = CRS.from_epsg(epsg)

        dat[-1].filename = ifile
        dat[-1].units = units
        dat[-1].datetime = date

        bmeta = dat[-1].metadata['Raster']
        fext = dat[-1].dataid[4:]

        bmeta['Sensor'] = f'ASTER {platform}'
        bmeta['WavelengthMin'] = satbands[fext][0]
        bmeta['WavelengthMax'] = satbands[fext][1]
        bmeta['wavelength'] = (satbands[fext][0]+satbands[fext][1])/2

        dataset.close()
        dataset1.close()

    showlog('Cleaning Extracted zip files...')
    for zfile in zipnames:
        os.remove(os.path.join(idir, zfile))

    return dat


def get_aster_hdf(ifile, piter=None, showlog=print, tnames=None,
                  metaonly=False):
    """
    Get ASTER hdf Data.

    This function needs the original filename to extract the date.

    Parameters
    ----------
    ifile : str
        filename to import
    piter : function, optional
        Progress bar iterable. Default is None.
    showlog : function, optional
        Routine to show text messages. The default is print.
    tnames : list, optional
        list of band names to import, in order. The default is None.
    metaonly : bool, optional
        Retrieve only the metadata for the file. The default is False.

    Returns
    -------
    dat : PyGMI raster Data
        dataset imported
    """
    if piter is None:
        piter = ProgressBarText().iter

    satbands = {'1': [520, 600],
                '2': [630, 690],
                '3N': [780, 860],
                '3B': [780, 860],
                '4': [1600, 1700],
                '5': [2145, 2185],
                '6': [2185, 2225],
                '7': [2235, 2285],
                '8': [2295, 2365],
                '9': [2360, 2430],
                '10': [8125, 8475],
                '11': [8475, 8825],
                '12': [8925, 9275],
                '13': [10250, 10950],
                '14': [10950, 11650]}

    ifile = ifile[:]

    if 'AST_07' in ifile:
        ptype = '07'
    elif 'AST_L1T' in ifile:
        ptype = 'L1T'
    elif 'AST_05' in ifile:
        ptype = '05'
    elif 'AST_08' in ifile:
        ptype = '08'
    else:
        return None

    # Get Date
    datetxt = os.path.basename(ifile).split('_')[2][3:]
    date = datetime.datetime.strptime(datetxt, '%m%d%Y%H%M%S')

    with rasterio.open(ifile) as dataset:
        meta = dataset.tags()
        subdata = dataset.subdatasets

    if ptype == 'L1T':
        ucc = {'ImageData1': float(meta['INCL1']),
               'ImageData2': float(meta['INCL2']),
               'ImageData3N': float(meta['INCL3N']),
               'ImageData4': float(meta['INCL4']),
               'ImageData5': float(meta['INCL5']),
               'ImageData6': float(meta['INCL6']),
               'ImageData7': float(meta['INCL7']),
               'ImageData8': float(meta['INCL8']),
               'ImageData9': float(meta['INCL9']),
               'ImageData10': float(meta['INCL10']),
               'ImageData11': float(meta['INCL11']),
               'ImageData12': float(meta['INCL12']),
               'ImageData13': float(meta['INCL13']),
               'ImageData14': float(meta['INCL14'])}

    solarelev = float(meta['SOLARDIRECTION'].split()[1])
    cdate = meta['CALENDARDATE']
    if len(cdate) == 8:
        fmt = '%Y%m%d'
    else:
        fmt = '%Y-%m-%d'
    dte = datetime.datetime.strptime(cdate, fmt)
    jdate = dte.timetuple().tm_yday

    if ptype == '07':
        subdata = [i for i in subdata if 'SurfaceReflectance' in i]
        scalefactor = 0.001
        units = 'Surface Reflectance'
    elif ptype == '05':
        subdata = [i for i in subdata if 'SurfaceEmissivity' in i]
        scalefactor = 0.001
        units = 'Surface Emissivity'
    elif ptype == '08':
        scalefactor = 0.1
        units = 'Surface Kinetic Temperature'
    elif ptype == 'L1T':
        subdata = [i for i in subdata if 'ImageData' in i]
        scalefactor = 1
        units = ''
    else:
        return None

    platform = os.path.basename(ifile).split('_')[1]

    if 'VNIR' in subdata[0]:
        platform += ' VNIR'
    elif 'SWIR' in subdata[0]:
        platform += ' SWIR'

    dat = []
    nval = 0
    calctoa = False
    for bfile in piter(subdata):
        if 'QA' in bfile:
            continue
        if ptype == 'L1T' and 'ImageData3B' in bfile:
            continue

        bname = bfile.split(':')[-1]
        if tnames is not None and bname not in tnames:
            continue

        dat.append(Data())

        dataset1 = rasterio.open(bfile)
        dataset = rasterio.vrt.WarpedVRT(dataset1)

        if metaonly is False:
            dat[-1].data = dataset.read(1)
            if ptype == '08':
                dat[-1].data[dat[-1].data == 2000] = nval
            dat[-1].data = np.ma.masked_invalid(dat[-1].data)*scalefactor
            dat[-1].data.mask = dat[-1].data.mask | (dat[-1].data == nval)
            if dat[-1].data.mask.size == 1:
                dat[-1].mask = np.ma.getmaskarray(dat[-1].data)

        dat[-1].meta_from_rasterio(dataset)

        dataset.close()
        dataset1.close()

        dat[-1].dataid = bname
        dat[-1].nodata = nval
        dat[-1].metadata['SolarElev'] = solarelev
        dat[-1].metadata['JulianDay'] = jdate
        dat[-1].metadata['CalendarDate'] = cdate
        dat[-1].metadata['ShortName'] = meta['SHORTNAME']
        dat[-1].filename = ifile
        dat[-1].units = units
        dat[-1].datetime = date

        if 'band' in dat[-1].dataid.lower():
            bmeta = dat[-1].metadata['Raster']
            fext = dat[-1].dataid[4:].split()[0]

            platform = os.path.basename(ifile).split('_')[1]
            bmeta['Sensor'] = f'ASTER {platform}'

            bmeta['WavelengthMin'] = satbands[fext][0]
            bmeta['WavelengthMax'] = satbands[fext][1]
            bmeta['wavelength'] = (satbands[fext][0] + satbands[fext][1])/2

            if ptype == 'L1T' and 'ImageData' in ifile:
                dat[-1].metadata['Gain'] = ucc[ifile[ifile.rindex('ImageData'):]]
                calctoa = True

    if not dat:
        dat = None

    elif ptype == 'L1T' and calctoa is True:
        dat = calculate_toa(dat)

    showlog('Import complete')
    return dat


def get_aster_ged(ifile, piter=None, showlog=print, tnames=None,
                  metaonly=False):
    """
    Get ASTER GED data.

    Parameters
    ----------
    ifile : str
        filename to import
    piter : function, optional
        Progress bar iterable. Default is None.
    showlog : function, optional
        Routine to show text messages. The default is print.
    tnames : list, optional
        list of band names to import, in order. The default is None.
    metaonly : bool, optional
        Retrieve only the metadata for the file. The default is False.

    Returns
    -------
    dat : PyGMI raster Data
        dataset imported
    """
    if piter is None:
        piter = ProgressBarText().iter

    dat = []
    ifile = ifile[:]

    with rasterio.open(ifile) as dataset:
        subdata = dataset.subdatasets
        meta = dataset.tags()
        try:
            year = int(meta['Emissivity_Mean_Description'].split('2000-')[-1])
            date = datetime.date(year, 1, 1)
        except:
            showlog('Date uncertian, setting it to 2000-01-01')
            date = datetime.date(2000, 1, 1)

    i = -1
    for ifile2 in subdata:
        units = ''

        if 'ASTER_GDEM' in ifile2:
            bandid = 'ASTER GDEM'
            units = 'meters'
        if 'Land_Water_Map' in ifile2:
            bandid = 'Land_water_map'
        if 'Observations' in ifile2:
            bandid = 'Observations'
            units = 'number per pixel'

        if tnames is not None and bandid not in tnames:
            continue

        dataset = rasterio.open(ifile2)

        nbands = 1

        if metaonly is False:
            rtmp2 = dataset.read()
            if 'Latitude' in ifile2:
                ymax = rtmp2.max()
                ydim = abs((rtmp2.max()-rtmp2.min())/rtmp2.shape[1])
                continue

            if 'Longitude' in ifile2:
                xmin = rtmp2.min()
                xdim = abs((rtmp2.max()-rtmp2.min())/rtmp2.shape[2])
                continue

            if rtmp2.shape[-1] == min(rtmp2.shape) and rtmp2.ndim == 3:
                rtmp2 = np.transpose(rtmp2, (2, 0, 1))

            if rtmp2.ndim == 3:
                nbands = rtmp2.shape[0]

        for i2 in range(nbands):
            nval = -9999
            i += 1

            dat.append(Data())
            if metaonly is False:
                if rtmp2.ndim == 3:
                    dat[i].data = rtmp2[i2]
                else:
                    dat[i].data = rtmp2

            dat[i].data = np.ma.masked_invalid(dat[i].data)
            dat[i].data.mask = (np.ma.getmaskarray(dat[i].data)
                                | (dat[i].data == nval))
            if dat[i].data.mask.size == 1:
                dat[-1].mask = np.ma.getmaskarray(dat[-1].data)

            dat[i].data = dat[i].data * 1.0
            if 'Emissivity/Mean' in ifile2:
                bandid = 'Emissivity_mean_band_'+str(10+i2)
                dat[i].data = dat[i].data * 0.001
            if 'Emissivity/SDev' in ifile2:
                bandid = 'Emissivity_std_dev_band_'+str(10+i2)
                dat[i].data = dat[i].data * 0.0001
            if 'NDVI/Mean' in ifile2:
                bandid = 'NDVI_mean'
                dat[i].data = dat[i].data * 0.01
            if 'NDVI/SDev' in ifile2:
                bandid = 'NDVI_std_dev'
                dat[i].data = dat[i].data * 0.01
            if 'Temperature/Mean' in ifile2:
                bandid = 'Temperature_mean'
                units = 'Kelvin'
                dat[i].data = dat[i].data * 0.01
            if 'Temperature/SDev' in ifile2:
                bandid = 'Temperature_std_dev'
                units = 'Kelvin'
                dat[i].data = dat[i].data * 0.01

            dat[i].dataid = bandid
            dat[i].nodata = nval
            dat[i].crs = CRS.from_epsg(4326)  # WGS84 geodetic
            dat[i].units = units
            dat[i].metadata['Raster']['Sensor'] = 'ASTER GED'
            dat[i].datetime = date
        dataset.close()

    if metaonly is False:
        for i in dat:
            i.set_transform(xdim, xmin, ydim, ymax)

    showlog('Import complete')
    return dat


def get_aster_ged_bin(ifile):
    """
    Get ASTER GED binary format.

    Emissivity_Mean_Description: Mean Emissivity for each pixel on grid-box
    using all ASTER data from 2000-2010
    Emissivity_SDev_Description: Emissivity Standard Deviation for each pixel
    on grid-box using all ASTER data from 2000-2010
    Temperature_Mean_Description: Mean Temperature (K) for each pixel on
    grid-box using all ASTER data from 2000-2010
    Temperature_SDev_Description: Temperature Standard Deviation for each pixel
    on grid-box using all ASTER data from 2000-2010
    NDVI_Mean_Description: Mean NDVI for each pixel on grid-box using all ASTER
    data from 2000-2010
    NDVI_SDev_Description: NDVI Standard Deviation for each pixel on grid-box
    using all ASTER data from 2000-2010
    Land_Water_Map_LWmap_Description: Land Water Map using ASTER visible bands
    Observations_NumObs_Description: Number of values used in computing mean
    and standard deviation for each pixel.
    Geolocation_Latitude_Description: Latitude
    Geolocation_Longitude_Description: Longitude
    ASTER_GDEM_ASTGDEM_Description: ASTER GDEM resampled to NAALSED

    Parameters
    ----------
    ifile : str
        filename to import

    Returns
    -------
    dat : PyGMI raster Data
        dataset imported
    """
    dat = []
    nval = -9999
    bandid = {}

    bandid[0] = 'Emissivity_mean_band_10'
    bandid[1] = 'Emissivity_mean_band_11'
    bandid[2] = 'Emissivity_mean_band_12'
    bandid[3] = 'Emissivity_mean_band_13'
    bandid[4] = 'Emissivity_mean_band_14'
    bandid[5] = 'Emissivity_std_dev_band_10'
    bandid[6] = 'Emissivity_std_dev_band_11'
    bandid[7] = 'Emissivity_std_dev_band_12'
    bandid[8] = 'Emissivity_std_dev_band_13'
    bandid[9] = 'Emissivity_std_dev_band_14'
    bandid[10] = 'Temperature_mean'
    bandid[11] = 'Temperature_std_dev'
    bandid[12] = 'NDVI_mean'
    bandid[13] = 'NDVI_std_dev'
    bandid[14] = 'Land_water_map'
    bandid[15] = 'Observations'
    bandid[16] = 'Latitude'
    bandid[17] = 'Longitude'
    bandid[18] = 'ASTER GDEM'

    scale = [0.001, 0.001, 0.001, 0.001, 0.001,
             0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
             0.01, 0.01, 0.01, 0.01,
             1, 1, 0.001, 0.001, 1]

    units = ['', '', '', '', '', '', '', '', '', '', 'Kelvin', 'Kelvin',
             '', '', '', 'Number per pixel', 'degrees', 'degrees', 'meters']

    data = np.fromfile(ifile, dtype=np.int32)
    rows_cols = int((data.size/19)**0.5)
    data.shape = (19, rows_cols, rows_cols)

    lats = data[16]*scale[16]
    lons = data[17]*scale[17]

    latsdim = (lats.max()-lats.min())/(lats.shape[0]-1)
    lonsdim = (lons.max()-lons.min())/(lons.shape[0]-1)

    tlx = lons.min()-abs(lonsdim/2)
    tly = lats.max()+abs(latsdim/2)

    for i in range(19):
        dat.append(Data())

        dat[i].data = data[i]*scale[i]

        dat[i].dataid = bandid[i]
        dat[i].nodata = nval*scale[i]
        dat[i].xdim = lonsdim
        dat[i].ydim = latsdim
        dat[i].units = units[i]

        rows, cols = dat[i].data.shape
        xmin = tlx
        ymax = tly
        ymin = ymax - rows*dat[i].ydim
        xmax = xmin + cols*dat[i].xdim

        dat[i].extent = [xmin, xmax, ymin, ymax]

    dat.pop(17)
    dat.pop(16)

    return dat


def get_ternary(dat, sunfile=None, clippercl=1., clippercu=1.,
                cell=25., alpha=.75, piter=iter, showlog=print):
    """
    Create a ternary image, with optional sunshading.

    Parameters
    ----------
    dat : list of PyGMI Data
        PyGMI Data.
    sunfile : str, optional
        Sunshading band or filename. The default is None.
    clippercl : float, optional
        Lower clip percentage for colour bar. The default is 1.
    clippercu : float, optional
        Upper clip percentage for colour bar. The default is 1.
    cell : float, optional
        Between 1 and 100 - controls sunshade detail. The default is 25.
    alpha : float, optional
        How much incident light is reflected (0 to 1). The default is .75.

    Returns
    -------
    newimg : list of PyGMI Data.
        list of PyGMI data.

    """
    data = lstack(dat, nodeepcopy=True, checkdataid=False, piter=piter,
                  showlog=showlog)
    sundata = None

    if sunfile is not None:
        sdata = None
        for i in data:
            if i.dataid == sunfile:
                sdata = i
                break

        if sdata is None:
            sundata = get_raster(sunfile, metaonly=True, piter=piter,
                                 showlog=showlog)

            owkt = sundata[0].crs.to_wkt()
            iwkt = data[0].crs.to_wkt()

            x = data[0].extent[:2]
            y = data[0].extent[2:]

            xx, yy = reprojxy(x, y, iwkt, owkt)

            bounds = [xx[0], yy[0], xx[1], yy[1]]

            sundata = get_raster(sunfile, bounds=bounds, piter=piter,
                                 showlog=showlog)
            sundata = data_reproject(sundata[0], data[0].crs)

            sdata = [data[0], sundata]
            masterid = sdata[0].dataid
            sdata = lstack(sdata, nodeepcopy=True, masterid=masterid,
                           resampling='bilinear', checkdataid=False,
                           piter=piter, showlog=showlog)[1]

    red = data[0].data
    green = data[1].data
    blue = data[2].data

    mask = np.logical_or(red.mask, green.mask)
    mask = np.logical_or(mask, blue.mask)
    mask = np.logical_not(mask)

    red, _, _ = histcomp(red, perc=clippercl, uperc=clippercu)
    green, _, _ = histcomp(green, perc=clippercl, uperc=clippercu)
    blue, _, _ = histcomp(blue, perc=clippercl, uperc=clippercu)

    img = np.ones((red.shape[0], red.shape[1], 4), dtype=np.uint8)
    img[:, :, 3] = mask*254+1

    img[:, :, 0] = norm255(red)
    img[:, :, 1] = norm255(green)
    img[:, :, 2] = norm255(blue)

    phi = -np.pi/4.
    theta = np.pi/4.

    if sundata is not None:
        sunshader = currentshader(sdata.data, cell, theta, phi, alpha)

        snorm = norm2(sunshader)

        img[:, :, 0] = img[:, :, 0]*snorm  # red
        img[:, :, 1] = img[:, :, 1]*snorm  # green
        img[:, :, 2] = img[:, :, 2]*snorm  # blue
        img = img.astype(np.uint8)

    newimg = [data[0].copy(),
              data[0].copy(),
              data[0].copy(),
              data[0].copy()]

    newimg[0].data = img[:, :, 0]
    newimg[1].data = img[:, :, 1]
    newimg[2].data = img[:, :, 2]
    newimg[3].data = img[:, :, 3]

    mask = img[:, :, 3]
    newimg[0].data[newimg[0].data == 0] = 1
    newimg[1].data[newimg[1].data == 0] = 1
    newimg[2].data[newimg[2].data == 0] = 1

    newimg[0].data[mask <= 1] = 0
    newimg[1].data[mask <= 1] = 0
    newimg[2].data[mask <= 1] = 0

    newimg[0].nodata = 0
    newimg[1].nodata = 0
    newimg[2].nodata = 0
    newimg[3].nodata = 0

    newimg[0].dataid = data[0].dataid
    newimg[1].dataid = data[1].dataid
    newimg[2].dataid = data[2].dataid
    newimg[3].dataid = 'Alpha'

    return newimg


def set_export_filename(dat, odir, otype=None):
    """
    Set the export filename according to convention.

    Different satellite products have different simplified conventions for
    output filenames to avoid names getting too long.

    Parameters
    ----------
    dat : list of PyGMI Data.
        List of PyGMI data.
    odir : str
        Output directory.
    otype : str, optional.
        Output file type. Default is None.

    Returns
    -------
    ofile : str
        Output file name.

    """
    sensor = dat[0].metadata['Raster']['Sensor']
    filename = os.path.basename(dat[0].filename)

    # Deal with Sentinel-2 directory case.
    if filename == 'MTD_MSIL2A.xml':
        filename = os.path.basename(os.path.dirname(dat[0].filename))

    filename = os.path.splitext(filename)[0]

    filename = filename.replace('_stack', '')
    filename = filename.replace('_ratio', '')

    if otype is None:
        otype = 'stack'

    if 'ASTER' in sensor:
        tmp = [os.path.basename(i.filename).split('_')[1] for i in dat]
        tmp = list(set(tmp))
        tmp.sort()

        tmp = filename.split('_')
        date = tmp[2][3:11]
        time = tmp[2][11:]
        ofile = f'AST_{date}_{time}'
    elif 'Landsat' in sensor:
        ofile = '_'.join(filename.split('_')[:4])
    elif 'Sentinel-2' in sensor:
        tmp = filename.split('_')
        mission = tmp[0]
        date = tmp[2].split('T')[0]
        row = tmp[4]
        tile = tmp[5]
        ofile = f'{mission}_{tile}_{row}_{date}'
    else:
        ofile = filename

    if otype == 'RGB':
        tmp = [i.dataid.split()[0] for i in dat]
        for i in tmp:
            ofile += f'_{i.lower().replace("band", "b")}'
    else:
        ofile += f'_{otype}'

    ofile = os.path.join(odir, ofile)

    return ofile


def utm_to_south(dat):
    """
    Make sure all UTM labels are for southern hemisphere.

    Parameters
    ----------
    dat : list of PyGMI Data
        List of Data.

    Returns
    -------
    dat : list of PyGMI Data
        List of Data.

    """
    for band in dat:
        epsgcode = band.crs.to_epsg()
        if epsgcode is None:
            continue
        if 32600 <= epsgcode <= 32660:
            epsgcode += 100
            band.crs = CRS.from_epsg(epsgcode)

            left, right, bottom, top = band.extent
            top = top + 10000000
            bottom = bottom + 10000000
            xdim = band.xdim
            ydim = band.ydim

            band.transform = Affine(xdim, 0, left, 0, -ydim, top)
            band.extent = (left, right, bottom, top)
            band.bounds = (left, bottom, right, top)

    return dat


def _test5P():
    """Test routine."""
    import matplotlib.pyplot as plt

    sfile = r"D:\Workdata\PyGMI Test Data\Remote Sensing\Import\Sentinel-5P\CCUS_Sept2021_25kmbuffer.shp"
    ifile = r"D:\Workdata\PyGMI Test Data\Remote Sensing\Import\Sentinel-5P\S5P_OFFL_L2__CH4____20230111T102529_20230111T120700_27184_03_020400_20230113T024518.nc"

    os.chdir(r"D:\Workdata\PyGMI Test Data\Remote Sensing\Import\Sentinel-5P")

    app = QtWidgets.QApplication(sys.argv)
    tmp = ImportSentinel5P()
    tmp.ifile = ifile
    tmp.settings()

    if 'Vector' not in tmp.outdata:
        print('No data')
        return

    shp = gpd.read_file(sfile)
    shp = shp.to_crs(4326)

    plt.figure(dpi=150)
    ax = plt.gca()
    shp.plot(ax=ax, fc='none', ec='black')

    tmp.outdata['Vector']['Point'].plot(ax=ax, column='data')
    plt.show()


def _testfn2():
    """Test routine."""
    app = QtWidgets.QApplication(sys.argv)

    tmp1 = ImportBatch()
    tmp1.idir = r"D:\Landsat"
    tmp1.idir = r'D:/Workdata/PyGMI Test Data/Remote Sensing/ConditionIndex'
    tmp1.get_sfile(True)
    tmp1.settings()

    dat = tmp1.outdata

    tmp2 = ExportBatch()
    tmp2.indata = dat
    tmp2.run()


def _testfn3():
    """Test routine."""
    import matplotlib.pyplot as plt

    ifile = r"D:\Workdata\PyGMI Test Data\Remote Sensing\Import\ASTER\AST_07XT_00304132006083806_20180608052446_30254.hdf"
    ifile = r"D:\AST_05_00307292006082045_20240308070825_1288044.zip"

    app = QtWidgets.QApplication(sys.argv)

    os.chdir(os.path.dirname(ifile))
    tmp1 = ImportData()
    tmp1.settings()

    dat = tmp1.outdata['Raster']

    ofile = set_export_filename(dat, odir='')
    breakpoint()

    print(dat[-1].datetime)

    for i in dat:
        plt.figure(dpi=150)
        plt.title(i.dataid)
        plt.imshow(i.data, extent=i.extent)
        plt.colorbar()
        plt.show()


def _testfn4():

    idir = r'D:/DRC'
    odir = r'D:/DRC/stack'

    ifiles = glob.glob(r'D:/DRC/*.zip')

    for ifile in ifiles:
        dat = get_data(ifile)

        idir = os.path.dirname(ifile)
        adate = os.path.basename(ifile).split('_')[2]
        ifiles = glob.glob(os.path.join(idir, '*'+adate+'*.hdf'))

        dat2 = get_data(ifiles[0])

        crs = dat[0].crs
        for i in dat2:
            i.crs = crs
            xdim = i.transform[0]
            ydim = i.transform[4]
            xmin = i.transform[2]
            ymax = i.transform[5]+10000000
            i.set_transform(xdim, xmin, ydim, ymax)

        dat = dat + dat2
        ofile = set_export_filename(dat, odir)

        export_raster(ofile+'.tif', dat)


if __name__ == "__main__":
    _testfn4()
