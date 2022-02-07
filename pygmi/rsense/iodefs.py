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
import copy
import glob
import tarfile
import zipfile
import datetime
from PyQt5 import QtWidgets, QtCore
import numpy as np
import pandas as pd

import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point
import rasterio
from rasterio.crs import CRS

from pygmi import menu_default
from pygmi.raster.datatypes import Data
from pygmi.misc import ProgressBarText

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


class ImportData():
    """
    Import Data - Interfaces with rasterio routines.

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    outdata : dictionary
        dictionary of output datasets
    ifile : str
        input file name. Used in main.py
    """

    def __init__(self, parent=None, extscene=None):
        self.ifile = ''
        self.filt = ''
        self.parent = parent
        self.indata = {}
        self.outdata = {}
        self.extscene = extscene
        if parent is None:
            self.showprocesslog = print
        else:
            self.showprocesslog = parent.showprocesslog

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
        piter = self.parent.pbar.iter

        if self.extscene is None:
            return False

        if not nodialog:
            self.ifile, self.filt = QtWidgets.QFileDialog.getOpenFileName(
                self.parent, 'Open File', '.', self.extscene)
            if self.ifile == '':
                return False
        os.chdir(os.path.dirname(self.ifile))

        dat = get_data(self.ifile, piter, self.showprocesslog, self.extscene)

        if dat is None:
            if self.filt == 'hdf (*.hdf *.h5)':
                QtWidgets.QMessageBox.warning(self.parent, 'Error',
                                              'Could not import the data.'
                                              'Currently only ASTER'
                                              'is supported.',
                                              QtWidgets.QMessageBox.Ok)
            else:
                QtWidgets.QMessageBox.warning(self.parent, 'Error',
                                              'Could not import the data.',
                                              QtWidgets.QMessageBox.Ok)
            return False

        output_type = 'Raster'
        self.outdata[output_type] = dat

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
        self.filt = projdata['filt']
        self.extscene = projdata['extscene']

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
        projdata['filt'] = self.filt
        projdata['extscene'] = self.extscene

        return projdata


class ImportBatch():
    """
    Batch Import Data Interface.

    This does not actually import data, but rather defines a list of datasets
    to be used by other routines.

    Attributes
    ----------
    parent : parent
        reference to the parent routine.
    idir : str
        Input directory.
    ifile : str
        Input file.
    indata : dictionary
        dictionary of input datasets.
    outdata : dictionary
        dictionary of output datasets.
    """

    def __init__(self, parent=None):
        self.ifile = ''
        self.idir = ''
        self.parent = parent
        self.indata = {}
        self.outdata = {}

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
        if not nodialog or self.idir == '':
            self.idir = QtWidgets.QFileDialog.getExistingDirectory(
                self.parent, 'Select Directory')
            if self.idir == '':
                return False
        os.chdir(self.idir)

        zipdat = glob.glob(self.idir+'//AST*.zip')
        hdfdat = glob.glob(self.idir+'//AST*.hdf')
#        tifdat = glob.glob(directory+'//AST*.tif')
        targzdat = glob.glob(self.idir+'//L*.tar*')
        mtldat = glob.glob(self.idir+'//L*MTL.txt')
        rasterdat = []
        for ftype in ['*.tif', '*.hdr', '*.img', '*.ers']:
            rasterdat += glob.glob(os.path.join(self.idir, ftype))

        sendat = []
        sendir = [f.path for f in os.scandir(self.idir) if f.is_dir() and
                  'SAFE' in f.path]
        for i in sendir:
            sendat.extend(glob.glob(i+'//MTD*.xml'))

        if (not hdfdat and not zipdat and not targzdat and not mtldat and not
                sendat and not rasterdat):
            QtWidgets.QMessageBox.warning(self.parent, 'Error',
                                          'No valid files in the directory.',
                                          QtWidgets.QMessageBox.Ok)
            return False

        dat = []
        for i in hdfdat:
            if 'met' not in i:
                dat.append(i)

        dat.extend(mtldat)
        dat.extend(targzdat)
        dat.extend(zipdat)
        dat.extend(sendat)
        dat.extend(rasterdat)

        # for i in tifdat:
        #     if i[:i.rindex('_')]+'_MTL.txt' in mtldat:
        #         continue
        #     dat.append(i)

        output_type = 'RasterFileList'
        self.outdata[output_type] = dat

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
        self.idir = projdata['idir']

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

        projdata['idir'] = self.idir

        return projdata


class ImportSentinel5P(QtWidgets.QDialog):
    """
    Import Sentinel 5P data to shapefile.

    This class imports Sentinel 5P data.

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    outdata : dictionary
        dictionary of output datasets
    ifile : str
        input file name. Used in main.py
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        if parent is None:
            self.showprocesslog = print
        else:
            self.showprocesslog = parent.showprocesslog

        self.parent = parent
        self.indata = {}
        self.outdata = {}
        self.ifile = ''
        self.filt = ''

        self.subdata = QtWidgets.QComboBox()
        self.lonmin = QtWidgets.QLineEdit('16')
        self.lonmax = QtWidgets.QLineEdit('34')
        self.latmin = QtWidgets.QLineEdit('-35')
        self.latmax = QtWidgets.QLineEdit('-21')

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
        helpdocs = menu_default.HelpButton('pygmi.rsense.iodefs.importsentinel5p')
        label_subdata = QtWidgets.QLabel('Product:')
        label_lonmin = QtWidgets.QLabel('Minimum Longitude:')
        label_lonmax = QtWidgets.QLabel('Maximum Longitude:')
        label_latmin = QtWidgets.QLabel('Minimum Latitude:')
        label_latmax = QtWidgets.QLabel('Maximum Latitude:')

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle(r'Import Sentinel-5P Data')

        gridlayout_main.addWidget(label_subdata, 0, 0, 1, 1)
        gridlayout_main.addWidget(self.subdata, 0, 1, 1, 1)

        gridlayout_main.addWidget(label_lonmin, 1, 0, 1, 1)
        gridlayout_main.addWidget(self.lonmin, 1, 1, 1, 1)

        gridlayout_main.addWidget(label_lonmax, 2, 0, 1, 1)
        gridlayout_main.addWidget(self.lonmax, 2, 1, 1, 1)

        gridlayout_main.addWidget(label_latmin, 3, 0, 1, 1)
        gridlayout_main.addWidget(self.latmin, 3, 1, 1, 1)

        gridlayout_main.addWidget(label_latmax, 4, 0, 1, 1)
        gridlayout_main.addWidget(self.latmax, 4, 1, 1, 1)

        gridlayout_main.addWidget(helpdocs, 5, 0, 1, 1)
        gridlayout_main.addWidget(buttonbox, 5, 1, 1, 3)

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
        if not nodialog:
            ext = ('Sentinel-5P (*.nc)')

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
            if i in ['latitude', 'longitude']:
                continue
            tmp.append(i)

        self.subdata.clear()
        self.subdata.addItems(tmp)
        self.subdata.setCurrentIndex(0)

        tmp = self.exec_()

        if tmp != 1:
            return tmp

        try:
            _ = float(self.lonmin.text())
            _ = float(self.latmin.text())
            _ = float(self.lonmax.text())
            _ = float(self.latmax.text())
        except ValueError:
            self.showprocesslog('Value error - abandoning import')
            return False

        gdf = self.get_5P_data(meta)

        if gdf is None:
            return False

        dat = {gdf.geom_type.iloc[0]: gdf}
        self.outdata['Vector'] = dat

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
        self.filt = projdata['filt']

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
        projdata['filt'] = self.filt

        return projdata

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
            if 'qa_value' in i:
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
        with rasterio.open(meta['latitude']) as dataset:
            lats = dataset.read(1)

        with rasterio.open(meta['longitude']) as dataset:
            lons = dataset.read(1)

        del meta['latitude']
        del meta['longitude']

        if lats is None:
            self.showprocesslog('No Latitudes in dataset')
            return None

        lats = lats.flatten()
        lons = lons.flatten()
        pnts = np.transpose([lons, lats])

        lonmin = float(self.lonmin.text())
        latmin = float(self.latmin.text())
        lonmax = float(self.lonmax.text())
        latmax = float(self.latmax.text())

        mask = ((lats > latmin) & (lats < latmax) & (lons < lonmax) &
                (lons > lonmin))

        idfile = self.subdata.currentText()

        dfile = meta[idfile]

        with rasterio.open(dfile) as dataset:
            dat = dataset.read(1)

        dat1 = dat.flatten()

        if mask.shape != dat1.shape:
            return None

        dat1 = dat1[mask]
        pnts1 = pnts[mask]

        pnts1 = pnts1[dat1 != 9.96921e+36]
        dat1 = dat1[dat1 != 9.96921e+36]

        if dat1.size == 0:
            self.showprocesslog(idfile, 'is empty.')
            return None

        df = pd.DataFrame({'lon': pnts1[:, 0], 'lat': pnts1[:, 1]})
        df['data'] = dat1

        gdf = GeoDataFrame(df.drop(['lon', 'lat'], axis=1),
                           geometry=[Point(xy) for xy in zip(df.lon, df.lat)])

        gdf = gdf.set_crs("EPSG:4326")

        return gdf


class ImportShapeData():
    """
    Import Shapefile Data.

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    outdata : dictionary
        dictionary of output datasets
    ifile : str
        input file name. Used in main.py
    """

    def __init__(self, parent=None):
        self.parent = parent
        self.indata = {}
        self.outdata = {}
        self.ifile = ''

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
            ext = 'Shapefile (*.shp);;' + 'All Files (*.*)'

            self.ifile, _ = QtWidgets.QFileDialog.getOpenFileName(self.parent,
                                                                  'Open File',
                                                                  '.', ext)
            if self.ifile == '':
                return False
        os.chdir(os.path.dirname(self.ifile))

        gdf = gpd.read_file(self.ifile)
        dat = {gdf.geom_type.iloc[0]: gdf}

        self.outdata['Vector'] = dat

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


def calculate_toa(dat, showprocesslog=print):
    """
    Top of atmosphere correction.

    Includes VNIR, SWIR and TIR bands.

    Parameters
    ----------
    dat : Data
        PyGMI raster dataset
    showprocesslog : function, optional
        Routine to show text messages. The default is print.

    Returns
    -------
    out : Data
        PyGMI raster dataset
    """
    showprocesslog('Calculating top of atmosphere...')

    datanew = {}
    for datai in dat:
        datanew[datai.dataid.split()[1]] = copy.deepcopy(datai)

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


def get_data(ifile, piter=iter, showprocesslog=print, extscene=None):
    """
    Load a raster dataset off the disk using the rasterio libraries.

    It returns the data in a PyGMI data object.

    Parameters
    ----------
    ifile : str
        filename to import
    piter : iter, optional
        Progress bar iterable. Default is iter
    showprocesslog : function, optional
        Routine to show text messages. The default is print.
    extscene : str or None
        String used currently to give an option to limit bands in Sentinel-2

    Returns
    -------
    dat : PyGMI raster Data
        dataset imported
    """
    ifile = ifile[:]
    bfile = os.path.basename(ifile)

    if extscene is None:
        extscene = ['']

    showprocesslog('Importing', bfile)

    if 'AST_' in bfile and 'hdf' in bfile.lower():
        dat = get_aster_hdf(ifile, piter)
    elif 'AST_' in bfile and 'zip' in bfile.lower():
        dat = get_aster_zip(ifile, piter, showprocesslog)
    elif bfile[:4] in ['LT04', 'LT05', 'LE07', 'LC08', 'LM05']:
        dat = get_landsat(ifile, piter, showprocesslog)
    elif ('.xml' in bfile and '.SAFE' in ifile) or 'Sentinel-2' in extscene:
        dat = get_sentinel2(ifile, piter, showprocesslog, extscene)
    elif (('MOD' in bfile or 'MCD' in bfile) and 'hdf' in bfile.lower() and
          '.006.' in bfile):
        dat = get_modisv6(ifile, piter)
    elif 'AG1' in bfile and 'h5' in bfile.lower():
        dat = get_aster_ged(ifile, piter)
    elif 'Hyperion' in extscene:
        dat = get_hyperion(ifile, piter, showprocesslog)
    else:
        dat = None

    if dat is not None:
        for i in dat:
            i.dataid = i.dataid.replace(',', ' ')

    return dat


def get_modisv6(ifile, piter=None):
    """
    Get MODIS v006 data.

    Parameters
    ----------
    ifile : str
        filename to import
    piter : iter, optional
        Progress bar iterable. Default is iter

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

        wkt = dataset.crs.wkt
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

        dat.append(Data())
        dat[-1].data = np.ma.array(rtmp2, mask=mask)
        # dat[-1].data = np.ma.masked_invalid(dat[-1].data)
        # dat[-1].data.mask = dat[-1].data.mask | (dat[-1].data == nval)
        # if dat[-1].data.mask.size == 1:
        #     dat[-1].data.mask = (np.ma.make_mask_none(dat[-1].data.shape) +
        #                          dat[-1].data.mask)

        dat[-1].dataid = bandid
        dat[-1].nodata = 1e+20
        dat[-1].crs = crs
        dat[-1].set_transform(transform=dataset.transform)
        dat[-1].filename = ifile
        dat[-1].units = dataset.units[0]

        dataset.close()

    if lulc is not None:
        dat.append(copy.deepcopy(dat[0]))
        dat[-1].data = lulc
        dat[-1].dataid = ('LULC: out of earth=7, water=6, barren=5, snow=4, '
                          'wetland=3, urban=2, unclassifed=1')
        dat[-1].nodata = 0

    return dat


def get_landsat(ifilet, piter=None, showprocesslog=print):
    """
    Get Landsat Data.

    Parameters
    ----------
    ifilet : str
        filename to import
    piter : iter, optional
        Progress bar iterable. Default is iter
    showprocesslog : function, optional
        Routine to show text messages. The default is print.

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
        satbands = {'1': [450, 520],
                    '2': [520, 600],
                    '3': [630, 690],
                    '4': [760, 900],
                    '5': [1550, 1750],
                    '6': [10400, 12500],
                    '7': [2080, 2350]}

    if platform == '07':
        satbands = {'1': [450, 520],
                    '2': [520, 600],
                    '3': [630, 690],
                    '4': [770, 900],
                    '5': [1550, 1750],
                    '6': [10400, 12500],
                    '7': [2090, 2350],
                    '8': [520, 900]}

    if platform == '08':
        satbands = {'1': [430, 450],
                    '2': [450, 510],
                    '3': [530, 590],
                    '4': [640, 670],
                    '5': [850, 880],
                    '6': [1570, 1650],
                    '7': [2110, 2290],
                    '8': [500, 680],
                    '9': [1360, 1380],
                    '10': [1060, 11190],
                    '11': [11500, 12510]}

    idir = os.path.dirname(ifilet)

    if '.tar' in ifilet:
        with tarfile.open(ifilet) as tar:
            tarnames = tar.getnames()
            ifile = next((i for i in tarnames if '_MTL.txt' in i), None)
            if ifile is None:
                showprocesslog('Could not find MTL.txt file in tar archive')
                return None
            showprocesslog('Extracting tar...')

            tar.extractall(idir)
            ifile = os.path.join(idir, ifile)
    elif '_MTL.txt' in ifilet:
        ifile = ifilet
    else:
        showprocesslog('Input needs to be tar.gz or _MTL.txt')
        return None

    files = glob.glob(ifile[:-7]+'*[0-9].tif')
    if glob.glob(ifile[:-7]+'*ST_QA.tif'):
        if 'LC08' in ifile:
            lstband = '10'
        else:
            lstband = '6'
        satbands['LST'] = satbands[lstband]
        # ['LT04', 'LT05', 'LE07', 'LC08', 'LM05']

        # haslst = True
        # breakpoint()

    showprocesslog('Importing Landsat data...')

    nval = 0
    dat = []
    for ifile2 in piter(files):
        if 'B6_VCID' in ifile2:
            fext = ifile2[-12:-4]
        elif ifile2[-6].isdigit():
            fext = ifile2[-6:-4]
        else:
            fext = ifile2[-5]

        if fext == lstband:
            fext = 'LST'

        showprocesslog('Importing Band '+fext)
        dataset = rasterio.open(ifile2)

        if dataset is None:
            showprocesslog('Problem with band '+fext)
            continue

        rtmp = dataset.read(1)

        dat.append(Data())
        dat[-1].data = rtmp
        dat[-1].data = np.ma.masked_invalid(dat[-1].data)
        dat[-1].data.mask = dat[-1].data.mask | (dat[-1].data == nval)
        if dat[-1].data.mask.size == 1:
            dat[-1].data.mask = (np.ma.make_mask_none(dat[-1].data.shape) +
                                 dat[-1].data.mask)

        if 'L2SP' in ifile2:
            if fext == 'LST':
                showprocesslog('Converting band '+lstband+' to Kelvin. '
                               'Band renamed as LST')
                dat[-1].data = dat[-1].data*0.00341802 + 149.0
            else:
                showprocesslog('Converting band '+lstband+' to reflectance.')
                dat[-1].data = dat[-1].data*0.0000275 - 0.2

        dat[-1].dataid = 'Band' + fext
        dat[-1].nodata = nval
        dat[-1].crs = dataset.crs
        dat[-1].set_transform(transform=dataset.transform)
        dat[-1].filename = ifile

        bmeta = dat[-1].metadata
        if satbands is not None and fext in satbands:
            bmeta['WavelengthMin'] = satbands[fext][0]
            bmeta['WavelengthMax'] = satbands[fext][1]
        bmeta['Raster']['wavelength'] = (satbands[fext][1]+satbands[fext][1])/2

        dataset.close()

    if dat == []:
        dat = None

    if '.tar' in ifilet:
        showprocesslog('Cleaning Extracted tar files...')
        for tfile in piter(tarnames):
            print(tfile)
            os.remove(os.path.join(os.path.dirname(ifile), tfile))
    print('Import complete')
    return dat


def get_hyperion(ifile, piter=None, showprocesslog=print):
    """
    Get Landsat Data.

    Parameters
    ----------
    ifile : str
        filename to import
    piter : iter, optional
        Progress bar iterable. Default is iter
    showprocesslog : function, optional
        Routine to show text messages. The default is print.

    Returns
    -------
    out : Data
        PyGMI raster dataset
    """
    if piter is None:
        piter = ProgressBarText().iter

    # bandnames = [f'Band {i+1}' for i in range(242)]
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

    # for i in range(242):
    #     print(bandnames[i], wavelength[i], fwhm[i])

    showprocesslog('Extracting zip...')

    idir = os.path.dirname(ifile)
    zfile = zipfile.ZipFile(ifile)

    zipnames = zfile.namelist()
    zfile.extractall(idir)
    zfile.close()

    zipnames2 = [i for i in zipnames if i[-3:].lower() == 'tif']

    showprocesslog('Importing Hyperion data...')

    nval = 0
    dat = []
    for ifile2 in piter(zipnames2):
        fext = ifile2.split('_')[1]
        bandno = int(fext[1:])

        # Eliminate bands not illuminated
        if bandno <= 7 or bandno >= 225:
            continue

        # Overlap Region
        if bandno >= 58 and bandno <= 78:
            continue

        showprocesslog(f'Importing band {bandno}: {wavelength[bandno-1]} nm')
        dataset = rasterio.open(os.path.join(idir, ifile2))

        if dataset is None:
            showprocesslog(f'Problem with band {bandno}')
            continue

        rtmp = dataset.read(1)

        dat.append(Data())
        dat[-1].data = rtmp

        dat[-1].data = np.ma.masked_invalid(dat[-1].data)
        dat[-1].data.mask = dat[-1].data.mask | (dat[-1].data == nval)
        if dat[-1].data.mask.size == 1:
            dat[-1].data.mask = (np.ma.make_mask_none(dat[-1].data.shape) +
                                 dat[-1].data.mask)

        dat[-1].dataid = f'Band {bandno}: {wavelength[bandno-1]} nm'
        dat[-1].nodata = nval
        dat[-1].crs = dataset.crs
        dat[-1].set_transform(transform=dataset.transform)
        dat[-1].filename = ifile

        bmeta = {}
        # if satbands is not None and fext in satbands:

        dat[-1].metadata['Raster']['wavelength'] = wavelength[bandno-1]
        bmeta['WavelengthMin'] = wavelength[bandno-1]-fwhm[bandno-1]/2
        bmeta['WavelengthMax'] = wavelength[bandno-1]+fwhm[bandno-1]/2

        dat[-1].metadata.update(bmeta)

        dataset.close()

    if dat == []:
        dat = None

    showprocesslog('Cleaning Extracted zip files...')
    for zfile in zipnames:
        os.remove(os.path.join(idir, zfile))

    # if '.tar' in ifilet:
    #     showprocesslog('Cleaning Extracted tar files...')
    #     for tfile in piter(tarnames):
    #         print(tfile)
    #         os.remove(os.path.join(os.path.dirname(ifile), tfile))
    print('Import complete')
    return dat


def get_sentinel2(ifile, piter=None, showprocesslog=print, extscene=None):
    """
    Get Sentinel-2 Data.

    Parameters
    ----------
    ifile : str
        filename to import
    piter : iter, optional
        Progress bar iterable. Default is iter
    showprocesslog : function, optional
        Routine to show text messages. The default is print.
    extscene : str or None
        String used currently to give an option to limit bands in Sentinel-2

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

    subdata = [i for i in subdata if 'TCI' not in i]  # TCI is true color

    nval = 0
    dat = []
    for bfile in subdata:
        dataset = rasterio.open(bfile)
        showprocesslog('Importing '+os.path.basename(bfile))
        if dataset is None:
            showprocesslog('Problem with '+ifile)
            continue

        for i in piter(dataset.indexes):
            rtmp = dataset.read(i)
            bname = dataset.descriptions[i-1]
            bmeta = dataset.tags(i)

            if ('Sentinel-2 Bands Only' in extscene and
                    'central wavelength' not in bname.lower()):
                continue

            dat.append(Data())
            dat[-1].data = rtmp
            dat[-1].data = np.ma.masked_invalid(dat[-1].data)
            dat[-1].data.mask = dat[-1].data.mask | (dat[-1].data == nval)
            if dat[-1].data.mask.size == 1:
                dat[-1].mask = np.ma.getmaskarray(dat[-1].data)
            dat[-1].data = dat[-1].data.astype(float)
            dat[-1].data = dat[-1].data / 10000.

            dat[-1].dataid = bname
            dat[-1].nodata = nval
            dat[-1].crs = dataset.crs
            dat[-1].set_transform(transform=dataset.transform)
            dat[-1].filename = ifile
            dat[-1].units = 'Reflectance'

            if 'WAVELENGTH' in bmeta and 'BANDWIDTH' in bmeta:
                wlen = float(bmeta['WAVELENGTH'])
                bwidth = float(bmeta['BANDWIDTH'])
                bmeta['WavelengthMin'] = wlen - bwidth/2
                bmeta['WavelengthMax'] = wlen + bwidth/2
                dat[-1].metadata['Raster']['wavelength'] = wlen

            dat[-1].metadata.update(bmeta)

            if 'SOLAR_IRRADIANCE_UNIT' in bmeta:
                dat[-1].units = bmeta['SOLAR_IRRADIANCE_UNIT']

        dataset.close()

    if dat == []:
        dat = None

    return dat


def get_aster_zip(ifile, piter=None, showprocesslog=print):
    """
    Get ASTER zip Data.

    Parameters
    ----------
    ifile : str
        filename to import
    piter : iter, optional
        Progress bar iterable. Default is iter
    showprocesslog : function, optional
        Routine to show text messages. The default is print.

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
    # elif 'AST_09' in ifile:
    #     scalefactor = None
    else:
        return None

    showprocesslog('Extracting zip...')

    idir = os.path.dirname(ifile)
    zfile = zipfile.ZipFile(ifile)

    zipnames = zfile.namelist()
    zfile.extractall(idir)
    zfile.close()

    dat = []
    nval = 0
    for zfile in piter(zipnames):
        if zfile.lower()[-4:] != '.tif':
            continue

        dataset1 = rasterio.open(os.path.join(idir, zfile))
        if dataset1 is None:
            showprocesslog('Problem with '+zfile)
            continue

        dataset = rasterio.vrt.WarpedVRT(dataset1)
        rtmp = dataset.read(1)

        dat.append(Data())
        dat[-1].data = rtmp
        dat[-1].data = np.ma.masked_invalid(dat[-1].data)*scalefactor
        dat[-1].data.mask = dat[-1].data.mask | (dat[-1].data == nval)
        if dat[-1].data.mask.size == 1:
            dat[-1].mask = np.ma.getmaskarray(dat[-1].data)

        dat[-1].dataid = zfile[zfile.index('Band'):zfile.index('.tif')]
        dat[-1].nodata = nval
        dat[-1].crs = dataset.crs
        dat[-1].set_transform(transform=dataset.transform)
        dat[-1].filename = ifile
        dat[-1].units = units

        bmeta = dat[-1].metadata
        fext = dat[-1].dataid[4:]
        bmeta['WavelengthMin'] = satbands[fext][0]
        bmeta['WavelengthMax'] = satbands[fext][1]
        bmeta['Raster']['wavelength'] = (satbands[fext][1]+satbands[fext][1])/2

        dataset.close()
        dataset1.close()

    showprocesslog('Cleaning Extracted zip files...')
    for zfile in zipnames:
        os.remove(os.path.join(idir, zfile))

    return dat


def get_aster_hdf(ifile, piter=None):
    """
    Get ASTER hdf Data.

    Parameters
    ----------
    ifile : str
        filename to import
    piter : iter, optional
        Progress bar iterable. Default is iter

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
    with rasterio.open(ifile) as dataset:
        meta = dataset.tags()
        subdata = dataset.subdatasets
        # crs = dataset.crs

    # zone = None
    # if crs is None:
    #     for i in meta:
    #         if 'UTMZONECODE' in i:
    #             zone = meta[i]
    #     if zone is not None:
    #         crs1 = CRS.from_epsg('326'+zone)

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

    dat = []
    nval = 0
    calctoa = False
    for bfile in piter(subdata):
        if 'QA' in bfile:
            continue
        if ptype == 'L1T' and 'ImageData3B' in bfile:
            continue

        dataset1 = rasterio.open(bfile)
        dataset = rasterio.vrt.WarpedVRT(dataset1)
        crs = dataset1.gcps[-1]

        # breakpoint()

        dat.append(Data())
        dat[-1].data = dataset.read(1)
        if ptype == '08':
            dat[-1].data[dat[-1].data == 2000] = nval
        dat[-1].data = np.ma.masked_invalid(dat[-1].data)*scalefactor
        dat[-1].data.mask = dat[-1].data.mask | (dat[-1].data == nval)
        if dat[-1].data.mask.size == 1:
            dat[-1].mask = np.ma.getmaskarray(dat[-1].data)

        dat[-1].dataid = bfile.split(':')[-1]
        dat[-1].nodata = nval
        dat[-1].set_transform(transform=dataset.transform)
        dat[-1].crs = crs
        dat[-1].metadata['SolarElev'] = solarelev
        dat[-1].metadata['JulianDay'] = jdate
        dat[-1].metadata['CalendarDate'] = cdate
        dat[-1].metadata['ShortName'] = meta['SHORTNAME']
        dat[-1].filename = ifile
        dat[-1].units = units

        if 'band' in dat[-1].dataid.lower():
            bmeta = dat[-1].metadata
            fext = dat[-1].dataid[4:].split()[0]
            bmeta['WavelengthMin'] = satbands[fext][0]
            bmeta['WavelengthMax'] = satbands[fext][1]
            bmeta['Raster']['wavelength'] = (satbands[fext][1]+satbands[fext][1])/2

            if ptype == 'L1T' and 'ImageData' in ifile:
                dat[-1].metadata['Gain'] = ucc[ifile[ifile.rindex('ImageData'):]]
                calctoa = True
        dataset.close()
        dataset1.close()

    if dat == []:
        dat = None

    if ptype == 'L1T' and calctoa is True:
        dat = calculate_toa(dat)

    return dat


def get_aster_ged(ifile, piter=None):
    """
    Get ASTER GED data.

    Parameters
    ----------
    ifile : str
        filename to import

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
        # meta = dataset.tags()
        subdata = dataset.subdatasets
        # crs = dataset.crs

    i = -1
    for ifile2 in subdata:
        dataset = rasterio.open(ifile2)
        units = ''

        if 'ASTER_GDEM' in ifile2:
            bandid = 'ASTER GDEM'
            units = 'meters'
        if 'Land_Water_Map' in ifile2:
            bandid = 'Land_water_map'
        if 'Observations' in ifile2:
            bandid = 'Observations'
            units = 'number per pixel'

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

        nbands = 1
        if rtmp2.ndim == 3:
            nbands = rtmp2.shape[0]

        for i2 in range(nbands):
            nval = -9999
            i += 1

            dat.append(Data())
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
            dat[i].crs = CRS.from_epsg(4326)  # WGS85 geodetic
            dat[i].units = units

        dataset.close()

    for i in dat:
        i.set_transform(xdim, xmin, ydim, ymax)

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


def _test5P():
    """test routine"""
    import sys
    import matplotlib.pyplot as plt

    ifile = r"E:\Workdata\PyGMI Test Data\Sentinel-5P\S5P_OFFL_L2__AER_AI_20200522T115244_20200522T133414_13508_01_010302_20200524T014436.nc"

    app = QtWidgets.QApplication(sys.argv)  # Necessary to test Qt Classes
    tmp = ImportSentinel5P()
    tmp.ifile = ifile
    tmp.settings(True)

    tmp.outdata['Vector']['Point'].plot(column='data')
    plt.show()


    breakpoint()


def _testfn():
    """Test routine."""
    import matplotlib.pyplot as plt

    extscene = None
    ifile = r"C:\Workdata\Remote Sensing\Modis\MOD16A2.A2013073.h20v11.006.2017101224330.hdf"
    ifile = r"C:\Workdata\Remote Sensing\Landsat\LC08_L1TP_176080_20190820_20190903_01_T1.tar.gz"
    ifile = r"C:\Workdata\Remote Sensing\Sentinel-2\S2B_MSIL2A_20201213T081239_N0214_R078_T34JGP_20201213T105149.SAFE\MTD_MSIL2A.xml"
    # ifile = r"C:\Workdata\Remote Sensing\ASTER\old\AST_07XT_00309042002082052_20200518021740_29313.zip"
    # ifile = r"C:\Workdata\Remote Sensing\ASTER\old\AST_07XT_00305282005083844_20180604061623_15509.hdf"
    # ifile = r"C:\Workdata\Remote Sensing\AG100.v003.-23.030.0001.h5"
    # ifile = r"E:\Workdata\bugs\AST_05_00309232013204629_20211004081945_4263.hdf"

    ifile = "C:/Workdata/Remote Sensing/AST_07XT_00307292005085059_20210608060928_376.hdf"
    ifile = "C:/Workdata/Remote Sensing/ASTER/AST_08_00305212003085056_20180604061050_13463.hdf"

    # ifile = r"c:\Workdata\Remote Sensing\ASTER\AST_05_00303132017211557_20180814030139_5621.hdf"

    # ifile = r"E:\Workdata\bugs\AST_08_00306272001204805_20211007060336_20853.hdf"
    # ifile = r"E:\Workdata\Remote Sensing\Landsat\LE07_L2SP_169076_20000822_20200917_02_T1.tar"

    ifile = "E:\Workdata\Remote Sensing\hyperion\EO1H1760802013198110KF_1T.ZIP"
    extscene = 'Hyperion'

    ifile = r"E:\Workdata\bugs\janine\MOD16A2.A2001153.h20v11.006.2017070120816.hdf"
    extscene = None

    dat = get_data(ifile, extscene = extscene)

    for i in dat:
        plt.figure(dpi=300)
        plt.title(i.dataid)
        plt.imshow(i.data, interpolation='none')
        plt.colorbar()
        plt.show()

    # ifile = r'C:/Work/Workdata/Remote Sensing/Modis/MOD11A2.A2013073.h20v11.006.2016155170529.hdf'
    # dat = get_modisv6(ifile)
    breakpoint()


if __name__ == "__main__":
    _testfn()
