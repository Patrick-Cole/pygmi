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
"""Import Data."""

import os
import copy
import glob
import tarfile
import zipfile
import datetime
from PyQt5 import QtWidgets, QtCore
import numpy as np
from osgeo import gdal, osr
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point
import pygmi.menu_default as menu_default
from pygmi.raster.datatypes import Data
from pygmi.raster.dataprep import quickgrid

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
    Import Data - Interfaces with GDAL routines

    Attributes
    ----------
    name : str
        item name
    pbar : progressbar
        reference to a progress bar.
    parent : parent
        reference to the parent routine
    outdata : dictionary
        dictionary of output datasets
    ifile : str
        input file name. Used in main.py
    ext : str
        filename extension
    """

    def __init__(self, parent=None, extscene=None):
        self.ifile = ''
        self.name = 'Import Data: '
        self.ext = ''
        self.pbar = None
        self.parent = parent
        self.indata = {}
        self.outdata = {}
        self.extscene = extscene

    def settings(self):
        """
        Entry point into item.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """

        if self.extscene is None:
            return
        else:
            ext = self.extscene

        filename, filt = QtWidgets.QFileDialog.getOpenFileName(self.parent,
                                                               'Open File',
                                                               '.', ext)
        if filename == '':
            return False
        os.chdir(os.path.dirname(filename))
        self.ifile = str(filename)
        self.ext = filename[-3:]
        self.ext = self.ext.lower()

        dat = get_data(self.ifile)

        if dat is None:
            if filt == 'hdf (*.hdf *.h5)':
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

class ImportBatch():
    """
    Import Data - Interfaces with GDAL routines

    Attributes
    ----------
    name : str
        item name
    pbar : progressbar
        reference to a progress bar.
    parent : parent
        reference to the parent routine
    outdata : dictionary
        dictionary of output datasets
    ifile : str
        input file name. Used in main.py
    ext : str
        filename extension
    """

    def __init__(self, parent=None, directory=None):
        self.ifile = ''
        self.name = 'Import Data: '
        self.ext = ''
        self.pbar = None
        self.parent = parent
        self.indata = {}
        self.outdata = {}

        if directory is not None:
            self.settings(directory)

    def settings(self, directory=None):
        """
        Entry point into item.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        if directory is None:
            directory = QtWidgets.QFileDialog.getExistingDirectory(
                self.parent, 'Select Directory')
        if directory == '':
            return False
        os.chdir(directory)

        zipdat = glob.glob(directory+'//AST*.zip')
        hdfdat = glob.glob(directory+'//AST*.hdf')
#        tifdat = glob.glob(directory+'//AST*.tif')
        targzdat = glob.glob(directory+'//L*.tar.gz')
        mtldat = glob.glob(directory+'//L*MTL.txt')

        sendat = []
        sendir = [f.path for f in os.scandir(directory) if f.is_dir() and 'SAFE' in f.path]
        for i in sendir:
            sendat.extend(glob.glob(i+'//MTD*.xml'))

        if not hdfdat and not zipdat and not targzdat and not mtldat and not sendat:
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

        # for i in tifdat:
        #     if i[:i.rindex('_')]+'_MTL.txt' in mtldat:
        #         continue
        #     dat.append(i)

        output_type = 'RasterFileList'
        self.outdata[output_type] = dat

        return True


class ImportSentinel5P(QtWidgets.QDialog):
    """
    Import Sentinel 5P data to shapefile.

    This class imports Sentinel 5P data.

    Attributes
    ----------
    name : str
        item name
    pbar : progressbar
        reference to a progress bar.
    parent : parent
        reference to the parent routine
    outdata : dictionary
        dictionary of output datasets
    ifile : str
        input file name. Used in main.py
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.name = 'Import Sentinel-5P Data: '
        self.pbar = None  # self.parent.pbar
        self.parent = parent
        self.indata = {}
        self.outdata = {}
        self.ifile = ''

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
        helpdocs = menu_default.HelpButton('pygmi.vector.iodefs.importpointdata')
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

    def settings(self):
        """
        Entry point into item.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        ext = ('Sentinel-5P (*.nc)')

        filename, filt = QtWidgets.QFileDialog.getOpenFileName(
            self.parent, 'Open File', '.', ext)
        if filename == '':
            return False

        os.chdir(os.path.dirname(filename))
        self.ifile = str(os.path.basename(filename))

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
            print('Value error - abandoning import')
            return False

        gdf = self.get_5P_data(meta)

        if gdf is None:
            return False

        dat = {gdf.geom_type.iloc[0]: gdf}
        self.outdata['Vector'] = dat

        return True

    def get_5P_meta(self):
        """
        Get metadata.

        Returns
        -------
        meta : Dictionary
            Dictionary containing metadata.

        """

        dataset = gdal.Open(self.ifile, gdal.GA_ReadOnly)
        if dataset is None:
            print('Problem! Unable to import')
            print(os.path.basename(self.ifile))
            return None

        subdata = dataset.GetSubDatasets()
        meta = {}
        for i in subdata:
            tmp = i[1].split()
            if 'SUPPORT_DATA' in i[0]:
                continue
            if 'METADATA' in i[0]:
                continue
            if 'time_utc' in i[0]:
                continue
            if 'delta_time' in i[0]:
                continue
            if 'qa_value' in i[0]:
                continue
            if 'precision' in i[0]:
                continue

            tmp = tmp[1].replace('//PRODUCT/', '')
            tmp = tmp.replace('/PRODUCT/', '')
            tmp = tmp.replace('/', '')

            meta[tmp] = i

        dataset = None

        return meta

    def get_5P_data(self, meta):
        """
        Get data.

        Parameters
        ----------
        meta : Dictionary
            Dictionary containing metadata.

        Returns
        -------
        gdf : DataFrame
            geopandas dataframe.

        """

        dataset = gdal.Open(meta['latitude'][0], gdal.GA_ReadOnly)
        rtmp = dataset.GetRasterBand(1)
        lats = rtmp.ReadAsArray()
        dataset = None

        dataset = gdal.Open(meta['longitude'][0], gdal.GA_ReadOnly)
        rtmp = dataset.GetRasterBand(1)
        lons = rtmp.ReadAsArray()
        dataset = None

        del meta['latitude']
        del meta['longitude']

        if lats is None:
            print('No Latitudes in dataset')
            return None

        lats = lats.flatten()
        lons = lons.flatten()
        pnts = np.transpose([lons, lats])

        lonmin = float(self.lonmin.text())
        latmin = float(self.latmin.text())
        lonmax = float(self.lonmax.text())
        latmax = float(self.latmax.text())

        mask = (lats > latmin) & (lats < latmax) & (lons < lonmax) & (lons > lonmin)

        idfile = self.subdata.currentText()

        dfile = meta[idfile][0]
        dataset = gdal.Open(dfile, gdal.GA_ReadOnly)
        rtmp = dataset.GetRasterBand(1)
        dat = rtmp.ReadAsArray()

        dataset = None
        dat1 = dat.flatten()

        if mask.shape != dat1.shape:
            return None

        dat1 = dat1[mask]
        pnts1 = pnts[mask]

        pnts1 = pnts1[dat1 != 9.96921e+36]
        dat1 = dat1[dat1 != 9.96921e+36]

        if dat1.size == 0:
            print(idfile, 'is empty.')
            return None

        df = pd.DataFrame({'lon': pnts1[:, 0], 'lat': pnts1[:, 1]})
        df['data'] = dat1

        gdf = GeoDataFrame(df.drop(['lon', 'lat'], axis=1),
                           geometry=[Point(xy) for xy in zip(df.lon, df.lat)])

        # tmp = os.path.join(idir, os.path.basename(ifile).split('T')[0])
        # tmp = tmp + '_' + idfile + '.shp'
        # tmp = tmp.replace('//PRODUCT/', '')
        # tmp = tmp.replace('/PRODUCT/', '')
        # tmp = tmp.replace('/', '')

#        gdf.to_file(tmp)
        return gdf


class ImportShapeData():
    """
    Import Shapefile Data.

    Attributes
    ----------
    name : str
        item name
    pbar : progressbar
        reference to a progress bar.
    parent : parent
        reference to the parent routine
    outdata : dictionary
        dictionary of output datasets
    ifile : str
        input file name. Used in main.py
    """

    def __init__(self, parent=None):
        self.name = 'Import Shapefile Data: '
        self.pbar = None
        self.parent = parent
        self.indata = {}
        self.outdata = {}
        self.ifile = ''

    def settings(self):
        """
        Entry point into item.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        ext = 'Shapefile (*.shp);;' + 'All Files (*.*)'

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self.parent,
                                                            'Open File',
                                                            '.', ext)
        if filename == '':
            return False
        os.chdir(os.path.dirname(filename))
        self.ifile = str(filename)

        ifile = str(filename)

        gdf = gpd.read_file(ifile)

        dat = {gdf.geom_type.iloc[0]: gdf}

        self.outdata['Vector'] = dat

        return True


def calculate_toa(dat):
    """
    Top of atmosphere correction. Includes VNIR, SWIR and TIR bands.

    Parameters
    ----------
    dat : Data
        PyGMI raster dataset

    Returns
    -------
    out : Data
        PyGMI raster dataset
    """

    print('Calculating top of atmosphere...')

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
        datai.data.set_fill_value(datai.nullvalue)
        dmask = datai.data.mask
        datai.data = np.ma.array(datai.data.filled(), mask=dmask)
        out.append(datai)

    return out


def get_data(ifile):
    """
    This function loads a raster dataset off the disk using the GDAL
    libraries. It returns the data in a PyGMI data object.

    Parameters
    ----------
    ifile : str
        filename to import

    Returns
    -------
    dat : PyGMI raster Data
        dataset imported
    """
    ifile = ifile[:]
    bfile = os.path.basename(ifile)

    print('Importing', bfile)

    if 'AST_' in bfile and 'hdf' in bfile.lower():
        dat = get_aster_hdf(ifile)
    elif 'AST_' in bfile and 'zip' in bfile.lower():
        dat = get_aster_zip(ifile)
    elif bfile[:4] in ['LT04', 'LT05', 'LE07', 'LC08', 'LM05']:
        dat = get_landsat(ifile)
    elif '.xml' in bfile and '.SAFE' in ifile:
        dat = get_sentinel2(ifile)
    else:
        dat = None

    return dat


def get_modis(ifile):
    """
    Gets MODIS data

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
    ifile = ifile[:]

    dataset = gdal.Open(ifile, gdal.GA_ReadOnly)

    subdata = dataset.GetSubDatasets()

    latentry = [i for i in subdata if 'Latitude' in i[1]]
    subdata.pop(subdata.index(latentry[0]))
    dataset = None

    dataset = gdal.Open(latentry[0][0], gdal.GA_ReadOnly)
    rtmp = dataset.GetRasterBand(1)
    lats = rtmp.ReadAsArray()
    latsdim = ((lats.max()-lats.min())/(lats.shape[0]-1))/2

    lonentry = [i for i in subdata if 'Longitude' in i[1]]
    subdata.pop(subdata.index(lonentry[0]))

    dataset = None
    dataset = gdal.Open(lonentry[0][0], gdal.GA_ReadOnly)
    rtmp = dataset.GetRasterBand(1)
    lons = rtmp.ReadAsArray()
    lonsdim = ((lons.max()-lons.min())/(lons.shape[1]-1))/2

    lonsdim = latsdim
    tlx = lons.min()-abs(lonsdim/2)
    tly = lats.max()+abs(latsdim/2)
    cols = int((lons.max()-lons.min())/lonsdim)+1
    rows = int((lats.max()-lats.min())/latsdim)+1

    newx2, newy2 = np.mgrid[0:rows, 0:cols]
    newx2 = newx2*lonsdim + tlx
    newy2 = tlx - newy2*latsdim

    tmp = []
    for i in subdata:
        if 'HDF4_EOS:EOS_SWATH' in i[0]:
            tmp.append(i)
    subdata = tmp

    i = -1
    for ifile2, bandid2 in subdata:
        dataset = None
        dataset = gdal.Open(ifile2, gdal.GA_ReadOnly)

        rtmp2 = dataset.ReadAsArray()

        if rtmp2.shape[-1] == min(rtmp2.shape) and rtmp2.ndim == 3:
            rtmp2 = np.transpose(rtmp2, (2, 0, 1))

        nbands = 1
        if rtmp2.ndim == 3:
            nbands = rtmp2.shape[0]

        for i2 in range(nbands):
            rtmp = dataset.GetRasterBand(i2+1)
            bandid = rtmp.GetDescription()
            nval = rtmp.GetNoDataValue()
            i += 1

            dat.append(Data())
            if rtmp2.ndim == 3:
                dat[i].data = rtmp2[i2]
            else:
                dat[i].data = rtmp2

            newx = lons[dat[i].data != nval]
            newy = lats[dat[i].data != nval]
            newz = dat[i].data[dat[i].data != nval]

            if newx.size == 0:
                dat[i].data = np.zeros((rows, cols)) + nval
            else:
                tmp = quickgrid(newx, newy, newz, latsdim)
                mask = np.ma.getmaskarray(tmp)
                gdat = tmp.data
                dat[i].data = np.ma.masked_invalid(gdat[::-1])
                dat[i].data.mask = mask[::-1]

            if dat[i].data.dtype.kind == 'i':
                if nval is None:
                    nval = 999999
                nval = int(nval)
            elif dat[i].data.dtype.kind == 'u':
                if nval is None:
                    nval = 0
                nval = int(nval)
            else:
                if nval is None:
                    nval = 1e+20
                nval = float(nval)

            dat[i].data = np.ma.masked_invalid(dat[i].data)
            dat[i].data.mask = (np.ma.getmaskarray(dat[i].data) |
                                (dat[i].data == nval))
            if dat[i].data.mask.size == 1:
                dat[i].mask = np.ma.getmaskarray(dat[i].data)

            dat[i].dataid = bandid2+' '+bandid
            dat[i].nullvalue = nval
            dat[i].xdim = abs(lonsdim)
            dat[i].ydim = abs(latsdim)

            rows, cols = dat[i].data.shape
            xmin = tlx
            ymax = tly
            ymin = ymax - rows*dat[i].ydim
            xmax = xmin + cols*dat[i].xdim

            dat[i].extent = [xmin, xmax, ymin, ymax]

            srs = osr.SpatialReference()
            srs.ImportFromWkt(dataset.GetProjection())
            srs.AutoIdentifyEPSG()
            srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

            dat[i].wkt = srs.ExportToWkt()

    dataset = None
    return dat


def get_landsat(ifilet):
    """
    Gets Landsat Data.

    Parameters
    ----------
    ifilet : str
        filename to import

    Returns
    -------
    out : Data
        PyGMI raster dataset
    """

    idir = os.path.dirname(ifilet)

    if 'tar.gz' in ifilet:
        with tarfile.open(ifilet) as tar:
            tarnames = tar.getnames()
            ifile = next((i for i in tarnames if '_MTL.txt' in i), None)
            if ifile is None:
                print('Could not find MTL.txt file in tar archive')
                return None
            print('Extracting tar...')
            tar.extractall(idir)
            ifile = os.path.join(idir, ifile)
    elif '_MTL.txt' in ifilet:
        ifile = ifilet
    else:
        print('Input needs to be tar.gz or _MTL.txt')
        return None

    files = glob.glob(ifile[:-7]+'*[0-9].tif')
#    files.extend(glob.glob(ifile[:-7]+'*[0-9][0-9].tif'))

    print('Importing Landsat data...')

    # mtlfile = open(ifile)
    # mtldat = mtlfile.read()
    # mtlfile.close()

    # mtldat = mtldat.splitlines()

    # mtldat2 = []
    # for i in mtldat:
    #     i = i.strip().split(' = ')
    #     mtldat2.append(i)

    # mtldat2.pop(0)
    # mtldat = {}
    # grpname = ''
    # for i in mtldat2:
    #     if i[0] == 'END_GROUP' or i[0] == 'END' or len(i) < 2:
    #         continue

    #     if i[0] == 'GROUP':
    #         grpname = i[1]
    #         mtldat[grpname] = {}
    #     else:
    #         if i[1][0] == '"':
    #             i[1] = i[1][1:-1]
    #         mtldat[grpname][i[0]] = i[1]

    # spacecraft = mtldat['PRODUCT_METADATA']['SPACECRAFT_ID']
    # sensor = mtldat['PRODUCT_METADATA']['SENSOR_ID']
    # bands = {}
    # bands['LANDSAT_8'] = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
    #                       '11']
    # bands['LANDSAT_7'] = ['1', '2', '3', '4', '5', '6_VCID_1', '6_VCID_2', '7',
    #                       '8']
    # bands['LANDSAT_5'] = ['1', '2', '3', '4', '5', '6', '7']
    # bands['LANDSAT_4'] = ['1', '2', '3', '4', '5', '6', '7']
    # if sensor == 'MSS':
    #     bands['LANDSAT_5'] = ['1', '2', '3', '4']
    #     bands['LANDSAT_4'] = ['1', '2', '3', '4']

    nval = 0
    dat = []
    for ifile2 in files:
    # for fext in bands[spacecraft]:
        if 'B6_VCID' in ifile2:
            fext = ifile2[-12:-4]
        elif ifile2[-6].isdigit():
            fext = ifile2[-6:-4]
        else:
            fext = ifile2[-5]

        print('Importing Band', fext)

#        ifile2 = os.path.join(idir, mtldat['PRODUCT_METADATA']['FILE_NAME_BAND_'+fext])

        dataset = gdal.Open(ifile2, gdal.GA_ReadOnly)

        if dataset is None:
            print('Problem with band '+fext)
            breakpoint()
            continue

        rtmp = dataset.GetRasterBand(1)

        dat.append(Data())
        dat[-1].data = rtmp.ReadAsArray()
        dat[-1].data = np.ma.masked_invalid(dat[-1].data)
        dat[-1].data.mask = dat[-1].data.mask | (dat[-1].data == nval)
        if dat[-1].data.mask.size == 1:
            dat[-1].data.mask = (np.ma.make_mask_none(dat[-1].data.shape) +
                                 dat[-1].data.mask)

        dat[-1].extent_from_gtr(dataset.GetGeoTransform())
        dat[-1].dataid = 'Band' + fext
        dat[-1].nullvalue = nval
        dat[-1].wkt = dataset.GetProjectionRef()

#        dat[-1].metadata = mtldat
        dataset = None

    if dat == []:
        dat = None

    if 'tar.gz' in ifilet:
        print('Cleaning Extracted tar files...')
        for tfile in tarnames:
            os.remove(os.path.join(os.path.dirname(ifile), tfile))

    return dat


def get_sentinel2(ifile):
    """
    Gets Sentinel-2 Data.

    Parameters
    ----------
    ifile : str
        filename to import

    Returns
    -------
    dat : PyGMI raster Data
        dataset imported
    """
    ifile = ifile[:]

    dataset = gdal.Open(ifile, gdal.GA_ReadOnly)

    subdata = dataset.GetSubDatasets()
    subdata = [i for i in subdata if 'True color' not in i[1]]

    nval = 0
    dat = []
    for ifile, bandid in subdata:
        dataset = gdal.Open(ifile, gdal.GA_ReadOnly)

        if dataset is None:
            print('Problem with', ifile)
            continue

        for i in range(dataset.RasterCount):
            rtmp = dataset.GetRasterBand(i+1)
            bname = rtmp.GetDescription()
            print('Importing', bname)

            dat.append(Data())
            dat[-1].data = rtmp.ReadAsArray()
            dat[-1].data = np.ma.masked_invalid(dat[-1].data)
            dat[-1].data.mask = dat[-1].data.mask | (dat[-1].data == nval)
            if dat[-1].data.mask.size == 1:
                dat[-1].mask = np.ma.getmaskarray(dat[-1].data)

            dat[-1].dataid = bname
            dat[-1].nullvalue = nval
            dat[-1].extent_from_gtr(dataset.GetGeoTransform())
            dat[-1].wkt = dataset.GetProjectionRef()

    if dat == []:
        dat = None

    return dat


def get_aster_zip(ifile):
    """
    Gets ASTER Data.

    Parameters
    ----------
    ifile : str
        filename to import

    Returns
    -------
    dat : PyGMI raster Data
        dataset imported
    """

    SF09 = {'Band1': 0.6760,
            'Band2': 0.7080,
            'Band3N': 0.8620,
            'Band4': 0.21740,
            'Band5': 0.06960,
            'Band6': 0.06250,
            'Band7': 0.05970,
            'Band8': 0.04178,
            'Band9': 0.03180,
            'Band10': 0.006882,
            'Band11': 0.006780,
            'Band12': 0.006590,
            'Band13': 0.005693,
            'Band14': 0.005225}

    if 'AST_07' in ifile:
        scalefactor = 0.001
    elif 'AST_05' in ifile:
        scalefactor = 0.001
    elif 'AST_08' in ifile:
        scalefactor = 0.1
    # elif 'AST_09' in ifile:
    #     scalefactor = None
    else:
        return None


    print('Extracting zip...')

    idir = os.path.dirname(ifile)
    zfile = zipfile.ZipFile(ifile)

    zipnames = zfile.namelist()
    zfile.extractall(idir)
    zfile.close()

    dat = []
    nval = 0
    for zfile in zipnames:
        if zfile.lower()[-4:] != '.tif':
            continue

        dataset = gdal.Open(os.path.join(idir, zfile), gdal.GA_ReadOnly)

        if dataset is None:
            print('Problem with', zfile)
            continue

        dataset = gdal.AutoCreateWarpedVRT(dataset)
        rtmp = dataset.GetRasterBand(1)

        dat.append(Data())
        dat[-1].data = rtmp.ReadAsArray()
        dat[-1].data = np.ma.masked_invalid(dat[-1].data)*scalefactor
        dat[-1].data.mask = dat[-1].data.mask | (dat[-1].data == nval)
        if dat[-1].data.mask.size == 1:
            dat[-1].mask = np.ma.getmaskarray(dat[-1].data)

        dat[-1].extent_from_gtr(dataset.GetGeoTransform())
        dat[-1].dataid = zfile[zfile.index('Band'):zfile.index('.tif')]
        dat[-1].nullvalue = nval
        dat[-1].wkt = dataset.GetProjectionRef()

        dataset = None

    print('Cleaning Extracted zip files...')
    for zfile in zipnames:
        os.remove(os.path.join(idir, zfile))

    return dat


def get_aster_hdf(ifile):
    """
    Gets ASTER Data.

    Parameters
    ----------
    ifile : str
        filename to import

    Returns
    -------
    dat : PyGMI raster Data
        dataset imported
    """
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

    dataset = gdal.Open(ifile, gdal.GA_ReadOnly)

    meta = dataset.GetMetadata()

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
    fmt = '%Y-%m-%d'
    dte = datetime.datetime.strptime(cdate, fmt)
    jdate = dte.timetuple().tm_yday

    subdata = dataset.GetSubDatasets()
    if ptype == '07':
        subdata = [i for i in subdata if 'SurfaceReflectance' in i[0]]
        scalefactor = 0.001
    elif ptype == '05':
        subdata = [i for i in subdata if 'SurfaceEmissivity' in i[0]]
        scalefactor = 0.001
    elif ptype == '08':
        scalefactor = 0.1
    elif ptype == 'L1T':
        subdata = [i for i in subdata if 'ImageData' in i[0]]
        scalefactor = 1
    else:
        return

    dat = []
    nval = 0
    for ifile, bandid in subdata:
        if 'QA' in ifile:
            continue
        if ptype == 'L1T' and 'ImageData3B' in ifile:
            continue

        bandid2 = bandid[bandid.lower().index(' band'):
                         bandid.lower().index('(')].strip()

        dataset = gdal.Open(ifile, gdal.GA_ReadOnly)

        tmpds = gdal.AutoCreateWarpedVRT(dataset)

        dat.append(Data())
        dat[-1].data = tmpds.ReadAsArray()
        dat[-1].data = np.ma.masked_invalid(dat[-1].data)*scalefactor
        dat[-1].data.mask = dat[-1].data.mask | (dat[-1].data == nval)
        if dat[-1].data.mask.size == 1:
            dat[-1].mask = np.ma.getmaskarray(dat[-1].data)

        dat[-1].dataid = bandid2
        dat[-1].nullvalue = nval
        dat[-1].extent_from_gtr(tmpds.GetGeoTransform())
        dat[-1].wkt = tmpds.GetProjectionRef()
        dat[-1].metadata['SolarElev'] = solarelev
        dat[-1].metadata['JulianDay'] = jdate
        dat[-1].metadata['CalendarDate'] = cdate
        dat[-1].metadata['ShortName'] = meta['SHORTNAME']

        if ptype == 'L1T':
            dat[-1].metadata['Gain'] = ucc[ifile[ifile.rindex('ImageData'):]]

    if dat == []:
        dat = None

    if ptype == 'L1T':
        dat = calculate_toa(dat)

    return dat


def get_aster_ged(ifile):
    """
    Gets ASTER GED data

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
    ifile = ifile[:]

    dataset = gdal.Open(ifile, gdal.GA_ReadOnly)

    subdata = dataset.GetSubDatasets()

    i = -1
    for ifile2, bandid2 in subdata:
        dataset = gdal.Open(ifile2, gdal.GA_ReadOnly)
        bandid = bandid2
        units = ''

        if 'ASTER_GDEM' in bandid2:
            bandid = 'ASTER GDEM'
            units = 'meters'
        if 'Land_Water_Map' in bandid2:
            bandid = 'Land_water_map'
        if 'Observations' in bandid2:
            bandid = 'Observations'
            units = 'number per pixel'

        rtmp2 = dataset.ReadAsArray()

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
            if 'Emissivity/Mean' in bandid2:
                bandid = 'Emissivity_mean_band_'+str(10+i2)
                dat[i].data = dat[i].data * 0.001
            if 'Emissivity/SDev' in bandid2:
                bandid = 'Emissivity_std_dev_band_'+str(10+i2)
                dat[i].data = dat[i].data * 0.0001
            if 'NDVI/Mean' in bandid2:
                bandid = 'NDVI_mean'
                dat[i].data = dat[i].data * 0.01
            if 'NDVI/SDev' in bandid2:
                bandid = 'NDVI_std_dev'
                dat[i].data = dat[i].data * 0.01
            if 'Temperature/Mean' in bandid2:
                bandid = 'Temperature_mean'
                units = 'Kelvin'
                dat[i].data = dat[i].data * 0.01
            if 'Temperature/SDev' in bandid2:
                bandid = 'Temperature_std_dev'
                units = 'Kelvin'
                dat[i].data = dat[i].data * 0.01

            dat[i].dataid = bandid
            dat[i].nullvalue = nval
            dat[i].extent_from_gtr(dataset.GetGeoTransform())
            dat[i].units = units
            dat[i].wkt = dataset.GetProjectionRef()

        dataset = None

    return dat


def get_aster_ged_bin(ifile):
    """
    Get ASTER GED binary format

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
        dat[i].nullvalue = nval*scale[i]
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


def testfn():
    """Main testing routine."""
    # ifile = r'C:\Work\Workdata\ASTER\S2A_MSIL2A_20170813T080011_N0205_R035_T35JKG_20170813T082818.SAFE\MTD_MSIL2A.xml'
    # dat = get_sentinel2(ifile)

    # ifile = r'C:\Work\Workdata\ASTER\AST_07XT_00309042002082052_20200518021739_29313.zip'
    # dat = get_aster_zip(ifile)

    ifile = r'C:\Work\Workdata\ASTER\LC081740432017101901T1-SC20180409064853.tar.gz'
    dat = get_landsat(ifile)

    breakpoint()

    # ifile = r'C:\Work\Workdata\ASTER\AST_05_00305282005083844_20180604061610_15573.hdf'
    # ofile = r'C:\Work\Workdata\ASTER\hope.tif'
    # dat = iodefs.get_data(ifile)
    # export_gdal(ofile, dat, 'GTiff')



if __name__ == "__main__":
    testfn()