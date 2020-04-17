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
import re
from PyQt5 import QtWidgets, QtCore
import numpy as np
from osgeo import ogr
import matplotlib.path as mplPath
from osgeo import gdal, osr, ogr
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point
import pygmi.menu_default as menu_default
from pygmi.raster.dataprep import GroupProj


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
            minx = float(self.lonmin.text())
            miny = float(self.latmin.text())
            maxx = float(self.lonmax.text())
            maxy = float(self.latmax.text())
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
