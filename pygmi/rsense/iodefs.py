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

        self.xchan = QtWidgets.QComboBox()
        self.ychan = QtWidgets.QComboBox()
        self.nodata = QtWidgets.QLineEdit('99999')

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
        label_xchan = QtWidgets.QLabel('X Channel:')
        label_ychan = QtWidgets.QLabel('Y Channel:')
        label_nodata = QtWidgets.QLabel('Null Value:')

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle(r'Import Point/Line Data')

        gridlayout_main.addWidget(label_xchan, 0, 0, 1, 1)
        gridlayout_main.addWidget(self.xchan, 0, 1, 1, 1)

        gridlayout_main.addWidget(label_ychan, 1, 0, 1, 1)
        gridlayout_main.addWidget(self.ychan, 1, 1, 1, 1)

        gridlayout_main.addWidget(label_nodata, 2, 0, 1, 1)
        gridlayout_main.addWidget(self.nodata, 2, 1, 1, 1)

        gridlayout_main.addWidget(helpdocs, 3, 0, 1, 1)
        gridlayout_main.addWidget(buttonbox, 3, 1, 1, 3)

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

        gdf = self.get_5P_meta()

        if gdf is None:
            return False

        ltmp = gdf.columns.values
        self.xchan.addItems(ltmp)
        self.ychan.addItems(ltmp)

        self.xchan.setCurrentIndex(0)
        self.ychan.setCurrentIndex(1)

        tmp = self.exec_()

        if tmp != 1:
            return tmp

        try:
            nodata = float(self.nodata.text())
        except ValueError:
            print('Null Value error - abandoning import')
            return False

        xcol = self.xchan.currentText()
        ycol = self.ychan.currentText()

        gdf['pygmiX'] = gdf[xcol]
        gdf['pygmiY'] = gdf[ycol]
        gdf['line'] = gdf['line'].astype(str)

        if 'Vector' not in self.outdata:
            self.outdata['Vector'] = {}

        gdf = gdf.replace(nodata, np.nan)
        self.outdata['Vector'][self.ifile] = gdf

        return True

    def get_5P_meta(self):
        """
        Get metadata.

        Returns
        -------
        dat : DataFrame
            Pandas dataframe.

        """
        df2 = None

        return df2


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
