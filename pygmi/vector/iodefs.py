# -----------------------------------------------------------------------------
# Name:        iodefs.py (part of PyGMI)
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


class ImportLineData(QtWidgets.QDialog):
    """
    Import Line Data.

    This class imports ASCII point data.

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

        self.name = 'Import Line Data: '
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

    def settings(self, nodialog=False):
        """
        Entry point into item.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        ext = ('Geosoft XYZ (*.xyz);;'
               'ASCII XYZ (*.xyz);;'
               'Comma Delimited (*.csv);;'
               'Tab Delimited (*.txt);;'
               'All Files (*.*)')

        filename, filt = QtWidgets.QFileDialog.getOpenFileName(
            self.parent, 'Open File', '.', ext)
        if filename == '':
            return False

        os.chdir(os.path.dirname(filename))
        self.ifile = str(os.path.basename(filename))

        if filt == 'Geosoft XYZ (*.xyz)':
            gdf = self.get_GXYZ()
        elif filt == 'ASCII XYZ (*.xyz)':
            gdf = self.get_delimited(' ')
        elif filt == 'Comma Delimited (*.csv)':
            gdf = self.get_delimited(',')
        elif filt == 'Tab Delimited (*.txt)':
            gdf = self.get_delimited('\t')
        else:
            return False

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

        if 'Line' not in self.outdata:
            self.outdata['Line'] = {}

        gdf = gdf.replace(nodata, np.nan)
        self.outdata['Line'][self.ifile] = gdf

        return True

    def loadproj(self, projdata):
        """
        Loads project data into class.

        Parameters
        ----------
        projdata : dictionary
            Project data loaded from JSON project file.

        Returns
        -------
        chk : bool
            A check to see if settings was successfully run.

        """

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

#        projdata['ftype'] = '2D Mean'

        return projdata

    def get_GXYZ(self):
        """
        Get geosoft XYZ.

        Returns
        -------
        dat : DataFrame
            Pandas dataframe.

        """

        with open(self.ifile) as fno:
            head = fno.readline()
            tmp = fno.read()
        if r'/' not in head:
            print('Not Geosoft XYZ format')
            return None
        head = head.split()
        head.pop(0)
        tmp = tmp.lower()
        tmp = re.split('(line|tie)', tmp)
        if tmp[0] == '':
            tmp.pop(0)

        dtype = {}
        dtype['names'] = head
        dtype['formats'] = ['f4']*len(head)

        df2 = None
        for i in range(0, len(tmp), 2):
            tmp2 = tmp[i+1]
            tmp2 = tmp2.split('\n')
            line = tmp[i]+tmp2.pop(0)
            tmp2 = np.genfromtxt(tmp2, names=head)
            df1 = pd.DataFrame(tmp2)
            df1['line'] = line
            if df2 is None:
                df2 = df1
            else:
                df2 = df2.append(df1, ignore_index=True)

        return df2

    def get_delimited(self, delimiter=','):
        """
        Get a delimited line file.

        Returns
        -------
        gdf : DataFrame
            Pandas dataframe.

        """

        with open(self.ifile) as fno:
            head = fno.readline()
            tmp = fno.read()

        head = head.split(delimiter)
        head = [i.lower() for i in head]
        tmp = tmp.lower()

        dtype = {}
        dtype['names'] = head
        dtype['formats'] = ['f4']*len(head)

        tmp = tmp.split('\n')
        tmp2 = np.genfromtxt(tmp, names=head, delimiter=delimiter, dtype=None,
                             encoding=None)
        gdf = pd.DataFrame(tmp2)

        if 'line' not in head:
            gdf['line'] = 'None'

        return gdf


class PointCut():
    """
    Cut Data using shapefiles.

    This class cuts point datasets using a boundary defined by a polygon
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

    def __init__(self, parent):
        self.ifile = ''
        self.name = 'Cut Data:'
        self.ext = ''
        self.pbar = parent.pbar
        self.parent = parent
        self.indata = {}
        self.outdata = {}

    def settings(self, nodialog=False):
        """
        Entry point into item.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        if 'Line' in self.indata:
            data = copy.deepcopy(self.indata['Line'])
            data = list(data.values())[0]
        else:
            print('No point data')
            return False

        ext = 'Shape file (*.shp)'

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.parent, 'Open Shape File', '.', ext)
        if filename == '':
            return False
        os.chdir(os.path.dirname(filename))

        self.ifile = str(filename)
        self.ext = filename[-3:]
        data = cut_point(data, self.ifile)

        if data is None:
            err = ('There was a problem importing the shapefile. Please make '
                   'sure you have at all the individual files which make up '
                   'the shapefile.')
            QtWidgets.QMessageBox.warning(self.parent, 'Error', err,
                                          QtWidgets.QMessageBox.Ok)
            return False

        self.pbar.to_max()
        self.outdata['Line'] = data

        return True

    def loadproj(self, projdata):
        """
        Loads project data into class.

        Parameters
        ----------
        projdata : dictionary
            Project data loaded from JSON project file.

        Returns
        -------
        chk : bool
            A check to see if settings was successfully run.

        """

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

#        projdata['ftype'] = '2D Mean'

        return projdata


class ExportLine():
    """
    Export Line Data.

    Attributes
    ----------
    name : str
        item name
    pbar : progressbar
        reference to a progress bar.
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    """

    def __init__(self, parent):
        self.name = 'Export Point: '
        self.pbar = None
        self.parent = parent
        self.indata = {}

    def run(self):
        """
        Run routine.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        if 'Line' not in self.indata:
            print('Error: You need to have line data first!')
            return False

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.parent, 'Save File', '.', 'csv (*.csv)')

        if filename == '':
            return False

        print('Export busy...')

        os.chdir(os.path.dirname(filename))
        data = self.indata['Line']
        data = list(data.values())[0]

        dfall = data.drop(['pygmiX', 'pygmiY'], axis=1)

        dfall.to_csv(filename, index=False)

        print('Export completed')

        return True


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

    def settings(self, nodialog=False):
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

    def loadproj(self, projdata):
        """
        Loads project data into class.

        Parameters
        ----------
        projdata : dictionary
            Project data loaded from JSON project file.

        Returns
        -------
        chk : bool
            A check to see if settings was successfully run.

        """

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

#        projdata['ftype'] = '2D Mean'

        return projdata


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

        self.indata = {}
        self.outdata = {}
        self.parent = parent
        self.pbar = self.parent.pbar

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
            print('Could not reproject')
            return

        key = list(self.indata['Line'].keys())[0]
        data = self.indata['Line'][key]

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

        dd = np.transpose([data.pygmiX, data.pygmiY])
        xy = ctrans.TransformPoints(dd)
        xy = np.array(xy)
        data = data.assign(Xnew=xy[:, 0])
        data = data.assign(Ynew=xy[:, 1])
        data.pygmiX = xy[:, 0]
        data.pygmiY = xy[:, 1]

        self.outdata['Line'] = {key: data}

    def settings(self, nodialog=False):
        """
        Entry point into item.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        if 'Line' not in self.indata:
            print('No line data.')
            return False

#        iwkt = self.in_proj.epsg_proj['WGS 84 / Geodetic Geographic'].wkt
        indx = self.in_proj.combobox.findText('WGS 84 / Geodetic Geographic')
        self.in_proj.combobox.setCurrentIndex(indx)

        indx = self.in_proj.combobox.findText('WGS 84 / UTM zone 35S')
        self.out_proj.combobox.setCurrentIndex(indx)

        tmp = self.exec_()

        if tmp != 1:
            return False

        self.acceptall()

        return True

    def loadproj(self, projdata):
        """
        Loads project data into class.

        Parameters
        ----------
        projdata : dictionary
            Project data loaded from JSON project file.

        Returns
        -------
        chk : bool
            A check to see if settings was successfully run.

        """

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

#        projdata['ftype'] = '2D Mean'

        return projdata


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
    Data
        PyGMI Dataset
    """
    shapef = ogr.Open(ifile)
    if shapef is None:
        return None
    lyr = shapef.GetLayer()
    poly = lyr.GetNextFeature()
    if lyr.GetGeomType() is not ogr.wkbPolygon or poly is None:
        shapef = None
        return None

    points = []
    geom = poly.GetGeometryRef()

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
    for pnt in range(pts.GetPointCount()):
        points.append((pts.GetX(pnt), pts.GetY(pnt)))

    bbpath = mplPath.Path(points)

    chk = bbpath.contains_points(np.transpose([data.pygmiX.values,
                                               data.pygmiY.values]))

    data = data[chk]

    shapef = None
    return data
