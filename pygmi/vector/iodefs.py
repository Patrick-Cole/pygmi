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
import pandas as pd
from pygmi.vector.datatypes import PData
from pygmi.vector.datatypes import LData
from pygmi.vector.datatypes import VData
import pygmi.menu_default as menu_default


class ImportLEMI417Data():
    """
    Import LEMI-417 ASCII MT Data.

    This is a class used to import LEMI-417 MT Data in ASCII format.

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
        self.name = 'Import LEMI-417 Data: '
        self.pbar = None
        self.parent = parent
        self.indata = {}
        self.outdata = {}
        self.ifile = ''

    def settings(self):
        """Entry point into item. Data imported from here."""
        ext = 'LEMI-417 Text DataAll Files (*.t*)'

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.parent, 'Open File', '.', ext)
        if filename == '':
            return False
        os.chdir(os.path.dirname(filename))
        self.ifile = str(filename)

        datatmp = np.loadtxt(filename, unpack=True)

        dat = []
        dataid = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second',
                  'Bx (nT)', 'By (nT)', 'Bz (nT)', 'TE (degrees C)',
                  'TF (degrees C)', 'E1 ('+chr(956)+'V/m)',
                  'E2 ('+chr(956)+'V/m)', 'E3 ('+chr(956)+'V/m)',
                  'E4 ('+chr(956)+'V/m)']

        for i in range(datatmp.shape[0]):
            dat.append(PData())
            dat[i].zdata = datatmp[i]
            dat[i].dataid = dataid[i]

        dat = dat[6:]
        dat = dat[0:3]+dat[5:]

        self.outdata['Point'] = dat
        return True


class ImportPointData(QtWidgets.QDialog):
    """
    Import Point Data.

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
        QtWidgets.QDialog.__init__(self, parent)

        self.name = 'Import Point/Line Data: '
        self.pbar = None  # self.parent.pbar
        self.parent = parent
        self.indata = {}
        self.outdata = {}
        self.ifile = ''

        self.xchan = QtWidgets.QComboBox()
        self.ychan = QtWidgets.QComboBox()

        self.setupui()

    def setupui(self):
        """Set up UI."""
        gridlayout_main = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.raster.iodefs.importpointdata')
        label_xchan = QtWidgets.QLabel('X Channel:')
        label_ychan = QtWidgets.QLabel('Y Channel:')

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle(r'Import Point Data')

        gridlayout_main.addWidget(label_xchan, 0, 0, 1, 1)
        gridlayout_main.addWidget(self.xchan, 0, 1, 1, 1)

        gridlayout_main.addWidget(label_ychan, 1, 0, 1, 1)
        gridlayout_main.addWidget(self.ychan, 1, 1, 1, 1)
        gridlayout_main.addWidget(helpdocs, 3, 0, 1, 1)
        gridlayout_main.addWidget(buttonbox, 3, 1, 1, 3)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)

    def settings(self):
        """Entry point into item. Data imported from here."""
        ext = ('Common Formats (*.csv *.dat *.xyz *.txt);;'
               'All Files (*.*)')

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.parent, 'Open File', '.', ext)
        if filename == '':
            return False

        os.chdir(os.path.dirname(filename))
        self.ifile = str(filename)

        datatmp = pd.read_csv(filename, sep=None, engine='python')
        ltmp = datatmp.columns.values

        self.xchan.addItems(ltmp)
        self.ychan.addItems(ltmp)

        self.xchan.setCurrentIndex(0)
        self.ychan.setCurrentIndex(1)

        tmp = self.exec_()

        if tmp != 1:
            return tmp

        xcol = self.xchan.currentText()
        ycol = self.ychan.currentText()

        ltmp = ltmp[ltmp != xcol]
        ltmp = ltmp[ltmp != ycol]

        dat = []
        for i in ltmp:
            dat.append(PData())
            dat[-1].xdata = datatmp[xcol].values
            dat[-1].ydata = datatmp[ycol].values
            dat[-1].zdata = datatmp[i].values
            dat[-1].dataid = i

        self.outdata['Point'] = dat
        return True


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
        QtWidgets.QDialog.__init__(self, parent)

        self.name = 'Import Line Data: '
        self.pbar = None  # self.parent.pbar
        self.parent = parent
        self.indata = {}
        self.outdata = {}
        self.ifile = ''

        self.xchan = QtWidgets.QComboBox()
        self.ychan = QtWidgets.QComboBox()
        self.nodata = QtWidgets.QLineEdit('-99999')

        self.setupui()

    def setupui(self):
        """Set up UI."""
        gridlayout_main = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.raster.iodefs.importpointdata')
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
        """Entry point into item. Data imported from here."""
        ext = ('Geosoft XYZ (*.xyz);;'
               'All Files (*.*)')

        filename, filt = QtWidgets.QFileDialog.getOpenFileName(
            self.parent, 'Open File', '.', ext)
        if filename == '':
            return False

        os.chdir(os.path.dirname(filename))
        self.ifile = str(filename)

        if filt == 'Geosoft XYZ (*.xyz)':
            dat = self.get_GXYZ()
        else:
            return False

        i = list(dat.keys())[0]
        ltmp = dat[i].dtype.names

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
            self.parent.showprocesslog('Null Value error - abandoning import')
            return False

        xcol = self.xchan.currentText()
        ycol = self.ychan.currentText()

        ltmp = ltmp[ltmp != xcol]
        ltmp = ltmp[ltmp != ycol]

        dat2 = LData()
        dat2.xchannel = xcol
        dat2.ychannel = ycol
        dat2.data = dat
        dat2.nullvalue = nodata

        self.outdata['Line'] = dat2
        return True

    def get_GXYZ(self):
        """Get geosoft XYZ."""
        dat = []

        with open(self.ifile) as fno:
            head = fno.readline()
            tmp = fno.read()

        head = head.split()
        head.pop(0)
        tmp = tmp.lower()
        tmp = re.split('(line|tie)', tmp)
        if tmp[0] == '':
            tmp.pop(0)

        dtype = {}
        dtype['names'] = head
        dtype['formats'] = ['f4']*len(head)

        dat = {}
        for i in range(0, len(tmp), 2):
            tmp2 = tmp[i+1]
            tmp2 = tmp2.split('\n')
            line = tmp[i]+tmp2.pop(0)
            tmp2 = np.genfromtxt(tmp2, names=head)
            dat[line] = tmp2

        return dat

    def get_delimited(self):
        """Get a delimited line file."""
        datatmp = pd.read_csv(self.ifile, sep=None, engine='python')
        ltmp = datatmp.columns.values

        self.xchan.addItems(ltmp)
        self.ychan.addItems(ltmp)

        self.xchan.setCurrentIndex(0)
        self.ychan.setCurrentIndex(1)

        tmp = self.exec_()

        if tmp != 1:
            return tmp

        xcol = self.xchan.currentText()
        ycol = self.ychan.currentText()

        ltmp = ltmp[ltmp != xcol]
        ltmp = ltmp[ltmp != ycol]

        dat = []
        for i in ltmp:
            dat.append(PData())
            dat[-1].xdata = datatmp[xcol].values
            dat[-1].ydata = datatmp[ycol].values
            dat[-1].zdata = datatmp[i].values
            dat[-1].dataid = i

        return dat


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

    def settings(self):
        """Show Info."""
        if 'Point' in self.indata:
            data = copy.deepcopy(self.indata['Point'])
        else:
            self.parent.showprocesslog('No point data')
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
        self.outdata['Point'] = data

        return True


class ExportPoint():
    """
    Export Point Data.

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
    showtext : text output
        reference to the show process log text output on the main interface
    """

    def __init__(self, parent):
        self.name = 'Export Point: '
        self.pbar = None
        self.parent = parent
        self.indata = {}
        self.showtext = self.parent.showprocesslog

    def run(self):
        """Run routine."""
        if 'Point' not in self.indata:
            self.showtext('Error: You need to have a point data first!')
            return False

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.parent, 'Save File', '.', 'csv (*.csv)')

        if filename == '':
            return False

        self.parent.showprocesslog('Export busy...')

        os.chdir(os.path.dirname(filename))
        ofile = os.path.basename(filename)[:-4]
        data = self.indata['Point']

        dfall = pd.DataFrame()
        dfall.loc[:, 'X'] = data[0].xdata
        dfall.loc[:, 'Y'] = data[0].ydata

        for i, datai in enumerate(data):
            datid = datai.dataid
            if datid == '':
                datid = str(i)

            tmp = datai.zdata.tolist()
            dfall.loc[:, datid] = tmp

        ofile2 = ofile+'_all.csv'
        dfall.to_csv(ofile2, index=False)

        self.parent.showprocesslog('Export completed')

        return True


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
    showtext : text output
        reference to the show process log text output on the main interface
    """

    def __init__(self, parent):
        self.name = 'Export Point: '
        self.pbar = None
        self.parent = parent
        self.indata = {}
        self.showtext = self.parent.showprocesslog

    def run(self):
        """Run routine."""
        if 'Line' not in self.indata:
            self.showtext('Error: You need to have line data first!')
            return False

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.parent, 'Save File', '.', 'csv (*.csv)')

        if filename == '':
            return False

        self.parent.showprocesslog('Export busy...')

        os.chdir(os.path.dirname(filename))
        ofile = os.path.basename(filename)[:-4]
        data = self.indata['Line']

        dfall = None
        for line in data.data:
            if dfall is None:
                dfall = pd.DataFrame(data.data[line])
                dfall.loc[:, 'LINE'] = line
            else:
                dtmp = pd.DataFrame(data.data[line])
                dtmp.loc[:, 'LINE'] = line
                dfall = dfall.append(dtmp)

        ofile2 = ofile+'_all.csv'
        dfall.to_csv(ofile2, index=False)

        self.parent.showprocesslog('Export completed')

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

    def settings(self):
        """Entry point into item. Data imported from here."""
        ext = 'Shapefile (*.shp);;' + 'All Files (*.*)'

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self.parent,
                                                            'Open File',
                                                            '.', ext)
        if filename == '':
            return False
        os.chdir(os.path.dirname(filename))
        self.ifile = str(filename)

        ifile = str(filename)

        # Line data
        shapef = ogr.Open(ifile)
        if shapef is None:
            err = ('There was a problem importing the shapefile. Please make '
                   'sure you have at all the individual files which make up '
                   'the shapefile.')
            QtWidgets.QMessageBox.warning(self.parent, 'Error', err,
                                          QtWidgets.QMessageBox.Ok)
            return False

        lyr = shapef.GetLayer()
        dat = VData()
        attrib = {}
        allcrds = []

        for i in range(lyr.GetFeatureCount()):
            feat = lyr.GetFeature(i)
            ftmp = feat.items()
            for j in ftmp:
                if attrib.get(j) is None:
                    attrib[j] = [ftmp[j]]
                else:
                    attrib[j] = attrib[j]+[ftmp[j]]

        if lyr.GetGeomType() is ogr.wkbPoint:
            for i in range(lyr.GetFeatureCount()):
                feat = lyr.GetFeature(i)
                geom = feat.GetGeometryRef()
                pnts = np.array(geom.GetPoints()).tolist()
                if pnts[0][0] > -1.e+308:
                    allcrds.append(pnts)
            dat.dtype = 'Point'

        if lyr.GetGeomType() is ogr.wkbPolygon:
            for j in range(lyr.GetFeatureCount()):
                feat = lyr.GetFeature(j)
                geom = feat.GetGeometryRef()
                ifin = 0
                if geom.GetGeometryName() == 'MULTIPOLYGON':
                    imax = 0
                    for i in range(geom.GetGeometryCount()):
                        geom.GetGeometryRef(i)
                        itmp = geom.GetGeometryRef(i)
                        itmp = itmp.GetGeometryRef(0).GetPointCount()
                        if itmp > imax:
                            imax = itmp
                            ifin = i
                geom = geom.GetGeometryRef(ifin)
                pnts = np.array(geom.GetPoints()).tolist()
                allcrds.append(pnts)
            dat.dtype = 'Poly'

        if lyr.GetGeomType() is ogr.wkbLineString:
            for i in range(lyr.GetFeatureCount()):
                feat = lyr.GetFeature(i)
                geom = feat.GetGeometryRef()
                pnts = np.array(geom.GetPoints()).tolist()
                allcrds.append(pnts)
            dat.dtype = 'Line'

        dat.crds = allcrds
        dat.attrib = attrib
        self.outdata['Vector'] = dat
        shapef = None
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

    chk = bbpath.contains_points(np.transpose([data[0].xdata, data[0].ydata]))

    for idata in data:
        idata.xdata = idata.xdata[chk]
        idata.ydata = idata.ydata[chk]
        idata.zdata = idata.zdata[chk]

    shapef = None
    return data
