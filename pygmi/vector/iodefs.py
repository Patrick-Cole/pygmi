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
""" Import Data """

import os
import copy
from PyQt5 import QtWidgets, QtCore
import numpy as np
from osgeo import ogr
import matplotlib.path as mplPath
import pandas as pd
from pygmi.vector.datatypes import PData
from pygmi.vector.datatypes import VData
import pygmi.menu_default as menu_default


class ImportLEMI417Data(object):
    """
    Import LEMI-417 ASCII MT Data

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
        self.name = "Import LEMI-417 Data: "
        self.pbar = None
        self.parent = parent
        self.outdata = {}
        self.ifile = ""

    def settings(self):
        """Entry point into item. Data imported from here."""
        ext = "LEMI-417 Text DataAll Files (*.t*)"

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.parent, 'Open File', '.', ext)
        if filename == '':
            return False
        os.chdir(filename.rpartition('/')[0])
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
    Import Point Data

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

        self.name = "Import Point/Line Data: "
        self.pbar = None  # self.parent.pbar
        self.parent = parent
        self.indata = {}
        self.outdata = {}
        self.ifile = ""

        self.xchan = QtWidgets.QComboBox()
        self.ychan = QtWidgets.QComboBox()

        self.setupui()

    def setupui(self):
        """ Setup UI """
        gridlayout_main = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.raster.iodefs.importpointdata')
        label_xchan = QtWidgets.QLabel()
        label_ychan = QtWidgets.QLabel()

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle(r"Import Point/Line Data")
        label_xchan.setText("X Channel:")
        label_ychan.setText("Y Channel:")

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
        ext = ("Common Formats (*.csv *.dat *.xyz *.txt);;"
               "All Files (*.*)")

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.parent, 'Open File', '.', ext)
        if filename == '':
            return False
        os.chdir(filename.rpartition('/')[0])
        self.ifile = str(filename)

        dlim = None
        if filename[-3:] == 'csv':
            dlim = ','

        pntfile = open(filename)
        ltmp = pntfile.readline()
        pntfile.close()
        ltmp = ltmp.lower()

        isheader = any(c.isalpha() for c in ltmp)

        if ',' in ltmp:
            dlim = ','

        srows = 0
        ltmp = ltmp.split(dlim)
        ltmp[-1] = ltmp[-1].strip('\n')

        self.xchan.addItems(ltmp)
        self.ychan.addItems(ltmp)

        self.xchan.setCurrentIndex(0)
        self.ychan.setCurrentIndex(1)

        tmp = self.exec_()

        if tmp != 1:
            return tmp

        xcol = self.xchan.currentIndex()
        ycol = self.ychan.currentIndex()

        if isheader:
            srows = 1
        else:
            ltmp = [str(c) for c in range(len(ltmp))]

#        datatmp = np.genfromtxt(filename, unpack=True, delimiter=dlim,
#                                skip_header=srows, usemask=True)
        datatmp = np.genfromtxt(filename, unpack=True, delimiter=dlim,
                                skip_header=srows, dtype=None)

        datanames = datatmp.dtype.names
        dat = []
        if datanames is None:
            datatmp = np.transpose(datatmp)
            dat = []
            for i, datatmpi in enumerate(datatmp):
                if i == xcol or i == ycol:
                    continue
                dat.append(PData())
                dat[-1].xdata = datatmp[xcol]
                dat[-1].ydata = datatmp[ycol]
                dat[-1].zdata = datatmpi
                dat[-1].dataid = ltmp[i]
        else:
            for i in datanames:
                if i == 'f'+str(xcol) or i == 'f'+str(ycol):
                    continue
                dat.append(PData())
                dat[-1].xdata = datatmp['f'+str(xcol)]
                dat[-1].ydata = datatmp['f'+str(ycol)]
                dat[-1].zdata = datatmp[i]
                dat[-1].dataid = ltmp[int(i[1:])]

        self.outdata['Point'] = dat
        return True



class PointCut(object):
    """
    Cut Data using shapefiles

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
        self.ifile = ""
        self.name = "Cut Data:"
        self.ext = ""
        self.pbar = parent.pbar
        self.parent = parent
        self.indata = {}
        self.outdata = {}

    def settings(self):
        """ Show Info """
        if 'Point' in self.indata:
            data = copy.deepcopy(self.indata['Point'])
        else:
            self.parent.showprocesslog('No point data')
            return

        ext = "Shape file (*.shp)"

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.parent, 'Open Shape File', '.', ext)
        if filename == '':
            return False
        os.chdir(filename.rpartition('/')[0])

        self.ifile = str(filename)
        self.ext = filename[-3:]
        data = cut_point(data, self.ifile)

        if data is None:
            err = ('There was a problem importing the shapefile. Please make '
                   'sure you have at all the individual files which make up '
                   'the shapefile.')
            QtWidgets.QMessageBox.warning(self.parent, 'Error', err,
                                          QtWidgets.QMessageBox.Ok,
                                          QtWidgets.QMessageBox.Ok)
            return False


        self.pbar.to_max()
        self.outdata['Point'] = data

        return True


class ExportPoint(object):
    """
    Export Point Data

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
        self.name = "Export Point: "
        self.pbar = None
        self.parent = parent
        self.indata = {}
        self.showtext = self.parent.showprocesslog

    def run(self):
        """ Runs routine """
        if 'Point' not in self.indata:
            self.showtext('Error: You need to have a point data first!')
            return

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.parent, 'Save File', '.', 'csv (*.csv)')

        if filename == '':
            return

        os.chdir(filename.rpartition('/')[0])
        ofile = str(filename.rpartition('/')[-1][:-4])
        data = self.indata['Point']
#        datall = []
#        head = ''

        dfall = pd.DataFrame()
        dfall.loc[:, 'X'] = data[0].xdata
        dfall.loc[:, 'Y'] = data[0].ydata

        for i, datai in enumerate(data):
#            datall.append(datai.zdata)
#            head += ','+datid

            datid = datai.dataid
            if datid == '':
                datid = str(i)

#            df = pd.DataFrame()
#            df.loc[:, 'X'] = datai.xdata
#            df.loc[:, 'Y'] = datai.ydata
#            df.loc[:, datai.dataid] = datai.zdata
            dfall.loc[:, datid] = datai.zdata

#            ofile2 = ofile+'_'+''.join(x for x in datid if x.isalnum())+'.csv'

#            df.to_csv(ofile2, index=False)
#            del df

        ofile2 = ofile+'_all.csv'
        dfall.to_csv(ofile2, index=False)

        self.parent.showprocesslog('Export completed')

        return True


class ImportShapeData(object):
    """
    Import Shapefile Data

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
        self.name = "Import Shapefile Data: "
        self.pbar = None
        self.parent = parent
        self.outdata = {}
        self.ifile = ""

    def settings(self):
        """Entry point into item. Data imported from here."""
        ext = "Shapefile (*.shp);;" + "All Files (*.*)"

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self.parent,
                                                            'Open File',
                                                            '.', ext)
        if filename == '':
            return False
        os.chdir(filename.rpartition('/')[0])
        self.ifile = str(filename)

        ifile = str(filename)

        # Line data
        shapef = ogr.Open(ifile)
        if shapef is None:
            err = ('There was a problem importing the shapefile. Please make '
                   'sure you have at all the individual files which make up '
                   'the shapefile.')
            QtWidgets.QMessageBox.warning(self.parent, 'Error', err,
                                          QtWidgets.QMessageBox.Ok,
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
        return True


def cut_point(data, ifile):
    """Cuts a point dataset

    Cut a point dataset using a shapefile

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
        return

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
    for p in range(pts.GetPointCount()):
        points.append((pts.GetX(p), pts.GetY(p)))

    bbpath = mplPath.Path(points)

    chk = bbpath.contains_points(np.transpose([data[0].xdata, data[0].ydata]))

    for idata in data:
        idata.xdata = idata.xdata[chk]
        idata.ydata = idata.ydata[chk]
        idata.zdata = idata.zdata[chk]

    return data