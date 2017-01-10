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
from PyQt4 import QtGui
import numpy as np
from osgeo import ogr
from pygmi.vector.datatypes import PData
from pygmi.vector.datatypes import VData


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

        filename = QtGui.QFileDialog.getOpenFileName(
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


class ImportPointData(object):
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
        self.name = "Import Point/Line Data: "
        self.pbar = None
        self.parent = parent
        self.indata = {}
        self.outdata = {}
        self.ifile = ""

    def settings(self):
        """Entry point into item. Data imported from here."""
        ext = ("Common Formats (*.csv *.dat *.xyz *.txt);;"
               "All Files (*.*)")

        filename = QtGui.QFileDialog.getOpenFileName(
            self.parent, 'Open File', '.', ext)
        if filename == '':
            return False
        os.chdir(filename.rpartition('/')[0])
        self.ifile = str(filename)

        tmp = QtGui.QMessageBox.question(self.parent, 'Data Query',
                                         'Are the first two columns X and Y?',
                                         QtGui.QMessageBox.Yes |
                                         QtGui.QMessageBox.No)

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

        xcol = 0
        ycol = 1
        if ltmp.index('lat') < ltmp.index('lon') and 'lat' in ltmp:
            xcol = 1
            ycol = 0

        srows = 0
        ltmp = ltmp.split(dlim)
        if isheader:
            srows = 1
        else:
            ltmp = [str(c) for c in range(len(ltmp))]

        datatmp = np.genfromtxt(filename, unpack=True, delimiter=dlim,
                                skip_header=srows, usemask=True)

#        datatmp.mask = np.logical_or(datatmp.mask, np.isnan(datatmp.data))
#        try:
#            datatmp = np.loadtxt(filename, unpack=True, delimiter=dlim,
#                                 skiprows=srows)
#        except ValueError:
#            QtGui.QMessageBox.critical(self.parent, 'Import Error',
#                                       'There was a problem loading the file.'
#                                       ' You may have a text character in one'
#                                       ' of your columns.')
#            return False

        dat = []
        if tmp == QtGui.QMessageBox.Yes:
            for i in range(2, datatmp.shape[0]):
                dat.append(PData())
                dat[-1].xdata = datatmp[xcol]
                dat[-1].ydata = datatmp[ycol]
                dat[-1].zdata = datatmp[i]
                dat[-1].dataid = ltmp[i]
        else:
            for i in range(datatmp.shape[0]):
                dat.append(PData())
                dat[i].zdata = datatmp[i]
                dat[i].dataid = ltmp[i]

        self.outdata['Point'] = dat
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

        filename = QtGui.QFileDialog.getSaveFileName(
            self.parent, 'Save File', '.', 'csv (*.csv)')

        if filename == '':
            return

        os.chdir(filename.rpartition('/')[0])
        ofile = str(filename.rpartition('/')[-1][:-4])
#        self.ext = filename[-3:]
        data = self.indata['Point']

        for i, datai in enumerate(data):
            datid = datai.dataid
            if datid is '':
                datid = str(i)

            dattmp = np.transpose([datai.xdata, datai.ydata, datai.zdata])

            ofile2 = ofile+'_'+''.join(x for x in datid if x.isalnum())+'.csv'

            np.savetxt(ofile2, dattmp, delimiter=',')


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

        filename = QtGui.QFileDialog.getOpenFileName(self.parent,
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
            QtGui.QMessageBox.warning(self.parent, 'Error', err,
                                      QtGui.QMessageBox.Ok,
                                      QtGui.QMessageBox.Ok)
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
            for i in range(lyr.GetFeatureCount()):
                feat = lyr.GetFeature(i)
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
