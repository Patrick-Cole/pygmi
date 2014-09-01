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

from PyQt4 import QtGui
import numpy as np
from osgeo import ogr
from .datatypes import PData
from .datatypes import VData
import os


class ImportLEMI417Data(object):
    """ Import LEMI-417 MT Data

    This is a class used to import LEMI-417 MT Data in ASCII format."""
    def __init__(self, parent=None):
        self.ifile = ""
        self.name = "Import LEMI-417 Data: "
        self.ext = ""
        self.pbar = None
        self.parent = parent
        self.indata = {}
        self.outdata = {}

    def settings(self):
        """ Settings section"""
        ext = \
            "LEMI-417 Text DataAll Files (*.t*)"

        filename = QtGui.QFileDialog.getOpenFileName(
            self.parent, 'Open File', '.', ext)
        if filename == '':
            return False
        os.chdir(filename.rpartition('/')[0])

        self.ifile = str(filename)
        self.ext = filename[-3:]

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
    """ Import Point Data

    This class imports ASCII point data."""
    def __init__(self, parent=None):
        self.ifile = ""
        self.name = "Import Point/Line Data: "
        self.ext = ""
        self.pbar = None
        self.parent = parent
        self.indata = {}
        self.outdata = {}

    def settings(self):
        """ Settings """
        ext = \
            "Common Formats (*.csv *.dat *.xyz *.txt);;" +\
            "All Files (*.*)"

        filename = QtGui.QFileDialog.getOpenFileName(
            self.parent, 'Open File', '.', ext)
        if filename == '':
            return False
        os.chdir(filename.rpartition('/')[0])

        self.ifile = str(filename)
        self.ext = filename[-3:]

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

        isheader = any(c.isalpha() for c in ltmp)

        srows = 0
        ltmp = ltmp.split(dlim)
        if isheader:
            srows = 1
        else:
            ltmp = [str(c) for c in range(len(ltmp))]

        try:
            datatmp = np.loadtxt(filename, unpack=True, delimiter=dlim,
                                 skiprows=srows)
        except ValueError:
            QtGui.QMessageBox.critical(self.parent, 'Import Error',
                                       'There was a problem loading the file.'
                                       ' You may have a text character in one'
                                       ' of your columns.')
            return False

        dat = []
        if tmp == QtGui.QMessageBox.Yes:
            for i in range(2, datatmp.shape[0]):
                dat.append(PData())
                dat[-1].xdata = datatmp[0]
                dat[-1].ydata = datatmp[1]
                dat[-1].zdata = datatmp[i]
                dat[-1].dataid = ltmp[i]
        else:
            for i in range(datatmp.shape[0]):
                dat.append(PData())
                dat[i].zdata = datatmp[i]
                dat[i].dataid = ltmp[i]

        self.outdata['Point'] = dat
        return True

#    def gdal(self):
#        """ Process """
#        dat = []
#        bname = self.ifile.split('/')[-1].rpartition('.')[0]+': '
#        ifile = self.ifile[:]
#        if self.ext == 'hdr':
#            ifile = self.ifile[:-4]
#
#        dataset = gdal.Open(ifile, gdal.GA_ReadOnly)
#        gtr = dataset.GetGeoTransform()
#
#        for i in range(dataset.RasterCount):
#            rtmp = dataset.GetRasterBand(i+1)
#            dat.append(Data())
#            dat[i].data = np.ma.array(rtmp.ReadAsArray())
#            dat[i].data[dat[i].data == rtmp.GetNoDataValue()] = np.nan
#            dat[i].data = np.ma.masked_invalid(dat[i].data)
#
#            dat[i].nrofbands = dataset.RasterCount
#            dat[i].tlx = gtr[0]
#            dat[i].tly = gtr[3]
#            dat[i].bandid = bname+str(i+1)
#            dat[i].nullvalue = rtmp.GetNoDataValue()
#            dat[i].rows = dataset.RasterYSize
#            dat[i].cols = dataset.RasterXSize
#            dat[i].xdim = abs(gtr[1])
#            dat[i].ydim = abs(gtr[5])
#            dat[i].gtr = gtr
#            dat[i].wkt = dataset.GetProjection()
#
#        self.outdata['Raster'] = dat


class ExportPoint(object):
    """ Export Data """
    def __init__(self, parent):
        self.ifile = ""
        self.name = "Export Point: "
        self.ext = ""
        self.pbar = None
        self.parent = parent
        self.indata = {}
        self.outdata = {}
        self.lmod = None
#        self.dirname = ""
        self.showtext = self.parent.showprocesslog

    def run(self):
        """ Show Info """
        if 'Point' not in self.indata:
            self.parent.showprocesslog(
                'Error: You need to have a point data first!')
            return

        data = self.indata['Point']

        filename = QtGui.QFileDialog.getSaveFileName(
            self.parent, 'Save File', '.', 'csv (*.csv)')

        if filename == '':
            return

        os.chdir(filename.rpartition('/')[0])
        ofile = str(filename.rpartition('/')[-1][:-4])
        self.ext = filename[-3:]

        for i in range(len(data)):
            datid = data[i].dataid
            if datid is '':
                datid = str(i)

            dattmp = np.transpose([data[i].xdata, data[i].ydata,
                                   data[i].zdata])

            ofile2 = ofile+'_'+''.join(x for x in datid if x.isalnum())+'.csv'

            np.savetxt(ofile2, dattmp, delimiter=',')


class ImportShapeData(object):
    """ Import Data """
    def __init__(self, parent=None):
        self.ifile = ""
        self.name = "Import Shapefile Data: "
        self.ext = ""
        self.pbar = None
        self.parent = parent
        self.indata = {}
        self.outdata = {}

    def settings(self):
        """ Settings """
        ext = \
            "Shapefile (*.shp);;" +\
            "All Files (*.*)"

        filename = QtGui.QFileDialog.getOpenFileName(
            self.parent, 'Open File', '.', ext)
        if filename == '':
            return False
        os.chdir(filename.rpartition('/')[0])

        self.ifile = str(filename)
        self.ext = filename[-3:]

        # Line data
        shapef = ogr.Open(self.ifile)
        lyr = shapef.GetLayer()
        dat = VData()
        attrib = {}
        allcrds = []

        for i in range(lyr.GetFeatureCount()):
            feat = lyr.GetFeature(i)
            ftmp = feat.items()
            for j in ftmp.keys():
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
