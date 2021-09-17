# -----------------------------------------------------------------------------
# Name:        dataprep.py (part of PyGMI)
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
"""Data Preparation for Vector Data."""

import os
import copy
from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
import matplotlib.path as mplPath
from osgeo import osr, ogr
from scipy.interpolate import griddata

import pygmi.menu_default as menu_default
from pygmi.raster.dataprep import GroupProj
from pygmi.raster.datatypes import Data
from pygmi.vector.minc import minc


class PointCut():
    """
    Cut Data using shapefiles.

    This class cuts point datasets using a boundary defined by a polygon
    shapefile.

    Attributes
    ----------
    ifile : str
        input file name.
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    outdata : dictionary
        dictionary of output datasets
    """

    def __init__(self, parent=None):
        self.ifile = ''
        self.pbar = parent.pbar
        self.parent = parent
        self.indata = {}
        self.outdata = {}
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
        if 'Line' in self.indata:
            data = copy.deepcopy(self.indata['Line'])
            key = list(data.keys())[0]
            data = data[key]
        else:
            self.showprocesslog('No point data')
            return False

        nodialog = False
        if not nodialog:
            ext = 'Shape file (*.shp)'
            self.ifile, _ = QtWidgets.QFileDialog.getOpenFileName(
                self.parent, 'Open Shape File', '.', ext)
            if self.ifile == '':
                return False

        os.chdir(os.path.dirname(self.ifile))
        data = cut_point(data, self.ifile)

        if data is None:
            err = ('There was a problem importing the shapefile. Please make '
                   'sure you have at all the individual files which make up '
                   'the shapefile.')
            QtWidgets.QMessageBox.warning(self.parent, 'Error', err,
                                          QtWidgets.QMessageBox.Ok)
            return False

        self.pbar.to_max()
        self.outdata['Line'] = {key: data}

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


class DataGrid(QtWidgets.QDialog):
    """
    Grid Point Data.

    This class grids point data using a nearest neighbourhood technique.

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
        if parent is None:
            self.showprocesslog = print
        else:
            self.showprocesslog = parent.showprocesslog

        self.indata = {}
        self.outdata = {}
        self.parent = parent
        self.dxy = None
        self.dataid_text = None

        self.dsb_dxy = QtWidgets.QLineEdit('1.0')
        self.dsb_null = QtWidgets.QLineEdit('0.0')
        self.bdist = QtWidgets.QLineEdit('4.0')

        self.dataid = QtWidgets.QComboBox()
        self.grid_method = QtWidgets.QComboBox()
        self.label_rows = QtWidgets.QLabel('Rows: 0')
        self.label_cols = QtWidgets.QLabel('Columns: 0')
        self.label_bdist = QtWidgets.QLabel('Blanking Distance:')

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
        helpdocs = menu_default.HelpButton('pygmi.vector.dataprep.datagrid')
        label_band = QtWidgets.QLabel('Column to Grid:')
        label_dxy = QtWidgets.QLabel('Cell Size:')
        label_null = QtWidgets.QLabel('Null Value:')
        label_method = QtWidgets.QLabel('Gridding Method:')

        val = QtGui.QDoubleValidator(0.0000001, 9999999999.0, 9)
        val.setNotation(QtGui.QDoubleValidator.ScientificNotation)
        val.setLocale(QtCore.QLocale(QtCore.QLocale.C))

        self.dsb_dxy.setValidator(val)
        self.dsb_null.setValidator(val)

        # self.grid_method.addItems(['Quick Grid', 'Nearest Neighbour',
        #                            'Linear', 'Cubic', 'Minimum Curvature'])

        self.grid_method.addItems(['Nearest Neighbour', 'Linear', 'Cubic',
                                   'Minimum Curvature'])

        self.label_bdist.hide()
        self.bdist.hide()

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Dataset Gridding')

        gridlayout_main.addWidget(label_method, 0, 0, 1, 1)
        gridlayout_main.addWidget(self.grid_method, 0, 1, 1, 1)
        gridlayout_main.addWidget(label_dxy, 1, 0, 1, 1)
        gridlayout_main.addWidget(self.dsb_dxy, 1, 1, 1, 1)
        gridlayout_main.addWidget(self.label_rows, 2, 0, 1, 2)
        gridlayout_main.addWidget(self.label_cols, 3, 0, 1, 2)
        gridlayout_main.addWidget(label_band, 4, 0, 1, 1)
        gridlayout_main.addWidget(self.dataid, 4, 1, 1, 1)
        gridlayout_main.addWidget(label_null, 5, 0, 1, 1)
        gridlayout_main.addWidget(self.dsb_null, 5, 1, 1, 1)
        gridlayout_main.addWidget(self.label_bdist, 6, 0, 1, 1)
        gridlayout_main.addWidget(self.bdist, 6, 1, 1, 1)
        gridlayout_main.addWidget(helpdocs, 7, 0, 1, 1)
        gridlayout_main.addWidget(buttonbox, 7, 1, 1, 3)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        self.dsb_dxy.textChanged.connect(self.dxy_change)
        self.grid_method.currentIndexChanged.connect(self.grid_method_change)

    def dxy_change(self):
        """
        When dxy is changed on the interface, this updates rows and columns.

        Returns
        -------
        None.

        """
        txt = str(self.dsb_dxy.text())
        if txt.replace('.', '', 1).isdigit():
            self.dxy = float(self.dsb_dxy.text())
        else:
            return

        key = list(self.indata['Line'].keys())[0]
        data = self.indata['Line'][key]
        data = data.dropna()

        x = data.pygmiX.values
        y = data.pygmiY.values

        cols = round(x.ptp()/self.dxy)
        rows = round(y.ptp()/self.dxy)

        self.label_rows.setText('Rows: '+str(rows))
        self.label_cols.setText('Columns: '+str(cols))

    def grid_method_change(self):
        """
        When grid method is changed, this updated hidden controls.

        Returns
        -------
        None.

        """
        if self.grid_method.currentText() == 'Minimum Curvature':
            self.label_bdist.show()
            self.bdist.show()
        else:
            self.label_bdist.hide()
            self.bdist.hide()

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
        tmp = []
        if 'Line' not in self.indata:
            self.showprocesslog('No Point Data')
            return False

        self.dataid.clear()

        key = list(self.indata['Line'].keys())[0]
        data = self.indata['Line'][key]
        data = data.dropna()

        filt = ((data.columns != 'geometry') &
                (data.columns != 'line') &
                (data.columns != 'pygmiX') &
                (data.columns != 'pygmiY'))

        cols = list(data.columns[filt])
        self.dataid.addItems(cols)

        if self.dataid_text is None:
            self.dataid_text = self.dataid.currentText()
        if self.dataid_text in cols:
            self.dataid.setCurrentText(self.dataid_text)

        if self.dxy is None:
            x = data.pygmiX.values
            y = data.pygmiY.values

            dx = x.ptp()/np.sqrt(x.size)
            dy = y.ptp()/np.sqrt(y.size)
            self.dxy = max(dx, dy)
            self.dxy = min([x.ptp(), y.ptp(), self.dxy])

        self.dsb_dxy.setText(f'{self.dxy:.8f}')
        self.dxy_change()

        if not nodialog:
            tmp = self.exec_()
            if tmp != 1:
                return False

        try:
            float(self.dsb_dxy.text())
            float(self.dsb_null.text())
            float(self.bdist.text())
        except ValueError:
            self.showprocesslog('Value Error')
            return False

        self.acceptall()

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
        self.dsb_dxy.textChanged.disconnect()
        self.dataid_text = projdata['band']
        self.dsb_dxy.setText(projdata['dxy'])
        self.dsb_null.setText(projdata['nullvalue'])
        self.bdist.setText(projdata['bdist'])

        self.dsb_dxy.textChanged.connect(self.dxy_change)

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

        projdata['dxy'] = self.dsb_dxy.text()
        projdata['nullvalue'] = self.dsb_null.text()
        projdata['bdist'] = self.bdist.text()
        projdata['band'] = self.dataid_text

        return projdata

    def acceptall(self):
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        None.

        """
        dxy = float(self.dsb_dxy.text())
        method = self.grid_method.currentText()
        nullvalue = float(self.dsb_null.text())
        bdist = float(self.bdist.text())
        key = list(self.indata['Line'].keys())[0]
        data = self.indata['Line'][key]
        data = data.dropna()
        newdat = []

        if bdist < 1:
            bdist = None
            self.showprocesslog('Blanking distance too small.')

        filt = (data[self.dataid.currentText()] != nullvalue)
        x = data.pygmiX.values[filt]
        y = data.pygmiY.values[filt]
        z = data[self.dataid.currentText()].values[filt]

        if method == 'Minimum Curvature':
            gdat = minc(x, y, z, dxy, showprocesslog=self.showprocesslog,
                        bdist=bdist)
            gdat = np.ma.filled(gdat, fill_value=nullvalue)
        else:
            extent = np.array([x.min(), x.max(), y.min(), y.max()])

            rows = int((extent[3] - extent[2])/dxy+1)
            cols = int((extent[1] - extent[0])/dxy+1)

            xxx = np.linspace(extent[0], extent[1], cols)
            yyy = np.linspace(extent[2], extent[3], rows)
            xxx, yyy = np.meshgrid(xxx, yyy)

            points = np.transpose([x.flatten(), y.flatten()])

            if method == 'Nearest Neighbour':
                gdat = griddata(points, z, (xxx, yyy), method='nearest')
            elif method == 'Linear':
                gdat = griddata(points, z, (xxx, yyy), method='linear',
                                fill_value=nullvalue)
            elif method == 'Cubic':
                gdat = griddata(points, z, (xxx, yyy), method='cubic',
                                fill_value=nullvalue)
            gdat = gdat[::-1]
        gdat = np.ma.masked_equal(gdat, nullvalue)

# Create dataset
        dat = Data()
        dat.data = gdat
        dat.nullvalue = nullvalue
        dat.dataid = self.dataid.currentText()
        dat.xdim = dxy
        dat.ydim = dxy
        # dat.extent = [x.min(), x.max(), y.min(), y.max()]

        rows, cols = dat.data.shape
        # left = x.min()
        # top = y.max()
        # right = left + dxy*cols
        # bottom = top - dxy*rows

        left = x.min()
        bottom = y.min()
        top = bottom + dxy*rows
        right = left + dxy*cols

        dat.extent = (left, right, bottom, top)

        newdat.append(dat)

        self.outdata['Raster'] = newdat
        self.outdata['Line'] = self.indata['Line']


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
        if parent is None:
            self.showprocesslog = print
        else:
            self.showprocesslog = parent.showprocesslog

        self.indata = {}
        self.outdata = {}
        self.parent = parent
        self.orig_wkt = None
        self.targ_wkt = None

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
            self.showprocesslog('Could not reproject')
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

        if np.inf in xy:
            self.showprocesslog('Note: inf values in reprojected results. '
                                'Please check your input and output '
                                'projections or input x and y data for '
                                'mistakes.')

        data = data.assign(Xnew=xy[:, 0])
        data = data.assign(Ynew=xy[:, 1])
        data.pygmiX = xy[:, 0]
        data.pygmiY = xy[:, 1]

        self.outdata['Line'] = {key: data}

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
        if 'Line' not in self.indata and 'Vector' not in self.indata:
            self.showprocesslog('No vector data.')
            return False

        if 'Vector' in self.indata:
            firstkey = next(iter(self.indata['Vector'].keys()))
            self.orig_wkt = self.indata['Vector'][firstkey].crs.to_wkt()

        if self.orig_wkt is None:
            indx = self.in_proj.combobox.findText(r'WGS 84 / '
                                                  r'Geodetic Geographic')
            self.in_proj.combobox.setCurrentIndex(indx)
            self.orig_wkt = self.in_proj.wkt
        else:
            self.in_proj.set_current(self.orig_wkt)
        if self.targ_wkt is None:
            indx = self.in_proj.combobox.findText(r'WGS 84 / UTM zone 35S')
            self.out_proj.combobox.setCurrentIndex(indx)
            self.targ_wkt = self.out_proj.wkt
        else:
            self.out_proj.set_current(self.targ_wkt)

#        iwkt = self.in_proj.epsg_proj['WGS 84 / Geodetic Geographic'].wkt
#        indx = self.in_proj.combobox.findText('WGS 84 / Geodetic Geographic')
#        self.in_proj.combobox.setCurrentIndex(indx)

#        indx = self.in_proj.combobox.findText('WGS 84 / UTM zone 35S')
#        self.out_proj.combobox.setCurrentIndex(indx)

        if not nodialog:
            tmp = self.exec_()

            if tmp != 1:
                return False

        if 'Vector' in self.indata:
            self.outdata['Vector'] = {}
            for i in self.indata['Vector']:
                ivec = self.indata['Vector'][i]
                ivec.set_crs(self.in_proj.wkt)
                self.outdata['Vector'][i] = ivec.to_crs(self.out_proj.wkt)
        else:
            self.acceptall()

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
        self.orig_wkt = projdata['orig_wkt']
        self.targ_wkt = projdata['targ_wkt']

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

        projdata['orig_wkt'] = self.in_proj.wkt
        projdata['targ_wkt'] = self.out_proj.wkt

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
    data : Data
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


def quickgrid(x, y, z, dxy, numits=4, showprocesslog=print):
    """
    Do a quick grid.

    Parameters
    ----------
    x : numpy array
        array of x coordinates
    y : numpy array
        array of y coordinates
    z : numpy array
        array of z values - this is the column being gridded
    dxy : float
        cell size for the grid, in both the x and y direction.
    numits : int
        number of iterations. By default its 4. If this is negative, a maximum
        will be calculated and used.
    showprocesslog : function, optional
        Routine to show text messages. The default is print.

    Returns
    -------
    newz : numpy array
        M x N array of z values
    """
    showprocesslog('Creating Grid')
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    newmask = np.array([1])
    j = -1
    rows = int((ymax-ymin)/dxy)+1
    cols = int((xmax-xmin)/dxy)+1

    if numits < 1:
        numits = int(max(np.log2(cols), np.log2(rows)))

    while np.max(newmask) > 0 and j < (numits-1):
        j += 1
        jj = 2**j
        dxy2 = dxy*jj
        rows = int((ymax-ymin)/dxy2)+1
        cols = int((xmax-xmin)/dxy2)+1

        newz = np.zeros([rows, cols])
        zdiv = np.zeros([rows, cols])

        xindex = ((x-xmin)/dxy2).astype(int)
        yindex = ((y-ymin)/dxy2).astype(int)

        for i in range(z.size):
            newz[yindex[i], xindex[i]] += z[i]
            zdiv[yindex[i], xindex[i]] += 1

        filt = zdiv > 0
        newz[filt] = newz[filt]/zdiv[filt]

        if j == 0:
            newmask = np.ones([rows, cols])
            for i in range(z.size):
                newmask[yindex[i], xindex[i]] = 0
            zfin = newz
        else:
            xx, yy = newmask.nonzero()
            xx2 = xx//jj
            yy2 = yy//jj
            zfin[xx, yy] = newz[xx2, yy2]
            newmask[xx, yy] = np.logical_not(zdiv[xx2, yy2])

        showprocesslog('Iteration done: '+str(j+1)+' of '+str(numits))

    showprocesslog('Finished!')

    newz = np.ma.array(zfin)
    newz.mask = newmask
    return newz


def _testfn():
    """Test routine."""
    import sys
    import matplotlib.pyplot as plt
    from pygmi.vector.iodefs import ImportLineData, ImportShapeData

    APP = QtWidgets.QApplication(sys.argv)  # Necessary to test Qt Classes

    ifile = r'C:\Workdata\vector\Line Data\MAGARCHIVE.XYZ'

    # ifile = r'D:\Workdata\vector\linecut\test2.csv'
    # sfile = r'D:\Workdata\vector\linecut\test2_cut_outline.shp'

    IO = ImportLineData()
    IO.ifile = ifile
    # IO.filt = 'Comma Delimited (*.csv)'
    IO.filt = 'Geosoft XYZ (*.xyz)'
    IO.settings(True)

    # line = list(IO.outdata['Line'].values())
    # plt.plot(line[0].x, line[0].y)
    # plt.show()

    DG = DataGrid()
    DG.indata = IO.outdata
    DG.settings()

    dat = DG.outdata['Raster'][0].data

    plt.imshow(dat)
    plt.show()

    # DR = DataReproj()
    # DR.indata = IO.outdata
    # DR.settings(True)

    breakpoint()


if __name__ == "__main__":
    _testfn()
