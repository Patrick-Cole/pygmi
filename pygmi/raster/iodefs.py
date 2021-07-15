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
"""Import raster data."""

import warnings
import os
import copy
import struct
from PyQt5 import QtWidgets, QtCore
import numpy as np
from osgeo import gdal, osr

from pygmi.raster.datatypes import Data
from pygmi.raster.dataprep import merge
from pygmi.misc import ProgressBarText


class ComboBoxBasic(QtWidgets.QDialog):
    """
    A combobox to select data bands.

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

        self.parent = parent
        self.indata = {}
        self.outdata = {}

        # create GUI
        self.setWindowTitle('Band Selection')

        self.vbox = QtWidgets.QVBoxLayout()
        self.setLayout(self.vbox)

        self.combo = QtWidgets.QListWidget()
        self.combo.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)

        self.vbox.addWidget(self.combo)

        self.buttonbox = QtWidgets.QDialogButtonBox()
        self.buttonbox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonbox.setCenterButtons(True)
        self.buttonbox.setStandardButtons(
            QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)

        self.vbox.addWidget(self.buttonbox)

        self.buttonbox.accepted.connect(self.accept)
        self.buttonbox.rejected.connect(self.reject)

    def run(self):
        """
        Run.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        self.parent.scene.selectedItems()[0].update_indata()
        my_class = self.parent.scene.selectedItems()[0].my_class

        data = my_class.indata.copy()

        tmp = []
        for i in data['Raster']:
            tmp.append(i.dataid)
        self.combo.addItems(tmp)

        if not tmp:
            return False

        tmp = self.exec_()

        if tmp != 1:
            return False

        atmp = [i.row() for i in self.combo.selectedIndexes()]

        if atmp:
            dtmp = []
            for i in atmp:
                dtmp.append(data['Raster'][i])
            data['Raster'] = dtmp

        my_class.indata = data
        if hasattr(my_class, 'data_reset'):
            my_class.data_reset()

        if hasattr(my_class, 'data_init'):
            my_class.data_init()

        self.parent.scene.selected_item_info()
        return True


class ImportData():
    """
    Import Data - Interfaces with GDAL routines.

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
        self.ifile = ''
        self.parent = parent
        self.indata = {}
        self.outdata = {}
        self.filt = ''
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
        piter = iter
        if not nodialog:
            if self.parent is not None:
                piter = self.parent.pbar.iter
            ext = ('Common formats (*.ers *.hdr *.tif *.tiff *.sdat *.img *.pix '
                   '*.bil);;'
                   'ArcGIS BIL (*.bil);;'
                   'Arcinfo Binary Grid (hdr.adf);;'
                   'ASCII with .hdr header (*.asc);;'
                   'ASCII XYZ (*.xyz);;'
                   'ENVI (*.hdr);;'
                   'ESRI ASCII (*.asc);;'
                   'ERMapper (*.ers);;'
                   'ERDAS Imagine (*.img);;'
                   'GeoPak grid (*.grd);;'
                   'Geosoft UNCOMPRESSED grid (*.grd);;'
                   'Geosoft (*.gxf);;'
                   'GeoTiff (*.tif *.tiff);;'
                   'PCI Geomatics Database File (*.pix);;'
                   'SAGA binary grid (*.sdat);;'
                   'Surfer grid (*.grd);;'
                   )

            self.ifile, self.filt = QtWidgets.QFileDialog.getOpenFileName(
                self.parent, 'Open File', '.', ext)
            if self.ifile == '':
                return False

        os.chdir(os.path.dirname(self.ifile))

        if self.filt == 'GeoPak grid (*.grd)':
            dat = get_geopak(self.ifile)
        elif self.filt == 'Geosoft UNCOMPRESSED grid (*.grd)':
            dat = get_geosoft(self.ifile)
        elif self.filt == 'ASCII with .hdr header (*.asc)':
            dat = get_ascii(self.ifile)
        elif self.filt == 'ESRI ASCII (*.asc)':
            dat = get_ascii(self.ifile)
        elif self.filt == 'ASCII XYZ (*.xyz)':
            nval = 0.0
            nval, ok = QtWidgets.QInputDialog.getDouble(self.parent,
                                                        'Null Value',
                                                        'Enter Null Value',
                                                        nval)
            if not ok:
                nval = 0.0
            dat = get_raster(self.ifile, nval, piter=piter,
                             showprocesslog=self.showprocesslog)
        else:
            dat = get_raster(self.ifile, piter=piter,
                             showprocesslog=self.showprocesslog)

        if dat is None:
            if self.filt == 'Surfer grid (v.6) (*.grd)':
                QtWidgets.QMessageBox.warning(self.parent, 'Error',
                                              'Could not import the surfer 6 '
                                              'grid. Please make sure it not '
                                              'another format, such as '
                                              'Geosoft.',
                                              QtWidgets.QMessageBox.Ok)
            elif self.filt == 'Geosoft UNCOMPRESSED grid (*.grd)':
                QtWidgets.QMessageBox.warning(self.parent, 'Error',
                                              'Could not import the grid. '
                                              'Please make sure it is a '
                                              'Geosoft FLOAT grid, and not a '
                                              'compressed grid. You can '
                                              'export your grid to '
                                              'this format using the Geosoft '
                                              'Viewer.',
                                              QtWidgets.QMessageBox.Ok)
            elif self.filt == 'hdf (*.hdf)':
                QtWidgets.QMessageBox.warning(self.parent, 'Error',
                                              'Could not import the data.'
                                              'Currently only ASTER and MODIS'
                                              'are supported.',
                                              QtWidgets.QMessageBox.Ok)
            else:
                QtWidgets.QMessageBox.warning(self.parent, 'Error',
                                              'Could not import the grid.',
                                              QtWidgets.QMessageBox.Ok)
            return False

        output_type = 'Raster'

        if 'Cluster' in dat[0].dataid:
            dat = clusterprep(dat)
            output_type = 'Cluster'

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


class ImportRGBData():
    """
    Import RGB Image - Interfaces with GDAL routines.

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
        self.ifile = ''
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
        ext = 'GeoTiff (*.tif)'

        if not nodialog:
            self.ifile, _ = QtWidgets.QFileDialog.getOpenFileName(
                self.parent, 'Open File', '.', ext)
            if self.ifile == '':
                return False

        os.chdir(os.path.dirname(self.ifile))

        dat = get_raster(self.ifile, showprocesslog=self.showprocesslog)

        if dat is None:
            QtWidgets.QMessageBox.warning(self.parent, 'Error',
                                          'Could not import the image.',
                                          QtWidgets.QMessageBox.Ok)
            return False

        if len(dat) < 3:
            QtWidgets.QMessageBox.warning(self.parent, 'Error',
                                          'Not RGB Image, less than 3 bands.',
                                          QtWidgets.QMessageBox.Ok)
            return False

        output_type = 'Raster'

        if len(dat) == 4:
            dat2 = np.ma.transpose([dat[0].data.T, dat[1].data.T,
                                    dat[2].data.T, dat[3].data.T])
        else:
            dat2 = np.ma.transpose([dat[0].data.T, dat[1].data.T,
                                    dat[2].data.T])
        dat = [dat[0]]
        dat[0].data = dat2
        dat[0].isrgb = True

        if dat[0].data.dtype == np.uint16:
            iidat = np.iinfo(dat[0].data.dtype)
            dat[0].data = dat[0].data.astype(float)
            dat[0].data = (dat[0].data-iidat.min)/(iidat.max-iidat.min)

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


def clusterprep(dat):
    """
    Prepare Cluster data from raster data.

    Parameters
    ----------
    dat : list
        List of PyGMI datasets.

    Returns
    -------
    dat2 : list
        List of PyGMI datasets.

    """
    dat2 = []
    for i in dat:
        if 'Cluster' in i.dataid and 'Membership' not in i.dataid:
            numclus = int(i.data.max())
            i.metadata['Cluster']['no_clusters'] = numclus
            i.metadata['Cluster']['memdat'] = [[]] * numclus
            for j in dat:
                if 'Membership' in j.dataid and i.dataid in j.dataid:
                    cnt = int(j.dataid.split(':')[0].split()[-1])-1
                    i.metadata['Cluster']['memdat'][cnt] = j.data
            dat2.append(i)

    return dat2


def get_ascii(ifile):
    """
    Import ascii raster dataset.

    Parameters
    ----------
    ifile : str
        filename to import

    Returns
    -------
    dat : PyGMI raster Data
        dataset imported
    """
    isESRI = False

    with open(ifile, 'r') as afile:
        adata = afile.read()

    adata = adata.split()

    if adata[0] == 'ncols':
        isESRI = True

    if isESRI:
        nbands = 1
        ncols = int(adata[1])
        nrows = int(adata[3])
        xdim = float(adata[9])
        ydim = float(adata[9])
        nval = float(adata[11])
        ulxmap = float(adata[5])
        ulymap = float(adata[7])+ydim*nrows
        if 'center' in adata[4].lower():
            ulxmap = ulxmap - xdim/2
        if 'center' in adata[6].lower():
            ulymap = ulymap - ydim/2
        adata = adata[12:]
    else:
        with open(ifile[:-3]+'hdr', 'r') as hfile:
            tmp = hfile.readlines()

        xdim = float(tmp[0].split()[-1])
        ydim = float(tmp[1].split()[-1])
        ncols = int(tmp[2].split()[-1])
        nrows = int(tmp[3].split()[-1])
        nbands = int(tmp[4].split()[-1])
        ulxmap = float(tmp[5].split()[-1])
        ulymap = float(tmp[6].split()[-1])
        nval = -9999.0

    bandid = ifile[:-4].rsplit('/')[-1]

    adata = np.array(adata, dtype=float)
    adata.shape = (nrows, ncols)

    if nbands > 1:
        warnings.warn('PyGMI only supports single band ASCII files. '
                      'Only first band will be exported.')

    dat = [Data()]
    i = 0

    dat[i].data = np.ma.masked_equal(adata, nval)
    if dat[i].data.mask.size == 1:
        dat[i].data.mask = (np.ma.make_mask_none(dat[i].data.shape) +
                            dat[i].data.mask)

    dat[i].dataid = bandid
    dat[i].nullvalue = nval
    dat[i].xdim = xdim
    dat[i].ydim = ydim
    dat[i].filename = ifile

    rows, cols = dat[i].data.shape
    xmin = ulxmap
    ymax = ulymap
    ymin = ymax - rows*ydim
    xmax = xmin + cols*xdim

    dat[i].extent = [xmin, xmax, ymin, ymax]

    return dat


def get_raster(ifile, nval=None, piter=iter, showprocesslog=print,
               iraster=None):
    """
    Get raster dataset.

    This function loads a raster dataset off the disk using the GDAL
    libraries. It returns the data in a PyGMI data object.

    Parameters
    ----------
    ifile : str
        filename to import
    nval : float, optional
        No data/null value. The default is None.
    piter : iterable from misc.ProgressBar or misc.ProgressBarText
        progress bar iterable
    showprocesslog : function, optional
        Routine to show text messages. The default is print.
    iraster : None or tuple
        Incremental raster import, to import a section of a file. The tuple is
        (xoff, yoff, xsize, ysize)

    Returns
    -------
    dat : PyGMI raster Data
        dataset imported
    """
    dat = []
    bname = ifile.split('/')[-1].rpartition('.')[0]
    ifile = ifile[:]
    ext = ifile[-3:]
    custom_wkt = None
    filename = ifile

    # Envi Case
    if ext == 'hdr':
        ifile = ifile[:-4]
        if os.path.exists(ifile+'.dat'):
            ifile = ifile+'.dat'
        elif os.path.exists(ifile+'.raw'):
            ifile = ifile+'.raw'
        elif os.path.exists(ifile+'.img'):
            ifile = ifile+'.img'
        elif not os.path.exists(ifile):
            return None

    if ext == 'ers':
        with open(ifile) as f:
            metadata = f.read()
            if 'STMLO' in metadata:
                clong = metadata.split('STMLO')[1][:2]

                orig = osr.SpatialReference()
                orig.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

                if 'CAPE' in metadata:
                    orig.ImportFromEPSG(4222)
                    orig.SetTM(0., float(clong), 1., 0., 0.)
                    orig.SetProjCS(r'Cape / TM'+clong)
                    custom_wkt = orig.ExportToWkt()
                elif 'WGS84' in metadata:
                    orig.ImportFromEPSG(4148)
                    orig.SetTM(0., float(clong), 1., 0., 0.)
                    orig.SetProjCS(r'Hartebeesthoek94 / TM'+clong)
                    custom_wkt = orig.ExportToWkt()

    dataset = gdal.Open(ifile, gdal.GA_ReadOnly)
    if dataset is None:
        return None

    istruct = dataset.GetMetadata('IMAGE_STRUCTURE')
    driver = dataset.GetDriver().GetDescription()

    dmeta = {}
    if driver == 'ENVI':
        dmeta = dataset.GetMetadata('ENVI')

    if 'INTERLEAVE' in istruct and driver in ['ENVI', 'ERS', 'EHdr']:
        if istruct['INTERLEAVE'] == 'LINE':
            dataset = None
            dat = get_bil(ifile, nval, piter, showprocesslog)
            return dat

    if custom_wkt is None:
        srs = osr.SpatialReference()
        srs.ImportFromWkt(dataset.GetProjection())
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        srs.AutoIdentifyEPSG()
        custom_wkt = srs.ExportToWkt()

    gtr = dataset.GetGeoTransform()
    for i in piter(range(dataset.RasterCount)):
        rtmp = dataset.GetRasterBand(i+1)
        bandid = rtmp.GetDescription()
        if nval is None:
            nval = rtmp.GetNoDataValue()

        dat.append(Data())
        if iraster is None:
            dat[i].data = rtmp.ReadAsArray()
        else:
            xoff, yoff, xsize, ysize = iraster
            dat[i].data = rtmp.ReadAsArray(xoff, yoff, xsize, ysize)

        if dat[i].data.dtype.kind == 'i':
            if nval is None:
                nval = 999999
                showprocesslog('Adjusting null value to '+str(nval))
            nval = int(nval)
        elif dat[i].data.dtype.kind == 'u':
            if nval is None:
                nval = 0
                showprocesslog('Adjusting null value to '+str(nval))
            nval = int(nval)
        else:
            if nval is None:
                nval = 1e+20
            nval = float(nval)
            if nval not in dat[i].data and np.isclose(dat[i].data.min(), nval):
                nval = dat[i].data.min()
                showprocesslog('Adjusting null value to '+str(nval))
            if nval not in dat[i].data and np.isclose(dat[i].data.max(), nval):
                nval = dat[i].data.max()
                showprocesslog('Adjusting null value to '+str(nval))

        if ext == 'ers' and nval == -1.0e+32:
            dat[i].data[np.ma.less_equal(dat[i].data, nval)] = -1.0e+32

# Note that because the data is stored in a masked array, the array ends up
# being double the size that it was on the disk.
        dat[i].data = np.ma.masked_invalid(dat[i].data)
        dat[i].data.mask = (np.ma.getmaskarray(dat[i].data) |
                            (dat[i].data == nval))
        if dat[i].data.mask.size == 1:
            dat[i].data.mask = (np.ma.make_mask_none(dat[i].data.shape)
                                + np.ma.getmaskarray(dat[i].data))

        dat[i].extent_from_gtr(gtr)

        if bandid == '':
            bandid = 'Band '+str(i+1)+' '+bname
        dat[i].dataid = bandid

        for unit in ['micrometers', 'nanometers', 'nT', 'mGal']:
            if unit in bandid or unit in bandid.lower():
                if unit == 'micrometers':
                    dat[i].units = 'μm'
                elif unit == 'nanometers':
                    dat[i].units = 'nm'
                else:
                    dat[i].units = unit

        # if bandid[-1] == ')':
        #     dat[i].units = bandid[bandid.rfind('(')+1:-1]

        dat[i].nullvalue = nval
        dat[i].wkt = custom_wkt
        dat[i].filename = filename

        dest = rtmp.GetMetadata()
        for j in ['Wavelength', 'WAVELENGTH']:
            if j in dest:
                dest[j.lower()] = dest[j]
                del dest[j]

        dat[i].metadata['Raster'] = {**dmeta, **dest}

    dataset = None

    return dat


def get_bil(ifile, nval, piter, showprocesslog=print):
    """
    Get BIL format file.

    Parameters
    ----------
    ifile : str
        filename to import
    nval : float
        No data/null value. The default is None.
    piter : iterable from misc.ProgressBar or misc.ProgressBarText
        progress bar iterable
    showprocesslog : function, optional
        Routine to show text messages. The default is print.

    Returns
    -------
    dat : PyGMI raster Data
        dataset imported

    """
    dat = []
    bname = ifile.split('/')[-1].rpartition('.')[0]
    ifile = ifile[:]
    ext = ifile[-3:]
    custom_wkt = None
    filename = ifile

    if ext == 'hdr':
        ifile = ifile[:-4]
        if os.path.exists(ifile+'.dat'):
            ifile = ifile+'.dat'
        elif os.path.exists(ifile+'.raw'):
            ifile = ifile+'.raw'
        elif os.path.exists(ifile+'.img'):
            ifile = ifile+'.img'
        elif not os.path.exists(ifile):
            return None

    if ext == 'ers':
        with open(ifile) as f:
            metadata = f.read()
            if 'STMLO' in metadata:
                clong = metadata.split('STMLO')[1][:2]

                orig = osr.SpatialReference()
                orig.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

                if 'CAPE' in metadata:
                    orig.ImportFromEPSG(4222)
                    orig.SetTM(0., float(clong), 1., 0., 0.)
                    orig.SetProjCS(r'Cape / TM'+clong)
                    custom_wkt = orig.ExportToWkt()
                elif 'WGS84' in metadata:
                    orig.ImportFromEPSG(4148)
                    orig.SetTM(0., float(clong), 1., 0., 0.)
                    orig.SetProjCS(r'Hartebeesthoek94 / TM'+clong)
                    custom_wkt = orig.ExportToWkt()

    dataset = gdal.Open(ifile, gdal.GA_ReadOnly)
    driver = dataset.GetDriver().GetDescription()

    dmeta = {}
    if driver == 'ENVI':
        dmeta = dataset.GetMetadata('ENVI')

    if custom_wkt is None:
        srs = osr.SpatialReference()
        srs.ImportFromWkt(dataset.GetProjection())
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        srs.AutoIdentifyEPSG()
        custom_wkt = srs.ExportToWkt()

    bands = dataset.RasterCount
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    # envimeta = dataset.GetMetadata('ENVI')
    gtr = dataset.GetGeoTransform()

    band = dataset.GetRasterBand(1)
    cols = band.XSize
    rows = band.YSize
    dtype = band.DataType

    dt2np = {}
    dt2np[0] = None
    dt2np[1] = np.uint8
    dt2np[2] = np.uint16
    dt2np[3] = np.int16
    dt2np[4] = np.uint32
    dt2np[5] = np.int32
    dt2np[6] = np.float32
    dt2np[7] = np.float64

    if dt2np[dtype] is None:
        return False

    count = bands*cols*rows

    offset = 0
    icount = count//10
    datin = []
    dsize = dt2np[dtype]().itemsize
    for i in piter(range(0, 10)):
        tmp = np.fromfile(ifile, dtype=dt2np[dtype], sep='', count=icount,
                          offset=offset)
        offset += icount*dsize
        datin.append(tmp)

    extra = int(count-offset/dsize)
    if extra > 0:
        tmp = np.fromfile(ifile, dtype=dt2np[dtype], sep='', count=extra,
                          offset=offset)
        datin.append(tmp)

    datin = np.concatenate(datin)

    datin.shape = (rows, bands, cols)

    datin = np.swapaxes(datin, 0, 1)

    if nval is None:
        nval = 0

    dat = []
    for i in piter(range(bands)):
        rtmp = dataset.GetRasterBand(i+1)
        bandid = rtmp.GetDescription()
        if nval is None:
            nval = rtmp.GetNoDataValue()
        dat.append(Data())

        dat[-1].data = datin[i]

        if ext == 'ers' and nval == -1.0e+32:
            dat[i].data[dat[-1].data <= nval] = -1.0e+32

        dat[i].data = np.ma.array(dat[i].data)

        dat[i].data = np.ma.masked_invalid(dat[i].data)
        dat[i].data.mask = (np.ma.getmaskarray(dat[i].data) |
                            (dat[i].data == nval))

        # dat[i].data.mask = (dat[i].data == nval)
        # dat[i].data = np.ma.masked_invalid(dat[i].data)

        dat[i].extent_from_gtr(gtr)

        if bandid == '':
            bandid = 'Band '+str(i+1)+' '+bname
        dat[i].dataid = bandid
        if bandid[-1] == ')':
            dat[i].units = bandid[bandid.rfind('(')+1:-1]

        dat[i].nullvalue = nval
        dat[i].wkt = custom_wkt
        dat[i].filename = filename
        dest = rtmp.GetMetadata()
        for j in ['Wavelength', 'WAVELENGTH']:
            if j in dest:
                dest[j.lower()] = dest[j]
                del dest[j]
        dat[i].metadata['Raster'] = {**dmeta, **dest}
    return dat


def get_geopak(hfile):
    """
    Geopak Import.

    Parameters
    ----------
    hfile : str
        filename to import

    Returns
    -------
    dat : PyGMI raster Data
        dataset imported

    Returns
    -------
    dat : PyGMI Data
        PyGMI raster dataset.

    """
    with open(hfile, 'rb') as fin:
        fall = fin.read()

    off = 0
    fnew = []
    while off < len(fall):
        off += 1
        breclen = np.frombuffer(fall, dtype=np.uint8, count=1, offset=off)[0]

        if breclen == 130:
            break

        reclen = breclen

        if breclen == 129:
            reclen = 128

        off += 1

        fnew.append(fall[off:off+reclen])
        off += reclen

    fnew = b''.join(fnew)
    header = np.frombuffer(fnew, dtype=np.float32, count=32, offset=0)

#     Lines in grid      1
#     Points per line    2
#     Grid factor        3
#     Grid base value    4
#     Grid X origin      5
#     Grid Y origin      6
#     Grid rotation      7
#     Grid dummy value   8
#     Map scale          9
#     Cell size (X)     10
#     Cell size (Y)     11
#     Inches/unit       12
#     Grid X offset     13
#     Grid Y offset     14
#     Grid hdr version  15
#
#     Lines in grid     17
#     Points per line   18
#     Grid factor       21
#     Grid base value   22
#     Z maximum         23
#     Z minimum         24
#
#     Grid dummy value  26

    nrows = int(header[0])
    ncols = int(header[1])
    gfactor = header[2]
    gbase = header[3]
    x0 = header[4]
    y0 = header[5]
#    rotation = header[6]
    nval = header[7]
#    mapscale = header[8]
    dx = header[9]
    dy = header[10]
#    inches_per_unit = header[11]
#    xoffset = header[12]
#    yoffset = header[13]
#    hver = header[14]
#    zmax = header[22]
#    zmin = header[23]

    data = np.frombuffer(fnew, dtype=np.int16, count=(nrows*ncols), offset=128)

    data = np.ma.masked_equal(data, nval)
    data = data/gfactor+gbase
    data.shape = (nrows, ncols)
    data = data[::-1]

    dat = []
    dat.append(Data())
    i = 0

    dat[i].data = data
    dat[i].dataid = hfile[:-4]

    dat[i].nullvalue = nval
    dat[i].xdim = dx
    dat[i].ydim = dy

    xmin = x0
    ymax = y0 + dy*nrows
    ymin = y0
    xmax = xmin + ncols*dx

    dat[i].extent = [xmin, xmax, ymin, ymax]
    dat[i].filename = hfile

    return dat


def get_geosoft(hfile):
    """
    Get Geosoft file.

    Parameters
    ----------
    ifile : str
        filename to import

    Returns
    -------
    dat : PyGMI Data
        Dataset imported
    """
    f = open(hfile, mode='rb')

    es = np.fromfile(f, dtype=np.int32, count=1)[0]  # 4
    sf = np.fromfile(f, dtype=np.int32, count=1)[0]  # signf
    ncols = np.fromfile(f, dtype=np.int32, count=1)[0]  # ncol
    nrows = np.fromfile(f, dtype=np.int32, count=1)[0]  # nrow
    kx = np.fromfile(f, dtype=np.int32, count=1)[0]  # 1

    dx = np.fromfile(f, dtype=np.float64, count=1)[0]  # dx
    dy = np.fromfile(f, dtype=np.float64, count=1)[0]  # dy
    x0 = np.fromfile(f, dtype=np.float64, count=1)[0]  # xllcor
    y0 = np.fromfile(f, dtype=np.float64, count=1)[0]  # yllcor
    rot = np.fromfile(f, dtype=np.float64, count=1)[0]  # rot
    zbase = np.fromfile(f, dtype=np.float64, count=1)[0]  # zbase
    zmult = np.fromfile(f, dtype=np.float64, count=1)[0]  # zmult

    label = np.fromfile(f, dtype='a48', count=1)[0]
    mapno = np.fromfile(f, dtype='a16', count=1)[0]

    proj = np.fromfile(f, dtype=np.int32, count=1)[0]
    unitx = np.fromfile(f, dtype=np.int32, count=1)[0]
    unity = np.fromfile(f, dtype=np.int32, count=1)[0]
    unitz = np.fromfile(f, dtype=np.int32, count=1)[0]
    nvpts = np.fromfile(f, dtype=np.int32, count=1)[0]
    izmin = np.fromfile(f, dtype=np.int32, count=1)[0]
    izmax = np.fromfile(f, dtype=np.int32, count=1)[0]
    izmed = np.fromfile(f, dtype=np.int32, count=1)[0]
    izmea = np.fromfile(f, dtype=np.int32, count=1)[0]

    zvar = np.fromfile(f, dtype=np.float64, count=1)[0]

    prcs = np.fromfile(f, dtype=np.int32, count=1)[0]

    temspc = np.fromfile(f, dtype='a324', count=1)[0]

    if es == 2:
        nval = -32767
        data = np.fromfile(f, dtype=np.int16, count=nrows*ncols)

    elif es == 4:
        data = np.fromfile(f, dtype=np.float32, count=nrows*ncols)
        nval = -1.0E+32
    else:
        return None

    data = np.ma.masked_equal(data, nval)

    data = data/zmult + zbase
    data.shape = (nrows, ncols)
    data = data[::-1]

    f.close()

    dat = []
    dat.append(Data())
    i = 0

    dat[i].data = data
    dat[i].dataid = hfile[:-4]
    dat[i].nullvalue = nval
    dat[i].xdim = dx
    dat[i].ydim = dy

    xmin = x0
    ymax = y0 + dy*nrows
    ymin = y0
    xmax = xmin + ncols*dx

    dat[i].extent = [xmin, xmax, ymin, ymax]
    dat[i].filename = hfile

    return dat


class ExportData():
    """
    Export Data.

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
        self.ifile = ''

        if parent is None:
            self.piter = ProgressBarText().iter
        else:
            self.piter = parent.pbar.iter
        self.parent = parent
        self.indata = {}
        self.outdata = {}
        if parent is None:
            self.showprocesslog = print
        else:
            self.showprocesslog = parent.showprocesslog

    def run(self):
        """
        Run.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        self.parent.process_is_active(True)

        if 'Cluster' in self.indata:
            data = self.indata['Cluster']
            newdat = copy.deepcopy(data)
            for i in data:
                if 'memdat' not in i.metadata['Cluster']:
                    continue
                for j, val in enumerate(i.metadata['Cluster']['memdat']):
                    tmp = copy.deepcopy(i)
                    tmp.memdat = None
                    tmp.data = val
                    tmp.dataid = ('Membership of class ' + str(j+1)
                                  + ': '+tmp.dataid)
                    newdat.append(tmp)
            data = newdat

        elif 'Raster' in self.indata:
            data = self.indata['Raster']
        else:
            self.showprocesslog('No raster data')
            self.parent.process_is_active(False)
            return False

        ext = ('GeoTiff (*.tif);;'
               'ENVI (*.hdr);;'
               'ERMapper (*.ers);;'
               'Geosoft (*.gxf);;'
               'ERDAS Imagine (*.img);;'
               'SAGA binary grid (*.sdat);;'
               'Surfer grid (v.6) (*.grd);;'
               'ArcInfo ASCII (*.asc);;'
               'ASCII XYZ (*.xyz);;'
               'ArcGIS BIL (*.bil)')

        filename, filt = QtWidgets.QFileDialog.getSaveFileName(
            self.parent, 'Save File', '.', ext)
        if filename == '':
            self.parent.process_is_active(False)
            return False
        os.chdir(os.path.dirname(filename))

        self.ifile = str(filename)

        self.showprocesslog('Export Data Busy...')

    # Pop up save dialog box
        if filt == 'ArcInfo ASCII (*.asc)':
            self.export_ascii(data)
        if filt == 'ASCII XYZ (*.xyz)':
            self.export_ascii_xyz(data)
        if filt == 'Geosoft (*.gxf)':
            self.export_gxf(data)
        if filt == 'Surfer grid (v.6) (*.grd)':
            self.export_surfer(data)
#            self.export_gdal(data, 'GSBG', piter=self.piter)
        if filt == 'ERDAS Imagine (*.img)':
            export_gdal(self.ifile, data, 'HFA', piter=self.piter)
        if filt == 'ERMapper (*.ers)':
            export_gdal(self.ifile, data, 'ERS', piter=self.piter)
        if filt == 'SAGA binary grid (*.sdat)':
            if len(data) > 1:
                for i, dat in enumerate(data):
                    file_out = self.get_filename(dat, 'sdat')
                    export_gdal(file_out, [dat], 'SAGA', piter=self.piter)
            else:
                export_gdal(self.ifile, data, 'SAGA', piter=self.piter)
        if filt == 'GeoTiff (*.tif)':
            export_gdal(self.ifile, data, 'GTiff', piter=self.piter)
        if filt == 'ENVI (*.hdr)':
            export_gdal(self.ifile, data, 'ENVI', piter=self.piter)
        if filt == 'ArcGIS BIL (*.bil)':
            export_gdal(self.ifile, data, 'EHdr', piter=self.piter)

        self.showprocesslog('Export Data Finished!')
        self.parent.process_is_active(False)
        return True

    def export_gxf(self, data):
        """
        Export GXF data.

        Parameters
        ----------
        data : PyGMI raster Data
            dataset to export

        Returns
        -------
        None.

        """
        if len(data) > 1:
            self.showprocesslog('Band names will be appended to the output '
                                'filenames since you have a multiple band '
                                'image')

        file_out = self.ifile.rpartition('.')[0]+'.gxf'
        for k in data:
            if len(data) > 1:
                file_out = self.get_filename(k, 'gxf')

            fno = open(file_out, 'w')

            xmin = k.extent[0]
            ymin = k.extent[2]

            krows, kcols = k.data.shape

            fno.write('#TITLE\n')
            fno.write('Export Data')
            fno.write('\n#POINTS\n')
            fno.write(str(kcols))
            fno.write('\n#ROWS\n')
            fno.write(str(krows))
            fno.write('\n#PTSEPARATION\n')
            fno.write(str(k.xdim))
            fno.write('\n#RWSEPARATION\n')
            fno.write(str(k.ydim))
            fno.write('\n#XORIGIN\n')
            fno.write(str(xmin))
            fno.write('\n#YORIGIN\n')
            fno.write(str(ymin))
            fno.write('\n#SENSE\n')
            fno.write('1')
            fno.write('\n#DUMMY\n')
            fno.write(str(k.nullvalue))
            fno.write('\n#GRID\n')
            tmp = k.data.filled(k.nullvalue)

            for i in range(k.data.shape[0]-1, -1, -1):
                kkk = 0
# write only 5 numbers in a row
                for j in range(k.data.shape[1]):
                    if kkk == 5:
                        kkk = 0
                    if kkk == 0:
                        fno.write('\n')

                    fno.write(str(tmp[i, j]) + '  ')
                    kkk += 1

            fno.close()

    def export_surfer(self, data):
        """
        Export a surfer binary grid.

        Parameters
        ----------
        data : PyGMI raster Data
            dataset to export

        Returns
        -------
        None.

        """
        if len(data) > 1:
            self.showprocesslog('Band names will be appended to the output '
                                'filenames since you have a multiple band '
                                'image')

        file_out = self.ifile.rpartition('.')[0] + '.grd'
        for k in data:
            if len(data) > 1:
                file_out = self.get_filename(k, 'grd')

            fno = open(file_out, 'wb')

            xmin, xmax, ymin, ymax = k.extent

            krows, kcols = k.data.shape
            bintmp = struct.pack('cccchhdddddd', b'D', b'S', b'B', b'B',
                                 kcols, krows,
                                 xmin, xmax,
                                 ymin, ymax,
                                 np.min(k.data),
                                 np.max(k.data))
            fno.write(bintmp)

            ntmp = 1.701410009187828e+38
            tmp = k.data.astype('f')
            tmp = tmp.filled(ntmp)
            tmp = tmp[::-1]
            fno.write(tmp.tostring())

            fno.close()

    def export_ascii(self, data):
        """
        Export Ascii file.

        Parameters
        ----------
        data : PyGMI raster Data
            dataset to export

        Returns
        -------
        None.

        """
        if len(data) > 1:
            self.showprocesslog('Band names will be appended to the output '
                                'filenames since you have a multiple band '
                                'image')

        file_out = self.ifile.rpartition('.')[0]+'.asc'
        for k in data:
            if len(data) > 1:
                file_out = self.get_filename(k, 'asc')
            fno = open(file_out, 'w')

            extent = k.extent
            xmin = extent[0]
            ymin = extent[2]
            krows, kcols = k.data.shape

            fno.write('ncols \t\t\t' + str(kcols))
            fno.write('\nnrows \t\t\t' + str(krows))
            fno.write('\nxllcorner \t\t\t' + str(xmin))
            fno.write('\nyllcorner \t\t\t' + str(ymin))
            fno.write('\ncellsize \t\t\t' + str(k.xdim))
            fno.write('\nnodata_value \t\t' + str(k.nullvalue))

            tmp = k.data.filled(k.nullvalue)
            krows, kcols = k.data.shape

            for j in range(krows):
                fno.write('\n')
                for i in range(kcols):
                    fno.write(str(tmp[j, i]) + ' ')

            fno.close()

    def export_ascii_xyz(self, data):
        """
        Export and xyz file.

        Parameters
        ----------
        data : PyGMI raster Data
            dataset to export

        Returns
        -------
        None.

        """
        if len(data) > 1:
            self.showprocesslog('Band names will be appended to the output '
                                'filenames since you have a multiple band '
                                'image')

        file_out = self.ifile.rpartition('.')[0]+'.xyz'
        for k in data:
            if len(data) > 1:
                file_out = self.get_filename(k, 'xyz')
            fno = open(file_out, 'w')

            tmp = k.data.filled(k.nullvalue)

            xmin = k.extent[0]
            ymax = k.extent[-1]
            krows, kcols = k.data.shape

            for j in range(krows):
                for i in range(kcols):
                    fno.write(str(xmin+i*k.xdim) + ' ' +
                              str(ymax-j*k.ydim) + ' ' +
                              str(tmp[j, i]) + '\n')
            fno.close()

    def get_filename(self, data, ext):
        """
        Get a valid filename in the case of multi band image.

        Parameters
        ----------
        data : PyGMI raster Data
            dataset to get filename from
        ext : str
            filename extension to use

        Returns
        -------
        file_out : str
            Output filename.

        """
        file_band = data.dataid.split('_')[0].strip('"')
        file_band = file_band.replace('/', '')
        file_band = file_band.replace(':', '')
        file_out = self.ifile.rpartition('.')[0]+'_'+file_band+'.'+ext

        return file_out


def export_gdal(ofile, dat, drv, envimeta='', piter=iter):
    """
    Export to GDAL format.

    Parameters
    ----------
    ofile : str
        Output file name.
    dat : PyGMI raster Data
        dataset to export
    drv : str
        name of the GDAL driver to use
    envimeta : str, optional
        ENVI metadata. The default is ''.
    piter : ProgressBar.iter/ProgressBarText.iter, optional
        Progressbar iterable from misc. The default is None.

    Returns
    -------
    None.

    """
    if isinstance(dat, dict):
        dat2 = []
        for i in dat:
            dat2.append(dat[i])
    else:
        dat2 = dat

    data = merge(dat2, piter)

    driver = gdal.GetDriverByName(drv)
    dtype = data[0].data.dtype

    if dtype == np.uint8:
        fmt = gdal.GDT_Byte
    elif dtype == np.uint16:
        fmt = gdal.GDT_UInt16
    elif dtype == np.int32:
        fmt = gdal.GDT_Int32
    elif dtype == np.float64:
        fmt = gdal.GDT_Float64
    else:
        fmt = gdal.GDT_Float32

    tmp = ofile.rpartition('.')

    if drv == 'GTiff':
        tmpfile = tmp[0] + '.tif'
    elif drv == 'EHdr':
        fmt = gdal.GDT_Float32
        dtype = np.float32
        tmpfile = tmp[0] + '.bil'
    elif drv == 'GSBG':
        tmpfile = tmp[0]+'.grd'
        fmt = gdal.GDT_Float32
        dtype = np.float32
    elif drv == 'SAGA':
        tmpfile = tmp[0]+'.sdat'
        data[0].nullvalue = -99999.0
    elif drv == 'HFA':
        tmpfile = tmp[0]+'.img'
    elif drv == 'ENVI':
        tmpfile = tmp[0]+'.dat'
    elif drv == 'ERS':  # ER Mapper
        tmpfile = tmp[0]
    else:
        tmpfile = ofile

    drows, dcols = data[0].data.shape
    if drv == 'GTiff':
        out = driver.Create(tmpfile, int(dcols), int(drows),
                            len(data), fmt, options=['COMPRESS=NONE',
                                                     'INTERLEAVE=BAND',
                                                     'TFW=YES',
                                                     'PROFILE=GeoTIFF',
                                                     'ESRI_XML_PAM=True'])
    elif drv == 'ERS' and 'Cape / TM' in data[0].wkt:
        tmp = data[0].wkt.split('TM')[1][:2]
        out = driver.Create(tmpfile, int(dcols), int(drows),
                            len(data), fmt,
                            options=['PROJ=STMLO'+tmp, 'DATUM=CAPE',
                                     'UNITS=METERS'])
    elif drv == 'ERS' and 'Hartebeesthoek94 / TM' in data[0].wkt:
        tmp = data[0].wkt.split('TM')[1][:2]
        out = driver.Create(tmpfile, int(dcols), int(drows),
                            len(data), fmt,
                            options=['PROJ=STMLO'+tmp, 'DATUM=WGS84',
                                     'UNITS=METERS'])
    else:
        out = driver.Create(tmpfile, int(dcols), int(drows),
                            len(data), fmt)

    out.SetGeoTransform(data[0].get_gtr())
    out.SetProjection(data[0].wkt)

    numbands = len(data)
    for i in piter(range(numbands)):
        datai = data[i]
        rtmp = out.GetRasterBand(i+1)
        rtmp.SetDescription(datai.dataid)
        rtmp.SetMetadataItem('BandName', datai.dataid)

        dtmp = np.ma.array(datai.data).astype(dtype)
        dtmp.set_fill_value(datai.nullvalue)
        dtmp = dtmp.filled()

        # This makes sure the nullvalue is of types GDAL accepts
        if dtype == np.uint8:
            datai.nullvalue = int(datai.nullvalue)
        elif dtype == np.float32 or dtype == np.float64:
            datai.nullvalue = float(datai.nullvalue)

        rtmp.SetNoDataValue(datai.nullvalue)
        rtmp.WriteArray(dtmp)
        rtmp.GetStatistics(False, True)

        if 'Raster' in datai.metadata:
            if 'wavelength' in datai.metadata['Raster']:
                rtmp.SetMetadataItem('wavelength',
                                     str(datai.metadata['Raster']['wavelength']))

            if 'reflectance_scale_factor' in datai.metadata['Raster']:
                rtmp.SetMetadataItem('reflectance_scale_factor',
                                     str(datai.metadata['Raster']['reflectance_scale_factor']))

        if 'WavelengthMin' in datai.metadata:
            rtmp.SetMetadataItem('WavelengthMin',
                                 str(datai.metadata['WavelengthMin']))
            rtmp.SetMetadataItem('WavelengthMax',
                                 str(datai.metadata['WavelengthMax']))

    out = None  # Close File
    if drv == 'ENVI':
        with open(tmpfile[:-4]+'.hdr', 'a') as myfile:
            myfile.write('data ignore value = ' + str(data[0].nullvalue)+'\n')
            myfile.write(envimeta)


def _filespeedtest():
    """Test."""
    from pygmi.misc import ProgressBarText

    print('Starting')
    pbar = ProgressBarText()
    ifile = r'E:\WorkData\HyperspectralScanner\Raw Data\LWIR(OWL)\bv1_17_118m16_125m79_2020-06-30_12-43-14\capture\BV1_17_118m16_125m79_2020-06-30_12-43-14.raw'
    ifile = r'E:\WorkData\Richtersveld\Reprocessed\RSarea_Hyper.dat'
    ifile = r'C:\Work\WorkData\Hyperspectral\056_0818-1125_ref_rect.dat'
    # ifile = r'C:\Work\WorkData\Hyperspectral\056_0818-1125_ref_rect_BSQ.dat'

    # xoff = 0
    # yoff = 0
    # xsize = None
    # ysize = 1000
    # iraster = (xoff, yoff, xsize, ysize)
    iraster = None

    dataset = get_raster(ifile, piter=pbar.iter, iraster=iraster)


if __name__ == "__main__":
    _filespeedtest()
