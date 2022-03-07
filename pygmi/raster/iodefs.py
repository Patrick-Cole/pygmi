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
from PyQt5 import QtWidgets, QtCore
import numpy as np
import rasterio
from rasterio.plot import plotting_extent
from rasterio.windows import Window
from rasterio.crs import CRS

from pygmi.raster.datatypes import Data
from pygmi.raster.dataprep import lstack
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
    Import Data - Interfaces with rasterio routines.

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
            self.piter = ProgressBarText().iter
        else:
            self.showprocesslog = parent.showprocesslog
            self.piter = parent.pbar.iter

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
        if not nodialog:
            ext = ('Common formats (*.ers *.hdr *.tif *.tiff *.sdat *.img '
                   '*.pix *.bil);;'
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
                   'GMT netCDF grid (*.grd);;'
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
            dat = get_raster(self.ifile, nval, piter=self.piter,
                             showprocesslog=self.showprocesslog)
        else:
            dat = get_raster(self.ifile, piter=self.piter,
                             showprocesslog=self.showprocesslog)

        if dat is None:
            if self.filt == 'Geosoft UNCOMPRESSED grid (*.grd)':
                QtWidgets.QMessageBox.warning(self.parent, 'Error',
                                              'Could not import the grid. '
                                              'Please make sure it is a '
                                              'Geosoft FLOAT grid, and not a '
                                              'compressed grid. You can '
                                              'export your grid to '
                                              'this format using the Geosoft '
                                              'Viewer.',
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

        if dat[0].crs is None:
            self.showprocesslog('Warning: Your data has no projection. '
                                'Please add a projection in the Display/Edit '
                                'Metadata interface.')

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
    Import RGB Image - Interfaces with rasterio routines.

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
            self.piter = ProgressBarText().iter
        else:
            self.showprocesslog = parent.showprocesslog
            self.piter = parent.pbar.iter

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

        dat = get_raster(self.ifile, piter=self.piter,
                         showprocesslog=self.showprocesslog)

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

    with open(ifile, 'r', encoding='utf-8') as afile:
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
        with open(ifile[:-3]+'hdr', 'r', encoding='utf-8') as hfile:
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
    dat[i].nodata = nval
    dat[i].filename = ifile

    xmin = ulxmap
    ymax = ulymap

    dat[i].set_transform(xdim, xmin, ydim, ymax)

    dat[i].crs = CRS.from_string('LOCAL_CS["Arbitrary",UNIT["metre",1,'
                                 'AUTHORITY["EPSG","9001"]],'
                                 'AXIS["Easting",EAST],'
                                 'AXIS["Northing",NORTH]]')

    return dat


def get_raster(ifile, nval=None, piter=None, showprocesslog=print,
               iraster=None, driver=None):
    """
    Get raster dataset.

    This function loads a raster dataset off the disk using the rasterio
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
    if piter is None:
        piter = ProgressBarText().iter

    dat = []
    bname = os.path.basename(ifile).rpartition('.')[0]
    ext = ifile[-3:]
    custom_wkt = ''
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
        with open(ifile, encoding='utf-8') as f:
            metadata = f.read()
            if 'STMLO' in metadata:
                clong = metadata.split('STMLO')[1][:2]

                if 'CAPE' in metadata:
                    custom_wkt = ('PROJCS["Cape / TM'+clong+'",'
                                  'GEOGCS["Cape",'
                                  'DATUM["Cape",'
                                  'SPHEROID["Clarke 1880 (Arc)",'
                                  '6378249.145,293.4663077,'
                                  'AUTHORITY["EPSG","7013"]],'
                                  'AUTHORITY["EPSG","6222"]],'
                                  'PRIMEM["Greenwich",0,'
                                  'AUTHORITY["EPSG","8901"]],'
                                  'UNIT["degree",0.0174532925199433,'
                                  'AUTHORITY["EPSG","9122"]],'
                                  'AUTHORITY["EPSG","4222"]],'
                                  'PROJECTION["Transverse_Mercator"],'
                                  'PARAMETER["latitude_of_origin",0],'
                                  'PARAMETER["central_meridian",'+clong+'],'
                                  'PARAMETER["scale_factor",1],'
                                  'PARAMETER["false_easting",0],'
                                  'PARAMETER["false_northing",0],'
                                  'UNIT["metre",1,AUTHORITY["EPSG","9001"]],'
                                  'AXIS["Easting",EAST],'
                                  'AXIS["Northing",NORTH]]')

                elif 'WGS84' in metadata:
                    custom_wkt = ('PROJCS["Hartebeesthoek94 / TM'+clong+'",'
                                  'GEOGCS["Hartebeesthoek94",'
                                  'DATUM["Hartebeesthoek94",'
                                  'SPHEROID["WGS 84",6378137,298.257223563,'
                                  'AUTHORITY["EPSG","7030"]],'
                                  'AUTHORITY["EPSG","6148"]],'
                                  'PRIMEM["Greenwich",0,'
                                  'AUTHORITY["EPSG","8901"]],'
                                  'UNIT["degree",0.0174532925199433,'
                                  'AUTHORITY["EPSG","9122"]],'
                                  'AUTHORITY["EPSG","4148"]],'
                                  'PROJECTION["Transverse_Mercator"],'
                                  'PARAMETER["latitude_of_origin",0],'
                                  'PARAMETER["central_meridian",'+clong+'],'
                                  'PARAMETER["scale_factor",1],'
                                  'PARAMETER["false_easting",0],'
                                  'PARAMETER["false_northing",0],'
                                  'UNIT["metre",1,AUTHORITY["EPSG","9001"]],'
                                  'AXIS["Easting",EAST],'
                                  'AXIS["Northing",NORTH]]')

    dmeta = {}
    with rasterio.open(ifile, driver=driver) as dataset:
        if dataset is None:
            return None

        # allns = dataset.tag_namespaces()

        gmeta = dataset.tags()
        istruct = dataset.tags(ns='IMAGE_STRUCTURE')
        driver = dataset.driver

        if driver == 'ENVI':
            dmeta = dataset.tags(ns='ENVI')

    if custom_wkt == '' and dataset.crs is not None:
        custom_wkt = dataset.crs.to_wkt()

    cols = dataset.width
    rows = dataset.height
    bands = dataset.count
    if nval is None:
        nval = dataset.nodata
    dtype = rasterio.band(dataset, 1).dtype
    if custom_wkt != '':
        crs = CRS.from_string(custom_wkt)
    else:
        showprocesslog('Warning: Your data does not have a projection. '
                       'Assigning local coordinate system.')
        crs = CRS.from_string('LOCAL_CS["Arbitrary",UNIT["metre",1,'
                              'AUTHORITY["EPSG","9001"]],'
                              'AXIS["Easting",EAST],'
                              'AXIS["Northing",NORTH]]')

    isbil = False
    if 'INTERLEAVE' in istruct and driver in ['ENVI', 'ERS', 'EHdr']:
        if istruct['INTERLEAVE'] == 'LINE' and iraster is None:
            isbil = True
            datin = get_bil(ifile, bands, cols, rows, dtype, piter)

    with rasterio.open(ifile) as dataset:
        for i in piter(range(dataset.count)):
            index = dataset.indexes[i]
            bandid = dataset.descriptions[i]

            if bandid == '' or bandid is None:
                bandid = 'Band '+str(index)+' '+bname

            unit = dataset.units[i]
            if unit is None:
                unit = ''
            if unit.lower() == 'micrometers':
                dat[i].units = 'μm'
            elif unit.lower() == 'nanometers':
                dat[i].units = 'nm'

            if nval is None:
                nval = dataset.nodata

            dat.append(Data())
            if isbil is True:
                dat[i].data = datin[i]
            elif iraster is None:
                dat[i].data = dataset.read(index)
            else:
                xoff, yoff, xsize, ysize = iraster
                dat[i].data = dataset.read(1, window=Window(xoff, yoff,
                                                            xsize, ysize))

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
                if nval not in dat[i].data and np.isclose(dat[i].data.min(),
                                                          nval):
                    nval = dat[i].data.min()
                    showprocesslog('Adjusting null value to '+str(nval))
                if nval not in dat[i].data and np.isclose(dat[i].data.max(),
                                                          nval):
                    nval = dat[i].data.max()
                    showprocesslog('Adjusting null value to '+str(nval))

            if ext == 'ers' and nval == -1.0e+32:
                dat[i].data[dat[i].data <= nval] = -1.0e+32

    # Note that because the data is stored in a masked array, the array ends up
    # being double the size that it was on the disk.
            dat[i].data = np.ma.masked_invalid(dat[i].data)
            dat[i].data.mask = (np.ma.getmaskarray(dat[i].data) |
                                (dat[i].data == nval))

            dat[i].extent = plotting_extent(dataset)
            dat[i].bounds = dataset.bounds
            dat[i].dataid = bandid
            dat[i].nodata = nval
            # dat[i].wkt = custom_wkt
            dat[i].filename = filename
            dat[i].units = unit
            dat[i].transform = dataset.transform

            if driver == 'netCDF' and dataset.crs is None:
                if 'x#actual_range' in gmeta and 'y#actual_range' in gmeta:
                    xrng = gmeta['x#actual_range']
                    xrng = xrng.strip('}{').split(',')
                    xrng = [float(i) for i in xrng]
                    xmin = min(xrng)
                    xdim = (xrng[1]-xrng[0])/cols

                    yrng = gmeta['y#actual_range']
                    yrng = yrng.strip('}{').split(',')
                    yrng = [float(i) for i in yrng]
                    ymin = min(yrng)
                    ydim = (yrng[1]-yrng[0])/rows
                    dat[i].set_transform(xdim, xmin, ydim, ymin)

            dat[i].crs = crs
            dat[i].xdim, dat[i].ydim = dataset.res
            dat[i].meta = dataset.meta

            dest = dataset.tags(index)
            for j in ['Wavelength', 'WAVELENGTH']:
                if j in dest:
                    dest[j.lower()] = dest[j]
                    del dest[j]

            if 'fwhm' in dmeta:
                fwhm = [float(i) for i in dmeta['fwhm'][1:-1].split(',')]
                dest['fwhm'] = fwhm[index-1]

            if '.raw' in ifile:
                dmeta['reflectance_scale_factor'] = 10000.

            if 'reflectance scale factor' in dmeta:
                dmeta['reflectance_scale_factor'] = dmeta['reflectance scale factor']

            dat[i].metadata['Raster'] = {**dmeta, **dest}

    return dat


def get_bil(ifile, bands, cols, rows, dtype, piter):
    """
    Get BIL format file.

    This routine is called from get_raster

    Parameters
    ----------
    ifile : str
        filename to import
    bands : int
        Number of bands.
    cols : int
        Number of columns.
    rows : int
        Number of rows.
    dtype : data type
        Data type.
    piter : iterable from misc.ProgressBar or misc.ProgressBarText
        progress bar iterable

    Returns
    -------
    datin : PyGMI raster Data
        dataset imported

    """
    dtype = np.dtype(dtype)

    count = bands*cols*rows

    offset = 0
    icount = count//10
    datin = []
    dsize = dtype.itemsize
    for _ in piter(range(0, 10)):
        tmp = np.fromfile(ifile, dtype=dtype, sep='', count=icount,
                          offset=offset)
        offset += icount*dsize
        datin.append(tmp)

    extra = int(count-offset/dsize)
    if extra > 0:
        tmp = np.fromfile(ifile, dtype=dtype, sep='', count=extra,
                          offset=offset)
        datin.append(tmp)

    datin = np.concatenate(datin)
    datin.shape = (rows, bands, cols)
    datin = np.swapaxes(datin, 0, 1)

    return datin


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

    dat[i].nodata = nval

    xmin = x0
    ymax = y0 + dy*nrows

    dat[i].set_transform(dx, xmin, dy, ymax)

    dat[i].filename = hfile
    dat[i].crs = CRS.from_string('LOCAL_CS["Arbitrary",UNIT["metre",1,'
                                 'AUTHORITY["EPSG","9001"]],'
                                 'AXIS["Easting",EAST],'
                                 'AXIS["Northing",NORTH]]')

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
    dat[i].nodata = nval

    xmin = x0
    ymax = y0 + dy*nrows

    dat[i].set_transform(dx, xmin, dy, ymax)
    dat[i].filename = hfile

    dat[i].crs = CRS.from_string('LOCAL_CS["Arbitrary",UNIT["metre",1,'
                                 'AUTHORITY["EPSG","9001"]],'
                                 'AXIS["Easting",EAST],'
                                 'AXIS["Northing",NORTH]]')

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
               'GeoTiff compressed using ZSTD (*.tif);;'
               'ENVI (*.hdr);;'
               'ERMapper (*.ers);;'
               'Geosoft (*.gxf);;'
               'ERDAS Imagine (*.img);;'
               'SAGA binary grid (*.sdat);;'
               'Surfer grid (*.grd);;'
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
        if filt == 'Surfer grid (*.grd)':
            self.export_surfer(data)
        if filt == 'ERDAS Imagine (*.img)':
            export_raster(self.ifile, data, 'HFA', piter=self.piter)
        if filt == 'ERMapper (*.ers)':
            export_raster(self.ifile, data, 'ERS', piter=self.piter)
        if filt == 'SAGA binary grid (*.sdat)':
            if len(data) > 1:
                for i, dat in enumerate(data):
                    file_out = self.get_filename(dat, 'sdat')
                    export_raster(file_out, [dat], 'SAGA', piter=self.piter)
            else:
                export_raster(self.ifile, data, 'SAGA', piter=self.piter)
        if filt == 'GeoTiff (*.tif)':
            export_raster(self.ifile, data, 'GTiff', piter=self.piter)
        if filt == 'GeoTiff compressed using ZSTD (*.tif)':
            export_raster(self.ifile, data, 'GTiff', piter=self.piter,
                          compression='ZSTD')
        if filt == 'ENVI (*.hdr)':
            export_raster(self.ifile, data, 'ENVI', piter=self.piter)
        if filt == 'ArcGIS BIL (*.bil)':
            export_raster(self.ifile, data, 'EHdr', piter=self.piter)

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

            fno = open(file_out, 'w', encoding='utf-8')

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
            fno.write(str(k.nodata))
            fno.write('\n#GRID\n')
            tmp = k.data.filled(k.nodata)

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
        for k0 in data:
            k = copy.deepcopy(k0)
            if len(data) > 1:
                file_out = self.get_filename(k, 'grd')

            k.data = k.data.filled(1.701410009187828e+38)
            k.nodata = 1.701410009187828e+38

            export_raster(file_out, [k], 'GS7BG', piter=self.piter)

    def export_ascii(self, data):
        """
        Export ASCII file.

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
            fno = open(file_out, 'w', encoding='utf-8')

            extent = k.extent
            xmin = extent[0]
            ymin = extent[2]
            krows, kcols = k.data.shape

            fno.write('ncols \t\t\t' + str(kcols))
            fno.write('\nnrows \t\t\t' + str(krows))
            fno.write('\nxllcorner \t\t\t' + str(xmin))
            fno.write('\nyllcorner \t\t\t' + str(ymin))
            fno.write('\ncellsize \t\t\t' + str(k.xdim))
            fno.write('\nnodata_value \t\t' + str(k.nodata))

            tmp = k.data.filled(k.nodata)
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
            fno = open(file_out, 'w', encoding='utf-8')

            tmp = k.data.filled(k.nodata)

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
        file_band = data.dataid.strip('"')
        file_band = file_band.replace('/', '')
        file_band = file_band.replace(':', '')

        file_out = self.ifile.rpartition('.')[0]+'_'+file_band+'.'+ext

        return file_out


def export_raster(ofile, dat, drv, envimeta='', piter=None,
                  compression='NONE'):
    """
    Export to rasterio format.

    Parameters
    ----------
    ofile : str
        Output file name.
    dat : PyGMI raster Data
        dataset to export
    drv : str
        name of the rasterio driver to use
    envimeta : str, optional
        ENVI metadata. The default is ''.
    piter : ProgressBar.iter/ProgressBarText.iter, optional
        Progressbar iterable from misc. The default is None.

    Returns
    -------
    None.

    """
    if piter is None:
        piter = ProgressBarText().iter

    if isinstance(dat, dict):
        dat2 = []
        for i in dat:
            dat2.append(dat[i])
    else:
        dat2 = dat

    data = lstack(dat2, piter)

    dtype = data[0].data.dtype
    nodata = dat[0].nodata
    trans = dat[0].transform
    crs = dat[0].crs

    try:
        nodata = dtype.type(nodata)
    except OverflowError:
        print('Invalid nodata for dtype, resetting to 0')
        nodata = 0

    if trans is None:
        trans = rasterio.transform.from_origin(dat[0].extent[0],
                                               dat[0].extent[3],
                                               dat[0].xdim, dat[0].ydim)

    tmp = ofile.rpartition('.')

    if drv == 'GTiff':
        tmpfile = tmp[0] + '.tif'
    elif drv == 'EHdr':
        dtype = np.float32
        tmpfile = tmp[0] + '.bil'
    elif drv == 'GSBG':
        tmpfile = tmp[0]+'.grd'
        dtype = np.float32
    elif drv == 'SAGA':
        tmpfile = tmp[0]+'.sdat'
        data[0].nodata = -99999.0
    elif drv == 'HFA':
        tmpfile = tmp[0]+'.img'
    elif drv == 'ENVI':
        tmpfile = tmp[0]+'.dat'
    elif drv == 'ERS':  # ER Mapper
        tmpfile = tmp[0]
    else:
        tmpfile = ofile

    drows, dcols = data[0].data.shape

    kwargs = {}
    if drv == 'GTiff':
        kwargs = {'COMPRESS': compression,
                  'ZLEVEL': '1',
                  'ZSTD_LEVEL': '1',
                  'BIGTIFF': 'YES',
                  'INTERLEAVE': 'BAND',
                  'TFW': 'YES',
                  'PROFILE': 'GeoTIFF'}
        if dtype == np.float32 or dtype == np.float64:
            kwargs['PREDICTOR'] = '3'

    with rasterio.open(tmpfile, 'w', driver=drv,
                       width=int(dcols), height=int(drows), count=len(data),
                       dtype=dtype, transform=trans, crs=crs,
                       nodata=nodata, **kwargs) as out:
        numbands = len(data)
        wavelength = []
        fwhm = []

        # cov = []
        # for idata in data:
        #     cov.append(idata.data.flatten())
        # cov = np.ma.array(cov)
        # cov = np.ma.cov(cov)

        for i in piter(range(numbands)):
            datai = data[i]
            out.set_band_description(i+1, datai.dataid)
            # rtmp.SetDescription(datai.dataid)
            # rtmp.SetMetadataItem('BandName', datai.dataid)

            dtmp = np.ma.array(datai.data)
            dtmp.set_fill_value(datai.nodata)
            dtmp = dtmp.filled()
            # rtmp.GetStatistics(False, True)

            out.write(dtmp, i+1)

            # icov = str(cov[i])[1:-1].replace(' ', ', ')
            # out.update_tags(i+1, STATISTICS_COVARIANCES=icov)
            out.update_tags(i+1, STATISTICS_EXCLUDEDVALUES='')
            out.update_tags(i+1, STATISTICS_MAXIMUM=datai.data.max())
            out.update_tags(i+1, STATISTICS_MEAN=datai.data.mean())
            # out.update_tags(i+1, STATISTICS_MEDIAN=np.ma.median(datai.data))
            out.update_tags(i+1, STATISTICS_MINIMUM=datai.data.min())
            out.update_tags(i+1, STATISTICS_SKIPFACTORX=1)
            out.update_tags(i+1, STATISTICS_SKIPFACTORY=1)
            out.update_tags(i+1, STATISTICS_STDDEV=datai.data.std())


            if 'Raster' in datai.metadata:
                if 'wavelength' in datai.metadata['Raster']:
                    out.update_tags(i+1, wavelength=str(datai.metadata['Raster']['wavelength']))
                    wavelength.append(datai.metadata['Raster']['wavelength'])

                if 'fwhm' in datai.metadata['Raster']:
                    fwhm.append(datai.metadata['Raster']['fwhm'])

                if 'reflectance_scale_factor' in datai.metadata['Raster']:
                    out.update_tags(i+1, reflectance_scale_factor=str(datai.metadata['Raster']['reflectance_scale_factor']))

            if 'WavelengthMin' in datai.metadata:
                out.update_tags(i+1, WavelengthMin=str(datai.metadata['WavelengthMin']))
                out.update_tags(i+1, WavelengthMax=str(datai.metadata['WavelengthMax']))

    if drv == 'ENVI':
        wout = ''
        if (wavelength and envimeta is not None and
                'wavelength' not in envimeta):
            wout = str(wavelength)
            wout = wout.replace('[', '{')
            wout = wout.replace(']', '}')
            wout = wout.replace("'", '')
            wout = 'wavelength = '+wout+'\n'
        if fwhm:
            fwhm = str(fwhm)
            fwhm = fwhm.replace('[', '{')
            fwhm = fwhm.replace(']', '}')
            fwhm = fwhm.replace("'", '')

            wout += 'fwhm = ' + fwhm+'\n'
        if 'reflectance_scale_factor' in datai.metadata['Raster']:
            wout += 'reflectance scale factor = '+ str(datai.metadata['Raster']['reflectance_scale_factor'])+'\n'

        with open(tmpfile[:-4]+'.hdr', 'a', encoding='utf-8') as myfile:
            myfile.write(wout)
            myfile.write(envimeta)


def _filespeedtest():
    """Test."""
    import matplotlib.pyplot as plt
    from pygmi.misc import getinfo
    print('Starting')
    pbar = ProgressBarText()
    # ifile = r'd:\WorkData\Richtersveld\Reprocessed\RSarea_Hyper.dat'
    # ifile = r'd:\WorkData\Hyperspectral\056_0818-1125_ref_rect.dat'
    # ifile = r'd:\WorkData\Hyperspectral\056_0818-1125_ref_rect_BSQ.dat'
    # ifile = r"d:\Workdata\testdata.hdr"
    # ifile = r"d:\Workdata\raster\rad_3bands.ers"
    # ofile = r"d:\Workdata\hope.tif"
    # xoff = 0
    # yoff = 0
    # xsize = None
    # ysize = 1000
    # iraster = (xoff, yoff, xsize, ysize)

    ifile = r"d:\Workdata\compress\New_max_22-55_iMNF15_ferriciron_UTM33s.tif"
    ifile = r"d:\Downloads\caldefo_o_unwrap_goldstein64_OrbAdj_FlatEarth-defo_raw11_ref20210226_dep20210322.pix"
    # ofile = r"d:\Workdata\compress\New_max_22-55_iMNF15_ferriciron_UTM33s_DEFLATE3ZL1.tif"

    ifile = r"C:\WorkProjects\Script6c_disp\disp_data.tif"


    # ifile = r'd:/Workdata/compress/017_0823-1146_ref_rect_BSQ_291div283_194div291_219div303.tif'
    # ofile = ifile[:-4]+'_DEFLATE3.tiff'


    # ifile = r"d:/Workdata/testdata.hdr"
    # ofile = r'd:/Workdata/testdata.grd'

    ifile = r"D:\Workdata\people\rahul\gravity_final.grd"
    ifile = r"D:\Workdata\people\rahul\grav.grd"

    iraster = None

    getinfo('Start')

    dataset = get_raster(ifile, iraster=iraster)

    # ofile = ifile[:-4]+'_hope.tif'
    # export_raster(ofile, dataset, 'GTiff')

    # k = dataset[0]
    # k.data = k.data.filled(1.701410009187828e+38)

    # export_raster(ofile, [k], 'GS7BG')
    # dataset = get_raster(ofile, iraster=iraster)

    plt.figure(dpi=150)
    plt.imshow(dataset[0].data, extent=dataset[0].extent)
    plt.colorbar()
    plt.show()

    # for i in dataset:
    #     i.data = i.data*10000
    #     i.data = i.data.astype(np.int16)

    # export_raster(ofile, dataset, 'GS7BG', piter=pbar.iter)

    # export_raster(ofile, dataset, 'GTiff', compression='PACKBITS')  # 182s
    # export_raster(ofile, dataset, 'GTiff', compression='LZW')  # 191, 140 with pred=3
    # export_raster(ofile, dataset, 'GTiff', compression='LZMA')  #
    # export_raster(ifile[:-4]+'_DEFLATE3ZL1.tiff', dataset, 'GTiff', compression='DEFLATE')  # 318, 277 PRED 3
    # export_raster(ifile[:-4]+'_ZSTD3ZL1.tiff', dataset, 'GTiff', compression='ZSTD')  # 241, 281 pred=3

    # best is zstd pred 3 zlvl 1
    # then deflade pred 3 zlvl 1


    getinfo('End')

    breakpoint()


if __name__ == "__main__":
    _filespeedtest()
