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
import datetime
import xml.etree.ElementTree as ET
from PyQt5 import QtWidgets, QtCore
import numpy as np
from natsort import natsorted
import rasterio
# from rasterio.plot import plotting_extent
from rasterio.windows import Window
from rasterio.crs import CRS

from pygmi.raster.datatypes import Data
from pygmi.raster.dataprep import lstack
from pygmi.misc import ProgressBarText, ContextModule, BasicModule

warnings.filterwarnings("ignore",
                        category=rasterio.errors.NotGeoreferencedWarning)


class BandSelect(ContextModule):
    """A combobox to select data bands."""

    def __init__(self, parent=None):
        super().__init__(parent)
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


class ImportData(BasicModule):
    """Import Data - Interfaces with rasterio routines."""

    def __init__(self, parent=None, ifile='', filt='', listimport=''):
        super().__init__(parent)

        self.ifile = ifile
        self.filt = filt
        self.listimport = listimport

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
        if not nodialog and not self.listimport:
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

            ifilelist, self.filt = QtWidgets.QFileDialog.getOpenFileNames(
                self.parent, 'Open File(s)', '.', ext)
            if not ifilelist:
                return False

            self.ifile = ifilelist.pop(0)

            for ifile in ifilelist:
                self.parent.item_insert('Io', 'Import Raster Data', ImportData,
                                        ifile=ifile, filt=self.filt,
                                        listimport=True)
        else:
            self.listimport = False

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


class ImportRGBData(BasicModule):
    """Import RGB Image - Interfaces with rasterio routines."""

    def __init__(self, parent=None):
        super().__init__(parent)

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
               iraster=None, driver=None, bounds=None, dataid=None,
               tnames=None, metaonly=False):
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
        progress bar iterable, default is None.
    showprocesslog : function, optional
        Routine to show text messages. The default is print.
    iraster : None or tuple
        Incremental raster import, to import a section of a file. The tuple is
        (xoff, yoff, xsize, ysize)
    driver : str
        GDAL raster driver name. The default is None.

    Returns
    -------
    dat : PyGMI raster Data
        dataset imported
    """
    # Exclusions
    if 'AG1' in ifile and 'h5' in ifile.lower():
        return None

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
    rdate = None
    try:
        with rasterio.open(ifile, driver=driver) as dataset:
            if dataset is None:
                return None
            # allns = dataset.tag_namespaces()

            gmeta = dataset.tags()
            istruct = dataset.tags(ns='IMAGE_STRUCTURE')
            driver = dataset.driver
            if 'TIFFTAG_DATETIME' in gmeta:
                dtimestr = gmeta['TIFFTAG_DATETIME']
                rdate = datetime.datetime.strptime(dtimestr,
                                                   '%Y:%m:%d %H:%M:%S')

            if driver == 'ENVI':
                dmeta = dataset.tags(ns='ENVI')

    except rasterio.errors.RasterioIOError:
        return None

    if custom_wkt == '' and dataset.crs is not None:
        custom_wkt = dataset.crs.to_wkt()

    cols = dataset.width
    rows = dataset.height
    bands = dataset.count
    if nval is None:
        nval = dataset.nodata
    dtype = rasterio.band(dataset, 1).dtype

    if bounds is not None:
        xdim, ydim = dataset.res
        xmin, ymin, xmax, ymax = dataset.bounds
        xmin1, ymin1, xmax1, ymax1 = bounds

        if xmin1 >= xmax or xmax1 <= xmin or ymin1 >= ymax or ymax1 <= ymin:
            showprocesslog('Warning: No data in polygon.')
            return None

        xmin2 = max(xmin, xmin1)
        ymin2 = max(ymin, ymin1)
        xmax2 = min(xmax, xmax1)
        ymax2 = min(ymax, ymax1)

        xoff = int((xmin2-xmin)//xdim)
        yoff = int((ymax-ymax2)//ydim)

        xsize = int((xmax2-xmin2)//xdim)
        ysize = int((ymax2-ymin2)//xdim)

        iraster = (xoff, yoff, xsize, ysize)
        newbounds = (xmin+xoff*xdim,
                     ymax-yoff*ydim-ysize*ydim,
                     xmin+xoff*xdim+xsize*xdim,
                     ymax-yoff*ydim)
    elif iraster is not None:
        xdim, ydim = dataset.res
        xmin, ymin, xmax, ymax = dataset.bounds
        xoff, yoff, xsize, ysize = iraster
        newbounds = (xmin+xoff*xdim,
                     ymax-yoff*ydim-ysize*ydim,
                     xmin+xoff*xdim+xsize*xdim,
                     ymax-yoff*ydim)
    else:
        newbounds = None

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
    if ('INTERLEAVE' in istruct and driver in ['ENVI', 'ERS', 'EHdr'] and
            dataid is None and metaonly is False):
        if istruct['INTERLEAVE'] == 'LINE':
            isbil = True
            datin = get_bil(ifile, bands, cols, rows, dtype, piter, iraster)

    with rasterio.open(ifile) as dataset:
        for i in piter(range(dataset.count)):
            index = dataset.indexes[i]
            bandid = dataset.descriptions[i]

            if bandid == '' or bandid is None:
                bandid = 'Band '+str(index)+' '+bname

            if dataid is not None and bandid != dataid:
                continue

            if tnames is not None and bandid not in tnames:
                continue

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
            if isbil is True and metaonly is False:
                dat[-1].data = datin[i]
            elif iraster is None and metaonly is False:
                dat[-1].data = dataset.read(index)
            elif metaonly is False:
                xoff, yoff, xsize, ysize = iraster
                dat[-1].data = dataset.read(index, window=Window(xoff, yoff,
                                                                 xsize, ysize))
            # print(dataset.meta['dtype'])
            if 'uint' in dataset.meta['dtype']:
                if nval is None or np.isnan(nval):
                    nval = 0
                    # showprocesslog(f'Adjusting null value to {nval}')
                nval = int(nval)

            elif 'int' in dataset.meta['dtype']:
                if nval is None or np.isnan(nval):
                    nval = 999999
                    # showprocesslog(f'Adjusting null value to {nval}')
                nval = int(nval)
            else:
                if nval is None or np.isnan(nval):
                    nval = 1e+20
                nval = float(nval)
                if nval not in dat[-1].data and np.isclose(dat[-1].data.min(),
                                                           nval):
                    nval = dat[-1].data.min()
                if nval not in dat[-1].data and np.isclose(dat[-1].data.max(),
                                                           nval):
                    nval = dat[-1].data.max()
                # showprocesslog(f'Adjusting null value to {nval}')

            if ext == 'ers' and nval == -1.0e+32 and metaonly is False:
                dat[-1].data[dat[-1].data <= nval] = -1.0e+32

    # Note that because the data is stored in a masked array, the array ends up
    # being double the size that it was on the disk.

            if metaonly is False:
                dat[-1].data = np.ma.masked_invalid(dat[-1].data)
                dat[-1].data = dat[-1].data.filled(nval)
                dat[-1].data = np.ma.masked_equal(dat[-1].data, nval)
                dat[-1].data.set_fill_value(nval)

            # dat[-1].data.mask = (np.ma.getmaskarray(dat[-1].data) |
            #                      (dat[-1].data == nval))
            if metaonly is True:
                rows = dataset.height
                cols = dataset.width
            else:
                rows = None
                cols = None

            if newbounds is not None:
                xmin, _, _, ymax = newbounds
                xdim, ydim = dataset.res
                dat[-1].set_transform(xdim, xmin, ydim, ymax, iraster=iraster,
                                      rows=rows, cols=cols)
            else:
                dat[-1].set_transform(transform=dataset.transform,
                                      rows=rows, cols=cols)

            dat[-1].dataid = bandid
            dat[-1].nodata = nval
            dat[-1].filename = filename
            dat[-1].units = unit
            dat[-1].datetime = rdate

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
                    dat[-1].set_transform(xdim, xmin, ydim, ymin)

            dat[-1].crs = crs
            # dat[i].xdim, dat[i].ydim = dataset.res
            dat[-1].meta = dataset.meta

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

            dat[-1].metadata['Raster'].update(dmeta)
            dat[-1].metadata['Raster'].update(dest)

    return dat


def get_bil(ifile, bands, cols, rows, dtype, piter, iraster=None):
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
    if iraster is not None:
        xoff, yoff, xsize, ysize = iraster
    else:
        xoff = 0
        yoff = 0
        ysize = rows
        xsize = cols

    dtype = np.dtype(dtype)
    dsize = dtype.itemsize

    count = bands*cols*ysize
    offset = yoff*dsize

    icount = count//10
    datin = []
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
    datin.shape = (ysize, bands, cols)
    datin = np.swapaxes(datin, 0, 1)

    if iraster is not None:
        datin = datin[:, :, xoff:xoff+xsize]

    return datin


def get_bil_old(ifile, bands, cols, rows, dtype, piter, iraster=None):
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

    if iraster is not None:
        xoff, yoff, xsize, ysize = iraster
        datin = datin[:, yoff:yoff+ysize, xoff:xoff+xsize]

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
    # rotation = header[6]
    nval = header[7]
    # mapscale = header[8]
    dx = header[9]
    dy = header[10]
    # inches_per_unit = header[11]
    # xoffset = header[12]
    # yoffset = header[13]
    # hver = header[14]
    # zmax = header[22]
    # zmin = header[23]

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
    with open(hfile, mode='rb') as f:

        es = np.fromfile(f, dtype=np.int32, count=1)[0]  # 4
        sf = np.fromfile(f, dtype=np.int32, count=1)[0]  # signf
        # ne - number of elements per vector or ncols
        ncols = np.fromfile(f, dtype=np.int32, count=1)[0]  # ncol/ne
        # nv - number of vectors or nrows
        nrows = np.fromfile(f, dtype=np.int32, count=1)[0]  # nrow/nv
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

        # elif es > 1024:
        #     esb = es-1024
        #     sig = np.fromfile(f, dtype=np.int32, count=1)[0]
        #     comp_type = np.fromfile(f, dtype=np.int32, count=1)[0]
        #     nb = np.fromfile(f, dtype=np.int32, count=1)[0]
        #     vpb = np.fromfile(f, dtype=np.int32, count=1)[0]

        #     ob = np.fromfile(f, dtype=np.int64, count=nb)
        #     cbs = np.fromfile(f, dtype=np.int32, count=nb)

        #     for i in range(nb):
        #         # breakpoint()
        #         blk = f.read(cbs[i])
        #         # breakpoint()
        #         blk2 = lzrw1.decompress_chunk(blk)

        #         breakpoint()

        else:
            return None

        data = np.ma.masked_equal(data, nval)

        data = data/zmult + zbase
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


class ExportData(BasicModule):
    """
    Export Data.

    Attributes
    ----------
    ofile : str
        output file name.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.ofile = ''

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
               'GeoTiff compressed using DEFLATE (*.tif);;'
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

        self.ofile, filt = QtWidgets.QFileDialog.getSaveFileName(
            self.parent, 'Save File', '.', ext)
        if self.ofile == '':
            self.parent.process_is_active(False)
            return False
        os.chdir(os.path.dirname(self.ofile))

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
            export_raster(self.ofile, data, 'HFA', piter=self.piter)
        if filt == 'ERMapper (*.ers)':
            export_raster(self.ofile, data, 'ERS', piter=self.piter)
        if filt == 'SAGA binary grid (*.sdat)':
            if len(data) > 1:
                for i, dat in enumerate(data):
                    file_out = self.get_filename(dat, 'sdat')
                    export_raster(file_out, [dat], 'SAGA', piter=self.piter)
            else:
                export_raster(self.ofile, data, 'SAGA', piter=self.piter)
        if 'GeoTiff' in filt:
            if 'ZSTD' in filt:
                compression = 'ZSTD'
            elif 'DEFLATE' in filt:
                compression = 'DEFLATE'
            else:
                compression = 'NONE'
            export_raster(self.ofile, data, 'GTiff', piter=self.piter,
                          compression=compression)
        if filt == 'ENVI (*.hdr)':
            export_raster(self.ofile, data, 'ENVI', piter=self.piter)
        if filt == 'ArcGIS BIL (*.bil)':
            export_raster(self.ofile, data, 'EHdr', piter=self.piter)

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

        file_out = self.ofile.rpartition('.')[0]+'.gxf'
        for k in data:
            if len(data) > 1:
                file_out = self.get_filename(k, 'gxf')

            with open(file_out, 'w', encoding='utf-8') as fno:
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

    def export_surfer(self, data):
        """
        Routine to export a surfer binary grid.

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

        file_out = self.ofile.rpartition('.')[0] + '.grd'
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

        file_out = self.ofile.rpartition('.')[0]+'.asc'
        for k in data:
            if len(data) > 1:
                file_out = self.get_filename(k, 'asc')
            with open(file_out, 'w', encoding='utf-8') as fno:
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

        file_out = self.ofile.rpartition('.')[0]+'.xyz'
        for k in data:
            if len(data) > 1:
                file_out = self.get_filename(k, 'xyz')
            with open(file_out, 'w', encoding='utf-8') as fno:
                tmp = k.data.filled(k.nodata)

                xmin = k.extent[0]
                ymax = k.extent[-1]
                krows, kcols = k.data.shape

                for j in range(krows):
                    for i in range(kcols):
                        fno.write(str(xmin+i*k.xdim) + ' ' +
                                  str(ymax-j*k.ydim) + ' ' +
                                  str(tmp[j, i]) + '\n')

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

        file_out = self.ofile.rpartition('.')[0]+'_'+file_band+'.'+ext

        return file_out


def export_raster(ofile, dat, drv='GTiff', envimeta='', piter=None,
                  compression='NONE', bandsort=True, pprint=print,
                  updatestats=True):
    """
    Export to rasterio format.

    Parameters
    ----------
    ofile : str
        Output file name.
    dat : list or dictionary of PyGMI raster Data
        dataset to export
    drv : str
        name of the rasterio driver to use
    envimeta : str, optional
        ENVI metadata. The default is ''.
    piter : ProgressBar.iter/ProgressBarText.iter, optional
        Progressbar iterable from misc. The default is None.
    compression : str, optional
        Compression for GeoTiff. Can be None or ZSTD. The default is None.
    bandsort : bool, optional
        sort the bands by dataid. The default is True

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

    data = lstack(dat2, piter, nodeepcopy=True)

    # Sort in band order.
    if bandsort is True:
        dataid = [i.dataid for i in data]
        data = [i for _, i in natsorted(zip(dataid, data))]

    dtype = data[0].data.dtype
    nodata = data[0].nodata
    trans = data[0].transform
    crs = data[0].crs

    try:
        nodata = dtype.type(nodata)
    except OverflowError:
        print('Invalid nodata for dtype, resetting to 0')
        nodata = 0

    if trans is None:
        trans = rasterio.transform.from_origin(data[0].extent[0],
                                               data[0].extent[3],
                                               data[0].xdim, data[0].ydim)

    tmp = os.path.splitext(ofile)

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
        nodata = -99999.0
    elif drv == 'HFA':
        tmpfile = tmp[0]+'.img'
    elif drv == 'ENVI':
        tmpfile = tmp[0]+'.dat'
    elif drv == 'ERS':
        tmpfile = tmp[0]
    else:
        tmpfile = ofile

    drows, dcols = data[0].data.shape

    kwargs = {}
    if drv == 'GTiff':
        kwargs = {'COMPRESS': compression,
                  'ZLEVEL': '1',
                  'BIGTIFF': 'YES',
                  'INTERLEAVE': 'BAND',
                  'TFW': 'YES',
                  'PROFILE': 'GeoTIFF'}
        if compression == 'ZSTD':
            kwargs['ZSTD_LEVEL'] = '1'
        if dtype in (np.float32, np.float64):
            kwargs['PREDICTOR'] = '3'

    with rasterio.open(tmpfile, 'w', driver=drv,
                       width=int(dcols), height=int(drows), count=len(data),
                       dtype=dtype, transform=trans, crs=crs,
                       nodata=nodata, **kwargs) as out:
        numbands = len(data)
        wavelength = []
        fwhm = []

        for i in piter(range(numbands)):
            datai = data[i]

            out.set_band_description(i+1, datai.dataid)

            dtmp = np.ma.array(datai.data)
            dtmp.set_fill_value(nodata)
            dtmp = dtmp.filled()

            out.write(dtmp, i+1)

            del dtmp

            # out.update_tags(i+1, STATISTICS_EXCLUDEDVALUES='')
            # out.update_tags(i+1, STATISTICS_MAXIMUM=datai.data.max())
            # out.update_tags(i+1, STATISTICS_MEAN=datai.data.mean())
            # out.update_tags(i+1, STATISTICS_MEDIAN=np.ma.median(datai.data))
            # out.update_tags(i+1, STATISTICS_MINIMUM=datai.data.min())
            # out.update_tags(i+1, STATISTICS_SKIPFACTORX=1)
            # out.update_tags(i+1, STATISTICS_SKIPFACTORY=1)
            # try:
            #     out.update_tags(i+1, STATISTICS_STDDEV=datai.data.std())
            # except MemoryError:
            #     pprint('Unable to calculate std deviation. Not enough memory')

            if 'Raster' in datai.metadata:
                rmeta = datai.metadata['Raster']
                if 'wavelength' in rmeta:
                    out.update_tags(i+1, wavelength=str(rmeta['wavelength']))
                    wavelength.append(rmeta['wavelength'])

                if 'fwhm' in rmeta:
                    fwhm.append(rmeta['fwhm'])

                if 'reflectance_scale_factor' in rmeta:
                    out.update_tags(i+1,
                                    reflectance_scale_factor=str(rmeta['reflectance_scale_factor']))

                if 'WavelengthMin' in rmeta:
                    out.update_tags(i+1,
                                    WavelengthMin=str(rmeta['WavelengthMin']))
                    out.update_tags(i+1,
                                    WavelengthMax=str(rmeta['WavelengthMax']))

    if updatestats is True:
        dcov = calccov(data, pprint)

        xfile = tmpfile+'.aux.xml'
        tree = ET.parse(xfile)
        root = tree.getroot()

        pprint('Calculating statistics...')
        for child in piter(root):
            band = int(child.attrib['band'])-1
            datai = data[band]
            donly = datai.data.compressed()

            # Histogram section
            dhist = np.histogram(donly, 256)
            dmin = str(dhist[1][0])
            dmax = str(dhist[1][-1])
            dhist = str(dhist[0].tolist()).replace(', ', '|')[1:-1]

            hist = ET.SubElement(child, 'Histograms')
            histitem = ET.SubElement(hist, 'HistItem')
            ET.SubElement(histitem, 'HistMin').text = dmin
            ET.SubElement(histitem, 'HistMax').text = dmax
            ET.SubElement(histitem, 'BucketCount').text = '256'
            ET.SubElement(histitem, 'IncludeOutOfRange').text = '1'
            ET.SubElement(histitem, 'Approximate').text = '0'
            ET.SubElement(histitem, 'HistCounts').text = dhist

            # Metadata, statistics
            dcovi = str(dcov[:, band].tolist()).replace(' ', '')[1:-1]
            dmin = str(donly.min())
            dmax = str(donly.max())
            dmean = str(donly.mean())
            dmedian = str(np.median(donly))
            dstd = str(donly.std())

            meta = child.find('Metadata')
            ET.SubElement(meta, 'MDI', key='STATISTICS_COVARIANCES').text = dcovi
            ET.SubElement(meta, 'MDI', key='STATISTICS_EXCLUDEDVALUES')
            ET.SubElement(meta, 'MDI', key='STATISTICS_MAXIMUM').text = dmax
            ET.SubElement(meta, 'MDI', key='STATISTICS_MEAN').text = dmean
            ET.SubElement(meta, 'MDI', key='STATISTICS_MEDIAN').text = dmedian
            ET.SubElement(meta, 'MDI', key='STATISTICS_MINIMUM').text = dmin
            ET.SubElement(meta, 'MDI', key='STATISTICS_SKIPFACTORX').text = '1'
            ET.SubElement(meta, 'MDI', key='STATISTICS_SKIPFACTORY').text = '2'
            ET.SubElement(meta, 'MDI', key='STATISTICS_STDDEV').text = dstd

            # meta[:] = sorted(meta, key=lambda x: x.tag)
            child[:] = sorted(child, key=lambda x: x.tag)

        ET.indent(tree)
        tree.write(xfile, encoding='utf-8')

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
            wout += ('reflectance scale factor = ' +
                     str(datai.metadata['Raster']['reflectance_scale_factor'])
                     + '\n')

        with open(tmpfile[:-4]+'.hdr', 'a', encoding='utf-8') as myfile:
            myfile.write(wout)
            myfile.write(envimeta)


def calccov(data, showprocesslog=print):
    """
    Calculate covariance from PyGMI Data.

    This routine assumes all bands are co-located, with the same size.
    Otherwise, run lstack first.

    Parameters
    ----------
    data : list
        List of PyGMI data.

    Returns
    -------
    dcov : numpy array
        Covariances.

    """
    from pygmi.misc import getinfo

    showprocesslog('Calculating covariances...')

    mask = data[0].data.mask
    for band in data:
        mask = np.logical_or(mask, band.data.mask)

    getinfo(0)

    data2 = []
    for band in data:
        data2.append(band.data.data[~mask])

    data2 = np.array(data2)
    dcov = np.cov(data2)

    del data2

    getinfo('covariance')
    return dcov


def _filespeedtest():
    """Test."""
    from pygmi.misc import getinfo
    print('Starting')

    ifile = r"D:\Ratios\S2A_MSIL2A_20220705T074621_N0400_R135_T35JPM_20220705T122811_ratio.tif"
    # ifile = ifile[:-4]+'_zstd.tif'
    dataset = get_raster(ifile)

    getinfo('Start')

    # export_raster(ifile[:-4]+'_NONE.tif', dataset, 'GTiff')  # 65s
    # export_raster(ifile[:-4]+'_PACKBITS.tif', dataset, 'GTiff', compression='PACKBITS')  # 82s
    # export_raster(ifile[:-4]+'_LZW.tif', dataset, 'GTiff', compression='LZW') # 132
    # export_raster(ifile[:-4]+'_LZWA.tif', dataset, 'GTiff', compression='LZMA')  #>900s
    export_raster(ifile[:-4]+'_DEFLATE.tif', dataset, 'GTiff', compression='DEFLATE')  # 104s, 4,246,330
    # export_raster(ifile[:-4]+'_ZSTD.tif', dataset, 'GTiff', compression='ZSTD')  # 74s

    # best is zstd pred 3 zlvl 1
    # then deflate pred 3 zlvl 1

    getinfo('End')


if __name__ == "__main__":
    _filespeedtest()
