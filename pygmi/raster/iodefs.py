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
import glob
import struct
from PyQt5 import QtWidgets
import numpy as np
from osgeo import gdal, osr
from pygmi.raster.datatypes import Data
from pygmi.clust.datatypes import Clust
from pygmi.raster.dataprep import merge
from pygmi.raster.dataprep import quickgrid


class ImportData(object):
    """
    Import Data - Interfaces with GDAL routines

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
    ext : str
        filename extension
    """
    def __init__(self, parent=None):
        self.ifile = ""
        self.name = "Import Data: "
        self.ext = ""
        self.pbar = None
        self.parent = parent
        self.indata = {}
        self.outdata = {}

    def settings(self):
        """ Settings """
        ext = \
            "Common formats (*.ers *.hdr *.tif *.sdat *.img *.pix *.bil);;" + \
            "hdf (*.hdf);;" + \
            "hdf (*.h5);;" + \
            "ASTER GED (*.bin);;" + \
            "ERMapper (*.ers);;" + \
            "ENVI (*.hdr);;" + \
            "ERDAS Imagine (*.img);;" + \
            "PCI Geomatics Database File (*.pix);;" + \
            "GeoTiff (*.tif);;" + \
            "SAGA binary grid (*.sdat);;" + \
            "Geosoft UNCOMPRESSED grid (*.grd);;" + \
            "Geosoft (*.gxf);;" + \
            "Surfer grid (v.6) (*.grd);;" + \
            "GeoPak grid (*.grd);;" + \
            "ASCII with .hdr header (*.asc);;" + \
            "ASCII XYZ (*.xyz);;" + \
            "Arcinfo Binary Grid (hdr.adf);;" + \
            "ArcGIS BIL (*.bil)"

        filename, filt = QtWidgets.QFileDialog.getOpenFileName(
            self.parent, 'Open File', '.', ext)
        if filename == '':
            return False
        os.chdir(filename.rpartition('/')[0])
        self.ifile = str(filename)
        self.ext = filename[-3:]
        self.ext = self.ext.lower()

        if filt == 'GeoPak grid (*.grd)':
            dat = get_geopak(self.ifile)
        elif filt == 'Geosoft UNCOMPRESSED grid (*.grd)':
            dat = get_geosoft(self.ifile)
        elif filt == 'hdf (*.hdf)':
            dat = get_hdf(self.ifile)
        elif filt == "hdf (*.h5)":
            dat = get_hdf(self.ifile)
        elif filt == 'ASCII with .hdr header (*.asc)':
            dat = get_ascii(self.ifile)
        elif filt == 'ASTER GED (*.bin)':
            dat = get_aster_ged_bin(self.ifile)
        else:
            dat = get_raster(self.ifile)

        if dat is None:
            if filt == 'Surfer grid (v.6) (*.grd)':
                QtWidgets.QMessageBox.warning(self.parent, 'Error',
                                              'Could not import the surfer 6 '
                                              'grid. Please make sure it not '
                                              'another format, such as '
                                              'geosoft.',
                                              QtWidgets.QMessageBox.Ok,
                                              QtWidgets.QMessageBox.Ok)
            elif filt == 'Geosoft UNCOMPRESSED grid (*.grd)':
                QtWidgets.QMessageBox.warning(self.parent, 'Error',
                                              'Could not import the grid. '
                                              'Please make sure it is a '
                                              'Geosoft FLOAT grid, and not a '
                                              'compressed grid. You can '
                                              'export your grid to '
                                              'this format using the Geosoft '
                                              'Viewer.',
                                              QtWidgets.QMessageBox.Ok,
                                              QtWidgets.QMessageBox.Ok)
            elif filt == 'hdf (*.hdf)':
                QtWidgets.QMessageBox.warning(self.parent, 'Error',
                                              'Could not import the data.'
                                              'Currently only ASTER and MODIS'
                                              'are supported.',
                                              QtWidgets.QMessageBox.Ok,
                                              QtWidgets.QMessageBox.Ok)
            else:
                QtWidgets.QMessageBox.warning(self.parent, 'Error',
                                              'Could not import the grid.',
                                              QtWidgets.QMessageBox.Ok,
                                              QtWidgets.QMessageBox.Ok)
            return False

        output_type = 'Raster'

        if 'Cluster' in dat[0].dataid:
            output_type = 'Cluster'

        self.outdata[output_type] = dat
        return True


def get_ascii(ifile):
    """
    Import ascii raster dataset

    Parameters
    ----------
    ifile : str
        filename to import

    Returns
    -------
    dat : PyGMI raster Data
        dataset imported
    """

    afile = open(ifile, 'r')
    adata = afile.read()

    adata = adata.split()
    adata = np.array(adata, dtype=float)

    hfile = open(ifile[:-3]+'hdr', 'r')
    tmp = hfile.readlines()

    xdim = float(tmp[0].split()[-1])
    ydim = float(tmp[1].split()[-1])
    ncols = int(tmp[2].split()[-1])
    nrows = int(tmp[3].split()[-1])
    nbands = int(tmp[4].split()[-1])
    ulxmap = float(tmp[5].split()[-1])
    ulymap = float(tmp[6].split()[-1])
    bandid = ifile[:-4].rsplit('/')[-1]

    adata.shape = (nrows, ncols)

    if nbands > 1:
        print('PyGMI only supports single band ASCII files')

    dat = [Data()]
    i = 0

    dat[i].data = adata

    nval = -9999.0

    dat[i].data = np.ma.masked_equal(dat[i].data, nval)
    if dat[i].data.mask.size == 1:
        dat[i].data.mask = (np.ma.make_mask_none(dat[i].data.shape) +
                            dat[i].data.mask)

    dat[i].nrofbands = nbands
    dat[i].tlx = ulxmap
    dat[i].tly = ulymap
    dat[i].dataid = bandid
    dat[i].nullvalue = nval
    dat[i].rows = nrows
    dat[i].cols = ncols
    dat[i].xdim = xdim
    dat[i].ydim = ydim

    return dat


def get_raster(ifile):
    """
    This function loads a raster dataset off the disk using the GDAL
    libraries. It returns the data in a PyGMI data object.

    Parameters
    ----------
    ifile : str
        filename to import

    Returns
    -------
    dat : PyGMI raster Data
        dataset imported
    """
    dat = []
    bname = ifile.split('/')[-1].rpartition('.')[0]+': '
    ifile = ifile[:]
    ext = ifile[-3:]
    custom_wkt = None

    # Envi Case
    if ext == 'hdr':
        ifile = ifile[:-4]
        tmp = glob.glob(ifile+'.dat')
        if tmp:
            ifile = tmp[0]

    if ext == 'ers':
        with open(ifile) as f:
            metadata = f.read()
            if 'STMLO' in metadata:
                clong = metadata.split('STMLO')[1][:2]

                orig = osr.SpatialReference()
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

    gtr = dataset.GetGeoTransform()

    for i in range(dataset.RasterCount):
        rtmp = dataset.GetRasterBand(i+1)
        bandid = rtmp.GetDescription()
        nval = rtmp.GetNoDataValue()

        if 'Cluster' in bandid:
            dat.append(Clust())
        else:
            dat.append(Data())
        dat[i].data = rtmp.ReadAsArray()
        if dat[i].data.dtype.kind == 'i':
            if nval is None:
                nval = 999999
            nval = int(nval)
        elif dat[i].data.dtype.kind == 'u':
            if nval is None:
                nval = 0
            nval = int(nval)
        else:
            if nval is None:
                nval = 1e+20
            nval = float(nval)
        if ext == 'ers' and nval == -1.0e+32:
            dat[i].data[np.ma.less_equal(dat[i].data, nval)] = -1.0e+32

# Note that because the data is stored in a masked array, the array ends up
# being double the size that it was on the disk.
        dat[i].data = np.ma.masked_invalid(dat[i].data)
        dat[i].data.mask = np.ma.getmaskarray(dat[i].data) | (dat[i].data == nval)
        if dat[i].data.mask.size == 1:
            dat[i].data.mask = (np.ma.make_mask_none(dat[i].data.shape) +
                                np.ma.getmaskarray(dat[i].data))

        dat[i].nrofbands = dataset.RasterCount
        dat[i].tlx = gtr[0]
        dat[i].tly = gtr[3]
        if bandid == '':
            bandid = bname+str(i+1)
        dat[i].dataid = bandid
        if bandid[-1] == ')':
            dat[i].units = bandid[bandid.rfind('(')+1:-1]

        dat[i].nullvalue = nval
        dat[i].rows = dataset.RasterYSize
        dat[i].cols = dataset.RasterXSize
        dat[i].xdim = abs(gtr[1])
        dat[i].ydim = abs(gtr[5])
        dat[i].gtr = gtr

        if custom_wkt is None:
            srs = osr.SpatialReference()
            srs.ImportFromWkt(dataset.GetProjection())
            srs.AutoIdentifyEPSG()
            dat[i].wkt = srs.ExportToWkt()
        else:
            dat[i].wkt = custom_wkt

        if 'Cluster' in bandid:
            dat[i].no_clusters = int(dat[i].data.max()+1)

    return dat


def get_hdf(ifile):
    """
    This function loads a raster dataset off the disk using the GDAL
    libraries. It returns the data in a PyGMI data object.

    Parameters
    ----------
    ifile : str
        filename to import

    Returns
    -------
    dat : PyGMI raster Data
        dataset imported
    """
    ifile = ifile[:]

    dataset = gdal.Open(ifile, gdal.GA_ReadOnly)

    if dataset is None:
        return None

    metadata = dataset.GetMetadata()

    if 'Moderate Resolution Imaging Spectroradiometer' in metadata.values():
        dat = get_modis(ifile)
    elif 'ASTER' in metadata.values():
        dat = get_aster(ifile)
    elif 'ASTER_GDEM_ASTGDEM_Description' in metadata:
        dat = get_aster_ged(ifile)
    else:
        dat = None

    return dat


def get_modis(ifile):
    """
    Gets MODIS data

    Parameters
    ----------
    ifile : str
        filename to import

    Returns
    -------
    dat : PyGMI raster Data
        dataset imported
    """
    dat = []
    ifile = ifile[:]

    dataset = gdal.Open(ifile, gdal.GA_ReadOnly)

    subdata = dataset.GetSubDatasets()

    latentry = [i for i in subdata if 'Latitude' in i[1]]
    subdata.pop(subdata.index(latentry[0]))
    dataset = gdal.Open(latentry[0][0], gdal.GA_ReadOnly)
    rtmp = dataset.GetRasterBand(1)
    lats = rtmp.ReadAsArray()
    latsdim = ((lats.max()-lats.min())/(lats.shape[0]-1))/2

    lonentry = [i for i in subdata if 'Longitude' in i[1]]
    subdata.pop(subdata.index(lonentry[0]))
    dataset = gdal.Open(lonentry[0][0], gdal.GA_ReadOnly)
    rtmp = dataset.GetRasterBand(1)
    lons = rtmp.ReadAsArray()
    lonsdim = ((lons.max()-lons.min())/(lons.shape[1]-1))/2

    lonsdim = latsdim
    tlx = lons.min()-abs(lonsdim/2)
    tly = lats.max()+abs(latsdim/2)
    cols = int((lons.max()-lons.min())/lonsdim)+1
    rows = int((lats.max()-lats.min())/latsdim)+1

    newx2, newy2 = np.mgrid[0:rows, 0:cols]
    newx2 = newx2*lonsdim + tlx
    newy2 = tlx - newy2*latsdim

    tmp = []
    for i in subdata:
        if 'HDF4_EOS:EOS_SWATH' in i[0]:
            tmp.append(i)
    subdata = tmp

    i = -1
    for ifile2, bandid2 in subdata:
        dataset = gdal.Open(ifile2, gdal.GA_ReadOnly)

        gtr = dataset.GetGeoTransform()
        rtmp2 = dataset.ReadAsArray()

        if rtmp2.shape[-1] == min(rtmp2.shape) and rtmp2.ndim == 3:
            rtmp2 = np.transpose(rtmp2, (2, 0, 1))

        nbands = 1
        if rtmp2.ndim == 3:
            nbands = rtmp2.shape[0]

        for i2 in range(nbands):
            rtmp = dataset.GetRasterBand(i2+1)
            bandid = rtmp.GetDescription()
            nval = rtmp.GetNoDataValue()
            i += 1

            dat.append(Data())
            if rtmp2.ndim == 3:
                dat[i].data = rtmp2[i2]
            else:
                dat[i].data = rtmp2

            newx = lons[dat[i].data != nval]
            newy = lats[dat[i].data != nval]
            newz = dat[i].data[dat[i].data != nval]

            if newx.size == 0:
                dat[i].data = np.zeros((rows, cols)) + nval
            else:
                tmp = quickgrid(newx, newy, newz, latsdim)
                mask = np.ma.getmaskarray(tmp)
                gdat = tmp.data
                dat[i].data = np.ma.masked_invalid(gdat[::-1])
                dat[i].data.mask = mask[::-1]

            if dat[i].data.dtype.kind == 'i':
                if nval is None:
                    nval = 999999
                nval = int(nval)
            elif dat[i].data.dtype.kind == 'u':
                if nval is None:
                    nval = 0
                nval = int(nval)
            else:
                if nval is None:
                    nval = 1e+20
                nval = float(nval)

            dat[i].data = np.ma.masked_invalid(dat[i].data)
            dat[i].data.mask = np.ma.getmaskarray(dat[i].data) | (dat[i].data == nval)
            if dat[i].data.mask.size == 1:
                dat[i].data.mask = (np.ma.make_mask_none(dat[i].data.shape) +
                                    np.ma.getmaskarray(dat[i].data))

            dat[i].nrofbands = dataset.RasterCount
            dat[i].tlx = tlx
            dat[i].tly = tly
            dat[i].dataid = bandid2+' '+bandid
            dat[i].nullvalue = nval
            dat[i].rows = dat[i].data.shape[0]
            dat[i].cols = dat[i].data.shape[1]
            dat[i].xdim = abs(lonsdim)
            dat[i].ydim = abs(latsdim)
            dat[i].gtr = gtr

            srs = osr.SpatialReference()
            srs.ImportFromWkt(dataset.GetProjection())
            srs.AutoIdentifyEPSG()

            dat[i].wkt = srs.ExportToWkt()

    return dat


def get_aster(ifile):
    """
    Gets ASTER Data

    Parameters
    ----------
    ifile : str
        filename to import

    Returns
    -------
    dat : PyGMI raster Data
        dataset imported
    """

    dat = []
    ifile = ifile[:]

    dataset = gdal.Open(ifile, gdal.GA_ReadOnly)

    subdata = dataset.GetSubDatasets()

    latentry = [i for i in subdata if 'Latitude' in i[1]]
    subdata.pop(subdata.index(latentry[0]))
    dataset = gdal.Open(latentry[0][0], gdal.GA_ReadOnly)
    rtmp = dataset.GetRasterBand(1)
    lats = rtmp.ReadAsArray()
    latsdim = ((lats.max()-lats.min())/(lats.shape[0]-1))/2

    lonentry = [i for i in subdata if 'Longitude' in i[1]]
    subdata.pop(subdata.index(lonentry[0]))
    dataset = gdal.Open(lonentry[0][0], gdal.GA_ReadOnly)
    rtmp = dataset.GetRasterBand(1)
    lons = rtmp.ReadAsArray()
    lonsdim = ((lons.max()-lons.min())/(lons.shape[1]-1))/2

    lonsdim = latsdim
    tlx = lons.min()-abs(lonsdim/2)
    tly = lats.max()+abs(latsdim/2)
    cols = int((lons.max()-lons.min())/lonsdim)+1
    rows = int((lats.max()-lats.min())/latsdim)+1

    newx2, newy2 = np.mgrid[0:rows, 0:cols]
    newx2 = newx2*lonsdim + tlx
    newy2 = tlx - newy2*latsdim

    subdata = [i for i in subdata if 'ImageData' in i[0]]

    i = -1
    for ifile2, bandid2 in subdata:
        dataset = gdal.Open(ifile2, gdal.GA_ReadOnly)

        rtmp2 = dataset.ReadAsArray()

        tmpds = gdal.AutoCreateWarpedVRT(dataset)
        rtmp2 = tmpds.ReadAsArray()
        gtr = tmpds.GetGeoTransform()
        tlx, lonsdim, _, tly, _, latsdim = gtr

        nval = 0

        i += 1

        dat.append(Data())
        dat[i].data = rtmp2

        if dat[i].data.dtype.kind == 'i':
            if nval is None:
                nval = 999999
            nval = int(nval)
        elif dat[i].data.dtype.kind == 'u':
            if nval is None:
                nval = 0
            nval = int(nval)
        else:
            if nval is None:
                nval = 1e+20
            nval = float(nval)

        dat[i].data = np.ma.masked_invalid(dat[i].data)
        dat[i].data.mask = dat[i].data.mask | (dat[i].data == nval)
        if dat[i].data.mask.size == 1:
            dat[i].data.mask = (np.ma.make_mask_none(dat[i].data.shape) +
                                dat[i].data.mask)

        dat[i].nrofbands = dataset.RasterCount
        dat[i].tlx = tlx
        dat[i].tly = tly
        dat[i].dataid = bandid2
        dat[i].nullvalue = nval
        dat[i].rows = dat[i].data.shape[0]
        dat[i].cols = dat[i].data.shape[1]
        dat[i].xdim = abs(lonsdim)
        dat[i].ydim = abs(latsdim)
        dat[i].gtr = gtr

        srs = osr.SpatialReference()
        srs.ImportFromWkt(dataset.GetProjection())
        srs.AutoIdentifyEPSG()

        dat[i].wkt = srs.ExportToWkt()

    if dat == []:
        dat = None
    return dat


def get_aster_ged(ifile):
    """
    Gets ASTER GED data

    Parameters
    ----------
    ifile : str
        filename to import

    Returns
    -------
    dat : PyGMI raster Data
        dataset imported
    """
    dat = []
    ifile = ifile[:]

    dataset = gdal.Open(ifile, gdal.GA_ReadOnly)

    subdata = dataset.GetSubDatasets()

    latentry = [i for i in subdata if 'Latitude' in i[1]]
    subdata.pop(subdata.index(latentry[0]))
    dataset = gdal.Open(latentry[0][0], gdal.GA_ReadOnly)
    rtmp = dataset.GetRasterBand(1)
    lats = rtmp.ReadAsArray()
    latsdim = (lats.max()-lats.min())/(lats.shape[0]-1)

    lonentry = [i for i in subdata if 'Longitude' in i[1]]
    subdata.pop(subdata.index(lonentry[0]))
    dataset = gdal.Open(lonentry[0][0], gdal.GA_ReadOnly)
    rtmp = dataset.GetRasterBand(1)
    lons = rtmp.ReadAsArray()
    lonsdim = (lons.max()-lons.min())/(lons.shape[0]-1)

    tlx = lons.min()-abs(lonsdim/2)
    tly = lats.max()+abs(latsdim/2)

    i = -1
    for ifile2, bandid2 in subdata:
        dataset = gdal.Open(ifile2, gdal.GA_ReadOnly)
        bandid = bandid2
        units = ''

        if 'ASTER_GDEM' in bandid2:
            bandid = 'ASTER GDEM'
            units = 'meters'
        if 'Land_Water_Map' in bandid2:
            bandid = 'Land_water_map'
        if 'Observations' in bandid2:
            bandid = 'Observations'
            units = 'number per pixel'

        gtr = dataset.GetGeoTransform()
        rtmp2 = dataset.ReadAsArray()

        if rtmp2.shape[-1] == min(rtmp2.shape) and rtmp2.ndim == 3:
            rtmp2 = np.transpose(rtmp2, (2, 0, 1))

        nbands = 1
        if rtmp2.ndim == 3:
            nbands = rtmp2.shape[0]

        for i2 in range(nbands):
            nval = -9999
            i += 1

            dat.append(Data())
            if rtmp2.ndim == 3:
                dat[i].data = rtmp2[i2]
            else:
                dat[i].data = rtmp2

            dat[i].data = np.ma.masked_invalid(dat[i].data)
            dat[i].data.mask = np.ma.getmaskarray(dat[i].data) | (dat[i].data == nval)
            if dat[i].data.mask.size == 1:
                dat[i].data.mask = (np.ma.make_mask_none(dat[i].data.shape) +
                                    np.ma.getmaskarray(dat[i].data))

            dat[i].data = dat[i].data * 1.0
            if 'Emissivity/Mean' in bandid2:
                bandid = 'Emissivity_mean_band_'+str(10+i2)
                dat[i].data = dat[i].data * 0.001
            if 'Emissivity/SDev' in bandid2:
                bandid = 'Emissivity_std_dev_band_'+str(10+i2)
                dat[i].data = dat[i].data * 0.0001
            if 'NDVI/Mean' in bandid2:
                bandid = 'NDVI_mean'
                dat[i].data = dat[i].data * 0.01
            if 'NDVI/SDev' in bandid2:
                bandid = 'NDVI_std_dev'
                dat[i].data = dat[i].data * 0.01
            if 'Temperature/Mean' in bandid2:
                bandid = 'Temperature_mean'
                units = 'Kelvin'
                dat[i].data = dat[i].data * 0.01
            if 'Temperature/SDev' in bandid2:
                bandid = 'Temperature_std_dev'
                units = 'Kelvin'
                dat[i].data = dat[i].data * 0.01

            dat[i].nrofbands = dataset.RasterCount
            dat[i].tlx = tlx
            dat[i].tly = tly
            dat[i].dataid = bandid
            dat[i].nullvalue = nval
            dat[i].rows = dat[i].data.shape[0]
            dat[i].cols = dat[i].data.shape[1]
            dat[i].xdim = abs(lonsdim)
            dat[i].ydim = abs(latsdim)
            dat[i].gtr = gtr
            dat[i].units = units

            srs = osr.SpatialReference()
            srs.ImportFromWkt(dataset.GetProjection())
            srs.AutoIdentifyEPSG()

            dat[i].wkt = srs.ExportToWkt()

    return dat


def get_aster_ged_bin(ifile):
    """
    Get ASTER GED binary format

    Emissivity_Mean_Description: Mean Emissivity for each pixel on grid-box
    using all ASTER data from 2000-2010
    Emissivity_SDev_Description: Emissivity Standard Deviation for each pixel
    on grid-box using all ASTER data from 2000-2010
    Temperature_Mean_Description: Mean Temperature (K) for each pixel on
    grid-box using all ASTER data from 2000-2010
    Temperature_SDev_Description: Temperature Standard Deviation for each pixel
    on grid-box using all ASTER data from 2000-2010
    NDVI_Mean_Description: Mean NDVI for each pixel on grid-box using all ASTER
    data from 2000-2010
    NDVI_SDev_Description: NDVI Standard Deviation for each pixel on grid-box
    using all ASTER data from 2000-2010
    Land_Water_Map_LWmap_Description: Land Water Map using ASTER visible bands
    Observations_NumObs_Description: Number of values used in computing mean
    and standard deviation for each pixel.
    Geolocation_Latitude_Description: Latitude
    Geolocation_Longitude_Description: Longitude
    ASTER_GDEM_ASTGDEM_Description: ASTER GDEM resampled to NAALSED

    Parameters
    ----------
    ifile : str
        filename to import

    Returns
    -------
    dat : PyGMI raster Data
        dataset imported
    """

    dat = []
    nval = -9999
    bandid = {}

    bandid[0] = 'Emissivity_mean_band_10'
    bandid[1] = 'Emissivity_mean_band_11'
    bandid[2] = 'Emissivity_mean_band_12'
    bandid[3] = 'Emissivity_mean_band_13'
    bandid[4] = 'Emissivity_mean_band_14'
    bandid[5] = 'Emissivity_std_dev_band_10'
    bandid[6] = 'Emissivity_std_dev_band_11'
    bandid[7] = 'Emissivity_std_dev_band_12'
    bandid[8] = 'Emissivity_std_dev_band_13'
    bandid[9] = 'Emissivity_std_dev_band_14'
    bandid[10] = 'Temperature_mean'
    bandid[11] = 'Temperature_std_dev'
    bandid[12] = 'NDVI_mean'
    bandid[13] = 'NDVI_std_dev'
    bandid[14] = 'Land_water_map'
    bandid[15] = 'Observations'
    bandid[16] = 'Latitude'
    bandid[17] = 'Longitude'
    bandid[18] = 'ASTER GDEM'

    scale = [0.001, 0.001, 0.001, 0.001, 0.001,
             0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
             0.01, 0.01, 0.01, 0.01,
             1, 1, 0.001, 0.001, 1]

    units = ['', '', '', '', '', '', '', '', '', '', 'Kelvin', 'Kelvin',
             '', '', '', 'Number per pixel', 'degrees', 'degrees', 'meters']

    data = np.fromfile(ifile, dtype=np.int32)
    rows_cols = int((data.size/19)**0.5)
    data.shape = (19, rows_cols, rows_cols)

    lats = data[16]*scale[16]
    lons = data[17]*scale[17]

    latsdim = (lats.max()-lats.min())/(lats.shape[0]-1)
    lonsdim = (lons.max()-lons.min())/(lons.shape[0]-1)

    tlx = lons.min()-abs(lonsdim/2)
    tly = lats.max()+abs(latsdim/2)

    for i in range(19):
        dat.append(Data())

        dat[i].data = data[i]*scale[i]

        dat[i].nrofbands = 1
        dat[i].tlx = tlx
        dat[i].tly = tly
        dat[i].dataid = bandid[i]
        dat[i].nullvalue = nval*scale[i]
        dat[i].rows = dat[i].data.shape[0]
        dat[i].cols = dat[i].data.shape[1]
        dat[i].xdim = lonsdim
        dat[i].ydim = latsdim
        dat[i].units = units[i]

    dat.pop(17)
    dat.pop(16)

    return dat


class ExportData(object):
    """
    Export Data

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
    ext : str
        filename extension
    """
    def __init__(self, parent):
        self.ifile = ""
        self.name = "Export Data: "
        self.ext = ""
        self.pbar = None
        self.parent = parent
        self.indata = {}
        self.outdata = {}

    def run(self):
        """ Show Info """
        self.parent.process_is_active(True)

        if 'Cluster' in self.indata:
            data = self.indata['Cluster']
        elif 'Raster' in self.indata:
            data = self.indata['Raster']
        else:
            self.parent.showprocesslog('No raster data')
            self.parent.process_is_active(False)
            return

        ext = \
            "GeoTiff (*.tif);;" + \
            "ENVI (*.hdr);;" + \
            "ERMapper (*.ers);;" + \
            "Geosoft (*.gxf);;" + \
            "Surfer grid (v.6) (*.grd);;" + \
            "ArcInfo ASCII (*.asc);;" + \
            "ASCII XYZ (*.xyz);;" + \
            "ArcGIS BIL (*.bil)"

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.parent, 'Save File', '.', ext)
        if filename == '':
            self.parent.process_is_active(False)
            return False
        os.chdir(filename.rpartition('/')[0])

        self.ifile = str(filename)
        self.ext = filename[-3:]

        self.parent.showprocesslog('Export Data Busy...')

    # Pop up save dialog box
        if self.ext == 'ers':
            self.export_gdal(data, 'ERS')
        if self.ext == 'gxf':
            self.export_gxf(data)
        if self.ext == 'grd':
            self.export_surfer(data)
        if self.ext == 'asc':
            self.export_ascii(data)
        if self.ext == 'xyz':
            self.export_ascii_xyz(data)
        if self.ext == 'tif':
            self.export_gdal(data, 'GTiff')
        if self.ext == 'hdr':
            self.export_gdal(data, 'ENVI')
        if self.ext == 'bil':
            self.export_gdal(data, 'EHdr')

        self.parent.showprocesslog('Export Data Finished!')
        self.parent.process_is_active(False)

    def export_gdal(self, dat, drv):
        """
        Export to GDAL format

        Parameters
        ----------
        dat : PyGMI raster Data
            dataset to export
        drv : str
            name of the GDAL driver to use
        """

        data = merge(dat)
        xmin = data[0].tlx
        ymax = data[0].tly

        driver = gdal.GetDriverByName(drv)
        dtype = data[0].data.dtype

        if dtype == np.uint8:
            fmt = gdal.GDT_Byte
        elif dtype == np.int32:
            fmt = gdal.GDT_Int32
        elif dtype == np.float64:
            fmt = gdal.GDT_Float64
        else:
            fmt = gdal.GDT_Float32

        tmp = self.ifile.rpartition('.')

        if drv == 'GTiff':
            tmpfile = tmp[0] + '.tif'
        elif drv == 'EHdr':
            fmt = gdal.GDT_Float32
            dtype = np.float32
            tmpfile = tmp[0] + '.bil'
        else:
            tmpfile = tmp[0]

        if drv == 'GTiff' and dtype == np.uint8:
            out = driver.Create(tmpfile, int(data[0].cols), int(data[0].rows),
                                len(data), fmt, options=['COMPRESS=NONE',
                                                         'TFW=YES'])
        elif drv == 'ERS' and 'Cape / TM' in data[0].wkt:
            tmp = data[0].wkt.split('TM')[1][:2]
            out = driver.Create(tmpfile, int(data[0].cols), int(data[0].rows),
                                len(data), fmt,
                                options=['PROJ=STMLO'+tmp, 'DATUM=CAPE',
                                         'UNITS=METERS'])
        elif drv == 'ERS' and 'Hartebeesthoek94 / TM' in data[0].wkt:
            tmp = data[0].wkt.split('TM')[1][:2]
            out = driver.Create(tmpfile, int(data[0].cols), int(data[0].rows),
                                len(data), fmt,
                                options=['PROJ=STMLO'+tmp, 'DATUM=WGS84',
                                         'UNITS=METERS'])
        else:
            out = driver.Create(tmpfile, int(data[0].cols), int(data[0].rows),
                                len(data), fmt)
        out.SetGeoTransform([xmin, data[0].xdim, 0, ymax, 0, -data[0].ydim])

        out.SetProjection(data[0].wkt)

        for i, datai in enumerate(data):
            rtmp = out.GetRasterBand(i+1)
            rtmp.SetDescription(datai.dataid)

            dtmp = np.ma.array(datai.data).astype(dtype)

            # This section tries to overcome null values with round off error
            # in 32-bit numbers.
            if dtype == np.float32:
                datai.nullvalue = np.float64(np.float32(datai.nullvalue))
                if datai.data.min() > -1e+10:
                    datai.nullvalue = np.float64(np.float32(-1e+10))
                elif datai.data.max() < 1e+10:
                    datai.nullvalue = np.float64(np.float32(1e+10))

            elif dtype == np.float or dtype == np.float64:
                datai.nullvalue = np.float64(dtmp.fill_value)

            elif dtype == np.uint8:
                datai.nullvalue = 0  # specify 0, since fill value is 999999
            elif dtype == np.int32:
                datai.nullvalue = np.uint32(dtmp.fill_value)

            dtmp.set_fill_value(datai.nullvalue)
            dtmp = dtmp.filled()

            if dtype == np.uint8:
                datai.nullvalue = int(datai.nullvalue)

            if drv != 'GTiff':
                rtmp.SetNoDataValue(datai.nullvalue)
            elif len(data) == 1:
                rtmp.SetNoDataValue(datai.nullvalue)
            rtmp.WriteArray(dtmp)

        out = None  # Close File
        if drv == 'ENVI':
            with open(tmpfile+'.hdr', 'a') as myfile:
                myfile.write('data ignore value = ' + str(data[0].nullvalue))

    def export_gxf(self, data):
        """
        Export GXF data

        Parameters
        ----------
        data : PyGMI raster Data
            dataset to export
        """
        for k in data:
            file_out = self.get_filename(k, 'gxf')
            fno = open(file_out, 'w')

            xmin = k.tlx
            ymin = k.tly - k.rows*k.ydim

            fno.write("#TITLE\n")
            fno.write(self.name)
            fno.write("\n#POINTS\n")
            fno.write(str(k.cols))
            fno.write("\n#ROWS\n")
            fno.write(str(k.rows))
            fno.write("\n#PTSEPARATION\n")
            fno.write(str(k.xdim))
            fno.write("\n#RWSEPARATION\n")
            fno.write(str(k.ydim))
            fno.write("\n#XORIGIN\n")
            fno.write(str(xmin))
            fno.write("\n#YORIGIN\n")
            fno.write(str(ymin))
            fno.write("\n#SENSE\n")
            fno.write("1")
            fno.write("\n#DUMMY\n")
            fno.write(str(k.nullvalue))
            fno.write("\n#GRID\n")
            tmp = k.data.filled(k.nullvalue)

            for i in range(k.data.shape[0]-1, -1, -1):
                kkk = 0
# write only 5 numbers in a row
                for j in range(k.data.shape[1]):
                    if kkk == 5:
                        kkk = 0
                    if kkk == 0:
                        fno.write("\n")

                    fno.write(str(tmp[i, j]) + "  ")
                    kkk += 1

            fno.close()

    def export_surfer(self, data):
        """
        Export a surfer binary grid

        Parameters
        ----------
        data : PyGMI raster Data
            dataset to export
        """
        for k in data:
            file_out = self.get_filename(k, 'grd')
            fno = open(file_out, 'wb')

            xmin = k.tlx
            xmax = k.tlx + k.cols*k.xdim
            ymin = k.tly - k.rows*k.ydim
            ymax = k.tly

            bintmp = struct.pack('cccchhdddddd', b'D', b'S', b'B', b'B',
                                 k.cols, k.rows,
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
        Export Ascii file

        Parameters
        ----------
        data : PyGMI raster Data
            dataset to export
        """
        for k in data:
            file_out = self.get_filename(k, 'asc')
            fno = open(file_out, 'w')

            xmin = k.tlx
            ymin = k.tly - k.rows*k.ydim

            fno.write("ncols \t\t\t" + str(k.cols))
            fno.write("\nnrows \t\t\t" + str(k.rows))
            fno.write("\nxllcorner \t\t\t" + str(xmin))
            fno.write("\nyllcorner \t\t\t" + str(ymin))
            fno.write("\ncellsize \t\t\t" + str(k.xdim))
            fno.write("\nnodata_value \t\t" + str(k.nullvalue))

            tmp = k.data.filled(k.nullvalue)

            for j in range(k.rows):
                fno.write("\n")
                for i in range(k.cols):
                    fno.write(str(tmp[j, i]) + " ")

            fno.close()

    def export_ascii_xyz(self, data):
        """
        Export and xyz file

        Parameters
        ----------
        data : PyGMI raster Data
            dataset to export
        """
        for k in data:
            file_out = self.get_filename(k, 'xyz')
            fno = open(file_out, 'w')

            tmp = k.data.filled(k.nullvalue)

            xmin = k.tlx
            ymax = k.tly

            for j in range(k.rows):
                for i in range(k.cols):
                    fno.write(str(xmin+i*k.xdim) + " " +
                              str(ymax-j*k.ydim) + " " +
                              str(tmp[j, i]) + "\n")
            fno.close()

    def get_filename(self, data, ext):
        """
        Gets a valid filename

        Parameters
        ----------
        data : PyGMI raster Data
            dataset to get filename from
        ext : str
            filename extension to use
        """
        file_band = data.dataid.split('_')[0].strip('"')
        file_band = file_band.replace('/', '')
        file_band = file_band.replace(':', '')
        file_out = self.ifile.rpartition(".")[0]+"_"+file_band+'.'+ext

        return file_out


def get_geopak(hfile):
    """
    GeoPak Import

    Parameters
    ----------
    hfile : str
        filename to import

    Returns
    -------
    dat : PyGMI raster Data
        dataset imported
    """

    fin = open(hfile, 'rb')
    fall = fin.read()
    fin.close()

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
    dat[i].nrofbands = 1
    dat[i].tlx = x0
    dat[i].tly = y0+dy*nrows
    dat[i].dataid = hfile[:-4]

    dat[i].nullvalue = nval
    dat[i].rows = nrows
    dat[i].cols = ncols
    dat[i].xdim = dx
    dat[i].ydim = dy

    return dat


def get_geosoft(hfile):
    """
    Get geosoft file

    Parameters
    ----------
    ifile : str
        filename to import

    Returns
    -------
    dat : PyGMI raster Data
        dataset imported
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
    dat[i].nrofbands = 1
    dat[i].tlx = x0
    dat[i].tly = y0+dy*nrows
    dat[i].dataid = hfile[:-4]

    dat[i].nullvalue = nval
    dat[i].rows = nrows
    dat[i].cols = ncols
    dat[i].xdim = dx
    dat[i].ydim = dy

    return dat
