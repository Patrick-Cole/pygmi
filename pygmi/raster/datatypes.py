# -----------------------------------------------------------------------------
# Name:        datatypes.py (part of PyGMI)
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
"""Class for raster data types and conversion routines."""

import warnings
import numpy as np
from rasterio.io import MemoryFile
from rasterio import Affine


def numpy_to_pygmi(data, pdata=None, dataid=None):
    """
    Convert an MxN numpy array into a PyGMI data object.

    For convenience, if pdata is defined, parameters from another dataset
    will be used (such as xdim, ydim etc).

    Parameters
    ----------
    data : numpy array
        MxN array

    pdata : Data
        PyGMI raster dataset

    dataid: str or None
        name for the band of data.

    Returns
    -------
    tmp : Data
        PyGMI raster dataset
    """
    if data.ndim != 2:
        warnings.warn('Error: you need 2 dimensions')
        return None

    tmp = Data()
    if np.ma.isMaskedArray(data):
        tmp.data = data
    else:
        tmp.data = np.ma.array(data)

    if isinstance(pdata, Data):
        if pdata.data.shape != data.shape:
            warnings.warn('Error: you need your data and pygmi data '
                          'shape to be the same')
            return None
        tmp.extent = pdata.extent
        tmp.bounds = pdata.bounds
        tmp.xdim = pdata.xdim
        tmp.ydim = pdata.ydim
        tmp.dataid = pdata.dataid
        tmp.nodata = pdata.nodata
        tmp.crs = pdata.crs
        tmp.transform = pdata.transform
        tmp.units = pdata.units
        tmp.isrgb = pdata.isrgb
        tmp.metadata = pdata.metadata

    if dataid is not None:
        tmp.dataid = str(dataid)

    return tmp


def pygmi_to_numpy(tmp):
    """
    Convert a PyGMI data object into an MxN numpy array.

    Parameters
    ----------
    tmp : Data
        PyGMI raster dataset

    Returns
    -------
    numpy array
        MxN numpy array
    """
    return np.array(tmp.data)


class Data():
    """
    PyGMI Data Object.

    Attributes
    ----------
    data : numpy masked array
        array to contain raster data
    extent : tuple
        Extent of data as (left, right, bottom, top)
    bounds : tuple
        Bounds of data as (left, bottom, right, top)
    xdim : float
        x-dimension of grid cell
    ydim : float
        y-dimension of grid cell
    dataid : str
        band name or id
    nodata : float
        grid null or no data value
    units : str
        description of units to be used with colour bars
    isrgb : bool
        Flag to signify an RGB image.
    metadata : dictionary
        Miscellaneous metadata for file.
    filename : str
        Filename of file.
    transform : list of Affine, optional
        rasterio transform. The default is None.
    crs : CRS
        rasterio crs of data
    """

    def __init__(self):
        self.data = np.ma.array([[0]])
        self.extent = None  # left, right, bottom, top
        self.bounds = None  # left, bottom, right, top
        self.xdim = None
        self.ydim = None
        self.dataid = ''
        self.nodata = None
        self.units = ''
        self.isrgb = False
        self.metadata = {'Cluster': {}, 'Raster': {'Sensor': 'Generic'}}
        self.meta = {}  # rasterio meta
        self.filename = ''
        self.transform = None
        self.crs = None
        self.datetime = None

        self.set_transform(1, 0, 1, 0)

    def meta_from_rasterio(self, dataset):
        """
        Set transform, bounds, extent, xdim and ydim from a rasterio dataset.

        Parameters
        ----------
        dataset : rasterio dataset
            Rasterio dataset.

        Returns
        -------
        None.

        """
        self.xdim = dataset.transform[0]
        self.ydim = abs(dataset.transform[4])

        left, bottom, right, top = dataset.bounds

        self.extent = (left, right, bottom, top)
        self.bounds = (left, bottom, right, top)
        self.transform = dataset.transform
        self.crs = dataset.crs
        self.meta = dataset.meta

    def set_transform(self, xdim=None, xmin=None, ydim=None, ymax=None,
                      transform=None, iraster=None, rows=None, cols=None):
        """
        Set the transform.

        This requires either transform as input OR xdim, ydim, xmin, ymax.

        Parameters
        ----------
        xdim : float, optional
            x dimension. The default is None.
        xmin : float, optional
            x minimum. The default is None.
        ydim : float, optional
            y dimension. The default is None.
        ymax : float, optional
            y maximum. The default is None.
        transform : list of Affine, optional
            transform. The default is None.
        iraster : list, optional
            list containing offsets etc in event of cutting data. The default
            is None.

        Returns
        -------
        None.

        """
        if transform is not None:
            xdim = transform[0]
            ydim = transform[4]
            xmin = transform[2]
            ymax = transform[5]

        ydim = abs(ydim)

        if iraster is None:
            xoff = 0
            yoff = 0
        else:
            xoff, yoff, _, _ = iraster

        # get rows and cols this way because RGB images have three dims
        if rows is None:
            rows = self.data.shape[0]
        if cols is None:
            cols = self.data.shape[1]

        left = xmin + xoff*xdim
        top = ymax - yoff*ydim
        right = left + xdim*cols
        bottom = top - ydim*rows

        self.transform = Affine(xdim, 0, left, 0, -ydim, top)
        self.xdim = xdim
        self.ydim = ydim

        self.extent = (left, right, bottom, top)
        self.bounds = (left, bottom, right, top)

    def to_mem(self):
        """
        Create a rasterio memory file from one band.

        Returns
        -------
        raster : MemoryFile
            rasterio memory file.

        """
        raster = MemoryFile().open(driver='GTiff',
                                   height=self.data.shape[0],
                                   width=self.data.shape[1],
                                   count=1,
                                   dtype=self.data.dtype,
                                   transform=self.transform,
                                   crs=self.crs,
                                   nodata=self.nodata)
        raster.write(self.data, 1)
        return raster

    def get_vmin_vmax(self, std=2.5):
        """
        Get vmin and vmax for use in imshow.

        Parameters
        ----------
        std : float, optional
            Multiplier for standard deviations to include about mean.
            The default is 2.5.

        Returns
        -------
        vmin : TYPE
            DESCRIPTION.
        vmax : TYPE
            DESCRIPTION.

        """
        mean = self.data.mean()
        std = self.data.std()
        vmin = mean-2*std
        vmax = mean+2*std

        return vmin, vmax


class RasterMeta():
    """
    PyGMI Raster Metadata Object.

    Attributes
    ----------
    sensor : str
        Sensor used to measure data.
    extent : tuple
        Extent of data as (left, right, bottom, top)
    bounds : tuple
        Bounds of data as (left, bottom, right, top)
    xdim : float
        x-dimension of grid cell
    ydim : float
        y-dimension of grid cell
    units : str
        description of units to be used with colour bars
    metadata : dictionary
        Miscellaneous metadata for file.
    filename : str
        Filename of file.
    transform : list of Affine, optional
        rasterio transform. The default is None.
    crs : CRS
        rasterio crs of data.
    datetime : datetime
        date of survey.
    bands : list
        list of bands in dataset.
    tnames : list
        list fo bands to process.
    """

    def __init__(self):
        self.sensor = 'Generic'
        self.crs = None
        self.filename = ''
        self.bands = []
        self.tnames = []
        self.banddata = []
        self.to_sutm = False

    def fromData(self, dat):
        """
        Populate class from a Data class.

        Parameters
        ----------
        dat : Data
            PyGMI data object.

        Returns
        -------
        None.

        """
        data = dat[0]
        self.sensor = data.metadata['Raster']['Sensor']
        self.crs = data.crs
        self.filename = data.filename

        self.bands = []
        self.tnames = []
        self.banddata = []
        for i in dat:
            self.bands.append(i.dataid)
            self.banddata.append(i)
            if i.dataid[0] == 'B':
                self.tnames.append(i.dataid)

        if not self.tnames:
            self.tnames = self.bands.copy()

        if 'ASTER' in self.sensor:
            self.sensor = 'ASTER'
            # if len(self.bands) == 3:
            #     self.sensor += 'v1'
            # else:
            #     self.sensor += 'v2'
