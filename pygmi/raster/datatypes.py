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
"""Classes for raster data types and conversion routines."""

import datetime
import warnings
from copy import deepcopy

import numpy as np
from rasterio import Affine
from rasterio.features import shapes, sieve
from rasterio.io import MemoryFile
from rasterio.windows import Window
from shapely.geometry import Polygon, shape


def numpy_to_pygmi(data, pdata=None, dataid=None):
    """
    Convert an MxN numpy array into a PyGMI data object.

    For convenience, if pdata is defined, parameters from another dataset
    will be used (such as xdim, ydim etc).

    Parameters
    ----------
    data : numpy array
        MxN array

    pdata : pygmi.raster.datatypes.Data
        PyGMI raster dataset

    dataid: str or None
        name for the band of data.

    Returns
    -------
    tmp : pygmi.raster.datatypes.Data
        PyGMI raster dataset
    """
    if data.ndim != 2:
        warnings.warn("Error: you need 2 dimensions")
        return None

    tmp = Data()
    if np.ma.isMaskedArray(data):
        tmp.data = data
    else:
        tmp.data = np.ma.array(data)

    if isinstance(pdata, Data):
        if pdata.data.shape != data.shape:
            warnings.warn(
                "Error: you need your data and pygmi data shape to be the same"
            )
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
    tmp : pygmi.raster.datatypes.Data
        PyGMI raster dataset

    Returns
    -------
    numpy array
        MxN numpy array
    """
    return np.array(tmp.data)


def bounds_to_transform(bounds, dxy):
    """
    Create a raster transform from vector grid bounds and dxy.

    This accounts for the situation where xmax and ymax need to be readjusted
    slightly because dxy does not divide perfectly into bounds. It also adds
    dxy/2 buffer. Therefore it cannot be used with raster bounds.

    Parameters
    ----------
    bounds : tuple
        Bounds of data as (left, bottom, right, top)
    dxy : float
        Raster pixel size.

    Returns
    -------
    transform : list of Affine
        rasterio transform.
    shape : tuple
        tuple of rows, cols.

    """
    xmin, ymin, xmax, ymax = bounds
    rows = int((ymax - ymin) // dxy) + 1
    cols = int((xmax - xmin) // dxy) + 1
    xmin -= dxy / 2
    ymin -= dxy / 2
    xmax = cols * dxy + xmin
    ymax = rows * dxy + ymin
    transform = Affine(dxy, 0, xmin, 0, -dxy, ymax)
    shape = (rows, cols)

    return transform, shape


def bounds_intersection(dataset, bounds, showlog=print):
    """
    Find the intersection between some bounds and a dataset.

    Parameters
    ----------
    dataset : rasterio dataset
        Rasterio dataset.
    bounds : tuple
        Bounds of data as (left, bottom, right, top).
    showlog : function, optional
        Display information. The default is print.

    Returns
    -------
    window : rasterio window
        Intersection area as window.
    newbounds : tuple
        Intersection area as bounds.

    """
    if bounds is not None:
        xdim, ydim = dataset.res
        xmin, ymin, xmax, ymax = dataset.bounds
        xmin1, ymin1, xmax1, ymax1 = bounds

        if xmin1 >= xmax or xmax1 <= xmin or ymin1 >= ymax or ymax1 <= ymin:
            showlog("Warning: No data in polygon.")
            return (False, False)

        xmin2 = max(xmin, xmin1)
        ymin2 = max(ymin, ymin1)
        xmax2 = min(xmax, xmax1)
        ymax2 = min(ymax, ymax1)

        xoff = int((xmin2 - xmin) // xdim)
        yoff = int((ymax - ymax2) // ydim)

        xsize = int((xmax2 - xmin2) // xdim)
        ysize = int((ymax2 - ymin2) // xdim)

        # iraster = (xoff, yoff, xsize, ysize)
        newbounds = (
            xmin + xoff * xdim,
            ymax - yoff * ydim - ysize * ydim,
            xmin + xoff * xdim + xsize * xdim,
            ymax - yoff * ydim,
        )
        window = Window(xoff, yoff, xsize, ysize)
    else:
        newbounds = None
        window = None

    return (window, newbounds)


class Data:
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
    meta : dictionary
        Rasterio metadata for file.
    filename : str
        Filename of file.
    transform : list of Affine, optional
        rasterio transform. The default is None.
    crs : CRS
        rasterio crs of data
    datetime : date
        Date of dataset.
    """

    def __init__(self):
        self.data = np.ma.array([[0]])
        self.extent = None  # left, right, bottom, top
        self.bounds = None  # left, bottom, right, top
        self.xdim = None
        self.ydim = None
        self.dataid = ""
        self.nodata = None
        self.units = ""
        self.isrgb = False
        self.metadata = {
            "Cluster": {},
            "Raster": {"Sensor": "Generic", "Section": False},
        }
        self.meta = {}  # rasterio meta
        self.filename = ""
        self.transform = None
        self.crs = None
        self.datetime = datetime.datetime(1900, 1, 1)
        self.geometry = None

        self.set_transform(1, 0, 1, 0)

    def copy(self, data0=None, resetmeta=False):
        """
        Make a deepcopy of the function.

        Parameters
        ----------
        data0 : numpy arraay
            Input data to replace old data. Must have same shape.
        resetmeta : bool, optional
            This will clear metadata during copy. The default is False.

        Returns
        -------
        data : pygmi.raster.datatypes.Data
            PyGMI data type.

        """
        data = Data()
        data.__dict__ = {key: deepcopy(value) for key, value in self.__dict__.items()}

        if resetmeta is True:
            data.metadata = {
                "Cluster": {},
                "Raster": {"Sensor": "Generic", "Section": False},
            }

        if data0 is not None:
            if data0.shape == data.data.shape:
                data.data = np.ma.array(data0)
            else:
                print("Datasets have different shapes")

        return data

    def in_bounds(self, bounds):
        """
        Check if dataset is in bounds supplied.

        Parameters
        ----------
        bounds : tuple
            Bounds of data as (left, bottom, right, top)

        Returns
        -------
        bool
            True if within bounds, otherwise False.

        """
        if self.bounds is None:
            return False

        xmin, ymin, xmax, ymax = self.bounds
        xmin1, ymin1, xmax1, ymax1 = bounds

        if xmin1 >= xmax or xmax1 <= xmin or ymin1 >= ymax or ymax1 <= ymin:
            return False

        return True

    def meta_from_rasterio(self, dataset, bounds=None):
        """
        Set transform, bounds, extent, xdim and ydim from a rasterio dataset.

        Parameters
        ----------
        dataset : rasterio dataset
            Rasterio dataset.
        bounds : tuple, optional
            Bounds of data as (left, bottom, right, top). The default is None.

        Returns
        -------
        None.

        """
        self.xdim = dataset.transform[0]
        self.ydim = abs(dataset.transform[4])
        self.crs = dataset.crs
        self.meta = dataset.meta

        if bounds is None:
            left, bottom, right, top = dataset.bounds
            self.transform = dataset.transform
        else:
            left, bottom, right, top = bounds
            self.transform = Affine(self.xdim, 0, left, 0, -self.ydim, top)

        self.extent = (left, right, bottom, top)
        self.bounds = (left, bottom, right, top)

    def modify_mask(self, mask, oper="or"):
        """
        Modify the existing mask with a new one.

        The routine also fills the masked areas with nodata.

        Parameters
        ----------
        mask : array
            Boolean array of new mask to modify old one.
        oper : str, optional
            Logical operation to be performed between masks. Can be 'or' or
            'and'. The default is 'or'.

        Returns
        -------
        None.

        """
        if oper == "or":
            self.data.mask = np.logical_or(self.data.mask, mask)
        else:
            self.data.mask = np.logical_and(self.data.mask, mask)

        self.data = self.data.filled(self.nodata)
        self.data = np.ma.masked_equal(self.data, self.nodata)

    def plot(self, ax):
        """
        Simple data plot.

        Parameters
        ----------
        ax : Matplotlib axes
            Matplotlib axes for plot.

        Returns
        -------
        None.

        """
        vmin, vmax = self.get_vmin_vmax()
        im = ax.imshow(
            self.data, vmin=vmin, vmax=vmax, extent=self.extent, interpolation="none"
        )
        return im

    def set_mask(self, mask=None):
        """
        Replace the existing mask with a new one.

        The routine also fills the masked areas with nodata.

        Parameters
        ----------
        mask : array
            Boolean array of new mask to modify old one.

        Returns
        -------
        None.

        """
        if mask is not None:
            self.data.mask = mask

        self.data = self.data.filled(self.nodata)
        self.data = np.ma.masked_equal(self.data, self.nodata)

    def set_transform(
        self,
        xdim=None,
        xmin=None,
        ydim=None,
        ymax=None,
        transform=None,
        iraster=None,
        rows=None,
        cols=None,
    ):
        """
        Set the transform, xdim, ydim, extent and bounds.

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
        iraster : None or tuple
            Incremental raster import, to import a section of a file.
            The tuple is (xoff, yoff, xsize, ysize). The default is None.
        rows : int, optional
            rows in dataset. The default is None.
        cols : int, optional
            columns in dataset. The default is None.

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

        left = xmin + xoff * xdim
        top = ymax - yoff * ydim
        right = left + xdim * cols
        bottom = top - ydim * rows

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
        raster = MemoryFile().open(
            driver="GTiff",
            height=self.data.shape[0],
            width=self.data.shape[1],
            count=1,
            dtype=self.data.dtype,
            transform=self.transform,
            crs=self.crs,
            nodata=self.nodata,
        )
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
        vmin : float
            Value minimum.
        vmax : float
            Value maximum.

        """
        mean = self.data.mean()
        std = self.data.std()
        vmin = mean - 2 * std
        vmax = mean + 2 * std

        return vmin, vmax

    def get_boundary(self):
        """
        Get raster boundary.

        Sets self.geometry to a Polygon of the raster boundary.

        """
        mask = ~np.ma.getmaskarray(self.data)
        mask = mask.astype(np.uint8)

        # minpixels = min(mask.sum() // 2, 100000)
        # minpixels = mask.sum() // 2
        minpixels = max(1, int(np.sqrt(mask.sum())))
        mask = sieve(mask, minpixels)  # , mask=self.data.mask)
        shape1 = None

        polys = []

        for shape1, _ in shapes(mask, mask=mask, transform=self.transform):
            polys.append(shape1)

        if len(polys) > 1:
            print("Warning, more than one polygon, choosing largest")
            lens = np.argmax([shape(i).area for i in polys])
            shape1 = polys[lens]

        geom = shape(shape1)
        geom = geom.simplify(tolerance=0.001)

        if geom.interiors:
            geom = Polygon(list(geom.exterior.coords))

        self.geometry = geom


class RasterMeta:
    """
    PyGMI Raster Metadata Object.

    Attributes
    ----------
    sensor : str
        Sensor used to measure data.
    filename : str
        Filename of file.
    crs : CRS
        rasterio crs of data.
    bands : list
        list of bands in dataset.
    tnames : list
        list fo bands to process.
    banddata : list
        list of band data.
    to_sutm : bool
        flag to convert a file to SUTM.
    datetime : date
        date and time of dataset.
    nodata : float
        grid null or no data value.

    """

    def __init__(self):
        self.sensor = "Generic"
        self.crs = None
        self.filename = ""
        self.bands = []
        self.tnames = []
        self.banddata = []
        self.to_sutm = False
        self.datetime = datetime.datetime(1900, 1, 1)
        self.nodata = None

    def fromData(self, dat):
        """
        Populate class from a Data class.

        Parameters
        ----------
        dat : pygmi.raster.datatypes.Data
            PyGMI data object.

        Returns
        -------
        None.

        """
        data = dat[0]
        self.sensor = data.metadata["Raster"]["Sensor"]
        self.crs = data.crs
        self.filename = data.filename
        self.datetime = data.datetime
        self.nodata = data.nodata

        self.bands = []
        self.tnames = []
        self.banddata = []
        for i in dat:
            self.bands.append(i.dataid)
            self.banddata.append(i)
            if i.dataid[0] == "B":
                self.tnames.append(i.dataid)

        if not self.tnames:
            self.tnames = self.bands.copy()

        if "ASTER" in self.sensor:
            self.sensor = "ASTER"
