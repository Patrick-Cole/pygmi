# -----------------------------------------------------------------------------
# Name:        misc.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2023 Council for Geoscience
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
"""Miscellaneous functions for raster data."""

from collections import Counter
from collections.abc import Callable, Iterable
from math import cos, sin, tan

import geopandas as gpd
import numexpr as ne
import numpy as np
import rasterio
from matplotlib.pyplot import colormaps
from numpy.typing import NDArray
from pyproj.crs import CRS
from pyproj.exceptions import ProjError
from rasterio.mask import mask as riomask
from rasterio.warp import reproject
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from shapely import Polygon

from pygmi.misc import ProgressBarText
from pygmi.raster.datatypes import Data


def aspect2(data: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """
    Aspect of a dataset.

    Parameters
    ----------
    data
        Input data used for the aspect calculation

    Returns
    -------
    adeg : ndarray
        aspect in degrees
    dzdx : ndarray
        gradient in x direction
    dzdy : ndarray
        gradient in y direction
    """
    cdy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    cdx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])

    dzdx = ndimage.convolve(data, cdx)  # Use convolve: matrix filtering
    dzdy = ndimage.convolve(data, cdy)  # 'valid' gets reduced array

    dzdx = ne.evaluate("dzdx/8.")
    dzdy = ne.evaluate("dzdy/8.")

    # Aspect Section
    pi = np.pi
    local = {"pi": pi, "dzdy": dzdy, "dzdx": dzdx}
    adeg = ne.evaluate("90-arctan2(dzdy, -dzdx)*180./pi", local_dict=local)
    adeg = np.ma.masked_invalid(adeg)
    adeg[np.ma.less(adeg, 0.0)] += 360.0
    adeg[np.logical_and(dzdx == 0, dzdy == 0)] = -1.0

    return adeg, dzdx, dzdy


def check_dataid(out: list[Data]) -> list[Data]:
    """
    Check dataid for duplicates and renames where necessary.

    Parameters
    ----------
    out
        PyGMI raster data.

    Returns
    -------
    list of Data
        PyGMI raster data.

    """
    tmplist = []
    for i in out:
        tmplist.append(i.dataid)

    tmpcnt = Counter(tmplist)
    for elt, count in tmpcnt.items():
        j = 1
        for i in out:
            if elt == i.dataid and count > 1:
                i.dataid += "(" + str(j) + ")"
                j += 1

    return out


def currentshader(
    data: NDArray,
    cell: float = 1.0,
    theta: float = np.pi / 4.0,
    phi: float = -np.pi / 4.0,
    alpha: float = 1.0,
) -> NDArray:
    """
    Blinn shader - used for sun shading.

    Parameters
    ----------
    data
        Dataset to be shaded.
    cell
        between 1 and 100 - controls sunshade detail.
    theta
        sun elevation (also called g in code below)
    phi
        azimuth
    alpha
        how much incident light is reflected (0 to 1)

    Returns
    -------
    ndarray
        array containing the shaded results.
    """
    if np.ma.is_masked(data):
        data = fill_nd_closest(data)

    local = {}

    _, pinit, qinit = aspect2(data)
    local["pinit"] = pinit
    local["qinit"] = qinit
    local["cell"] = cell
    local["alpha"] = alpha
    n = 2
    # pinit = asp[1]
    # qinit = asp[2]

    p = ne.evaluate("pinit/cell")
    q = ne.evaluate("qinit/cell")

    local["n"] = n
    local["p"] = p
    local["q"] = q

    sqrt_1p2q2 = ne.evaluate("sqrt(1+p**2+q**2)", local_dict=local)
    local["sqrt_1p2q2"] = sqrt_1p2q2

    cosg2 = cos(theta / 2)
    p0 = -cos(phi) * tan(theta)
    q0 = -sin(phi) * tan(theta)

    local["cosg2"] = cosg2
    local["p0"] = p0
    local["q0"] = q0

    sqrttmp = ne.evaluate("(1+sqrt(1+p0**2+q0**2))", local_dict=local)
    local["sqrttmp"] = sqrttmp
    p1 = ne.evaluate("p0 / sqrttmp", local_dict=local)
    q1 = ne.evaluate("q0 / sqrttmp", local_dict=local)
    local["p1"] = p1
    local["q1"] = q1

    cosi = ne.evaluate(
        "((1+p0*p+q0*q)/(sqrt_1p2q2*sqrt(1+p0**2+q0**2)))", local_dict=local
    )
    coss = ne.evaluate(
        "((1+p1*p+q1*q)/(sqrt_1p2q2*sqrt(1+p1**2+q1**2)))", local_dict=local
    )
    local["cosi"] = cosi
    local["coss"] = coss
    Ps = ne.evaluate("coss**n", local_dict=local)
    local["Ps"] = Ps
    R = np.ma.masked_invalid(
        ne.evaluate("((1-alpha)+alpha*Ps)*cosi/cosg2", local_dict=local)
    )

    return R


def cut_raster(
    data: list[Data],
    ibnd: str | gpd.GeoDataFrame | tuple,
    showlog: Callable[..., None] = print,
    deepcopy: bool = True,
) -> list[Data]:
    """
    Cut a raster dataset.

    Parameters
    ----------
    data
        |Input PyGMI Dataset
    ibnd
        shapefile or GeoDataFrame or tuple of bounds used to cut data.
    showlog
        Function for printing text. The default is print.
    deepcopy
        Make a copy of the data array before use, by default True.

    Returns
    -------
    list of Data
        Cut version of Dataset
    """
    if ibnd is None:
        return data

    if deepcopy is True:
        data = [i.copy() for i in data]

    if isinstance(ibnd, gpd.GeoDataFrame):
        gdf = ibnd
    elif isinstance(ibnd, (list, tuple)):
        x0, y0, x1, y1 = ibnd
        poly = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)])
        gdf = gpd.GeoDataFrame({"geometry": [poly]})
    else:
        gdf = gpd.read_file(ibnd)

    if gdf.crs is None:
        gdf = gdf.set_crs(data[0].crs)
    else:
        try:
            gdf = gdf.to_crs(data[0].crs)
        except ProjError:
            showlog(
                "There was a problem converting the shapefile projection "
                "to the raster projection. Check to see that both files "
                "have valid projections."
            )
            return None
    gdf = gdf[gdf.geometry.notna()]

    if "Polygon" not in gdf.geom_type.iloc[0]:
        showlog("You need a polygon in that shape file")
        return None

    for idata in data:
        # Convert the layer extent to image pixel coordinates
        dext = idata.bounds
        lext = gdf["geometry"].total_bounds

        if (
            (dext[0] > lext[2])
            or (dext[2] < lext[0])
            or (dext[1] > lext[3])
            or (dext[3] < lext[1])
        ):
            showlog(
                "The shapefile or bounds is not in the same area as the "
                "raster dataset. Please check its coordinates and make "
                "sure its projection is the same as the raster dataset"
            )
            return None

        # This section converts PolygonZ to Polygon, and takes first polygon.
        coords = gdf["geometry"]

        dat, trans = riomask(idata.to_mem(), coords, crop=True, all_touched=True)

        idata.data = np.ma.masked_equal(dat.squeeze(), idata.nodata)

        idata.set_transform(transform=trans)

    # data = trim_raster(data)

    return data


def fill_nd_closest(arr: NDArray) -> NDArray:
    """
    Fill array using closest value.

    Parameters
    ----------
    arr
        Input array.

    Returns
    -------
    NDArray
        Filled array.
    """
    mask = np.ma.getmaskarray(arr)

    # distance_transform_edt finds distance/index to the nearest ZERO (False) element.
    # We pass the mask directly, so masked elements (True) lookup the nearest unmasked (False) elements.
    indices = distance_transform_edt(mask, return_distances=False, return_indices=True)

    # Extract data using the mapped coordinates
    # indices shape: (ndim, dim1, dim2, ...)
    filled_data = arr.data[tuple(indices)]

    return filled_data


def histcomp(img, perc=5.0, uperc=None):
    """
    Histogram Compaction.

    This compacts a % of the outliers in data, allowing for a cleaner, linear
    representation of the data.

    Parameters
    ----------
    img : numpy array
        data to compact
    perc : float
        percentage of histogram to clip. If uperc is not None, then this is
        the lower percentage, default is 5.
    uperc : float
        upper percentage to clip. If uperc is None, then it is set to the
        same value as perc, default is None

    Returns
    -------
    img2 : numpy array
        compacted array
    svalue : float
        Start value
    evalue : float
        End value

    """
    if uperc is None:
        uperc = perc

    # get image histogram
    imask = np.ma.getmaskarray(img)

    svalue, evalue = np.percentile(img.compressed(), (perc, 100 - uperc))

    # img2 = np.empty_like(img, dtype=np.float32)
    # np.copyto(img2, img)

    img2 = img.copy()

    filt = np.ma.less(img2, svalue)
    img2[filt] = svalue

    filt = np.ma.greater(img2, evalue)
    img2[filt] = evalue

    img2 = np.ma.array(img2, mask=imask)

    return img2, svalue, evalue


def histeq(img, nbrbins=32768):
    """
    Histogram Equalization.

    Equalizes the histogram to colours. This allows for seeing as much data as
    possible in the image, at the expense of knowing the real value of the
    data at a point. It bins the data equally - flattening the distribution.

    Parameters
    ----------
    img : numpy array
        input data to be equalised
    nbrbins : integer
        number of bins to be used in the calculation, default is 32768

    Returns
    -------
    im2 : numpy array
        output data
    """
    # get image histogram
    imhist, bins = np.histogram(img.compressed(), nbrbins)
    bins = (bins[1:] - bins[:-1]) / 2 + bins[:-1]  # get bin center point

    cdf = imhist.cumsum()  # cumulative distribution function
    cdf = cdf - cdf[0]  # subtract min, which is first val in cdf
    cdf = cdf.astype(np.int64)
    cdf = nbrbins * cdf / cdf[-1]  # norm to nbr_bins

    # use linear interpolation of cdf to find new pixel values
    im2 = np.interp(img, bins, cdf)
    im2 = np.ma.array(im2, mask=img.mask)

    return im2


def img2rgb(img: NDArray, cbar: colormaps = colormaps["jet"]) -> NDArray:
    """
    Image to RGB.

    convert image to 4 channel rgba colour image.

    Parameters
    ----------
    img
        array to be converted to rgba image.
    cbar : matplotlib colour map
        colormap to apply to the image, default is jet.

    Returns
    -------
    ndarray
        Output RGBA image.
    """
    im2 = img.copy()
    im2 = norm255(im2)
    cbartmp = cbar(range(255))
    cbartmp = np.array([[0.0, 0.0, 0.0, 1.0]] + cbartmp.tolist()) * 255
    cbartmp = cbartmp.round()
    cbartmp = cbartmp.astype(np.uint8)
    im2 = cbartmp[im2]
    im2[:, :, 3] = np.logical_not(img.mask) * 254 + 1

    return im2


def lstack(
    dat: list[Data],
    *,
    piter: Iterable | None = None,
    dxy: float | None = None,
    showlog: Callable[..., None] = print,
    commonmask: bool = False,
    masterid: str | None = None,
    nodeepcopy: bool = False,
    resampling: str = "cubic_spline",  # "nearest",
    checkdataid: bool = True,
) -> list[Data]:
    """
    Layer stack datasets found in a single PyGMI data object.

    The aim is to ensure that all datasets have the same number of rows and
    columns.

    Parameters
    ----------
    dat
        data object which stores datasets
    piter
        Progress bar iterator. The default is None.
    dxy
        Cell size. The default is None.
    showlog
        Display information. The default is print.
    commonmask
        Create a common mask for all bands. The default is False.
    masterid
        ID of master dataset. The default is None.
    nodeepcopy
        Flag to avoid making a copy of the input data, by default False.
    resampling
        The resampling to be used on output date. The default is 'nearest'.
    checkdataid
        Check to make sure there are no duplicate data ids. The default is True

    Returns
    -------
    list of Data
        list of raster data.

    """
    if piter is None:
        piter = ProgressBarText().iter

    if dat[0].isrgb:
        return dat

    resampling = rasterio.enums.Resampling[resampling]
    needsmerge = False
    rows, cols = dat[0].data.shape

    for i in dat:
        irows, icols = i.data.shape
        if irows != rows or icols != cols:
            needsmerge = True
        if dxy is not None and (i.xdim != dxy or i.ydim != dxy):
            needsmerge = True
        if commonmask is True:
            needsmerge = True
        if i.extent != dat[0].extent:
            needsmerge = True

    if needsmerge is False:
        if not nodeepcopy:
            dat = [i.copy() for i in dat]
        if checkdataid is True:
            dat = check_dataid(dat)
        return dat

    if masterid is not None:
        data = dat[0]
        for i in dat:
            if i.dataid == masterid:
                data = i
                break

        xmin, xmax, ymin, ymax = data.extent

        if dxy is None:
            dxy = min(data.xdim, data.ydim)
    else:
        data = dat[0]

        if dxy is None:
            dxy = min(data.xdim, data.ydim)
            for data in dat:
                dxy = min(dxy, data.xdim, data.ydim)

        xmin, xmax, ymin, ymax = data.extent
        for data in dat:
            xmin0, xmax0, ymin0, ymax0 = data.extent
            xmin = min(xmin, xmin0)
            xmax = max(xmax, xmax0)
            ymin = min(ymin, ymin0)
            ymax = max(ymax, ymax0)

    cols = int(round((xmax - xmin) / dxy, 9))
    rows = int(round((ymax - ymin) / dxy, 9))
    trans = rasterio.Affine(dxy, 0, float(xmin), 0, -1 * dxy, float(ymax))

    if cols == 0 or rows == 0:
        showlog("Your rows or cols are zero. Your input projection may be wrong")
        return None

    dat2 = []
    cmask = None
    for data in piter(dat):
        # if dtype is not None:
        #     data.data = data.data.astype(dtype)
        #     data.nodata = nodata

        if data.crs is None:
            showlog(f"{data.dataid} has no defined projection. Assigning local.")

            data.crs = CRS.from_string(
                'LOCAL_CS["Arbitrary",UNIT["metre",1,'
                'AUTHORITY["EPSG","9001"]],'
                'AXIS["Easting",EAST],'
                'AXIS["Northing",NORTH]]'
            )

        doffset = 0.0
        # data.data.set_fill_value(data.nodata)
        # data.data = np.ma.array(data.data.filled(), mask=data.data.mask)
        # data.data.mask = np.ma.getmaskarray(data.data)

        trans0 = data.transform
        if trans0 == trans and data.data.shape == (rows, cols):
            if not nodeepcopy:
                dat2.append(data.copy())
            else:
                dat2.append(data)
        else:
            if data.data.min() <= 0:
                doffset = data.data.min() - 1.0
                data.data = data.data - doffset
            # height, width = data.data.shape

            # data.data = data.data.filled(0)
            odata = np.zeros((rows, cols), dtype=data.data.dtype)
            odata, _ = reproject(
                source=data.data,
                destination=odata,
                src_transform=trans0,
                src_crs=data.crs,
                src_nodata=data.nodata,
                dst_transform=trans,
                dst_crs=data.crs,
                resampling=resampling,
            )

            data2 = Data()
            # odata[odata == 0] = data.nodata
            data2.data = np.ma.masked_equal(odata, data.nodata)
            # data2.data.set_fill_value(data.nodata)
            # data2.data = np.ma.array(data2.data.filled(), mask=data2.data.mask)
            data2.data.mask = np.ma.getmaskarray(data2.data)
            data2.nodata = data.nodata
            data2.crs = data.crs
            data2.set_transform(transform=trans)
            # data2.data = data2.data.astype(data.data.dtype)
            data2.dataid = data.dataid
            data2.filename = data.filename
            data2.datetime = data.datetime

            dat2.append(data2)

            dat2[-1].metadata = data.metadata
            dat2[-1].data = dat2[-1].data + doffset

            data.data[data.data == 0] = data.nodata
            data.data = np.ma.masked_equal(data.data, data.nodata)

            if doffset != 0.0:
                data.data = data.data + doffset

        if cmask is None:
            cmask = np.ma.getmaskarray(dat2[-1].data)
        else:
            cmask = np.logical_or(cmask, np.ma.getmaskarray(dat2[-1].data))

    if commonmask is True:
        for idat in piter(dat2):
            idat.data.mask = cmask
            idat.data = np.ma.array(idat.data.filled(idat.nodata), mask=cmask)

    if checkdataid is True:
        out = check_dataid(dat2)
    else:
        out = dat2

    return out


def norm2(
    dat: NDArray, datmin: float | None = None, datmax: float | None = None
) -> NDArray:
    """
    Normalise array vector between 0 and 1.

    Parameters
    ----------
    dat
        array to be normalised
    datmin
        data minimum, default is None
    datmax
        data maximum, default is None

    Returns
    -------
    out : ndarray of floats
        normalised array
    """
    if datmin is None:
        datmin = float(dat.min())
    if datmax is None:
        datmax = float(dat.max())
    datptp = datmax - datmin
    local = {"datmin": datmin, "dat": dat, "datptp": datptp}
    out = np.ma.array(ne.evaluate("(dat-datmin)/datptp", local_dict=local))
    out.mask = np.ma.getmaskarray(dat)
    out[out < 0] = 0.0
    out[out > 1] = 1.0

    return out


def norm255(dat: NDArray) -> NDArray:
    """
    Normalise array vector between 1 and 255.

    Parameters
    ----------
    dat
        array to be normalised.

    Returns
    -------
    ndarray of 8 bit integers
        normalised array
    """
    datmin = float(dat.min())
    datptp = float(np.ma.ptp(dat))

    local = {"datmin": datmin, "dat": dat, "datptp": datptp}

    out = ne.evaluate("254*(dat-datmin)/datptp+1", local_dict=local)
    out = out.round()
    out = out.astype(np.uint8)
    return out


def _testfn():
    """Test."""
    import matplotlib.pyplot as plt

    from pygmi.raster.iodefs import get_raster

    ifile1 = r"D:\Workdata\Mosaic\3122DC_RegMag_hbhk94tm23.hdr"

    dat1 = get_raster(ifile1)

    dat3 = lstack(dat1, dxy=100, commonmask=True, resampling="cubic")

    plt.imshow(dat3[0].data)
    plt.show()


if __name__ == "__main__":
    _testfn()
