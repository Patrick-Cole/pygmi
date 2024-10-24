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
"""Miscellaneous functions."""

from math import cos, sin, tan
from collections import Counter
import numpy as np
import numexpr as ne
from scipy import ndimage
from matplotlib.pyplot import colormaps
from pyproj.crs import CRS
import rasterio
from rasterio.warp import reproject

from pygmi.misc import ProgressBarText
from pygmi.raster.datatypes import Data


def aspect2(data):
    """
    Aspect of a dataset.

    Parameters
    ----------
    data : numpy MxN array
        input data used for the aspect calculation

    Returns
    -------
    adeg : numpy masked array
        aspect in degrees
    dzdx : numpy array
        gradient in x direction
    dzdy : numpy array
        gradient in y direction
    """
    cdy = np.array([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])
    cdx = np.array([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]])

    dzdx = ndimage.convolve(data, cdx)  # Use convolve: matrix filtering
    dzdy = ndimage.convolve(data, cdy)  # 'valid' gets reduced array

    dzdx = ne.evaluate('dzdx/8.')
    dzdy = ne.evaluate('dzdy/8.')

    # Aspect Section
    pi = np.pi
    adeg = ne.evaluate('90-arctan2(dzdy, -dzdx)*180./pi')
    adeg = np.ma.masked_invalid(adeg)
    adeg[np.ma.less(adeg, 0.)] += 360.
    adeg[np.logical_and(dzdx == 0, dzdy == 0)] = -1.

    return [adeg, dzdx, dzdy]


def check_dataid(out):
    """
    Check dataid for duplicates and renames where necessary.

    Parameters
    ----------
    out : list of PyGMI Data
        PyGMI raster data.

    Returns
    -------
    out : list of PyGMI Data
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
                i.dataid += '('+str(j)+')'
                j += 1

    return out


def currentshader(data, cell=1., theta=np.pi/4., phi=-np.pi/4., alpha=1.0):
    """
    Blinn shader - used for sun shading.

    Parameters
    ----------
    data : numpy array
        Dataset to be shaded.
    cell : float
        between 1 and 100 - controls sunshade detail.
    theta : float
        sun elevation (also called g in code below)
    phi : float
        azimuth
    alpha : float
        how much incident light is reflected (0 to 1)

    Returns
    -------
    R : numpy array
        array containing the shaded results.

        self.phi = -np.pi/4.
        self.theta = np.pi/4.
        self.cell = 100.
        self.alpha = .0


    """
    asp = aspect2(data)
    n = 2
    pinit = asp[1]
    qinit = asp[2]
    p = ne.evaluate('pinit/cell')
    q = ne.evaluate('qinit/cell')
    sqrt_1p2q2 = ne.evaluate('sqrt(1+p**2+q**2)')

    cosg2 = cos(theta/2)
    p0 = -cos(phi)*tan(theta)
    q0 = -sin(phi)*tan(theta)
    sqrttmp = ne.evaluate('(1+sqrt(1+p0**2+q0**2))')
    p1 = ne.evaluate('p0 / sqrttmp')
    q1 = ne.evaluate('q0 / sqrttmp')

    cosi = ne.evaluate('((1+p0*p+q0*q)/(sqrt_1p2q2*sqrt(1+p0**2+q0**2)))')
    coss = ne.evaluate('((1+p1*p+q1*q)/(sqrt_1p2q2*sqrt(1+p1**2+q1**2)))')
    Ps = ne.evaluate('coss**n')
    R = np.ma.masked_invalid(ne.evaluate('((1-alpha)+alpha*Ps)*cosi/cosg2'))

    return R


def histcomp(img, nbr_bins=None, perc=5., uperc=None):
    """
    Histogram Compaction.

    This compacts a % of the outliers in data, allowing for a cleaner, linear
    representation of the data.

    Parameters
    ----------
    img : numpy array
        data to compact
    nbr_bins : int
        number of bins to use in compaction, default is None
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

    if nbr_bins is None:
        nbr_bins = max(img.shape)
        nbr_bins = max(nbr_bins, 256)

    # get image histogram
    imask = np.ma.getmaskarray(img)

    svalue, evalue = np.percentile(img.compressed(), (perc, 100-uperc))

    img2 = np.empty_like(img, dtype=np.float32)
    np.copyto(img2, img)

    filt = np.ma.less(img2, svalue)
    img2[filt] = svalue

    filt = np.ma.greater(img2, evalue)
    img2[filt] = evalue

    img2 = np.ma.array(img2, mask=imask)

    return img2, svalue, evalue


def histeq(img, nbr_bins=32768):
    """
    Histogram Equalization.

    Equalizes the histogram to colours. This allows for seeing as much data as
    possible in the image, at the expense of knowing the real value of the
    data at a point. It bins the data equally - flattening the distribution.

    Parameters
    ----------
    img : numpy array
        input data to be equalised
    nbr_bins : integer
        number of bins to be used in the calculation, default is 32768

    Returns
    -------
    im2 : numpy array
        output data
    """
    # get image histogram
    imhist, bins = np.histogram(img.compressed(), nbr_bins)
    bins = (bins[1:]-bins[:-1])/2+bins[:-1]  # get bin center point

    cdf = imhist.cumsum()  # cumulative distribution function
    cdf = cdf - cdf[0]  # subtract min, which is first val in cdf
    cdf = cdf.astype(np.int64)
    cdf = nbr_bins * cdf / cdf[-1]  # norm to nbr_bins

    # use linear interpolation of cdf to find new pixel values
    im2 = np.interp(img, bins, cdf)
    im2 = np.ma.array(im2, mask=img.mask)

    return im2


def img2rgb(img, cbar=colormaps['jet']):
    """
    Image to RGB.

    convert image to 4 channel rgba colour image.

    Parameters
    ----------
    img : numpy array
        array to be converted to rgba image.
    cbar : matplotlib colour map
        colormap to apply to the image, default is jet.

    Returns
    -------
    im2 : numpy array
        output rgba image
    """
    im2 = img.copy()
    im2 = norm255(im2)
    cbartmp = cbar(range(255))
    cbartmp = np.array([[0., 0., 0., 1.]]+cbartmp.tolist())*255
    cbartmp = cbartmp.round()
    cbartmp = cbartmp.astype(np.uint8)
    im2 = cbartmp[im2]
    im2[:, :, 3] = np.logical_not(img.mask)*254+1

    return im2


def lstack(dat, piter=None, dxy=None, showlog=print, commonmask=False,
           masterid=None, nodeepcopy=False, resampling='nearest',
           checkdataid=True):
    """
    Layer stack datasets found in a single PyGMI data object.

    The aim is to ensure that all datasets have the same number of rows and
    columns.

    Parameters
    ----------
    dat : list of PyGMI Data
        data object which stores datasets
    piter : function, optional
        Progress bar iterator. The default is None.
    dxy : float, optional
        Cell size. The default is None.
    showlog : function, optional
        Display information. The default is print.
    commonmask : bool, optional
        Create a common mask for all bands. The default is False.
    masterid : str, optional
        ID of master dataset. The default is None.

    Returns
    -------
    out : list of PyGMI Data
        data object which stores datasets

    """
    if piter is None:
        piter = ProgressBarText().iter

    if dat[0].isrgb:
        return dat

    resampling = rasterio.enums.Resampling[resampling]
    needsmerge = False
    rows, cols = dat[0].data.shape

    dtypes = []
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
        dtypes.append(i.data.dtype)

    dtypes = np.unique(dtypes)
    dtype = None
    nodata = None
    if len(dtypes) > 1:
        needsmerge = True
        for i in dtypes:
            if np.issubdtype(i, np.floating):
                dtype = np.float64
                nodata = 1e+20
            elif dtype is None:
                dtype = np.int32
                nodata = 999999

    if needsmerge is False:
        if not nodeepcopy:
            dat = [i.copy() for i in dat]
        if checkdataid is True:
            dat = check_dataid(dat)
        return dat

    # showlog('Merging data...')
    if masterid is not None:
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

        xmin0, xmax0, ymin0, ymax0 = data.extent
        for data in dat:
            xmin, xmax, ymin, ymax = data.extent
            xmin = min(xmin, xmin0)
            xmax = max(xmax, xmax0)
            ymin = min(ymin, ymin0)
            ymax = max(ymax, ymax0)

    cols = int((xmax - xmin)/dxy)
    rows = int((ymax - ymin)/dxy)
    trans = rasterio.Affine(dxy, 0, xmin, 0, -1*dxy, ymax)

    if cols == 0 or rows == 0:
        showlog('Your rows or cols are zero. '
                'Your input projection may be wrong')
        return None

    dat2 = []
    cmask = None
    for data in piter(dat):

        if dtype is not None:
            data.data = data.data.astype(dtype)
            data.nodata = nodata

        if data.crs is None:
            showlog(f'{data.dataid} has no defined projection. '
                    'Assigning local.')

            data.crs = CRS.from_string('LOCAL_CS["Arbitrary",UNIT["metre",1,'
                                       'AUTHORITY["EPSG","9001"]],'
                                       'AXIS["Easting",EAST],'
                                       'AXIS["Northing",NORTH]]')

        doffset = 0.0
        data.data.set_fill_value(data.nodata)
        data.data = np.ma.array(data.data.filled(), mask=data.data.mask)

        trans0 = data.transform
        if trans0 == trans:
            dat2.append(data.copy())
        else:
            if data.data.min() <= 0:
                doffset = data.data.min()-1.
                data.data = data.data - doffset
            height, width = data.data.shape

            odata = np.zeros((rows, cols), dtype=data.data.dtype)
            odata, _ = reproject(source=data.data,
                                 destination=odata,
                                 src_transform=trans0,
                                 src_crs=data.crs,
                                 src_nodata=data.nodata,
                                 dst_transform=trans,
                                 dst_crs=data.crs,
                                 resampling=resampling)

            data2 = Data()
            data2.data = np.ma.masked_equal(odata, data.nodata)
            data2.data.mask = np.ma.getmaskarray(data2.data)
            data2.nodata = data.nodata
            data2.crs = data.crs
            data2.set_transform(transform=trans)
            data2.data = data2.data.astype(data.data.dtype)
            data2.dataid = data.dataid
            data2.filename = data.filename
            data2.datetime = data.datetime

            dat2.append(data2)

            dat2[-1].metadata = data.metadata
            dat2[-1].data = dat2[-1].data + doffset

            dat2[-1].nodata = data.nodata
            dat2[-1].data.set_fill_value(data.nodata)
            dat2[-1].data = np.ma.array(dat2[-1].data.filled(),
                                        mask=dat2[-1].data.mask)

            if doffset != 0.:
                data.data = data.data + doffset

        if cmask is None:
            cmask = dat2[-1].data.mask
        else:
            cmask = np.logical_or(cmask, dat2[-1].data.mask)

    if commonmask is True:
        for idat in piter(dat2):
            idat.data.mask = cmask
            idat.data = np.ma.array(idat.data.filled(idat.nodata), mask=cmask)

    if checkdataid is True:
        out = check_dataid(dat2)
    else:
        out = dat2

    return out


def norm2(dat, datmin=None, datmax=None):
    """
    Normalise array vector between 0 and 1.

    Parameters
    ----------
    dat : numpy array
        array to be normalised
    datmin : float
        data minimum, default is None
    datmax : float
        data maximum, default is None

    Returns
    -------
    out : numpy array of floats
        normalised array
    """
    if datmin is None:
        datmin = float(dat.min())
    if datmax is None:
        datmax = float(dat.max())
    datptp = datmax - datmin
    out = np.ma.array(ne.evaluate('(dat-datmin)/datptp'))
    out.mask = np.ma.getmaskarray(dat)
    out[out < 0] = 0.
    out[out > 1] = 1.

    return out


def norm255(dat):
    """
    Normalise array vector between 1 and 255.

    Parameters
    ----------
    dat : numpy array
        array to be normalised.

    Returns
    -------
    out : numpy array of 8 bit integers
        normalised array
    """
    datmin = float(dat.min())
    datptp = float(dat.ptp())
    out = ne.evaluate('254*(dat-datmin)/datptp+1')
    out = out.round()
    out = out.astype(np.uint8)
    return out


def _testfn():
    """Test."""
    from pygmi.raster.iodefs import get_raster

    ifile1 = r"D:\Landslides\JTNdem.tif"
    ifile2 = r"D:\Landslides\GeoTiff\S2B_T36JTN_R092_20220428_stack.tif"

    dat1 = get_raster(ifile1)
    dat2 = get_raster(ifile2)

    dat3 = lstack(dat1+dat2, dxy=10, commonmask=True)


if __name__ == "__main__":
    _testfn()
