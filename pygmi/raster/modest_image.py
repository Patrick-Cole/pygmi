"""
Modest Image.

Modification of Chris Beaumont's mpl-modest-image package to allow the use of
set_extent.

pcole, 2021  - Bug fix to allow for correct zooming if origin is set to 'upper'
"""
from __future__ import print_function, division

from math import cos, sin, tan
import numpy as np
import numexpr as ne
from scipy import ndimage

from matplotlib import rcParams
from matplotlib import cm
import matplotlib.image as mi
import matplotlib.colors as mcolors
from matplotlib import cbook
from matplotlib.transforms import IdentityTransform, Affine2D

IDENTITY_TRANSFORM = IdentityTransform()


class ModestImage(mi.AxesImage):
    """
    Computationally modest image class.

    ModestImage is an extension of the Matplotlib AxesImage class
    better suited for the interactive display of larger images. Before
    drawing, ModestImage resamples the data array based on the screen
    resolution and view window. This has very little affect on the
    appearance of the image, but can substantially cut down on
    computation since calculations of unresolved or clipped pixels
    are skipped.

    The interface of ModestImage is the same as AxesImage. However, it
    does not currently support setting the 'extent' property. There
    may also be weird coordinate warping operations for images that
    I'm not aware of. Don't expect those to work either.
    """

    def __init__(self, *args, **kwargs):
        self._full_res = None
        self._full_extent = kwargs.get('extent', None)
        self.origin = kwargs.get('origin', 'lower')
        super(ModestImage, self).__init__(*args, **kwargs)
        self.invalidate_cache()

        # Custom lines for PyGMI
        self.shade = None
        self.rgbmode = ''  # Can be None, CMY Ternary or RGB Ternary
        self.rgbclip = [[None, None], [None, None], [None, None]]
        self.dohisteq = False
        self.kval = 0.01  # For CMYK Ternary

    def set_data(self, A):
        """


        Parameters
        ----------
        A : numpy/PIL Image A
            A numpy or PIL image.

        Raises
        ------
        TypeError
            Error when data has incorrect dimensions.

        Returns
        -------
        None.

        """
        self._full_res = A
        self._A = A

        if self._A.dtype != np.uint8 and not np.can_cast(self._A.dtype,
                                                         float):
            raise TypeError("Image data can not convert to float")

        if self._A.ndim not in (2, 3):
            raise TypeError("Invalid dimensions for image data")
        elif (self._A.ndim == 3 and self._A.shape[-1] not in (3, 4) and
              self.shade is False):
            raise TypeError("Invalid dimensions for image data")

        self.invalidate_cache()

    def invalidate_cache(self):
        """
        Invalidate cache.

        Returns
        -------
        None.

        """
        self._bounds = None
        self._imcache = None
        self._rgbacache = None
        self._oldxslice = None
        self._oldyslice = None
        self._sx, self._sy = None, None
        self._pixel2world_cache = None
        self._world2pixel_cache = None

    def set_extent(self, extent):
        """
        Set extent.

        Parameters
        ----------
        extent : tuple
            Extent of data.

        Returns
        -------
        None.

        """
        self._full_extent = extent
        self.invalidate_cache()
        mi.AxesImage.set_extent(self, extent)

    def get_array(self):
        """
        Override to return the full-resolution array.

        Returns
        -------
        numpy array
            Return data array of full resolution.

        """
        return self._full_res

    def get_cursor_data(self, event):
        """
        Correct z-value display when zoomed.

        Parameters
        ----------
        event : matpltolib cursor event.
            Cursor event.

        Returns
        -------
        float
            z-value or NAN.

        """
        x = event.xdata
        y = event.ydata

        # if self._full_extent is None:
        #     col = int(x + 0.5)
        #     row = int(y + 0.5)
        # else:
        #     col, row = self._world2pixel.transform((x, y))
        #     col = int(col + 0.5)
        #     row = int(row + 0.5)

        col, row = self._world2pixel.transform((x, y))
        col = int(col + 0.5)
        row = int(row + 0.5)

        numrows, numcols = self._full_res.shape[:2]

        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            # -1 because we are reversing rows.
            z = self._full_res[numrows-row-1, col]
            return z

        return np.nan

    def format_cursor_data(self, data):
        """
        Format z data on graph.

        Parameters
        ----------
        data : float
            Data value to display.

        Returns
        -------
        zval : str
            Formatted string to display.

        """
        if np.ma.is_masked(data) or isinstance(data, (list, np.ndarray)):
            zval = 'z = None'
        else:
            zval = f'z = {data:,.5f}'

        return zval

    @property
    def _pixel2world(self):

        if self._pixel2world_cache is None:

            # Pre-compute affine transforms to convert between the 'world'
            # coordinates of the axes (what is shown by the axis labels) to
            # 'pixel' coordinates in the underlying array.

            extent = self._full_extent

            if extent is None:

                self._pixel2world_cache = IDENTITY_TRANSFORM

            else:

                self._pixel2world_cache = Affine2D()

                self._pixel2world.translate(+0.5, +0.5)

                self._pixel2world.scale((extent[1] - extent[0]) /
                                        self._full_res.shape[1],
                                        (extent[3] - extent[2]) /
                                        self._full_res.shape[0])

                self._pixel2world.translate(extent[0], extent[2])

            self._world2pixel_cache = None

        return self._pixel2world_cache

    @property
    def _world2pixel(self):
        if self._world2pixel_cache is None:
            self._world2pixel_cache = self._pixel2world.inverted()
        return self._world2pixel_cache

    def _scale_to_res(self):
        """
        Scale to resolution.

        Change self._A and _extent to render an image whose resolution is
        matched to the eventual rendering.
        """
        # Find out how we need to slice the array to make sure we match the
        # resolution of the display. We pass self._world2pixel which matters
        # for cases where the extent has been set.
        x0, x1, sx, y0, y1, sy = extract_matched_slices(axes=self.axes,
                                                        shape=self._full_res.shape,
                                                        transform=self._world2pixel)

        # Check whether we've already calculated what we need, and if so just
        # return without doing anything further.
        # if (self._bounds is not None and
        #         sx >= self._sx and sy >= self._sy and
        #         x0 >= self._bounds[0] and x1 <= self._bounds[1] and
        #         y0 >= self._bounds[2] and y1 <= self._bounds[3]):
        #     return

        # Slice the array using the slices determined previously to optimally
        # match the display
        # sx=1
        # sy=1

        if self.origin == 'upper':
            self._A = self._full_res[::-1][y0:y1:sy, x0:x1:sx]
            self._A = cbook.safe_masked_invalid(self._A)
            self._A = self._A[::-1]
        else:
            self._A = self._full_res[y0:y1:sy, x0:x1:sx]
            self._A = cbook.safe_masked_invalid(self._A)

        # We now determine the extent of the subset of the image, by
        # determining it first in pixel space, and converting it to the
        # 'world' coordinates.

        # See https://github.com/matplotlib/matplotlib/issues/8693 for a
        # demonstration of why origin='upper' and extent=None needs to be
        # special-cased.

        if self.origin == 'upper' and self._full_extent is None:
            xmin, xmax, ymin, ymax = x0 - .5, x1 - .5, y1 - .5, y0 - .5
        else:
            xmin, xmax, ymin, ymax = x0 - .5, x1 - .5, y0 - .5, y1 - .5

        xmin, ymin, xmax, ymax = self._pixel2world.transform([(xmin, ymin),
                                                              (xmax, ymax)]).ravel()

        mi.AxesImage.set_extent(self, [xmin, xmax, ymin, ymax])
        # self.set_extent([xmin, xmax, ymin, ymax])

        # Finally, we cache the current settings to avoid re-computing similar
        # arrays in future.
        self._sx = sx
        self._sy = sy
        self._bounds = (x0, x1, y0, y1)

        self.changed()

    def draw(self, renderer, *args, **kwargs):
        """Draw."""
        if self._full_res.shape is None:
            return
        self._scale_to_res()
        if (self.dohisteq is True and self.shade is None and
                'Ternary' not in self.rgbmode):
            self._A = norm2(histeq(self._A))
            self.set_clim(0, 1)

        if 'Ternary' in self.rgbmode:
            colormap = self.draw_ternary()
        else:
            colormap = self._A

        if self.shade is not None:
            colormap = self.draw_sunshade(colormap)

        self._A = colormap

        super(ModestImage, self).draw(renderer, *args, **kwargs)

    def draw_ternary(self):
        """
        Draw ternary.

        Returns
        -------
        None.

        """
        colormap = np.ma.ones((self._A.shape[0], self._A.shape[1], 4))
        if self.dohisteq:
            colormap[:, :, 0] = norm2(histeq(self._A[:, :, 0]))
            colormap[:, :, 1] = norm2(histeq(self._A[:, :, 1]))
            colormap[:, :, 2] = norm2(histeq(self._A[:, :, 2]))
        else:
            colormap[:, :, 0] = norm2(self._A[:, :, 0],
                                      self.rgbclip[0][0], self.rgbclip[0][1])
            colormap[:, :, 1] = norm2(self._A[:, :, 1],
                                      self.rgbclip[1][0], self.rgbclip[1][1])
            colormap[:, :, 2] = norm2(self._A[:, :, 2],
                                      self.rgbclip[2][0], self.rgbclip[2][1])

        if 'CMY' in self.rgbmode:
            colormap[:, :, 0] = (1-colormap[:, :, 0])*(1-self.kval)
            colormap[:, :, 1] = (1-colormap[:, :, 1])*(1-self.kval)
            colormap[:, :, 2] = (1-colormap[:, :, 2])*(1-self.kval)

        if np.ma.isMaskedArray(self._A):
            mask = np.logical_or(self._A[:, :, 0].mask,
                                 self._A[:, :, 1].mask)
            mask = np.logical_or(mask, self._A[:, :, 2].mask)
            colormap[:, :, 3] = np.logical_not(mask)
        return colormap

    def draw_sunshade(self, colormap=None):
        """
        Apply sunshading.

        Returns
        -------
        None.

        """
        sun = self._A[:, :, -1]
        self._A = self._A[:, :, :-1]
        self._A = self._A.squeeze()
        if self.dohisteq is True:
            self._A = norm2(histeq(self._A))
        else:
            datmin, datmax = self.get_clim()
            self._A = norm2(self._A, datmin, datmax)

        cell, theta, phi, alpha = self.shade
        sunshader = currentshader(sun.data, cell, theta, phi, alpha)
        snorm = norm2(sunshader)

        if 'Ternary' not in self.rgbmode:
            colormap = self.cmap(self._A)

        colormap[:, :, 0] *= snorm  # red
        colormap[:, :, 1] *= snorm  # green
        colormap[:, :, 2] *= snorm  # blue
        if np.ma.isMaskedArray(sun):
            colormap[:, :, 3] = np.logical_not(sun.mask)

        return colormap

    def set_clim_std(self, mult):
        """
        Set the vmin and vmax to mult*std(self._A).

        This routine only works on a 2D array.

        Parameters
        ----------
        mult : float
            Multiplier.

        Returns
        -------
        None.

        """
        self._scale_to_res()

        if self._A.ndim > 2:
            raise TypeError("Invalid dimensions for image data. Should be 2D.")

        vstd = self._A.std()
        vmean = self._A.mean()
        vmin = vmean - mult*vstd
        vmax = vmean + mult*vstd

        self.set_clim(vmin, vmax)


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


def currentshader(data, cell, theta, phi, alpha):
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
        number of bins to use in compaction
    perc : float
        percentage of histogram to clip. If uperc is not None, then this is
        the lower percentage
    uperc : float
        upper percentage to clip. If uperc is None, then it is set to the
        same value as perc

    Returns
    -------
    img2 : numpy array
        compacted array
    """
    if uperc is None:
        uperc = perc

    if nbr_bins is None:
        nbr_bins = max(img.shape)
        nbr_bins = max(nbr_bins, 256)

# get image histogram
    imask = np.ma.getmaskarray(img)
    tmp = img.compressed()
    imhist, bins = np.histogram(tmp, nbr_bins)

    cdf = imhist.cumsum()  # cumulative distribution function
    if cdf[-1] == 0:
        return img
    cdf = cdf / float(cdf[-1])  # normalize

    perc = perc/100.
    uperc = uperc/100.

    sindx = np.arange(nbr_bins)[cdf > perc][0]
    if cdf[0] > (1-uperc):
        eindx = 1
    else:
        eindx = np.arange(nbr_bins)[cdf < (1-uperc)][-1]+1
    svalue = bins[sindx]
    evalue = bins[eindx]

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

    Equalizes the histogram to colors. This allows for seeing as much data as
    possible in the image, at the expense of knowing the real value of the
    data at a point. It bins the data equally - flattening the distribution.

    Parameters
    ----------
    img : numpy array
        input data to be equalised
    nbr_bins : integer
        number of bins to be used in the calculation

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


def img2rgb(img, cbar=cm.get_cmap('jet')):
    """
    Image to RGB.

    convert image to 4 channel rgba color image.

    Parameters
    ----------
    img : numpy array
        array to be converted to rgba image.
    cbar : matplotlib color map
        colormap to apply to the image

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


def norm2(dat, datmin=None, datmax=None):
    """
    Normalise array vector between 0 and 1.

    Parameters
    ----------
    dat : numpy array
        array to be normalised

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
        array to be normalised

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


def main():
    """Main."""
    from time import time
    import matplotlib.pyplot as plt
    x, y = np.mgrid[0:2000, 0:2000]
    data = np.sin(x / 10.) * np.cos(y / 30.)

    f = plt.figure()
    ax = f.add_subplot(111)

    # try switching between
    artist = ModestImage(ax, data=data)

    ax.set_aspect('equal')
    artist.norm.vmin = -1
    artist.norm.vmax = 1

    ax.add_artist(artist)

    t0 = time()
    plt.gcf().canvas.draw()
    t1 = time()

    print("Draw time for %s: %0.1f ms" % (artist.__class__.__name__,
                                          (t1 - t0) * 1000))

    plt.show()


def imshow(axes, X, cmap=None, norm=None, aspect=None,
           interpolation=None, alpha=None, vmin=None, vmax=None,
           origin=None, extent=None, shape=None, filternorm=1,
           filterrad=4.0, imlim=None, resample=None, url=None, **kwargs):
    """
    Similar to matplotlib's imshow command, but produces a ModestImage.

    Unlike matplotlib version, must explicitly specify axes.
    """
    if norm is not None:
        assert(isinstance(norm, mcolors.Normalize))
    if aspect is None:
        aspect = rcParams['image.aspect']
    axes.set_aspect(aspect)

    if interpolation is None:
        interpolation = 'none'

    im = ModestImage(axes, cmap=cmap, norm=norm, interpolation=interpolation,
                     origin=origin, extent=extent, filternorm=filternorm,
                     filterrad=filterrad, resample=resample, **kwargs)

    im.set_data(X)
    im.set_alpha(alpha)
    axes._set_artist_props(im)

    axes.format_coord = lambda x, y: f'x = {x:,.5f}, y = {y:,.5f}'

    if im.get_clip_path() is None:
        # image does not already have clipping set, clip to axes patch
        im.set_clip_path(axes.patch)

    # if norm is None and shape is None:
    #    im.set_clim(vmin, vmax)
    if vmin is not None or vmax is not None:
        im.set_clim(vmin, vmax)
    elif norm is None:
        im.autoscale_None()

    im.set_url(url)

    # update ax.dataLim, and, if autoscaling, set viewLim
    # to tightly fit the image, regardless of dataLim.
    im.set_extent(im.get_extent())

    axes.images.append(im)
    im._remove_method = lambda h: axes.images.remove(h)

    return im


def extract_matched_slices(axes=None, shape=None, extent=None,
                           transform=IDENTITY_TRANSFORM):
    """Determine the slice parameters to use, matched to the screen.

    :param ax: Axes object to query. It's extent and pixel size
               determine the slice parameters

    :param shape: Tuple of the full image shape to slice into. Upper
               boundaries for slices will be cropped to fit within
               this shape.

    :rtype: tuple of x0, x1, sx, y0, y1, sy

    Indexing the full resolution array as array[y0:y1:sy, x0:x1:sx] returns
    a view well-matched to the axes' resolution and extent
    """
    # Find extent in display pixels (this gives the resolution we need
    # to sample the array to)
    ext = (axes.transAxes.transform([(1, 1)]) -
           axes.transAxes.transform([(0, 0)]))[0]

    # Find the extent of the axes in 'world' coordinates
    xlim, ylim = axes.get_xlim(), axes.get_ylim()
    # breakpoint()

    # Transform the limits to pixel coordinates
    ind0 = transform.transform([min(xlim), min(ylim)])
    ind1 = transform.transform([max(xlim), max(ylim)])

    def _clip(val, lo, hi):
        return int(max(min(val, hi), lo))

    # Determine the range of pixels to extract from the array, including a 5
    # pixel margin all around. We ensure that the shape of the resulting array
    # will always be at least (1, 1) even if there is really no overlap, to
    # avoid issues.
    y0 = _clip(ind0[1] - 5, 0, shape[0] - 1)
    y1 = _clip(ind1[1] + 5, 1, shape[0])
    x0 = _clip(ind0[0] - 5, 0, shape[1] - 1)
    x1 = _clip(ind1[0] + 5, 1, shape[1])

    # Determine the strides that can be used when extracting the array
    sy = int(max(1, min((y1 - y0) / 5., np.ceil(abs((ind1[1] - ind0[1]) /
                                                    ext[1])))))
    sx = int(max(1, min((x1 - x0) / 5., np.ceil(abs((ind1[0] - ind0[0]) /
                                                    ext[0])))))

    return x0, x1, sx, y0, y1, sy


def main2():
    """main2."""
    from pygmi.raster.iodefs import get_raster
    from pygmi.misc import ProgressBarText
    import matplotlib.pyplot as plt
    from IPython import get_ipython
    get_ipython().magic('matplotlib qt5')

    ifile = r'd:\Work\Programming\mpl-modest-image-master\test.tif'
    ifile = r'd:\workdata\testdata.hdr'

    pbar = ProgressBarText()

    data = get_raster(ifile, piter=pbar.iter)

    cdat = data[1].data
    extent = data[1].extent

    f = plt.figure()
    ax = f.add_subplot(121)
    imshow(ax, cdat, extent=extent)
    ax.grid(True)

    plt.subplot(122)
    numrows, numcols = cdat.shape

    im = plt.imshow(cdat, extent=extent)

    im.format_cursor_data = lambda data: f'z = {data:,.5f}'

    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main2()
