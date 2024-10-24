"""
Modest IO Image.

Modification of Chris Beaumont's mpl-modest-image package to allow the use of
set_extent as well as better integration into PyGMI. It is changed to read data
directly from disk.
"""
from __future__ import print_function, division

import numpy as np

from matplotlib import rcParams
import matplotlib.image as mi
import matplotlib.colors as mcolors
from matplotlib.transforms import IdentityTransform, Affine2D

from pygmi.raster.iodefs import get_raster
from pygmi.raster.misc import currentshader, histeq, norm2

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
        super().__init__(*args, **kwargs)
        self.invalidate_cache()

        # Custom lines for PyGMI
        self.shade = None
        self.rgbmode = ''  # Can be None, CMY Ternary or RGB Ternary
        # self.rgbclip = [[None, None], [None, None], [None, None]]
        self.rgbclip = None
        self.dohisteq = False
        self.kval = 0.01  # For CMYK Ternary
        self.dmeta = None
        self.piter = iter
        self.showlog = print

    def set_data(self, A):
        """
        Set data.

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
        if len(A) == 3:
            fshape = (A[0].meta['height'], A[0].meta['width'], 4)
        else:
            fshape = (A[0].meta['height'], A[0].meta['width'])

        self._full_res = np.zeros(fshape)
        self._A = self._full_res
        self.dmeta = A

        # if self._A.dtype != np.uint8 and not np.can_cast(self._A.dtype,
        #                                                  float):
        #     raise TypeError("Image data can not convert to float")

        # if self._A.ndim not in (2, 3):
        #     raise TypeError("Invalid dimensions for image data")
        # if (self._A.ndim == 3 and self._A.shape[-1] not in (3, 4) and
        #         self.shade is False):
        #     raise TypeError("Invalid dimensions for image data")

        self.invalidate_cache()

    def set_shade(self, doshade, cell=None, theta=None, phi=None, alpha=None):
        """
        Set the shade information.

        Parameters
        ----------
        doshade : bool
            Check for whether to shade or not.
        cell : float, optional
            Sunshade detail, between 1 and 100. The default is None.
        theta : float, optional
            Sun inclination or elevation. The default is None.
        phi : float, optional
            Sun declination or azimuth. The default is None.
        alpha : float, optional
            Light reflectance, between 0 and 1. The default is None.

        Returns
        -------
        None.

        """
        if doshade is True:
            self.shade = [cell, theta, phi, alpha]
            if self._A.ndim == 2:
                tmp = np.ma.stack([self._A, self._A])
                tmp = np.moveaxis(tmp, 0, -1)
                self.set_data(tmp)
        else:
            self.shade = None

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

    def set_extent(self, extent, **kwargs):
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
        event : matplotlib cursor event.
            Cursor event.

        Returns
        -------
        float
            z-value or NAN.

        """
        x = event.xdata
        y = event.ydata

        col, row = self._world2pixel.transform((x, y))
        col = int(col + 0.5)
        row = int(row + 0.5)

        numrows, numcols = self._full_res.shape[:2]

        if 0 <= col < numcols and 0 <= row < numrows:
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

        ytot = self._full_res.shape[0]

        out_shape = ((y1-y0)//sy, (x1-x0)//sx)
        iraster = (x0, ytot-y1, (x1-x0), (y1-y0))
        tnames = [i.dataid for i in self.dmeta]
        ifile = self.dmeta[0].filename

        dat = get_raster(ifile, out_shape=out_shape, iraster=iraster,
                         tnames=tnames)

        datd = {}
        for i in dat:
            datd[i.dataid] = i

        if len(tnames) == 3:
            red = datd[tnames[0]].data
            green = datd[tnames[1]].data
            blue = datd[tnames[2]].data

            A = np.ma.array([red, green, blue])
            A = np.moveaxis(A, 0, 2)
            self._A = A

            if self.rgbclip is None:
                self.rgbclip = []
                lclip, uclip = np.percentile(red.compressed(), [1, 99])
                self.rgbclip.append([lclip, uclip])
                lclip, uclip = np.percentile(green.compressed(), [1, 99])
                self.rgbclip.append([lclip, uclip])
                lclip, uclip = np.percentile(blue.compressed(), [1, 99])
                self.rgbclip.append([lclip, uclip])
        else:
            self._A = datd[tnames[0]].data
            self.rgbclip = []
            lclip, uclip = np.percentile(self._A.compressed(), [1, 99])
            self.rgbclip.append([lclip, uclip])
            self.rgbclip.append([lclip, uclip])
            self.rgbclip.append([lclip, uclip])

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
            self.set_clim(0, 1)

        if 'Ternary' in self.rgbmode:
            colormap = self.draw_ternary()
        else:
            colormap = norm2(self._A, self.rgbclip[0][0], self.rgbclip[0][1])

        if self.shade is not None:
            colormap = self.draw_sunshade(colormap)

        self._A = colormap

        super().draw(renderer, *args, **kwargs)

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
        self.set_clim(vmin, vmax)


def imshow(axes, X, cmap=None, norm=None, aspect=None,
           interpolation=None, alpha=None, vmin=None, vmax=None,
           origin=None, extent=None, shape=None, filternorm=1,
           filterrad=4.0, imlim=None, resample=None, url=None,
           suncell=None, suntheta=None, sunphi=None, sunalpha=None,
           piter=iter, showlog=print, **kwargs):
    """
    Similar to matplotlib's imshow command, but produces a ModestImage.

    Unlike matplotlib version, must explicitly specify axes.
    """
    if norm is not None:
        assert isinstance(norm, mcolors.Normalize)
    if aspect is None:
        aspect = rcParams['image.aspect']
    axes.set_aspect(aspect)

    if interpolation is None:
        interpolation = 'none'

    im = ModestImage(axes, cmap=cmap, norm=norm, interpolation=interpolation,
                     origin=origin, extent=extent, filternorm=filternorm,
                     filterrad=filterrad, resample=resample, **kwargs)

    im.piter = piter
    im.showlog = showlog
    im.set_data(X)
    im.set_alpha(alpha)

    axes._set_artist_props(im)

    axes.format_coord = lambda x, y: f'x = {x:,.5f}, y = {y:,.5f}'

    if im.get_clip_path() is None:
        # image does not already have clipping set, clip to axes patch
        im.set_clip_path(axes.patch)

    if vmin is not None or vmax is not None:
        im.set_clim(vmin, vmax)
        im.set_clim(vmin, vmax)
    # elif norm is None:
    #     im.autoscale_None()

    if suncell is not None:
        im.set_shade(True, suncell, suntheta, sunphi, sunalpha)

    im.set_url(url)

    # update ax.dataLim, and, if autoscaling, set viewLim
    # to tightly fit the image, regardless of dataLim.
    im.set_extent(im.get_extent())

    # axes.images.append(im)
    axes.add_image(im)
    im._remove_method = lambda h: axes.images.remove(h)

    return im


def extract_matched_slices(axes=None, shape=None,
                           transform=IDENTITY_TRANSFORM):
    """
    Determine the slice parameters to use, matched to the screen.

    Indexing the full resolution array as array[y0:y1:sy, x0:x1:sx] returns
    a view well-matched to the axes' resolution and extent

    Parameters
    ----------
    axes : Axes, optional
        Axes object to query. It's extent and pixel size determine the slice
        parameters. The default is None.
    shape : tuple, optional
        Tuple of the full image shape to slice into. Upper boundaries for
        slices will be cropped to fit within this shape. The default is None.
    transform : rasterio transform, optional
        Rasterio transform. The default is IDENTITY_TRANSFORM.

    Returns
    -------
    x0 : int
        x minimum.
    x1 : int
        x maximum.
    sx : int
        x stride.
    y0 : int
        y minimum.
    y1 : int
        y maximum.
    sy : int
        y stride.

    """
    # Find extent in display pixels (this gives the resolution we need
    # to sample the array to)
    ext = (axes.transAxes.transform([(1, 1)]) -
           axes.transAxes.transform([(0, 0)]))[0]

    # Find the extent of the axes in 'world' coordinates
    xlim, ylim = axes.get_xlim(), axes.get_ylim()

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
