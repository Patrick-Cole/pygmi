pygmi.raster.modest_image
=========================

.. py:module:: pygmi.raster.modest_image

.. autoapi-nested-parse::

   Modest Image.

   Modification of Chris Beaumont's mpl-modest-image package to allow the use of
   set_extent as well as better integration into PyGMI

   pcole, 2021  - Bug fix to allow for correct zooming if origin is set to 'upper'



Classes
-------

.. autoapisummary::

   pygmi.raster.modest_image.ModestImage


Functions
---------

.. autoapisummary::

   pygmi.raster.modest_image.imshow
   pygmi.raster.modest_image.extract_matched_slices


Module Contents
---------------

.. py:class:: ModestImage(*args, **kwargs)

   Bases: :py:obj:`matplotlib.image.AxesImage`


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


   .. py:method:: set_data(A)

      Set data.

      :param A: A numpy or PIL image.
      :type A: numpy/PIL Image A

      :raises TypeError: Error when data has incorrect dimensions.

      :rtype: None.



   .. py:method:: set_shade(doshade, cell=None, theta=None, phi=None, alpha=None)

      Set the shade information.

      :param doshade: Check for whether to shade or not.
      :type doshade: bool
      :param cell: Sunshade detail, between 1 and 100. The default is None.
      :type cell: float, optional
      :param theta: Sun inclination or elevation. The default is None.
      :type theta: float, optional
      :param phi: Sun declination or azimuth. The default is None.
      :type phi: float, optional
      :param alpha: Light reflectance, between 0 and 1. The default is None.
      :type alpha: float, optional

      :rtype: None.



   .. py:method:: invalidate_cache()

      Invalidate cache.

      :rtype: None.



   .. py:method:: set_extent(extent, **kwargs)

      Set extent.

      :param extent: Extent of data.
      :type extent: tuple

      :rtype: None.



   .. py:method:: get_array()

      Override to return the full-resolution array.

      :returns: Return data array of full resolution.
      :rtype: numpy array



   .. py:method:: get_cursor_data(event)

      Correct z-value display when zoomed.

      :param event: Cursor event.
      :type event: matpltolib cursor event.

      :returns: z-value or NAN.
      :rtype: float



   .. py:method:: format_cursor_data(data)

      Format z data on graph.

      :param data: Data value to display.
      :type data: float

      :returns: **zval** -- Formatted string to display.
      :rtype: str



   .. py:method:: draw(renderer, *args, **kwargs)

      Draw.



   .. py:method:: draw_ternary()

      Draw ternary.

      :rtype: None.



   .. py:method:: draw_sunshade(colormap=None)

      Apply sunshading.

      :rtype: None.



   .. py:method:: set_clim_std(mult)

      Set the vmin and vmax to mult*std(self._A).

      This routine only works on a 2D array.

      :param mult: Multiplier.
      :type mult: float

      :rtype: None.



.. py:function:: imshow(axes, X, cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=None, vmax=None, origin=None, extent=None, shape=None, filternorm=1, filterrad=4.0, imlim=None, resample=None, url=None, suncell=None, suntheta=None, sunphi=None, sunalpha=None, **kwargs)

   Similar to matplotlib's imshow command, but produces a ModestImage.

   Unlike matplotlib version, must explicitly specify axes.


.. py:function:: extract_matched_slices(axes=None, shape=None, transform=IDENTITY_TRANSFORM)

   Determine the slice parameters to use, matched to the screen.

   Indexing the full resolution array as array[y0:y1:sy, x0:x1:sx] returns
   a view well-matched to the axes' resolution and extent

   :param axes: Axes object to query. It's extent and pixel size determine the slice
                parameters. The default is None.
   :type axes: Axes, optional
   :param shape: Tuple of the full image shape to slice into. Upper boundaries for
                 slices will be cropped to fit within this shape. The default is None.
   :type shape: tuple, optional
   :param transform: Rasterio transform. The default is IDENTITY_TRANSFORM.
   :type transform: rasterio transform, optional

   :returns: * **x0** (*int*) -- x minimum.
             * **x1** (*int*) -- x maximum.
             * **sx** (*int*) -- x stride.
             * **y0** (*int*) -- y minimum.
             * **y1** (*int*) -- y maximum.
             * **sy** (*int*) -- y stride.


