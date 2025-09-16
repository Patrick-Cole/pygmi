pygmi.raster.datatypes
======================

.. py:module:: pygmi.raster.datatypes

.. autoapi-nested-parse::

   Class for raster data types and conversion routines.



Classes
-------

.. autoapisummary::

   pygmi.raster.datatypes.Data
   pygmi.raster.datatypes.RasterMeta


Functions
---------

.. autoapisummary::

   pygmi.raster.datatypes.numpy_to_pygmi
   pygmi.raster.datatypes.pygmi_to_numpy
   pygmi.raster.datatypes.bounds_to_transform
   pygmi.raster.datatypes.bounds_intersection


Module Contents
---------------

.. py:function:: numpy_to_pygmi(data, pdata=None, dataid=None)

   Convert an MxN numpy array into a PyGMI data object.

   For convenience, if pdata is defined, parameters from another dataset
   will be used (such as xdim, ydim etc).

   :param data: MxN array
   :type data: numpy array
   :param pdata: PyGMI raster dataset
   :type pdata: pygmi.raster.datatypes.Data
   :param dataid: name for the band of data.
   :type dataid: str or None

   :returns: **tmp** -- PyGMI raster dataset
   :rtype: pygmi.raster.datatypes.Data


.. py:function:: pygmi_to_numpy(tmp)

   Convert a PyGMI data object into an MxN numpy array.

   :param tmp: PyGMI raster dataset
   :type tmp: pygmi.raster.datatypes.Data

   :returns: MxN numpy array
   :rtype: numpy array


.. py:function:: bounds_to_transform(bounds, dxy)

   Create a raster transform from vector grid bounds and dxy.

   This accounts for the situation where xmax and ymax need to be readjusted
   slightly because dxy does not divide perfectly into bounds. It also adds
   dxy/2 buffer. Therefore it cannot be used with raster bounds.

   :param bounds: Bounds of data as (left, bottom, right, top)
   :type bounds: tuple
   :param dxy: Raster pixel size.
   :type dxy: float

   :returns: * **transform** (*list of Affine*) -- rasterio transform.
             * **shape** (*tuple*) -- tuple of rows, cols.


.. py:function:: bounds_intersection(dataset, bounds, showlog=print)

   Find the intersection between some bounds and a dataset.

   :param dataset: Rasterio dataset.
   :type dataset: rasterio dataset
   :param bounds: Bounds of data as (left, bottom, right, top).
   :type bounds: tuple
   :param showlog: Display information. The default is print.
   :type showlog: function, optional

   :returns: * **window** (*rasterio window*) -- Intersection area as window.
             * **newbounds** (*tuple*) -- Intersection area as bounds.


.. py:class:: Data

   PyGMI Data Object.

   .. attribute:: data

      array to contain raster data

      :type: numpy masked array

   .. attribute:: extent

      Extent of data as (left, right, bottom, top)

      :type: tuple

   .. attribute:: bounds

      Bounds of data as (left, bottom, right, top)

      :type: tuple

   .. attribute:: xdim

      x-dimension of grid cell

      :type: float

   .. attribute:: ydim

      y-dimension of grid cell

      :type: float

   .. attribute:: dataid

      band name or id

      :type: str

   .. attribute:: nodata

      grid null or no data value

      :type: float

   .. attribute:: units

      description of units to be used with colour bars

      :type: str

   .. attribute:: isrgb

      Flag to signify an RGB image.

      :type: bool

   .. attribute:: metadata

      Miscellaneous metadata for file.

      :type: dictionary

   .. attribute:: meta

      Rasterio metadata for file.

      :type: dictionary

   .. attribute:: filename

      Filename of file.

      :type: str

   .. attribute:: transform

      rasterio transform. The default is None.

      :type: list of Affine, optional

   .. attribute:: crs

      rasterio crs of data

      :type: CRS

   .. attribute:: datetime

      Date of dataset.

      :type: date


   .. py:method:: copy(data0=None, resetmeta=False)

      Make a deepcopy of the function.

      :param data0: Input data to replace old ddata. Must have same shape.
      :type data0: numpy arraay
      :param resetmeta: This will clear metadata during copy. The default is False.
      :type resetmeta: bool, optional

      :returns: **data** -- PyGMI data type.
      :rtype: pygmi.raster.datatypes.Data



   .. py:method:: in_bounds(bounds)

      Check if dataset is in bounds supplied.

      :param bounds: Bounds of data as (left, bottom, right, top)
      :type bounds: tuple

      :returns: True if within bounds, otherwise False.
      :rtype: bool



   .. py:method:: meta_from_rasterio(dataset, bounds=None)

      Set transform, bounds, extent, xdim and ydim from a rasterio dataset.

      :param dataset: Rasterio dataset.
      :type dataset: rasterio dataset
      :param bounds: Bounds of data as (left, bottom, right, top). The default is None.
      :type bounds: tuple, optional

      :rtype: None.



   .. py:method:: modify_mask(mask, oper='or')

      Modify the existing mask with a new one.

      The routine also fills the masked areas with nodata.

      :param mask: Boolean array of new mask to modify old one.
      :type mask: array
      :param oper: Logical operation to be performed between masks. Can be 'or' or
                   'and'. The default is 'or'.
      :type oper: str, optional

      :rtype: None.



   .. py:method:: plot(ax)

      Simple data plot.



   .. py:method:: set_mask(mask=None)

      Replace the existing mask with a new one.

      The routine also fills the masked areas with nodata.

      :param mask: Boolean array of new mask to modify old one.
      :type mask: array

      :rtype: None.



   .. py:method:: set_transform(xdim=None, xmin=None, ydim=None, ymax=None, transform=None, iraster=None, rows=None, cols=None)

      Set the transform, xdim, ydim, extent and bounds.

      This requires either transform as input OR xdim, ydim, xmin, ymax.

      :param xdim: x dimension. The default is None.
      :type xdim: float, optional
      :param xmin: x minimum. The default is None.
      :type xmin: float, optional
      :param ydim: y dimension. The default is None.
      :type ydim: float, optional
      :param ymax: y maximum. The default is None.
      :type ymax: float, optional
      :param transform: transform. The default is None.
      :type transform: list of Affine, optional
      :param iraster: Incremental raster import, to import a section of a file.
                      The tuple is (xoff, yoff, xsize, ysize). The default is None.
      :type iraster: None or tuple
      :param rows: rows in dataset. The default is None.
      :type rows: int, optional
      :param cols: columns in dataset. The default is None.
      :type cols: int, optional

      :rtype: None.



   .. py:method:: to_mem()

      Create a rasterio memory file from one band.

      :returns: **raster** -- rasterio memory file.
      :rtype: MemoryFile



   .. py:method:: get_vmin_vmax(std=2.5)

      Get vmin and vmax for use in imshow.

      :param std: Multiplier for standard deviations to include about mean.
                  The default is 2.5.
      :type std: float, optional

      :returns: * **vmin** (*float*) -- Value minimum.
                * **vmax** (*float*) -- Value maximum.



   .. py:method:: get_boundary()

      Get raster boundary.



.. py:class:: RasterMeta

   PyGMI Raster Metadata Object.

   .. attribute:: sensor

      Sensor used to measure data.

      :type: str

   .. attribute:: filename

      Filename of file.

      :type: str

   .. attribute:: crs

      rasterio crs of data.

      :type: CRS

   .. attribute:: bands

      list of bands in dataset.

      :type: list

   .. attribute:: tnames

      list fo bands to process.

      :type: list

   .. attribute:: banddata

      list of band data.

      :type: list

   .. attribute:: to_sutm

      flag to convert a file to SUTM.

      :type: bool


   .. py:method:: fromData(dat)

      Populate class from a Data class.

      :param dat: PyGMI data object.
      :type dat: pygmi.raster.datatypes.Data

      :rtype: None.



