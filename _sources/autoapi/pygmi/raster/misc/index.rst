pygmi.raster.misc
=================

.. py:module:: pygmi.raster.misc

.. autoapi-nested-parse::

   Miscellaneous functions for raster data.



Functions
---------

.. autoapisummary::

   pygmi.raster.misc.aspect2
   pygmi.raster.misc.check_dataid
   pygmi.raster.misc.currentshader
   pygmi.raster.misc.cut_raster
   pygmi.raster.misc.histcomp
   pygmi.raster.misc.histeq
   pygmi.raster.misc.img2rgb
   pygmi.raster.misc.lstack
   pygmi.raster.misc.norm2
   pygmi.raster.misc.norm255


Module Contents
---------------

.. py:function:: aspect2(data)

   Aspect of a dataset.

   :param data: input data used for the aspect calculation
   :type data: numpy MxN array

   :returns: * **adeg** (*numpy masked array*) -- aspect in degrees
             * **dzdx** (*numpy array*) -- gradient in x direction
             * **dzdy** (*numpy array*) -- gradient in y direction


.. py:function:: check_dataid(out)

   Check dataid for duplicates and renames where necessary.

   :param out: PyGMI raster data.
   :type out: list of pygmi.raster.datatypes.Data

   :returns: **out** -- PyGMI raster data.
   :rtype: list of pygmi.raster.datatypes.Data


.. py:function:: currentshader(data, cell=1.0, theta=np.pi / 4.0, phi=-np.pi / 4.0, alpha=1.0)

   Blinn shader - used for sun shading.

   :param data: Dataset to be shaded.
   :type data: numpy array
   :param cell: between 1 and 100 - controls sunshade detail.
   :type cell: float
   :param theta: sun elevation (also called g in code below)
   :type theta: float
   :param phi: azimuth
   :type phi: float
   :param alpha: how much incident light is reflected (0 to 1)
   :type alpha: float

   :returns: **R** -- array containing the shaded results.

             self.phi = -np.pi/4.
             self.theta = np.pi/4.
             self.cell = 100.
             self.alpha = .0
   :rtype: numpy array


.. py:function:: cut_raster(data, ibnd, showlog=print, deepcopy=True)

   Cut a raster dataset.

   :param data: PyGMI Dataset
   :type data: list of pygmi.raster.datatypes.Data
   :param ibnd: shapefile or GeoDataFrame used to cut data.
   :type ibnd: str or GeoDataFrame, or tuple of bounds
   :param showlog: Function for printing text. The default is print.
   :type showlog: function, optional
   :param deepcopy: Make a copy of the data array before use.
   :type deepcopy: bool

   :returns: **data** -- PyGMI Dataset
   :rtype: list of pygmi.raster.datatypes.Data


.. py:function:: histcomp(img, perc=5.0, uperc=None)

   Histogram Compaction.

   This compacts a % of the outliers in data, allowing for a cleaner, linear
   representation of the data.

   :param img: data to compact
   :type img: numpy array
   :param perc: percentage of histogram to clip. If uperc is not None, then this is
                the lower percentage, default is 5.
   :type perc: float
   :param uperc: upper percentage to clip. If uperc is None, then it is set to the
                 same value as perc, default is None
   :type uperc: float

   :returns: * **img2** (*numpy array*) -- compacted array
             * **svalue** (*float*) -- Start value
             * **evalue** (*float*) -- End value


.. py:function:: histeq(img, nbrbins=32768)

   Histogram Equalization.

   Equalizes the histogram to colours. This allows for seeing as much data as
   possible in the image, at the expense of knowing the real value of the
   data at a point. It bins the data equally - flattening the distribution.

   :param img: input data to be equalised
   :type img: numpy array
   :param nbrbins: number of bins to be used in the calculation, default is 32768
   :type nbrbins: integer

   :returns: **im2** -- output data
   :rtype: numpy array


.. py:function:: img2rgb(img, cbar=colormaps['jet'])

   Image to RGB.

   convert image to 4 channel rgba colour image.

   :param img: array to be converted to rgba image.
   :type img: numpy array
   :param cbar: colormap to apply to the image, default is jet.
   :type cbar: matplotlib colour map

   :returns: **im2** -- output rgba image
   :rtype: numpy array


.. py:function:: lstack(dat, *, piter=None, dxy=None, showlog=print, commonmask=False, masterid=None, nodeepcopy=False, resampling='nearest', checkdataid=True)

   Layer stack datasets found in a single PyGMI data object.

   The aim is to ensure that all datasets have the same number of rows and
   columns.

   :param dat: data object which stores datasets
   :type dat: list of pygmi.raster.datatypes.Data
   :param piter: Progress bar iterator. The default is None.
   :type piter: function, optional
   :param dxy: Cell size. The default is None.
   :type dxy: float, optional
   :param showlog: Display information. The default is print.
   :type showlog: function, optional
   :param commonmask: Create a common mask for all bands. The default is False.
   :type commonmask: bool, optional
   :param masterid: ID of master dataset. The default is None.
   :type masterid: str, optional
   :param nodeepcopy: Flag to avoid making a copy of the input data, by default False.
   :type nodeepcopy: bool
   :param resampling: The resampling to be used on output date. The default is 'nearest'.
   :type resampling: str
   :param checkdataid: Check to make sure there are no duplicate data ids. The default is True
   :type checkdataid: bool

   :returns: **out** -- data object which stores datasets
   :rtype: list of pygmi.raster.datatypes.Data


.. py:function:: norm2(dat, datmin=None, datmax=None)

   Normalise array vector between 0 and 1.

   :param dat: array to be normalised
   :type dat: numpy array
   :param datmin: data minimum, default is None
   :type datmin: float
   :param datmax: data maximum, default is None
   :type datmax: float

   :returns: **out** -- normalised array
   :rtype: numpy array of floats


.. py:function:: norm255(dat)

   Normalise array vector between 1 and 255.

   :param dat: array to be normalised.
   :type dat: numpy array

   :returns: **out** -- normalised array
   :rtype: numpy array of 8 bit integers


