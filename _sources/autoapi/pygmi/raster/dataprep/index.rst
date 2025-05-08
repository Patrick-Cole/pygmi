pygmi.raster.dataprep
=====================

.. py:module:: pygmi.raster.dataprep

.. autoapi-nested-parse::

   A set of Raster Data Preparation routines.



Classes
-------

.. autoapisummary::

   pygmi.raster.dataprep.Continuation
   pygmi.raster.dataprep.DataCut
   pygmi.raster.dataprep.DataLayerStack
   pygmi.raster.dataprep.DataMerge
   pygmi.raster.dataprep.DataReproj
   pygmi.raster.dataprep.GetProf
   pygmi.raster.dataprep.Metadata


Functions
---------

.. autoapisummary::

   pygmi.raster.dataprep.cluster_to_raster
   pygmi.raster.dataprep.fftprep
   pygmi.raster.dataprep.fft_getkxy
   pygmi.raster.dataprep.fftcont
   pygmi.raster.dataprep.get_shape_bounds
   pygmi.raster.dataprep.merge_median
   pygmi.raster.dataprep.merge_min
   pygmi.raster.dataprep.merge_max
   pygmi.raster.dataprep.mosaic
   pygmi.raster.dataprep.redistribute_vertices
   pygmi.raster.dataprep.taylorcont
   pygmi.raster.dataprep.trim_raster
   pygmi.raster.dataprep.verticalp


Module Contents
---------------

.. py:class:: Continuation(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   GUI to perform upward and downward continuation on potential field data.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: settings(nodialog=False)

      Entry point into item.

      :param nodialog: Run settings without a dialog. The default is False.
      :type nodialog: bool, optional

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: saveproj()

      Save project data from class.

      :rtype: None.



   .. py:method:: acceptall()

      Accept option.

      Updates self.outdata, which is used as input to other modules.

      :rtype: None.



.. py:class:: DataCut(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   GUI to Cut Data using shapefiles.

   This class cuts raster datasets using a boundary defined by a polygon
   shapefile.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: settings(nodialog=False)

      Entry point into item.

      :param nodialog: Run settings without a dialog. The default is False.
      :type nodialog: bool, optional

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: saveproj()

      Save project data from class.

      :rtype: None.



.. py:class:: DataLayerStack(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Data Layer Stack GUI.

   This class merges datasets which have different rows and columns. It
   resamples them so that they have the same rows and columns.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: dxy_change()

      Update dxy.

      This is the size of a grid cell in the x and y directions.

      :rtype: None.



   .. py:method:: settings(nodialog=False)

      Entry point into item.

      :param nodialog: Run settings without a dialog. The default is False.
      :type nodialog: bool, optional

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: saveproj()

      Save project data from class.

      :rtype: None.



   .. py:method:: acceptall()

      Accept option.

      Updates self.outdata, which is used as input to other modules.

      :rtype: None.



.. py:class:: DataMerge(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Data merge or mosaic GUI.

   This class merges datasets which have different rows and columns. It
   resamples them so that they have the same rows and columns.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: method_change()

      Change method.

      :rtype: None.



   .. py:method:: get_idir()

      Get the input directory.

      :rtype: None.



   .. py:method:: get_sfile()

      Get the input shapefile.

      :rtype: None.



   .. py:method:: settings(nodialog=False)

      Entry point into item.

      :param nodialog: Run settings without a dialog. The default is False.
      :type nodialog: bool, optional

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: saveproj()

      Save project data from class.

      :rtype: None.



   .. py:method:: merge_different()

      Merge files with different numbers of bands and/or band order.

      This uses more memory, but is flexible.

      :returns: Success of routine.
      :rtype: bool



.. py:class:: DataReproj(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Raster reprojection GUI.

   This class reprojects datasets using the rasterio routines.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: acceptall()

      Accept option.

      Updates self.outdata, which is used as input to other modules.

      :rtype: None.



   .. py:method:: settings(nodialog=False)

      Entry point into item.

      :param nodialog: Run settings without a dialog. The default is False.
      :type nodialog: bool, optional

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: saveproj()

      Save project data from class.

      :rtype: None.



.. py:class:: GetProf(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   GUI to extract a profile from a raster dataset.

   This class extracts a profile from a raster dataset using a line shapefile.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: settings(nodialog=False)

      Entry point into item.

      :param nodialog: Run settings without a dialog. The default is False.
      :type nodialog: bool, optional

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: saveproj()

      Save project data from class.

      :rtype: None.



.. py:class:: Metadata(parent=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   Edit raster metadata.

   This class allows the editing of the metadata for a raster dataset using a
   GUI.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional

   .. attribute:: banddata

      band data

      :type: dictionary

   .. attribute:: bandid

      dictionary of strings containing band names.

      :type: dictionary


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: acceptall()

      Accept option.

      :rtype: None.



   .. py:method:: rename_id()

      Rename the band name.

      :rtype: None.



   .. py:method:: update_vals()

      Update the values on the interface.

      :rtype: None.



   .. py:method:: run()

      Entry point into the routine, used to run context menu item.

      :returns: **tmp** -- True if successful, False otherwise.
      :rtype: bool



.. py:function:: cluster_to_raster(indata)

   Convert cluster datasets to raster datasets.

   Some routines will not understand the datasets produced by cluster
   analysis routines, since they are designated 'Cluster' and not 'Raster'.
   This provides a work-around for that.

   :param indata: Dictionary of PyGMI datasets.
   :type indata: dict

   :returns: **indata** -- Dictionary of PyGMI datasets.
   :rtype: dict


.. py:function:: fftprep(data)

   FFT preparation.

   :param data: Input dataset.
   :type data: pygmi.raster.datatypes.Data

   :returns: * **zfin** (*numpy array.*) -- Output prepared data.
             * **rdiff** (*int*) -- rows divided by 2.
             * **cdiff** (*int*) -- columns divided by 2.
             * **datamedian** (*float*) -- Median of data.


.. py:function:: fft_getkxy(fftmod, xdim, ydim)

   Get KX and KY.

   :param fftmod: FFT data.
   :type fftmod: numpy array
   :param xdim: cell x dimension.
   :type xdim: float
   :param ydim: cell y dimension.
   :type ydim: float

   :returns: * **KX** (*numpy array*) -- x sample frequencies.
             * **KY** (*numpy array*) -- y sample frequencies.


.. py:function:: fftcont(data, h)

   Continuation.

   :param data: PyGMI raster data.
   :type data: pygmi.raster.datatypes.Data
   :param h: Height.
   :type h: float

   :returns: **dat** -- PyGMI raster data.
   :rtype: pygmi.raster.datatypes.Data


.. py:function:: get_shape_bounds(sfile, crs=None, showlog=print)

   Get bounds from a shape file.

   :param sfile: Filename for shapefile.
   :type sfile: str
   :param crs: target crs for shapefile
   :type crs: rasterio CRS
   :param showlog: Display information. The default is print.
   :type showlog: function, optional

   :returns: **bounds** -- Rasterio bounds.
   :rtype: list


.. py:function:: merge_median(merged_data, new_data, merged_mask, new_mask, index=None, roff=None, coff=None)

   Merge using median for rasterio, taking minimum value.

   :param merged_data: Old data.
   :type merged_data: numpy array
   :param new_data: New data to merge to old data.
   :type new_data: numpy array
   :param merged_mask: Old mask.
   :type merged_mask: float
   :param new_mask: New mask.
   :type new_mask: float
   :param index: index of the current dataset within the merged dataset collection.
                 The default is None.
   :type index: int, optional
   :param roff: row offset in base array. The default is None.
   :type roff: int, optional
   :param coff: col offset in base array. The default is None.
   :type coff: int, optional

   :rtype: None.


.. py:function:: merge_min(merged_data, new_data, merged_mask, new_mask, index=None, roff=None, coff=None)

   Merge using minimum for rasterio, taking minimum value.

   :param merged_data: Old data.
   :type merged_data: numpy array
   :param new_data: New data to merge to old data.
   :type new_data: numpy array
   :param merged_mask: Old mask.
   :type merged_mask: float
   :param new_mask: New mask.
   :type new_mask: float
   :param index: index of the current dataset within the merged dataset collection.
                 The default is None.
   :type index: int, optional
   :param roff: row offset in base array. The default is None.
   :type roff: int, optional
   :param coff: col offset in base array. The default is None.
   :type coff: int, optional

   :rtype: None.


.. py:function:: merge_max(merged_data, new_data, merged_mask, new_mask, index=None, roff=None, coff=None)

   Merge using maximum for rasterio, taking maximum value.

   :param merged_data: Old data.
   :type merged_data: numpy array
   :param new_data: New data to merge to old data.
   :type new_data: numpy array
   :param merged_mask: Old mask.
   :type merged_mask: float
   :param new_mask: New mask.
   :type new_mask: float
   :param index: index of the current dataset within the merged dataset collection.
                 The default is None.
   :type index: int, optional
   :param roff: row offset in base array. The default is None.
   :type roff: int, optional
   :param coff: col offset in base array. The default is None.
   :type coff: int, optional

   :rtype: None.


.. py:function:: mosaic(dat, *, idir=None, bfile=None, bandstofiles=False, piter=iter, showlog=print, singleband=False, forcetype=None, shifttomedian=False, tmpdir=None, nodata=None, method='first', res=None)

   Merge files with different numbers of bands and/or band order.

   This uses more memory, but is flexible.

   :param dat: List of PyGMI data bands to be merged. Can be empty if idir is provided.
   :type dat: list
   :param idir: Directory where file to be mosaiced are found. The default is None.
   :type idir: str, optional
   :param bfile: Path to boundary file. Can be shapefile or raster. The default is None.
   :type bfile: str, optional
   :param bandstofiles: Export output bands to files. The default is False.
   :type bandstofiles: bool, optional
   :param piter: Progress bar iterable. The default is iter.
   :type piter: function, optional
   :param showlog: Function for printing text. The default is print.
   :type showlog: function, optional
   :param singleband: Ignore band names, since there is only one band. The default is False.
   :type singleband: bool, optional
   :param forcetype: Force input data type. The default is None.
   :type forcetype: bool, optional
   :param shifttomedian: Shift bands to median value. The default is False.
   :type shifttomedian: bool, optional
   :param tmpdir: Alternate directory for temporary files. The default is None.
   :type tmpdir: str, optional
   :param nodata: Nodata value. The default is None.
   :type nodata: float, optional
   :param method: Mosaic method. Can be 'first', 'last', 'merge_min', 'merge_max' or
                  'merge_median. The default is 'first'.
   :type method: str, optional
   :param res: Output resolution. Can be a tuple. The default is None.
   :type res: float, optional

   :returns: **outdat** -- Output mosaiced dataset.
   :rtype: PyGMI raster data


.. py:function:: redistribute_vertices(geom, distance)

   Redistribute vertices in a geometry.

   From https://stackoverflow.com/questions/34906124/interpolating-every-x-distance-along-multiline-in-shapely,
   and by Mike-T.

   :param geom: Geometry from geopandas.
   :type geom: shapely geometry
   :param distance: sampling distance.
   :type distance: float

   :raises ValueError: Error when there is an unknown geometry.

   :returns: New geometry.
   :rtype: shapely geometry


.. py:function:: taylorcont(data, h)

   Taylor Continuation.

   :param data: PyGMI raster data.
   :type data: pygmi.raster.datatypes.Data
   :param h: Height.
   :type h: float

   :returns: **dat** -- PyGMI raster data.
   :rtype: pygmi.raster.datatypes.Data


.. py:function:: trim_raster(olddata)

   Trim nulls from a raster dataset.

   This function trims entire rows or columns of data which are masked,
   and are on the edges of the dataset. Masked values are set to the null
   value.

   :param olddata: PyGMI dataset.
   :type olddata: list of pygmi.raster.datatypes.Data

   :returns: **olddata** -- PyGMI dataset.
   :rtype: list of pygmi.raster.datatypes.Data


.. py:function:: verticalp(data, order=1)

   Vertical derivative.

   :param data: Input data.
   :type data: numpy array
   :param order: Order. The default is 1.
   :type order: float, optional

   :returns: **dout** -- Output data
   :rtype: numpy array


