pygmi.raster.iodefs
===================

.. py:module:: pygmi.raster.iodefs

.. autoapi-nested-parse::

   Import and export routines for raster data.



Classes
-------

.. autoapisummary::

   pygmi.raster.iodefs.BandSelect
   pygmi.raster.iodefs.ImportData
   pygmi.raster.iodefs.ImportRGBData
   pygmi.raster.iodefs.ExportData


Functions
---------

.. autoapisummary::

   pygmi.raster.iodefs.clusterprep
   pygmi.raster.iodefs.get_ascii
   pygmi.raster.iodefs.get_raster
   pygmi.raster.iodefs.get_bil
   pygmi.raster.iodefs.get_geopak
   pygmi.raster.iodefs.get_geosoft
   pygmi.raster.iodefs.export_raster
   pygmi.raster.iodefs.calccov


Module Contents
---------------

.. py:class:: BandSelect(parent=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   A combobox to select data bands.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional


   .. py:method:: run()

      Entry point into the routine, used to run context menu item.

      :returns: True if successful, False otherwise.
      :rtype: bool



.. py:class:: ImportData(parent=None, ifile='', filt='')

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Import Data GUI - Interfaces with rasterio routines.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional
   :param ifile: Input file. The default is ''.
   :type ifile: str, optional
   :param filt: File filter. The default is ''.
   :type filt: str, optional


   .. py:method:: settings(nodialog=False)

      Entry point into item.

      :param nodialog: Run settings without a dialog. The default is False.
      :type nodialog: bool, optional

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: saveproj()

      Save project data from class.

      :rtype: None



.. py:class:: ImportRGBData(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Import RGB Image GUI- Interfaces with rasterio routines.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional


   .. py:method:: settings(nodialog=False)

      Entry point into item.

      :param nodialog: Run settings without a dialog. The default is False.
      :type nodialog: bool, optional

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: saveproj()

      Save project data from class.

      :rtype: None.



.. py:function:: clusterprep(dat)

   Prepare Cluster data from raster data.

   :param dat: List of PyGMI datasets.
   :type dat: list of pygmi.raster.datatypes.Data

   :returns: **dat2** -- List of PyGMI datasets.
   :rtype: list of pygmi.raster.datatypes.Data


.. py:function:: get_ascii(ifile)

   Import ascii raster dataset.

   :param ifile: filename to import
   :type ifile: str

   :returns: **dat** -- dataset imported
   :rtype: PyGMI raster Data


.. py:function:: get_raster(ifile, *, nval=None, piter=None, showlog=print, iraster=None, driver=None, bounds=None, tnames=None, metaonly=False, out_shape=None)

   Get raster dataset.

   This function loads a raster dataset off the disk using the rasterio
   libraries. It returns the data in a PyGMI data object.

   :param ifile: filename to import
   :type ifile: str
   :param nval: Nodata/null value. The default is None.
   :type nval: float, optional
   :param piter: progress bar iterable, default is None.
   :type piter: function, optional
   :param showlog: Routine to show text messages. The default is print.
   :type showlog: function, optional
   :param iraster: Incremental raster import, to import a section of a file.
                   The tuple is (xoff, yoff, xsize, ysize). The default is None.
   :type iraster: None or tuple
   :param driver: GDAL raster driver name. The default is None.
   :type driver: str
   :param bounds: Bounds of data to import as (left, bottom, right, top)
   :type bounds: tuple
   :param tnames: list of band names to import, in order. The default is None.
   :type tnames: list, optional
   :param metaonly: Retrieve only the metadata for the file. The default is False.
   :type metaonly: bool, optional
   :param out_shape: Tuple describing the output array's shape.
   :type out_shape: tuple, optional

   :returns: **dat** -- Raster dataset imported
   :rtype: list of pygmi.raster.datatypes.Data


.. py:function:: get_bil(ifile, bands, cols, rows, dtype, *, piter=iter, iraster=None, interleave='LINE')

   Get BIL format file.

   This routine is called from get_raster

   :param ifile: filename to import
   :type ifile: str
   :param bands: Number of bands.
   :type bands: int
   :param cols: Number of columns.
   :type cols: int
   :param rows: Number of rows.
   :type rows: int
   :param dtype: Data type.
   :type dtype: data type
   :param piter: progress bar iterable.
   :type piter: function
   :param iraster: Incremental raster import, to import a section of a file.
                   The tuple is (xoff, yoff, xsize, ysize). The default is None.
   :type iraster: None or tuple
   :param interleave: Band interleave. Default is 'LINE'
   :type interleave: str

   :returns: **datin** -- dataset imported
   :rtype: PyGMI raster Data


.. py:function:: get_geopak(hfile)

   Geopak Import.

   :param hfile: filename to import
   :type hfile: str

   :returns: **dat** -- PyGMI raster dataset.
   :rtype: list of pygmi.raster.datatypes.Data


.. py:function:: get_geosoft(hfile)

   Get Geosoft file (uncompressed).

   :param hfile: filename to import
   :type hfile: str

   :returns: **dat** -- Dataset imported
   :rtype: list of pygmi.raster.datatypes.Data


.. py:class:: ExportData(parent=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   Export Data GUI.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional

   .. attribute:: ofile

      output file name.

      :type: str


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: run(option=None)

      Entry point into the routine, used to run context menu item.

      :returns: * *bool* -- True if successful, False otherwise.
                * **option** (*str*) -- A string option. The default is None.



   .. py:method:: acceptall()

      Accept choice.



   .. py:method:: export_ubc(data)

      Export a section to a 3D UBC mesh and model.

      :param data: dataset to export
      :type data: PyGMI raster Data

      :rtype: None.



   .. py:method:: export_gxf(data)

      Export GXF data.

      :param data: dataset to export
      :type data: PyGMI raster Data

      :rtype: None.



   .. py:method:: export_surfer(data)

      Routine to export a surfer binary grid.

      :param data: dataset to export
      :type data: PyGMI raster Data

      :rtype: None.



   .. py:method:: export_ascii(data)

      Export ASCII file.

      :param data: dataset to export
      :type data: PyGMI raster Data

      :rtype: None.



   .. py:method:: export_ascii_xyz(data)

      Export and xyz file.

      :param data: dataset to export
      :type data: PyGMI raster Data

      :rtype: None.



   .. py:method:: get_filename(data, ext)

      Get a valid filename in the case of multi band image.

      :param data: dataset to get filename from
      :type data: PyGMI raster Data
      :param ext: filename extension to use
      :type ext: str

      :returns: **file_out** -- Output filename.
      :rtype: str



   .. py:method:: get_ofile()

      Get output directory.



.. py:function:: export_raster(ofile, dat, *, drv='GTiff', piter=None, compression='NONE', bandsort=True, showlog=print, updatestats=True)

   Export to rasterio format.

   :param ofile: Output file name.
   :type ofile: str
   :param dat: dataset to export
   :type dat: list or dictionary of PyGMI raster Data
   :param drv: name of the rasterio driver to use
   :type drv: str
   :param piter: Progressbar iterable. The default is None.
   :type piter: function, optional
   :param compression: Compression for GeoTIFF. Can be NONE, DEFLATE or ZSTD. The default is
                       NONE.
   :type compression: str, optional
   :param bandsort: sort the bands by dataid. The default is True
   :type bandsort: bool, optional
   :param showlog: Show information using a function. The default is print.
   :type showlog: function, optional
   :param updatestats: Update statistics in exported file.
   :type updatestats: bool, optional

   :rtype: None.


.. py:function:: calccov(data, showlog=print)

   Calculate covariance from PyGMI Data.

   This routine assumes all bands are co-located, with the same size.
   Otherwise, run lstack first.

   :param data: List of PyGMI data.
   :type data: list of pygmi.raster.datatypes.Data
   :param showlog: Show information using a function. The default is print.
   :type showlog: function, optional

   :returns: **dcov** -- Covariances.
   :rtype: numpy array


