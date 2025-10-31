pygmi.rsense.iodefs
===================

.. py:module:: pygmi.rsense.iodefs

.. autoapi-nested-parse::

   Import remote sensing data.



Classes
-------

.. autoapisummary::

   pygmi.rsense.iodefs.ImportData
   pygmi.rsense.iodefs.ImportBatch
   pygmi.rsense.iodefs.ImportSentinel5P
   pygmi.rsense.iodefs.ExportBatch


Functions
---------

.. autoapisummary::

   pygmi.rsense.iodefs.calculate_toa
   pygmi.rsense.iodefs.consolidate_aster_list
   pygmi.rsense.iodefs.convert_ll_to_utm
   pygmi.rsense.iodefs.etree_to_dict
   pygmi.rsense.iodefs.export_batch
   pygmi.rsense.iodefs.files_to_rastermeta
   pygmi.rsense.iodefs.get_data
   pygmi.rsense.iodefs.get_from_rastermeta
   pygmi.rsense.iodefs.get_emit
   pygmi.rsense.iodefs.get_modisv6
   pygmi.rsense.iodefs.get_landsat
   pygmi.rsense.iodefs.get_worldview
   pygmi.rsense.iodefs.get_hyperion
   pygmi.rsense.iodefs.get_sentinel1
   pygmi.rsense.iodefs.get_sentinel2
   pygmi.rsense.iodefs.get_sentinel2_metadata
   pygmi.rsense.iodefs.get_spot
   pygmi.rsense.iodefs.get_aster_zip
   pygmi.rsense.iodefs.get_aster_tif
   pygmi.rsense.iodefs.get_aster_metadata
   pygmi.rsense.iodefs.get_aster_hdf
   pygmi.rsense.iodefs.get_aster_ged
   pygmi.rsense.iodefs.get_aster_ged_bin
   pygmi.rsense.iodefs.get_ternary
   pygmi.rsense.iodefs.set_export_filename
   pygmi.rsense.iodefs.utm_to_south


Module Contents
---------------

.. py:class:: ImportData(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Import Data GUI - Interfaces with rasterio routines.

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



   .. py:method:: get_sfile()

      Get the satellite filename.



   .. py:method:: saveproj()

      Save project data from class.

      :rtype: None.



.. py:class:: ImportBatch(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Batch Import Data Interface.

   This does not actually import data, but rather defines a list of datasets
   to be used by other routines.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional

   .. attribute:: idir

      Input directory.

      :type: str


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: settings(nodialog=False)

      Entry point into item.

      :param nodialog: Run settings without a dialog. The default is False.
      :type nodialog: bool, optional

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: get_sfile(nodialog=False)

      Get the satellite filenames.



   .. py:method:: setsensor()

      Set the sensor band data.

      :rtype: None.



   .. py:method:: saveproj()

      Save project data from class.

      :rtype: None.



.. py:class:: ImportSentinel5P(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   GUI import Sentinel 5P data and export to shapefile.

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



   .. py:method:: clipchoice()

      Choose clip style.

      :rtype: None.



   .. py:method:: loadshp()

      Load shapefile filename.

      :rtype: None.



   .. py:method:: saveproj()

      Save project data from class.

      :rtype: None.



   .. py:method:: get_5P_meta()

      Get 5P metadata.

      :returns: **meta** -- Dictionary containing metadata.
      :rtype: Dictionary



   .. py:method:: get_5P_data(meta)

      Get 5P data.

      :param meta: Dictionary containing metadata.
      :type meta: Dictionary

      :returns: **gdf** -- geopandas dataframe.
      :rtype: DataFrame



.. py:class:: ExportBatch(parent=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   GUI to export Raster File List.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: click_ternary()

      Click ternary event.

      :rtype: None.



   .. py:method:: run()

      Entry point into the routine, used to run context menu item.

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: acceptall()

      Accept choice.



   .. py:method:: get_odir(odir='')

      Get output directory.

      :param odir: Output directory submitted for testing. The default is ''.
      :type odir: str, optional

      :rtype: None.



.. py:function:: calculate_toa(dat, showlog=print)

   Top of atmosphere correction.

   Includes VNIR, SWIR and TIR bands.

   :param dat: PyGMI raster dataset
   :type dat: pygmi.raster.datatypes.Data
   :param showlog: Routine to show text messages. The default is print.
   :type showlog: function, optional

   :returns: **out** -- PyGMI raster dataset
   :rtype: pygmi.raster.datatypes.Data


.. py:function:: consolidate_aster_list(flist)

   Consolidate ASTER files from a file list, getting rid of extra files.

   :param flist: List of filenames.
   :type flist: list

   :returns: **flist** -- List of filenames.
   :rtype: list


.. py:function:: convert_ll_to_utm(lon, lat)

   Convert latitude and longitude to UTM.

   https://stackoverflow.com/a/40140326/4556479

   :param lon: Longitude.
   :type lon: float
   :param lat: latitude.
   :type lat: float

   :returns: **epsg_code** -- EPSG code.
   :rtype: str


.. py:function:: etree_to_dict(t)

   Convert an ElementTree to dictionary.

   From K3--rnc: https://stackoverflow.com/questions/7684333/converting-xml-to-dictionary-using-elementtree

   :param t: Root.
   :type t: Elementtree

   :returns: **d** -- Dictionary of ElementTree items.
   :rtype: dictionary


.. py:function:: export_batch(indata, odir, filt, *, tnames=None, piter=None, showlog=print, otype=None, sunfile=None, cell=25.0, alpha=0.75)

   Export a batch of files directly from satellite format to disk.

   :param indata: Dictionary containing 'RasterFileList' as one of its keys.
   :type indata: dictionary
   :param odir: Output Directory.
   :type odir: str
   :param filt: type of file to export.
   :type filt: str
   :param tnames: list of band names to import, in order. the default is None.
   :type tnames: list, optional
   :param piter: Progress bar iterable. Default is None.
   :type piter: function, optional
   :param showlog: Routine to show text messages. The default is print.
   :type showlog: function, optional
   :param otype: output type of file, regular or RGB ternary (with possible sunshading)
   :type otype: str
   :param sunfile: either a filename of an external file to be used for sunshading, or an
                   existing band name. the default is None.
   :type sunfile: str
   :param cell: Between 1 and 100 - controls sunshade detail. The default is 25.
   :type cell: float
   :param alpha: How much incident light is reflected (0 to 1). The default is .75.
   :type alpha: float

   :rtype: None.


.. py:function:: files_to_rastermeta(allfiles, piter=iter, showlog=print)

   Import files to a RasterMeta item.

   :param allfiles: List of filenames.
   :type allfiles: list
   :param piter: Progress bar iterable. Default is iter.
   :type piter: function, optional
   :param showlog: Routine to show text messages. The default is print.
   :type showlog: function, optional

   :returns: * **bands** (*dict*) -- Bands
             * **tnames** (*dict*) -- Sensor types
             * **filelist** (*list*) -- List of RasterMeta data.


.. py:function:: get_data(ifile, *, piter=None, showlog=print, tnames=None, metaonly=False, bounds=None)

   Load a raster dataset off the disk using the rasterio libraries.

   It returns the data in a PyGMI data object.

   :param ifile: filename to import
   :type ifile: str
   :param piter: Progress bar iterable. Default is None.
   :type piter: function, optional
   :param showlog: Routine to show text messages. The default is print.
   :type showlog: function, optional
   :param tnames: list of band names to import, in order. The default is None.
   :type tnames: list, optional
   :param metaonly: Retrieve only the metadata for the file. The default is False.
   :type metaonly: bool, optional
   :param bounds: Bounds of data to import as (left, bottom, right, top)
   :type bounds: tuple

   :returns: **dat** -- dataset imported
   :rtype: list of pygmi.raster.datatypes.Data


.. py:function:: get_from_rastermeta(ldata, *, piter=None, showlog=print, tnames=None, metaonly=False, bounds=None)

   Import data from a RasterMeta item.

   For convenience a Data object is also accepted as input.

   :param ldata: RasterMeta item.
   :type ldata: RasterMeta or pygmi.raster.datatypes.Data
   :param piter: Progress bar iterable. Default is None.
   :type piter: function, optional
   :param showlog: Routine to show text messages. The default is print.
   :type showlog: function, optional
   :param tnames: list of band names to import, in order. The default is None.
   :type tnames: list, optional
   :param metaonly: Retrieve only the metadata for the files. The default is False.
   :type metaonly: bool, optional
   :param bounds: Bounds of data to import as (left, bottom, right, top)
   :type bounds: tuple

   :returns: **dat** -- List of data.
   :rtype: list  of pygmi.raster.datatypes.Data


.. py:function:: get_emit(ifile, piter=None, showlog=print, tnames=None, metaonly=False)

   Get EMIT Data.

   :param ifile: filename to import
   :type ifile: str
   :param piter: Progress bar iterable. Default is None.
   :type piter: function, optional
   :param showlog: Routine to show text messages. The default is print.
   :type showlog: function, optional
   :param tnames: list of band names to import, in order. The default is None.
   :type tnames: list, optional
   :param metaonly: Retrieve only the metadata for the file. The default is False.
   :type metaonly: bool, optional

   :returns: **dat** -- dataset imported
   :rtype: PyGMI raster Data


.. py:function:: get_modisv6(ifile, piter=None, showlog=print, tnames=None, metaonly=False)

   Get MODIS v006 data.

   :param ifile: filename to import
   :type ifile: str
   :param piter: Progress bar iterable. Default is None.
   :type piter: function, optional
   :param showlog: Routine to show text messages. The default is print.
   :type showlog: function, optional
   :param tnames: list of band names to import, in order. The default is None.
   :type tnames: list, optional
   :param metaonly: Retrieve only the metadata for the file. The default is False.
   :type metaonly: bool, optional

   :returns: **dat** -- dataset imported
   :rtype: PyGMI raster Data


.. py:function:: get_landsat(ifilet, piter=None, showlog=print, tnames=None, metaonly=False)

   Get Landsat Data.

   :param ifilet: filename to import
   :type ifilet: str
   :param piter: Progress bar iterable. Default is None.
   :type piter: function, optional
   :param showlog: Routine to show text messages. The default is print.
   :type showlog: function, optional
   :param tnames: list of band names to import, in order. The default is None.
   :type tnames: list, optional
   :param metaonly: Retrieve only the metadata for the file. The default is False.
   :type metaonly: bool, optional

   :returns: **out** -- PyGMI raster dataset
   :rtype: pygmi.raster.datatypes.Data


.. py:function:: get_worldview(ifilet, piter=None, showlog=print, tnames=None, metaonly=False)

   Get WorldView Data.

   :param ifilet: filename to import
   :type ifilet: str
   :param piter: Progress bar iterable. Default is None.
   :type piter: function, optional
   :param showlog: Routine to show text messages. The default is print.
   :type showlog: function, optional
   :param tnames: list of band names to import, in order. The default is None.
   :type tnames: list, optional
   :param metaonly: Retrieve only the metadata for the file. The default is False.
   :type metaonly: bool, optional

   :returns: **out** -- PyGMI raster dataset
   :rtype: pygmi.raster.datatypes.Data


.. py:function:: get_hyperion(ifile, piter=None, showlog=print, tnames=None, metaonly=False)

   Get Hyperion Data.

   :param ifile: filename to import
   :type ifile: str
   :param piter: Progress bar iterable. Default is None.
   :type piter: function, optional
   :param showlog: Routine to show text messages. The default is print.
   :type showlog: function, optional
   :param tnames: list of band names to import, in order. The default is None.
   :type tnames: list, optional
   :param metaonly: Retrieve only the metadata for the file. The default is False.
   :type metaonly: bool, optional

   :returns: **out** -- PyGMI raster dataset
   :rtype: pygmi.raster.datatypes.Data


.. py:function:: get_sentinel1(ifile, piter=None, showlog=print, tnames=None, metaonly=False)

   Get Sentinel-1 Data.

   :param ifile: filename to import
   :type ifile: str
   :param piter: Progress bar iterable. Default is None.
   :type piter: function, optional
   :param showlog: Routine to show text messages. The default is print.
   :type showlog: function, optional
   :param tnames: list of band names to import, in order. The default is None.
   :type tnames: list, optional
   :param metaonly: Retrieve only the metadata for the file. The default is False.
   :type metaonly: bool, optional

   :returns: **dat** -- dataset imported
   :rtype: PyGMI raster Data


.. py:function:: get_sentinel2(ifile, *, piter=None, showlog=print, tnames=None, metaonly=False, bounds=None)

   Get Sentinel-2 Data.

   :param ifile: filename to import
   :type ifile: str
   :param piter: Progress bar iterable. Default is None.
   :type piter: function, optional
   :param showlog: Routine to show text messages. The default is print.
   :type showlog: function, optional
   :param tnames: list of band names to import, in order. The default is None.
   :type tnames: list, optional
   :param metaonly: Retrieve only the metadata for the file. The default is False.
   :type metaonly: bool, optional
   :param bounds: Bounds of data to import as (left, bottom, right, top)
   :type bounds: tuple

   :returns: **dat** -- dataset imported
   :rtype: PyGMI raster Data


.. py:function:: get_sentinel2_metadata(ifile)

   Get extra metadata from xml files which rasterio does not access.

   :param ifile: Input filename.
   :type ifile: str

   :returns: **meta** -- Output metadata.
   :rtype: dictionary


.. py:function:: get_spot(ifile, piter=None, showlog=print, tnames=None, metaonly=False)

   Get Spot DIMAP Data.

   :param ifile: filename to import
   :type ifile: str
   :param piter: Progress bar iterable. Default is None.
   :type piter: function, optional
   :param showlog: Routine to show text messages. The default is print.
   :type showlog: function, optional
   :param tnames: list of band names to import, in order. The default is None.
   :type tnames: list, optional
   :param metaonly: Retrieve only the metadata for the file. The default is False.
   :type metaonly: bool, optional

   :returns: **dat** -- dataset imported
   :rtype: PyGMI raster Data


.. py:function:: get_aster_zip(ifile, piter=None, showlog=print, tnames=None, metaonly=False)

   Get ASTER zip Data.

   :param ifile: filename to import
   :type ifile: str
   :param piter: Progress bar iterable. Default is None.
   :type piter: function, optional
   :param showlog: Routine to show text messages. The default is print.
   :type showlog: function, optional
   :param tnames: list of band names to import, in order. The default is None.
   :type tnames: list, optional
   :param metaonly: Retrieve only the metadata for the file. The default is False.
   :type metaonly: bool, optional

   :returns: **dat** -- dataset imported
   :rtype: PyGMI raster Data


.. py:function:: get_aster_tif(ifiles, piter=None, showlog=print, tnames=None, metaonly=False)

   Get ASTER zip Data.

   :param ifile: filename to import
   :type ifile: str
   :param piter: Progress bar iterable. Default is None.
   :type piter: function, optional
   :param showlog: Routine to show text messages. The default is print.
   :type showlog: function, optional
   :param tnames: list of band names to import, in order. The default is None.
   :type tnames: list, optional
   :param metaonly: Retrieve only the metadata for the file. The default is False.
   :type metaonly: bool, optional

   :returns: **dat** -- dataset imported
   :rtype: PyGMI raster Data


.. py:function:: get_aster_metadata(ifile)

   Get extra metadata from met files which rasterio does not access.

   :param ifile: Input filename.
   :type ifile: str

   :returns: **meta** -- Output metadata.
   :rtype: dictionary


.. py:function:: get_aster_hdf(ifile, piter=None, showlog=print, tnames=None, metaonly=False)

   Get ASTER hdf Data.

   This function needs the original filename to extract the date.

   :param ifile: filename to import
   :type ifile: str
   :param piter: Progress bar iterable. Default is None.
   :type piter: function, optional
   :param showlog: Routine to show text messages. The default is print.
   :type showlog: function, optional
   :param tnames: list of band names to import, in order. The default is None.
   :type tnames: list, optional
   :param metaonly: Retrieve only the metadata for the file. The default is False.
   :type metaonly: bool, optional

   :returns: **dat** -- dataset imported
   :rtype: PyGMI raster Data


.. py:function:: get_aster_ged(ifile, piter=None, showlog=print, tnames=None, metaonly=False)

   Get ASTER GED data.

   :param ifile: filename to import
   :type ifile: str
   :param piter: Progress bar iterable. Default is None.
   :type piter: function, optional
   :param showlog: Routine to show text messages. The default is print.
   :type showlog: function, optional
   :param tnames: list of band names to import, in order. The default is None.
   :type tnames: list, optional
   :param metaonly: Retrieve only the metadata for the file. The default is False.
   :type metaonly: bool, optional

   :returns: **dat** -- dataset imported
   :rtype: PyGMI raster Data


.. py:function:: get_aster_ged_bin(ifile)

   Get ASTER GED binary format.

   Emissivity_Mean_Description: Mean Emissivity for each pixel on grid-box
   using all ASTER data from 2000-2010
   Emissivity_SDev_Description: Emissivity Standard Deviation for each pixel
   on grid-box using all ASTER data from 2000-2010
   Temperature_Mean_Description: Mean Temperature (K) for each pixel on
   grid-box using all ASTER data from 2000-2010
   Temperature_SDev_Description: Temperature Standard Deviation for each pixel
   on grid-box using all ASTER data from 2000-2010
   NDVI_Mean_Description: Mean NDVI for each pixel on grid-box using all ASTER
   data from 2000-2010
   NDVI_SDev_Description: NDVI Standard Deviation for each pixel on grid-box
   using all ASTER data from 2000-2010
   Land_Water_Map_LWmap_Description: Land Water Map using ASTER visible bands
   Observations_NumObs_Description: Number of values used in computing mean
   and standard deviation for each pixel.
   Geolocation_Latitude_Description: Latitude
   Geolocation_Longitude_Description: Longitude
   ASTER_GDEM_ASTGDEM_Description: ASTER GDEM resampled to NAALSED

   :param ifile: filename to import
   :type ifile: str

   :returns: **dat** -- dataset imported
   :rtype: PyGMI raster Data


.. py:function:: get_ternary(dat, sunfile=None, clippercl=1.0, clippercu=1.0, cell=25.0, alpha=0.75, piter=iter, showlog=print)

   Create a ternary image, with optional sunshading.

   :param dat: PyGMI Data.
   :type dat: list of pygmi.raster.datatypes.Data
   :param sunfile: Sunshading band or filename. The default is None.
   :type sunfile: str, optional
   :param clippercl: Lower clip percentage for colour bar. The default is 1.
   :type clippercl: float, optional
   :param clippercu: Upper clip percentage for colour bar. The default is 1.
   :type clippercu: float, optional
   :param cell: Between 1 and 100 - controls sunshade detail. The default is 25.
   :type cell: float, optional
   :param alpha: How much incident light is reflected (0 to 1). The default is .75.
   :type alpha: float, optional

   :returns: **newimg** -- list of PyGMI data.
   :rtype: list of pygmi.raster.datatypes.Data.


.. py:function:: set_export_filename(dat, odir, otype=None)

   Set the export filename according to convention.

   Different satellite products have different simplified conventions for
   output filenames to avoid names getting too long.

   :param dat: List of PyGMI data.
   :type dat: list of pygmi.raster.datatypes.Data.
   :param odir: Output directory.
   :type odir: str
   :param otype: Output file type. Default is None.
   :type otype: str, optional.

   :returns: **ofile** -- Output file name.
   :rtype: str


.. py:function:: utm_to_south(dat)

   Make sure all UTM labels are for southern hemisphere.

   :param dat: List of Data.
   :type dat: list of pygmi.raster.datatypes.Data

   :returns: **dat** -- List of Data.
   :rtype: list of pygmi.raster.datatypes.Data


