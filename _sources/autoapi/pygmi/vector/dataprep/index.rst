pygmi.vector.dataprep
=====================

.. py:module:: pygmi.vector.dataprep

.. autoapi-nested-parse::

   Data Preparation for Vector Data.



Classes
-------

.. autoapisummary::

   pygmi.vector.dataprep.PointCut
   pygmi.vector.dataprep.DataGrid
   pygmi.vector.dataprep.DataReproj
   pygmi.vector.dataprep.Metadata
   pygmi.vector.dataprep.TextFileSplit


Functions
---------

.. autoapisummary::

   pygmi.vector.dataprep.blanking
   pygmi.vector.dataprep.cut_point
   pygmi.vector.dataprep.txtlinecnt
   pygmi.vector.dataprep.filesplit
   pygmi.vector.dataprep.gridxyz
   pygmi.vector.dataprep.lltomap
   pygmi.vector.dataprep.maptobounds
   pygmi.vector.dataprep.maptovector
   pygmi.vector.dataprep.quickgrid
   pygmi.vector.dataprep.reprojxy


Module Contents
---------------

.. py:class:: PointCut(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   GUI to cut data using shapefiles.

   This class cuts point datasets using a boundary defined by a polygon
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



.. py:class:: DataGrid(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   GUI to grid point data.

   This class grids point data using a nearest neighbourhood technique.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: dxy_change()

      When dxy is changed on the interface, this updates rows and columns.

      :rtype: None.



   .. py:method:: grid_method_change()

      When grid method is changed, this updated hidden controls.

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



.. py:class:: DataReproj(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   GUI to reproject vector data.

   This class reprojects datasets using the GeoPandas routines.

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



.. py:class:: Metadata(parent=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   GUI to display and edit vector metadata.

   This class allows the editing of the metadata for a vector dataset using a
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



   .. py:method:: run()

      Entry point into the routine, used to run context menu item.

      :returns: **tmp** -- True if successful, False otherwise.
      :rtype: bool



.. py:class:: TextFileSplit(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   GUI to split a text file into smaller text files.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: change_method()

      Update fields when method changes.



   .. py:method:: get_ifile()

      Get input file information.

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



.. py:function:: blanking(gdat, x, y, bdist, extent, dxy, nullvalue)

   Blanks area further than a defined number of cells from input data.

   :param gdat: grid data to blank.
   :type gdat: numpy array
   :param x: x coordinates.
   :type x: numpy array
   :param y: y coordinates.
   :type y: numpy array
   :param bdist: Blanking distance in units for cell.
   :type bdist: int
   :param extent: extent of grid.
   :type extent: list
   :param dxy: Cell size.
   :type dxy: float
   :param Nullvalue: Null or nodata value.
   :type Nullvalue: float

   :returns: **mask** -- Mask to be used for blanking.
   :rtype: numpy array


.. py:function:: cut_point(data, ifile, showlog=print)

   Cuts a point dataset.

   Cut a point dataset using a shapefile.

   :param data: GeoPandas GeoDataFrame
   :type data: GeoDataFrame
   :param ifile: shapefile used to cut data
   :type ifile: str

   :returns: **data** -- GeoPandas GeoDataFrame
   :rtype: GeoDataFrame


.. py:function:: txtlinecnt(filename)

   Count lines in text file.

   :param filename: filename of text file.
   :type filename: str

   :returns: Total number of lines in a file.
   :rtype: int


.. py:function:: filesplit(ifile, num, mode='bytes', showlog=print, piter=None)

   Split an input file into a number of output files.

   :param ifile: Input filename.
   :type ifile: str
   :param num: Number of bytes or lines to split by.
   :type num: int
   :param mode: Can be 'bytes', 'files' or 'lines'. The default is 'bytes'.
   :type mode: str, optional
   :param showlog: Display information. The default is print.
   :type showlog: function, optional
   :param piter: Progress iterator. The default is None.
   :type piter: iter, optional

   :rtype: None.


.. py:function:: gridxyz(x, y, z, dxy, *, nullvalue=1e+20, method='Nearest Neighbour', bdist=4.0, showlog=print)

   Grid xyz data.

   :param x: X coordinate values.
   :type x: numpy array
   :param y: Y coordinate values.
   :type y: numpy array
   :param z: Z or data values.
   :type z: numpy array
   :param dxy: Grid cell size, in distance units.
   :type dxy: float
   :param nullvalue: null or nodata value. The default is 1e+20.
   :type nullvalue: float, optional
   :param method: Gridding method. The default is 'Nearest Neighbour'.
   :type method: str, optional
   :param bdist: Blanking distance. The default is 4.0.
   :type bdist: float, optional
   :param showlog: Display information. The default is print.
   :type showlog: function, optional

   :returns: **dat** -- Output raster dataset.
   :rtype: pygmi.raster.datatypes.Data.


.. py:function:: lltomap(lat, lon)

   Convert a latitude and longitude to a 1:50,000 sheet name.

   :param lat: Latitude.
   :type lat: float
   :param lon: Longitude.
   :type lon: float

   :returns: **mapsheet** -- Map sheet number.
   :rtype: str


.. py:function:: maptobounds(mapsheet, crs_to=None, showlog=print)

   Convert a South African map sheet name to bounds.

   :param mapsheet: Map sheet number. Four numbers and up to two letters denoting NE corner
                    in latitude and longitude and quadrants (A to D). Eg, 2928AB is
                    latitude 29, longitude 28, quadrant B of quadrant A.
   :type mapsheet: str
   :param crs_to: Destination projection. The default is None.
   :type crs_to: CRS, optional
   :param showlog: Display information. The default is print.
   :type showlog: function, optional

   :returns: **bounds** -- output bounds.
   :rtype: list


.. py:function:: maptovector(maplist)

   Create a vector layer from map numbers.

   :param maplist: List of strings containing map sheet numbers.
   :type maplist: list

   :returns: **data** -- GeoPandas GeoDataFrame
   :rtype: GeoDataFrame


.. py:function:: quickgrid(x, y, z, dxy, *, numits=4, showlog=print)

   Do a quick grid.

   :param x: array of x coordinates
   :type x: numpy array
   :param y: array of y coordinates
   :type y: numpy array
   :param z: array of z values - this is the column being gridded
   :type z: numpy array
   :param dxy: cell size for the grid, in both the x and y direction.
   :type dxy: float
   :param numits: number of iterations. By default its 4. If this is negative, a maximum
                  will be calculated and used.
   :type numits: int
   :param showlog: Routine to show text messages. The default is print.
   :type showlog: function, optional

   :returns: **newz** -- M x N array of z values
   :rtype: numpy array


.. py:function:: reprojxy(x, y, iwkt, owkt, showlog=print)

   Reproject x and y coordinates.

   :param x: x coordinates
   :type x: numpy array or float
   :param y: y coordinates
   :type y: numpy array or float
   :param iwkt: Input wkt description or EPSG code (int) or CRS
   :type iwkt: str, int, CRS
   :param owkt: Output wkt description or EPSG code (int) or CRS
   :type owkt: str, int, CRS

   :returns: * **xout** (*numpy array*) -- x coordinates.
             * **yout** (*numpy array*) -- y coordinates.


