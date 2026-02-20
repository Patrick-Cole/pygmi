pygmi.vector.iodefs
===================

.. py:module:: pygmi.vector.iodefs

.. autoapi-nested-parse::

   Import and export vector data.



Classes
-------

.. autoapisummary::

   pygmi.vector.iodefs.ColumnSelect
   pygmi.vector.iodefs.ImportVector
   pygmi.vector.iodefs.ImportXYZ
   pygmi.vector.iodefs.ImportVoxel
   pygmi.vector.iodefs.ExportXYZ
   pygmi.vector.iodefs.ExportVector
   pygmi.vector.iodefs.ExportVoxel


Functions
---------

.. autoapisummary::

   pygmi.vector.iodefs.import_ubc
   pygmi.vector.iodefs.export_ubc
   pygmi.vector.iodefs.get_GXYZ
   pygmi.vector.iodefs.get_intrepid


Module Contents
---------------

.. py:class:: ColumnSelect(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   A combobox to select vector columns.

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



.. py:class:: ImportVector(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   GUI to import vector data.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: ppygmi.main.MainWidget, optional


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: settings(nodialog=False)

      Entry point into item.

      :param nodialog: Run settings without a dialog. The default is False.
      :type nodialog: bool, optional

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: change_bounds()

      Change the bounds combo.



   .. py:method:: get_sfile()

      Get the filename and crs and bounds.



   .. py:method:: set_bounds(bounds)

      Set the bounds.

      :param bounds: Bounds defined as (xmin, ymin, xmax, ymax).
      :type bounds: list or numpy array

      :rtype: None.



   .. py:method:: saveproj()

      Save project data from class.

      :rtype: None.



.. py:class:: ImportXYZ(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   GUI to import XYZ data.

   This class imports tabular data.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional


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



   .. py:method:: get_GXYZ()

      Get Geosoft XYZ.

      :returns: **df** -- Pandas dataframe.
      :rtype: DataFrame



   .. py:method:: get_delimited(delimiter=',')

      Get a delimited file.

      :param delimiter: Delimiter type. The default is ','.
      :type delimiter: str, optional

      :returns: **gdf** -- Pandas dataframe.
      :rtype: Dataframe



   .. py:method:: get_excel()

      Get an Excel spreadsheet.

      :returns: **gdf** -- Pandas dataframe.
      :rtype: Dataframe



.. py:class:: ImportVoxel(parent=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   GUI to import voxel data.

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



.. py:class:: ExportXYZ(parent=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   GUI to export XYZ data.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional


   .. py:method:: run()

      Entry point into the routine, used to run context menu item.

      :returns: True if successful, False otherwise.
      :rtype: bool



.. py:class:: ExportVector(parent=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   GUI to export vector data.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional


   .. py:method:: run()

      Entry point into the routine, used to run context menu item.

      :returns: True if successful, False otherwise.
      :rtype: bool



.. py:class:: ExportVoxel(parent=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   GUI to export voxel data.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional


   .. py:method:: run()

      Entry point into the routine, used to run context menu item.

      :returns: True if successful, False otherwise.
      :rtype: bool



.. py:function:: import_ubc(ifile)

   Import a 3D UBC mesh and model.

   :param ifile: Input file name.
   :type ifile: str

   :returns: **vdat** -- Imported voxel model.
   :rtype: pygmi.vector.datatypes.VoxModel


.. py:function:: export_ubc(ofile, data)

   Export a section to a 3D UBC mesh and model.

   :param data: Dataset to export
   :type data: pygmi.vector.datatypes.VoxModel

   :rtype: None.


.. py:function:: get_GXYZ(ifile, showlog=print, piter=iter)

   Get Geosoft XYZ.

   :param ifile: Input file name.
   :type ifile: str
   :param showlog: Routine to show text messages. The default is print.
   :type showlog: function, optional
   :param piter: progress bar iterable, default is iter.
   :type piter: function, optional

   :returns: **df2** -- Pandas dataframe.
   :rtype: DataFrame


.. py:function:: get_intrepid(ifile, showlog=print, piter=iter)

   Get Intrepid Database.

   :param ifile: Input file name.
   :type ifile: str
   :param showlog: Routine to show text messages. The default is print.
   :type showlog: function, optional
   :param piter: progress bar iterable, default is iter.
   :type piter: function, optional

   :returns: **df** -- Pandas Dataframe.
   :rtype: DataFrame


