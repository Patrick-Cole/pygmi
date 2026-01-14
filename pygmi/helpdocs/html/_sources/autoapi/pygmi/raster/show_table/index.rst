pygmi.raster.show_table
=======================

.. py:module:: pygmi.raster.show_table

.. autoapi-nested-parse::

   Routine which displays a table graphically with various stats.



Classes
-------

.. autoapisummary::

   pygmi.raster.show_table.BasicStats
   pygmi.raster.show_table.ClusterStats


Functions
---------

.. autoapisummary::

   pygmi.raster.show_table.basicstats_calc
   pygmi.raster.show_table.savetable


Module Contents
---------------

.. py:class:: BasicStats(parent=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   Show a summary of basic stats.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: combo()

      Combo.

      :rtype: None.



   .. py:method:: run()

      Entry point into the routine, used to run context menu item.

      :rtype: None.



   .. py:method:: save()

      Save Table.

      :returns: True if successful, False otherwise.
      :rtype: bool



.. py:function:: basicstats_calc(data)

   Calculate statistics.

   :param data: PyGMI raster dataset.
   :type data: pygmi.raster.datatypes.Data.

   :returns: * **bands** (*list*) -- Band list, currently only 'Data Column'
             * **cols** (*list*) -- Columns for the table
             * **dattmp** (*list*) -- List of arrays containing statistics.


.. py:class:: ClusterStats(parent=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   Show a summary of basic statistics.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: combo()

      Combo.

      :rtype: None.



   .. py:method:: run()

      Entry point into the routine, used to run context menu item.

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: save()

      Save Table.

      :returns: True if successful, False otherwise.
      :rtype: bool



.. py:function:: savetable(ofile, bands, cols, data)

   Save tabular data.

   :param ofile: Output file name.
   :type ofile: str
   :param bands: List of bands.
   :type bands: list
   :param cols: List of column headings.
   :type cols: list
   :param data: List of arrays containing statistics.
   :type data: list

   :rtype: None.


