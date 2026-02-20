pygmi.grav.iodefs
=================

.. py:module:: pygmi.grav.iodefs

.. autoapi-nested-parse::

   Routines to import gravity data and associated GPS data.



Classes
-------

.. autoapisummary::

   pygmi.grav.iodefs.ImportCG5


Functions
---------

.. autoapisummary::

   pygmi.grav.iodefs.get_cg5
   pygmi.grav.iodefs.get_cg6
   pygmi.grav.iodefs.get_gps
   pygmi.grav.iodefs.merge_gpsmag


Module Contents
---------------

.. py:class:: ImportCG5(parent)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Import CG-5 data.

   This class imports CG-5 gravimeter data with associated GPS data.

   :param parent: Reference to the parent routine.
   :type parent: pygmi.main.MainWidget


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



   .. py:method:: get_cg5(filename='')

      Get CG-5 filename and load data.

      :param filename: CG-5 filename submitted for testing. The default is ''.
      :type filename: str, optional

      :rtype: None.



   .. py:method:: get_gps(filename='')

      Get GPS filename and load data.

      :param filename: GPS filename (csv). The default is ''.
      :type filename: str, optional

      :rtype: None.



.. py:function:: get_cg5(filename)

   Get CG-5 filename and load data.

   :param filename: CG-5 filename.
   :type filename: str

   :returns: **df_cg5** -- Gravity data
   :rtype: Pandas DataFrame


.. py:function:: get_cg6(filename)

   Get CG-6 filename and load data.

   :param filename: CG-6 filename.
   :type filename: str

   :returns: **df** -- Gravity data
   :rtype: Pandas DataFrame


.. py:function:: get_gps(filename)

   Get GPS filename and load data.

   :param filename: GPS filename (csv).
   :type filename: str

   :returns: **df2** -- GPS data.
   :rtype: Pandas DataFrame


.. py:function:: merge_gpsmag(cg5file, gpsfile, basethres=10000.0, cren=None, showlog=print)

   Import and merge GPS and gravity data.

   :param cg5file: Gravity filename for data in CG format.
   :type cg5file: str
   :param gpsfile: GPS filename in CSV format.
   :type gpsfile: str
   :param basethres: Threshold for base station numbers. The default is 10000.
   :type basethres: float, optional
   :param cren: Dictionary of columns to rename. The default is None.
   :type cren: dict
   :param showlog: Display information. The default is print.
   :type showlog: function, optional

   :returns: Dataframe with merged data.
   :rtype: Pandas DataFrame


