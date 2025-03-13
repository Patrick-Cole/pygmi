pygmi.rsense.landsat_composite
==============================

.. py:module:: pygmi.rsense.landsat_composite

.. autoapi-nested-parse::

   Calculate Landsat composite scenes.



Classes
-------

.. autoapisummary::

   pygmi.rsense.landsat_composite.LandsatComposite


Functions
---------

.. autoapisummary::

   pygmi.rsense.landsat_composite.composite
   pygmi.rsense.landsat_composite.import_and_score


Module Contents
---------------

.. py:class:: LandsatComposite(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Landsat Composite Interface.

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



   .. py:method:: get_idir()

      Get the input directory.

      :rtype: None.



   .. py:method:: saveproj()

      Save project data from class.

      :rtype: None.



.. py:function:: composite(idir, dreq=10, mean=None, showlog=print, piter=None)

   Create a Landsat composite.

   :param idir: Input directory.
   :type idir: str
   :param dreq: Distance to cloud in pixels. The default is 10.
   :type dreq: int, optional
   :param mean: The mean or target day. If not specified, it is calculated
                automatically. The default is None.
   :type mean: float, optional
   :param showlog: Function for printing text. The default is print.
   :type showlog: function, optional
   :param piter: Progress bar iterable. The default is None.
   :type piter: function, optional

   :returns: **datfin** -- List of PyGMI Data.
   :rtype: list of pygmi.raster.datatypes.Data.


.. py:function:: import_and_score(ifile, dreq, mean, std, *, showlog=print, piter=None)

   Import data and score it.

   :param ifile: Input filename.
   :type ifile: str
   :param dreq: Distance to cloud in pixels. The default is 10.
   :type dreq: int, optional
   :param mean: The mean or target day.
   :type mean: float
   :param std: The standard deviation of all days.
   :type std: float
   :param showlog: Function for printing text. The default is print.
   :type showlog: function, optional
   :param piter: Progress bar iterable. The default is None.
   :type piter: function, optional

   :returns: **dat** -- Dictionary of bands imported.
   :rtype: dictionary.


