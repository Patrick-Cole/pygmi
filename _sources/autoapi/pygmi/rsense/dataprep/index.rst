pygmi.rsense.dataprep
=====================

.. py:module:: pygmi.rsense.dataprep

.. autoapi-nested-parse::

   Data preparation for satellite data.

   This focuses on routines to either prepare Sentinel 2 data for topographic
   correction, or doing the topographic correction itself.



Classes
-------

.. autoapisummary::

   pygmi.rsense.dataprep.TopoCorrect
   pygmi.rsense.dataprep.Sen2Cor


Functions
---------

.. autoapisummary::

   pygmi.rsense.dataprep.c_correction


Module Contents
---------------

.. py:class:: TopoCorrect(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   GUI to calculate topographic correction.

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



   .. py:method:: acceptall()

      Accept option.

      Updates self.outdata, which is used as input to other modules.

      :rtype: None.



.. py:class:: Sen2Cor(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   GUI to calculate atmospheric correction using Sen2Cor.

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



   .. py:method:: get_sdir(nodialog=False)

      Get the satellite directory.



   .. py:method:: get_sen2cor(nodialog=False)

      Get the sen2cor directory.



   .. py:method:: saveproj()

      Save project data from class.

      :rtype: None.



   .. py:method:: acceptall()

      Accept option.

      Updates self.outdata, which is used as input to other modules.

      :rtype: None.



.. py:function:: c_correction(data, dem, azimuth, zenith, *, showlog=print, piter=iter)

   Calculate C correction.

   :param data: Data to be corrected.
   :type data: pygmi.raster.datatypes.Data
   :param dem: DEM data used in correction.
   :type dem: pygmi.raster.datatypes.Data
   :param azimuth: Solar azimuth in degrees.
   :type azimuth: float
   :param zenith: Solar zenith in degrees.
   :type zenith: float
   :param showlog: Display information. The default is print.
   :type showlog: function, optional
   :param piter: Progress bar iterator. The default is iter.
   :type piter: function, optional

   :returns: **data2** -- List of c-corrected data arrays.
   :rtype: list


