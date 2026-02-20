pygmi.rsense.ratios
===================

.. py:module:: pygmi.rsense.ratios

.. autoapi-nested-parse::

   Calculate remote sensing ratios and condition indices.



Classes
-------

.. autoapisummary::

   pygmi.rsense.ratios.SatRatios
   pygmi.rsense.ratios.ConditionIndices


Functions
---------

.. autoapisummary::

   pygmi.rsense.ratios.calc_ratios
   pygmi.rsense.ratios.correct_bands
   pygmi.rsense.ratios.get_aster_list
   pygmi.rsense.ratios.get_landsat_list
   pygmi.rsense.ratios.get_sentinel_list
   pygmi.rsense.ratios.get_TCI
   pygmi.rsense.ratios.get_VCI
   pygmi.rsense.ratios.get_VHI
   pygmi.rsense.ratios.landslide_index


Module Contents
---------------

.. py:class:: SatRatios(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   GUI to calculate satellite ratios.

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



   .. py:method:: setratios()

      Set the available ratios.

      The ratio definitions are for the ASTER satellite. Band 0 refers to
      an imaginary blue band.

      :rtype: None.



   .. py:method:: invert_selection()

      Invert the selected ratios.

      :rtype: None.



.. py:class:: ConditionIndices(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   GUI to calculate satellite condition indices.

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



   .. py:method:: setratios()

      Set the available indices.

      :rtype: None.



   .. py:method:: invert_selection()

      Invert the selected ratios.

      :rtype: None.



   .. py:method:: set_selected_ratios()

      Set the selected ratios.

      :rtype: None.



.. py:function:: calc_ratios(dat, rlist, showlog=print, piter=iter, sensor=None)

   Calculate Band ratios.

   Note that this routine assumes that the ratio you supply is correct for
   your data.

   :param dat: List of PyGMI Data.
   :type dat: list of pygmi.raster.datatypes.Data.
   :param rlist: List of strings, containing ratios to calculate..
   :type rlist: list
   :param showlog: Display information. The default is print.
   :type showlog: function, optional
   :param piter: Progress bar iterator. The default is iter.
   :type piter: function, optional
   :param sensor: The sensor being processed. The default is None.
   :type sensor: str

   :returns: **datfin** -- List of PyGMI Data.
   :rtype: list of pygmi.raster.datatypes.Data.


.. py:function:: correct_bands(rlist, sensor, bfile=None)

   Correct the band designations.

   Ratio formula are defined in terms of ASTER bands. This converts that to
   the target sensor.

   :param rlist: List of input ratios.
   :type rlist: list
   :param sensor: Target sensor.
   :type sensor: str
   :param bfile: Data filename. The default is None.
   :type bfile: str

   :returns: **rlist2** -- List of converted ratios.
   :rtype: list


.. py:function:: get_aster_list(flist)

   Get ASTER files from a file list.

   :param flist: List of filenames.
   :type flist: list

   :returns: **flist2** -- List of filenames.
   :rtype: list


.. py:function:: get_landsat_list(flist, sensor=None, allsats=False)

   Get Landsat files from a file list.

   :param flist: List of filenames.
   :type flist: list

   :returns: **flist2** -- List of filenames.
   :rtype: list


.. py:function:: get_sentinel_list(flist)

   Get Sentinel-2 files from a file list.

   :param flist: List of filenames.
   :type flist: list

   :returns: **flist2** -- List of filenames.
   :rtype: list


.. py:function:: get_TCI(lst)

   Calculate TCI.

   :param lst: list of PyGMI datasets - land surface temperatures.
   :type lst: list of pygmi.raster.datatypes.Data.

   :returns: **tci** -- output TCI datasets.
   :rtype: list of pygmi.raster.datatypes.Data.


.. py:function:: get_VCI(evi, index)

   Calculate VCI.

   :param evi: list of EVI datasets.
   :type evi: list of pygmi.raster.datatypes.Data
   :param index: index for dataid.
   :type index: str

   :returns: **vci** -- output VCI datasets.
   :rtype: list of pygmi.raster.datatypes.Data


.. py:function:: get_VHI(tci, vci, alpha=0.5)

   Calculate VHI.

   :param tci: TCI dataset list.
   :type tci: list
   :param vci: VCI dataset list.
   :type vci: list
   :param alpha: Weight for proportion of TCI and VCI. The default is 0.5.
   :type alpha: float, optional

   :returns: **vhi** -- Output VHI datasets.
   :rtype: list of pygmi.raster.datatypes.Data


.. py:function:: landslide_index(dat, sensor=None, showlog=print, piter=iter)

   Calculate Band ratios.

   Note that this routine assumes that the ratio you supply is correct for
   your data.

   :param dat: List of PyGMI Data.
   :type dat: list of pygmi.raster.datatypes.Data.
   :param sensor: The sensor being processed. The default is None.
   :type sensor: str
   :param showlog: Display information. The default is print.
   :type showlog: function, optional
   :param piter: Progress bar iterator. The default is iter.
   :type piter: function, optional

   :returns: **datfin** -- Red, green and blue PyGMI Data.
   :rtype: list of pygmi.raster.datatypes.Data.


