pygmi.raster.normalisation
==========================

.. py:module:: pygmi.raster.normalisation

.. autoapi-nested-parse::

   Raster normalisation routine.



Classes
-------

.. autoapisummary::

   pygmi.raster.normalisation.Normalisation


Functions
---------

.. autoapisummary::

   pygmi.raster.normalisation.datacommon
   pygmi.raster.normalisation.norm


Module Contents
---------------

.. py:class:: Normalisation(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Class Normalisation GUI.

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



.. py:function:: datacommon(data, tmp1, tmp2)

   Variables used in the process routine.

   :param data: PyGMI raster dataset.
   :type data: pygmi.raster.datatypes.Data.
   :param tmp1: Parameter 1. Can be min, mean or median.
   :type tmp1: float
   :param tmp2: Parameter 2. Can be range, std, or mad.
   :type tmp2: float

   :returns: * **data** (*pygmi.raster.datatypes.Data*) -- PyGMI raster dataset.
             * **transform** (*numpy array.*) -- Transformation applied to data.


.. py:function:: norm(data, ntype)

   Normalise data.

   :param data: PyGMI Data in a list.
   :type data: list
   :param ntype: Normalisation type.Can be 'interval', 'mean', 'median' or '8bit'.
   :type ntype: str

   :returns: **data** -- PyGMI Data in a list.
   :rtype: list


