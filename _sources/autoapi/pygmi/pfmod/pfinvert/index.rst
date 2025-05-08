pygmi.pfmod.pfinvert
====================

.. py:module:: pygmi.pfmod.pfinvert

.. autoapi-nested-parse::

   Magnetic inversion using the SimPEG inversion library.



Classes
-------

.. autoapisummary::

   pygmi.pfmod.pfinvert.MagInvert


Module Contents
---------------

.. py:class:: MagInvert(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   MagInvert - GUI class for magnetic inversion.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: apply_changes()

      Apply changes.

      :rtype: None.



   .. py:method:: choose_combo(combo, dtxt)

      Combo box choice routine.

      :param combo: Combo box.
      :type combo: QComboBox
      :param dtxt: Text to describe new raster data entry.
      :type dtxt: str

      :rtype: None.



   .. py:method:: choose_dtm()

      Combo box to choose current DTM.

      :rtype: None.



   .. py:method:: choose_model()

      Choose model file.

      :rtype: None.



   .. py:method:: extgrid(gdata)

      Extrapolates the grid to get rid of nulls.

      Uses a masked grid.

      :param gdata: Raster dataset.
      :type gdata: numpy array

      :returns: Output dataset.
      :rtype: numpy masked array



   .. py:method:: get_area()

      Get current grid extents and parameters.

      :rtype: None.



   .. py:method:: init()

      Initialise parameters.

      :rtype: None.



   .. py:method:: upd_layers()

      Update layers.

      :rtype: None.



   .. py:method:: update_model_combos()

      Update model combos.

      :rtype: None.



   .. py:method:: update_combos()

      Update combos.

      :rtype: None.



   .. py:method:: update_vals()

      Update the visible model extent parameters.

      :rtype: None.



   .. py:method:: xycell(dxy)

      Adjust XY dimensions when cell size changes.

      :param dxy: Cell dimension.
      :type dxy: float

      :rtype: None.



   .. py:method:: zcell(d_z)

      Adjust Z dimension when cell size changes.

      :param d_z: Layer thickness.
      :type d_z: float

      :rtype: None.



   .. py:method:: settings(nodialog=False)

      Entry point.

      :rtype: None.



   .. py:method:: saveproj()

      Save project data from class.

      :rtype: None.



   .. py:method:: acceptall()

      Accept All.

      Based on the SimPEG example.

      :rtype: None.



