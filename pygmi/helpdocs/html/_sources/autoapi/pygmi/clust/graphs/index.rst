pygmi.clust.graphs
==================

.. py:module:: pygmi.clust.graphs

.. autoapi-nested-parse::

   Routines to plot cluster data.



Classes
-------

.. autoapisummary::

   pygmi.clust.graphs.MyMplCanvas
   pygmi.clust.graphs.GraphWindow
   pygmi.clust.graphs.PlotRaster
   pygmi.clust.graphs.PlotMembership
   pygmi.clust.graphs.PlotVRCetc


Module Contents
---------------

.. py:class:: MyMplCanvas(parent=None)

   Bases: :py:obj:`matplotlib.backends.backend_qtagg.FigureCanvasQTAgg`


   Matplotlib canvas widget for the actual plot.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: update_classes(data1)

      Update the class plot.

      :param data1: Input raster dataset.
      :type data1: pygmi.raster.datatypes.Data

      :rtype: None.



   .. py:method:: update_scatter(x, y)

      Update the scatter plot.

      :param x: X coordinates (Number of classes).
      :type x: numpy array
      :param y: Y Coordinates.
      :type y: numpy array

      :rtype: None.



   .. py:method:: update_wireframe(x, y, z)

      Update the wireframe plot.

      :param x: Iteration number.
      :type x: numpy array
      :param y: Number of classes.
      :type y: numpy array
      :param z: z coordinate.
      :type z: numpy array

      :rtype: None.



   .. py:method:: update_membership(data1, mem)

      Update membership plot.

      :param data1: Raster dataset.
      :type data1: pygmi.raster.datatypes.Data
      :param mem: Membership.
      :type mem: int

      :rtype: None.



.. py:class:: GraphWindow(parent=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   Graph Window GUI.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: change_band()

      Combo to change band.

      :rtype: None.



.. py:class:: PlotRaster(parent=None)

   Bases: :py:obj:`GraphWindow`


   Plot Raster Class GUI.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: change_band()

      Combo to change band.

      :rtype: None.



   .. py:method:: run()

      Entry point into the routine, used to run context menu item.

      :rtype: None.



.. py:class:: PlotMembership(parent=None)

   Bases: :py:obj:`GraphWindow`


   Plot Fuzzy Membership data GUI.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: change_band()

      Combo to change band.

      :rtype: None.



   .. py:method:: run()

      Entry point into the routine, used to run context menu item.

      :rtype: None.



   .. py:method:: change_band_two()

      Combo box to choose band.



.. py:class:: PlotVRCetc(parent=None)

   Bases: :py:obj:`GraphWindow`


   Plot VRC, NCE, OBJ and XBI GUI.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: change_band()

      Combo to change band.

      :rtype: None.



   .. py:method:: run()

      Entry point into the routine, used to run context menu item.

      :rtype: None.



