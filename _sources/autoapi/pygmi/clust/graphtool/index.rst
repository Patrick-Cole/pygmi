pygmi.clust.graphtool
=====================

.. py:module:: pygmi.clust.graphtool

.. autoapi-nested-parse::

   Multi-function graphing tool for use with cluster analysis.



Classes
-------

.. autoapisummary::

   pygmi.clust.graphtool.GraphHist
   pygmi.clust.graphtool.GraphMap
   pygmi.clust.graphtool.PolygonInteractor
   pygmi.clust.graphtool.ScatterPlot


Functions
---------

.. autoapisummary::

   pygmi.clust.graphtool.dist_point_to_segment


Module Contents
---------------

.. py:class:: GraphHist(parent=None)

   Bases: :py:obj:`matplotlib.backends.backend_qtagg.FigureCanvasQTAgg`


   Histogram graph widget.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: get_hist(bins)

      Routine to get the scattergram with histogram overlay.

      :param bins: Number of bins.
      :type bins: int

      :returns: **xymahist** -- Output data.
      :rtype: numpy array



   .. py:method:: get_clust_scat(bins, dattmp, ctmp)

      Routine to get the scattergram with cluster overlay.

      :param bins: Number of bins.
      :type bins: int
      :param dattmp: List of PyGMI raster data (pygmi.raster.datatypes.Data).
      :type dattmp: list
      :param ctmp: Cluster indices.
      :type ctmp: list

      :returns: **xymahist** -- Output data.
      :rtype: numpy array



   .. py:method:: init_graph()

      Initialize the Graph.

      :rtype: None.



   .. py:method:: polyint()

      Polygon Interactor routine.

      :rtype: None.



   .. py:method:: setup_coords()

      Routine to setup the coordinates for the scattergram.

      :rtype: None.



   .. py:method:: setup_hist()

      Routine to setup the 1D histograms.

      :rtype: None.



   .. py:method:: update_graph(clearaxis=False)

      Draw Routine.

      :param clearaxis: True to clear the axis. The default is False.
      :type clearaxis: bool, optional

      :rtype: None.



.. py:class:: GraphMap(parent=None)

   Bases: :py:obj:`matplotlib.backends.backend_qtagg.FigureCanvasQTAgg`


   Map widget.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: init_graph()

      Initialize the Graph.

      :rtype: None.



   .. py:method:: polyint()

      Polygon Integrator.

      :rtype: None.



   .. py:method:: update_graph()

      Draw routine.

      :rtype: None.



.. py:class:: PolygonInteractor(axtmp, pntxy)

   Bases: :py:obj:`PyQt6.QtCore.QObject`


   Polygon Interactor for the graph tool.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional

   .. attribute:: epsilon

      Epsilon tolerance for index detection.

      :type: int

   .. attribute:: polyi_changed

      Qt signal when polygon has changed.

      :type: QtCore.pyqtSignal


   .. py:method:: draw_callback()

      Draw callback.

      :rtype: None.



   .. py:method:: new_poly(npoly)

      Create new Polygon.

      :param npoly: New polygon coordinates.
      :type npoly: list

      :rtype: None.



   .. py:method:: get_ind_under_point(event)

      Get the index of vertex under point if within epsilon tolerance.

      :param event: Mouse event.
      :type event: matplotlib.backend_bases.MouseEvent

      :returns: **ind** -- Index of vertex under point.
      :rtype: int or None



   .. py:method:: button_press_callback(event)

      Button press callback.

      :param event: Mouse event.
      :type event: matplotlib.backend_bases.MouseEvent

      :rtype: None.



   .. py:method:: button_release_callback(event)

      Button release callback.

      :param event: Mouse event.
      :type event: matplotlib.backend_bases.MouseEvent

      :rtype: None.



   .. py:method:: update_plots()

      Update plots.

      :rtype: None.



   .. py:method:: motion_notify_callback(event)

      Mouse notify callback.

      :param event: Mouse event.
      :type event: matplotlib.backend_bases.MouseEvent

      :rtype: None.



.. py:class:: ScatterPlot(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Main graph tool GUI routine.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: on_cp_dpoly()

      On cross plot, delete polygon.

      :rtype: None.



   .. py:method:: on_map_dpoly()

      On map delete polygon.

      :rtype: None.



   .. py:method:: on_cp_combo()

      On cross plot, combo.

      :rtype: None.



   .. py:method:: on_cp_combo2()

      On cross plot, combo 2.

      :rtype: None.



   .. py:method:: on_cp_combo3()

      On cross plot, combo 3.

      :rtype: None.



   .. py:method:: on_map_combo()

      On map combo.

      :rtype: None.



   .. py:method:: on_map_combo2()

      On map combo 2.

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



   .. py:method:: update_map()

      Update map.

      :rtype: None.



   .. py:method:: update_hist()

      Update histogram.

      :rtype: None.



.. py:function:: dist_point_to_segment(p, s0, s1)

   Distance of a point to a line segment.

   Reimplementation of Matplotlib's dist_point_to_segment, after it was
   depreciated. Follows http://geomalgorithms.com/a02-_lines.html

   :param p: Point.
   :type p: numpy array
   :param s0: Start of segment.
   :type s0: numpy array
   :param s1: End of segment.
   :type s1: numpy array

   :returns: Distance of point to segment.
   :rtype: numpy array


