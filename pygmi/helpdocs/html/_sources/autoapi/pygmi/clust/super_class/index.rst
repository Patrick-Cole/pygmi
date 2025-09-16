pygmi.clust.super_class
=======================

.. py:module:: pygmi.clust.super_class

.. autoapi-nested-parse::

   Supervised Classification tool.



Classes
-------

.. autoapisummary::

   pygmi.clust.super_class.GraphMap
   pygmi.clust.super_class.PolygonInteractor
   pygmi.clust.super_class.SuperClass


Functions
---------

.. autoapisummary::

   pygmi.clust.super_class.dist_point_to_segment


Module Contents
---------------

.. py:class:: GraphMap(parent=None)

   Bases: :py:obj:`matplotlib.backends.backend_qtagg.FigureCanvasQTAgg`


   Graph map widget.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: polyint(dat)

      Polygon integrator.

      :rtype: None.



   .. py:method:: compute_initial_figure(dat)

      Compute initial figure.

      :param dat: PyGMI dataset/s (pygmi.raster.datatypes.Data) in a dictionary.
      :type dat: dict

      :rtype: None.



   .. py:method:: update_plot(dat)

      Update plot.

      :param dat: PyGMI dataset/s (pygmi.raster.datatypes.Data) in a dictionary.
      :type dat: dict

      :rtype: None.



   .. py:method:: update_class(dat)

      Update plot.

      :param dat: PyGMI dataset/s (pygmi.raster.datatypes.Data) in a dictionary.
      :type dat: dict

      :rtype: None.



.. py:class:: PolygonInteractor(axtmp, pntxy)

   Bases: :py:obj:`PySide6.QtCore.QObject`


   Polygon Interactor for the supervised classification tool.

   :param axtmp: Matplotlib axis.
   :type axtmp: matplotlib.axes._axes.Axes
   :param pntxy: X and Y mouse coordinates in N by 2  array.
   :type pntxy: numpy array

   .. attribute:: epsilon

      Epsilon tolerance for index detection.

      :type: int

   .. attribute:: polyi_changed

      Qt signal when polygon has changed.

      :type: QtCore.Signal


   .. py:method:: draw_callback(event=None)

      Draw callback.

      :param event: Draw event object. The default is None.
      :type event: matplotlib.backend_bases.DrawEvent, optional

      :rtype: None.



   .. py:method:: new_poly(npoly=None)

      Create new polygon.

      :param npoly: New polygon coordinates.
      :type npoly: list or None, optional

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

      :param event: Mouse Event.
      :type event: matplotlib.backend_bases.MouseEvent

      :rtype: None.



   .. py:method:: update_plots()

      Update plots.

      :rtype: None.



   .. py:method:: motion_notify_callback(event)

      Motion notify on mouse movement.

      :param event: Mouse event.
      :type event: matplotlib.backend_bases.MouseEvent

      :rtype: None.



.. py:class:: SuperClass(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Main supervised classification GUI.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: calculate()

      Calculate new clusters.

      :rtype: None.



   .. py:method:: class_change()

      Routine called when current classification choice changes.

      :rtype: None.



   .. py:method:: calc_metrics()

      Calculate metrics.

      :rtype: None.



   .. py:method:: updatepoly(xycoords=None)

      Update polygon.

      :param xycoords: x, y coordinates. The default is None.
      :type xycoords: numpy array, optional

      :rtype: None.



   .. py:method:: oncellchange(row, col)

      Routine activated whenever a cell is changed.

      :param row: Current row.
      :type row: int
      :param col: Current column.
      :type col: int

      :rtype: None.



   .. py:method:: onrowchange(current, previous)

      Routine activated whenever a row is changed.

      :param current: current item.
      :type current: QTableWidgetItem
      :param previous: previous item.
      :type previous: QTableWidgetItem

      :rtype: None.



   .. py:method:: on_apoly()

      On add polygon.

      :rtype: None.



   .. py:method:: on_dpoly()

      On delete polygon.

      :rtype: None.



   .. py:method:: on_combo()

      On combo to choose type of plot for data.

      :rtype: None.



   .. py:method:: on_radio()

      On radiobutton to choose type of plot for data.

      :rtype: None.



   .. py:method:: load_shape()

      Load shapefile.

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: save_shape()

      Save shapefile.

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: settings(nodialog=False)

      Entry point into item.

      :param nodialog: Run settings without a dialog. The default is False.
      :type nodialog: bool, optional

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: saveproj()

      Save project data from class.

      :rtype: None.



   .. py:method:: init_classifier()

      Initialise classifier.

      :returns: * **classifier** (*object*) -- Scikit learn classification object.
                * **lbls** (*numpy array*) -- Class labels.
                * **datall** (*numpy array*) -- Dataset.
                * **X_test** (*numpy array*) -- X test dataset.
                * **y_test** (*numpy array*) -- Y test dataset.
                * **tlbls** (*numpy array*) -- Class labels.



   .. py:method:: update_class_polys()

      Update class poly summaries.



.. py:function:: dist_point_to_segment(p, s0, s1)

   Dist point to segment.

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


