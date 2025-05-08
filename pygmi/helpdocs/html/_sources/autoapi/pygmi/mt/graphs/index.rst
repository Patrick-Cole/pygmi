pygmi.mt.graphs
===============

.. py:module:: pygmi.mt.graphs

.. autoapi-nested-parse::

   Plot Data using Matplotlib.



Classes
-------

.. autoapisummary::

   pygmi.mt.graphs.MyMplCanvas
   pygmi.mt.graphs.PlotPoints


Module Contents
---------------

.. py:class:: MyMplCanvas(parent=None, width=8, height=6, dpi=100)

   Bases: :py:obj:`matplotlib.backends.backend_qtagg.FigureCanvasQTAgg`


   Matplotlib canvas widget for the actual plot.

   This routine will also allow the picking and movement of nodes of data.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: button_release_callback(event)

      Mouse button release callback.

      :param event: event variable.
      :type event: event

      :rtype: None.



   .. py:method:: motion_notify_callback(event)

      Move mouse callback.

      :param event: event variable.
      :type event: event

      :rtype: None.



   .. py:method:: onpick(event)

      Picker event.

      :param event: event variable.
      :type event: event

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: update_line(data, ival, itype)

      Update the plot from point data.

      :param data: EDI data.
      :type data: EDI data object
      :param ival: dictionary key.
      :type ival: str
      :param itype: dictionary key.
      :type itype: str

      :rtype: None.



.. py:class:: PlotPoints(parent=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   Plot points class.


   .. py:method:: change_band()

      Combo to choose band.

      :rtype: None.



   .. py:method:: run()

      Entry point into the routine, used to run context menu item.

      :rtype: None.



