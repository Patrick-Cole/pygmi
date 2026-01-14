pygmi.raster.cliptozoom
=======================

.. py:module:: pygmi.raster.cliptozoom

.. autoapi-nested-parse::

   Clip to Zoom.

   This module allows a raster dataset to be clipped to the current zoomed
   extents.



Classes
-------

.. autoapisummary::

   pygmi.raster.cliptozoom.MyMplCanvas
   pygmi.raster.cliptozoom.ClipToZoom


Module Contents
---------------

.. py:class:: MyMplCanvas

   Bases: :py:obj:`matplotlib.backends.backend_qtagg.FigureCanvasQTAgg`


   Matplotlib canvas widget for the actual plot.


   .. py:method:: update_raster(data1, cmap)

      Update the raster plot.

      :param data1: raster dataset to be used in contouring
      :type data1: PyGMI raster Data
      :param cmap: Matplotlib colormap description
      :type cmap: str

      :rtype: None.



.. py:class:: ClipToZoom(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Clip to zoom GUI Class.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional


   .. py:method:: change_band()

      Combo box to choose band.

      :rtype: None.



   .. py:method:: settings(nodialog=False)

      Run.

      :rtype: None.



