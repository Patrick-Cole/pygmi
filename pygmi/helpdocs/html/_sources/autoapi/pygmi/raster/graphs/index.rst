pygmi.raster.graphs
===================

.. py:module:: pygmi.raster.graphs

.. autoapi-nested-parse::

   Plot Raster Data.

   This module provides a variety of methods to plot raster data via the context
   menu. The following are supported:

    * Correlation coefficients
    * Images
    * Surfaces
    * Histograms



Classes
-------

.. autoapisummary::

   pygmi.raster.graphs.MyMplCanvas
   pygmi.raster.graphs.PlotCCoef
   pygmi.raster.graphs.PlotRaster
   pygmi.raster.graphs.PlotSurface
   pygmi.raster.graphs.PlotScatter
   pygmi.raster.graphs.PlotHist


Functions
---------

.. autoapisummary::

   pygmi.raster.graphs.check_bands
   pygmi.raster.graphs.corr2d


Module Contents
---------------

.. py:class:: MyMplCanvas(parent=None)

   Bases: :py:obj:`matplotlib.backends.backend_qtagg.FigureCanvasQTAgg`


   Matplotlib canvas widget for the actual plot.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: update_ccoef(data1, dmat)

      Update the correlation coefficient plot.

      :param data1: raster dataset to be used.
      :type data1: PyGMI raster Data
      :param dmat: dummy matrix of numbers to be plotted using pcolor.
      :type dmat: numpy array

      :rtype: None.



   .. py:method:: update_raster(data1, cmap)

      Update the raster plot.

      :param data1: raster dataset to be used in contouring
      :type data1: PyGMI raster Data
      :param cmap: Matplotlib colormap description
      :type cmap: str

      :rtype: None.



   .. py:method:: update_hexbin(data1, data2)

      Update the hexbin plot.

      :param data1: raster dataset to be used
      :type data1: PyGMI raster Data
      :param data2: raster dataset to be used
      :type data2: PyGMI raster Data

      :rtype: None.



   .. py:method:: update_surface(data, cmap)

      Update the surface plot.

      :param data: raster dataset to be used
      :type data: PyGMI raster Data
      :param cmap: Matplotlib colormap description
      :type cmap: str

      :rtype: None.



   .. py:method:: update_hist(data1, ylog, iscum)

      Update the histogram plot.

      :param data1: raster dataset to be used
      :type data1: PyGMI raster Data
      :param ylog: Boolean for a log scale on y-axis.
      :type ylog: bool
      :param iscum: Boolean for a cumulative distribution.
      :type iscum: bool

      :rtype: None.



.. py:class:: PlotCCoef(parent=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   Plot 2D Correlation Coefficients.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: run()

      Entry point into the routine, used to run context menu item.

      :rtype: None.



.. py:class:: PlotRaster(parent=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   Plot Raster Class.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: change_band()

      Combo box to choose band.

      :rtype: None.



   .. py:method:: run()

      Entry point into the routine, used to run context menu item.

      :rtype: None.



.. py:class:: PlotSurface(parent=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   Plot Surface Class.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: change_band()

      Combo box to choose band.

      :rtype: None.



   .. py:method:: run()

      Entry point into the routine, used to run context menu item.

      :rtype: None.



.. py:class:: PlotScatter(parent=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   Plot Hexbin Class.

   A Hexbin is a type of scatter plot which is raster.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: change_band()

      Combo box to choose band.

      :rtype: None.



   .. py:method:: run()

      Entry point into the routine, used to run context menu item.

      :rtype: None.



.. py:class:: PlotHist(parent=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   Plot Histogram Class.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: change_band()

      Combo box to choose band.

      :rtype: None.



   .. py:method:: run()

      Entry point into the routine, used to run context menu item.

      :rtype: None.



.. py:function:: check_bands(data)

   Check that band sizes are the same.

   :param data: PyGMI raster dataset.
   :type data: list of pygmi.raster.datatypes.Data

   :returns: **chk** -- True if sizes are the same, False otherwise.
   :rtype: bool


.. py:function:: corr2d(dat1, dat2)

   Calculate the 2D correlation.

   :param dat1: dataset 1 for use in correlation calculation.
   :type dat1: numpy array
   :param dat2: dataset 2 for use in correlation calculation.
   :type dat2: numpy array

   :returns: **out** -- array of correlation coefficients
   :rtype: numpy array


