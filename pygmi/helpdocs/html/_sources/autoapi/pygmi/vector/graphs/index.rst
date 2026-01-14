pygmi.vector.graphs
===================

.. py:module:: pygmi.vector.graphs

.. autoapi-nested-parse::

   Plot Vector Data using Matplotlib.



Classes
-------

.. autoapisummary::

   pygmi.vector.graphs.MyMplCanvas
   pygmi.vector.graphs.PlotCCoef
   pygmi.vector.graphs.PlotHist
   pygmi.vector.graphs.PlotLines
   pygmi.vector.graphs.PlotLineMap
   pygmi.vector.graphs.PlotRose
   pygmi.vector.graphs.PlotVector


Functions
---------

.. autoapisummary::

   pygmi.vector.graphs.heatmap
   pygmi.vector.graphs.annotate_heatmap
   pygmi.vector.graphs.histogram
   pygmi.vector.graphs.rotate


Module Contents
---------------

.. py:class:: MyMplCanvas

   Bases: :py:obj:`matplotlib.backends.backend_qtagg.FigureCanvasQTAgg`


   Matplotlib canvas widget for the actual plot.

   This routine will also allow the picking and movement of nodes of data.


   .. py:method:: button_release_callback(event)

      Mouse button release callback.

      :param event: Button release event.
      :type event: matplotlib.backend_bases.MouseEvent

      :rtype: None.



   .. py:method:: format_coord(x, y)

      Set format coordinate for correlation coefficient plot.

      :param x: x coordinate.
      :type x: float
      :param y: y coordinate.
      :type y: float

      :returns: Output string to display.
      :rtype: str



   .. py:method:: motion_notify_callback(event)

      Move mouse callback.

      :param event: Motion notify event.
      :type event: matplotlib.backend_bases.MouseEvent

      :rtype: None.



   .. py:method:: onpick(event)

      Picker event.

      :param event: Pick event.
      :type event: matplotlib.backend_bases.PickEvent

      :returns: Return TRUE if pick succeeded, False otherwise.
      :rtype: bool



   .. py:method:: resizeline(event)

      Resize event.

      :param event: Resize event.
      :type event: matplotlib.backend_bases.ResizeEvent

      :rtype: None.



   .. py:method:: textresize(axes)

      Resize the text on a correlation plot when zooming.

      :param axes: Current Matplotlib axes.
      :type axes: Matplotlib axes

      :rtype: None.



   .. py:method:: update_ccoef(data, style='Normal')

      Update the plot from point data.

      :param data: GeoPandas data in a dictionary.
      :type data: dictionary
      :param style: Style of colour mapping.
      :type style: str

      :rtype: None.



   .. py:method:: update_lines(r, data)

      Update the plot from point data.

      :param r: array of distances, for the x-axis
      :type r: numpy array
      :param data: array of data to be plotted on the y-axis
      :type data: numpy array

      :rtype: None.



   .. py:method:: update_lmap(data, ival, scale, uselabels)

      Update the plot from line data.

      :param data: Line data
      :type data: Pandas dataframe
      :param ival: dictionary key representing the line data channel to be plotted.
      :type ival: dictionary key
      :param scale: scale of exaggeration for the profile data on the map.
      :type scale: float
      :param uselabels: boolean choice whether to use labels or not.
      :type uselabels: bool

      :rtype: None.



   .. py:method:: update_vector(data, col, style=None)

      Update the plot from vector data.

      :param data: GeoPandas data in a dictionary.
      :type data: dictionary
      :param col: Label for column to extract.
      :type col: str
      :param style: Style of colour mapping.
      :type style: str or None

      :rtype: None.



   .. py:method:: update_rose(data, rtype, nbins=8, equal=False)

      Update the rose diagram plot using vector data.

      :param data: GeoPandas data in a dictionary. It should be 'LineString'
      :type data: dictionary
      :param rtype: Rose diagram type. Can be either 0 or 1.
      :type rtype: int
      :param nbins: Number of bins used in rose diagram. The default is 8.
      :type nbins: int, optional
      :param equal: Option for an equal area rose diagram. The default is False.
      :type equal: bool, optional

      :rtype: None.



   .. py:method:: update_hist(data, col, ylog, iscum)

      Update the histogram plot.

      :param data: GeoPandas data in a dictionary.
      :type data: dictionary
      :param col: Label for column to extract.
      :type col: str
      :param ylog: Boolean for a log scale on y-axis.
      :type ylog: bool
      :param iscum: Boolean for a cumulative distribution.
      :type iscum: bool

      :rtype: None.



.. py:class:: PlotCCoef(parent=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   GUI to plot correlation coefficients.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional


   .. py:method:: change_band()

      Combo box to choose band.

      :rtype: None.



   .. py:method:: run()

      Entry point into the routine, used to run context menu item.

      :rtype: None.



.. py:class:: PlotHist(parent=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   GUI to plot histogram from vectors.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional


   .. py:method:: change_band()

      Combo box to choose band.

      :rtype: None.



   .. py:method:: run()

      Entry point into the routine, used to run context menu item.

      :rtype: None.



.. py:class:: PlotLines(parent=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   GUI to plot lines.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional


   .. py:method:: change_band()

      Combo box to choose band.

      :rtype: None.



   .. py:method:: run()

      Entry point into the routine, used to run context menu item.

      :rtype: None.



.. py:class:: PlotLineMap(parent=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   GUI to plot a line map.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional


   .. py:method:: change_band()

      Combo box to choose band.

      :rtype: None.



   .. py:method:: run()

      Entry point into the routine, used to run context menu item.

      :rtype: None.



.. py:class:: PlotRose(parent=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   GUI to plot rose diagrams.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional


   .. py:method:: change_band()

      Combo box to choose band.

      :rtype: None.



   .. py:method:: run()

      Entry point into the routine, used to run context menu item.

      :rtype: None.



.. py:class:: PlotVector(parent=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   GUI to plot vectors.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional


   .. py:method:: change_band()

      Combo box to choose band.

      :rtype: None.



   .. py:method:: run()

      Entry point into the routine, used to run context menu item.

      :rtype: None.



.. py:function:: heatmap(data, row_labels, col_labels, ax, *, cbar_kw=None, cbarlabel='', **kwargs)

   Create a heatmap from a numpy array and two lists of labels.

   From Matplotlib.org

   :param data: A 2D numpy array of shape (M, N).
   :param row_labels: A list or array of length M with the labels for the rows.
   :param col_labels: A list or array of length N with the labels for the columns.
   :param ax: A `matplotlib.axes.Axes` instance to which the heatmap is plotted.
   :param cbar_kw: A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
   :param cbarlabel: The label for the colorbar.  Optional.
   :param \*\*kwargs: All other arguments are forwarded to `imshow`.


.. py:function:: annotate_heatmap(im, data=None, valfmt='{x:.2f}', textcolors=('black', 'white'), threshold=None, **textkw)

   Annotate a heatmap.

   From Matplotlib.org.

   :param im: The AxesImage to be labelled.
   :param data: Data used to annotate.  If None, the image's data is used.  Optional.
   :param valfmt: The format of the annotations inside the heatmap.  This should either
                  use the string format method, e.g. "$ {x:.2f}", or be a
                  `matplotlib.ticker.Formatter`.  Optional.
   :param textcolors: A pair of colours.  The first is used for values below a threshold,
                      the second for those above.  Optional.
   :param threshold: Value in data units according to which the colours from textcolors are
                     applied.  If None (the default) uses the middle of the colormap as
                     separation.  Optional.
   :param \*\*kwargs: All other arguments are forwarded to each call to `text` used to create
                      the text labels.


.. py:function:: histogram(x, y=None, xmin=None, xmax=None, bins=10)

   Histogram.

   Calculate histogram of a set of data. It is different from a
   conventional histogram in that instead of summing elements of
   specific values, this allows the sum of weights/probabilities on a per
   element basis.

   :param x: Input data
   :type x: numpy array
   :param y: Input data weights. The default is None.
   :type y: numpy array
   :param xmin: Lower value for the bins. The default is None.
   :type xmin: float
   :param xmax: Upper value for the bins. The default is None.
   :type xmax: float
   :param bins: number of bins. The default is 10.
   :type bins: int

   :returns: * **hist** (*numpy array*) -- The values of the histogram
             * **bin_edges** (*numpy array*) -- bin edges of the histogram


.. py:function:: rotate(origin, point, angle)

   Rotate a point counterclockwise by a given angle around a given origin.

   The angle should be given in radians.

   :param origin: List containing origin point (ox, oy)
   :type origin: list
   :param point: List containing point to be rotated (px, py)
   :type point: list
   :param angle: Angle in radians.
   :type angle: float

   :returns: * **qx** (*float*) -- Rotated x-coordinate.
             * **qy** (*float*) -- Rotated y-coordinate.


