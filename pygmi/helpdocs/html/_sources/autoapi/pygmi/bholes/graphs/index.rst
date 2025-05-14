pygmi.bholes.graphs
===================

.. py:module:: pygmi.bholes.graphs

.. autoapi-nested-parse::

   Methods to plot borehole data via the context menu.



Classes
-------

.. autoapisummary::

   pygmi.bholes.graphs.MyMplCanvas
   pygmi.bholes.graphs.PlotLog


Functions
---------

.. autoapisummary::

   pygmi.bholes.graphs.gethatch
   pygmi.bholes.graphs.commentprep
   pygmi.bholes.graphs.chkname


Module Contents
---------------

.. py:class:: MyMplCanvas(parent=None)

   Bases: :py:obj:`matplotlib.backends.backend_qtagg.FigureCanvasQTAgg`


   Matplotlib canvas widget for the actual plot.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: update_legend(data1)

      Update the plot legend.

      :param data1: Dictionary containing the data.
      :type data1: dictionary

      :rtype: None.



   .. py:method:: update_log(data1)

      Update the borehole log plot.

      :param data1: PyGMI log dataset to be used.
      :type data1: dictionary.

      :rtype: None.



.. py:class:: PlotLog(parent=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   Class to plot the borehole log.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: change_band()

      Combo box to choose the borehole to display.

      :rtype: None.



   .. py:method:: run()

      Entry point into the routine, used to run context menu item.

      :rtype: None.



.. py:function:: gethatch(svgfile)

   Get hatching from an SVG file, to be used on the log.

   :param svgfile: SVG filename.
   :type svgfile: str

   :rtype: None.


.. py:function:: commentprep(mystring, slen=50)

   Create the correct case for a string and inserts carriage returns.

   :param mystring: String to correct.
   :type mystring: str
   :param slen: String length. The default is 50.
   :type slen: int, optional

   :returns: **finstring** -- Output string.
   :rtype: str


.. py:function:: chkname(iname)

   Check a filename for illegal characters.

   :param iname: Input filename.
   :type iname: str

   :returns: **iname** -- Corrected filename.
   :rtype: str


