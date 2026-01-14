pygmi.mag.tiltdepth
===================

.. py:module:: pygmi.mag.tiltdepth

.. autoapi-nested-parse::

   Tilt Depth Routine.

   Based on work by EH Stettler

   .. rubric:: References

   Salem et al., 2007, Leading Edge, Dec,p1502-5



Classes
-------

.. autoapisummary::

   pygmi.mag.tiltdepth.TiltDepth


Functions
---------

.. autoapisummary::

   pygmi.mag.tiltdepth.tiltdepth
   pygmi.mag.tiltdepth.distpc
   pygmi.mag.tiltdepth.vgrad


Module Contents
---------------

.. py:class:: TiltDepth(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Primary class for the Tilt Depth.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional

   .. attribute:: self.mmc

      main canvas containing the image

      :type: FigureCanvas


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: rtp_choice()

      Check if RTP must be done.

      :rtype: None.



   .. py:method:: save_depths()

      Save depths.

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: change_cbar()

      Change the colour map for the colour bar.

      :rtype: None.



   .. py:method:: calculate()

      Routine which occurs when apply button is pressed.

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



.. py:function:: tiltdepth(data, inc=None, dec=None, pbar=None, showlog=print)

   Calculate tilt depth.

   Output is stored in self.outdata.

   :param data: PyGMI raster dataset.
   :type data: pygmi.raster.datatypes.Data

   :rtype: None.


.. py:function:: distpc(dx, dy, dx0, dy0, dcnt)

   Find closest distances.

   :param dx: X array.
   :type dx: numpy array
   :param dy: Y array.
   :type dy: numpy array
   :param dx0: X point to measure distance from.
   :type dx0: float
   :param dy0: Y point to measure distance from.
   :type dy0: float
   :param dcnt: Starting index to measure distance from.
   :type dcnt: int

   :returns: **dcnt** -- Index of closest distance found in x and y arrays.
   :rtype: int


.. py:function:: vgrad(cnt)

   Get contour gradients at vertices.

   :param cnt: Output from Matplotlib's axes.contour.
   :type cnt: axes.contour

   :returns: * **gx** (*numpy array*) -- X gradients.
             * **gy** (*numpy array*) -- Y gradients.
             * **cgrad** (*numpy array*) -- Contour gradient.
             * **cntid** (*numpy array*) -- Contour index.


