pygmi.raster.cooper
===================

.. py:module:: pygmi.raster.cooper

.. autoapi-nested-parse::

   A collection of routines by Gordon Cooper for filtering raster data.

   |    School of Geosciences, University of the Witwatersrand
   |    Johannesburg, South Africa
   |    cooperg@geosciences.wits.ac.za
   |    http://www.wits.ac.za/science/geophysics/gc.htm



Classes
-------

.. autoapisummary::

   pygmi.raster.cooper.Gradients
   pygmi.raster.cooper.Visibility2d
   pygmi.raster.cooper.AGC


Functions
---------

.. autoapisummary::

   pygmi.raster.cooper.gradients
   pygmi.raster.cooper.thgrad
   pygmi.raster.cooper.derivative_ratio
   pygmi.raster.cooper.visibility2d
   pygmi.raster.cooper.visibilitytot
   pygmi.raster.cooper.nextpow2
   pygmi.raster.cooper.agc


Module Contents
---------------

.. py:class:: Gradients(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Class used to gather information via a GUI, for function gradients.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional

   .. attribute:: azi

      Azimuth/filter direction (degrees)

      :type: float

   .. attribute:: elev

      Elevation (for sunshading, degrees from horizontal)

      :type: float

   .. attribute:: order

      Order of DR filter - see paper. Try 1 first.

      :type: int


   .. py:method:: setupui()

      Set up UI.

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



   .. py:method:: radiochange()

      Check radio button state.

      :rtype: None.



.. py:function:: gradients(data, azi, xint, yint)

   Gradients.

   Compute directional derivative of image data. Based on code by
   Gordon Cooper.

   :param data: input numpy data array
   :type data: numpy array
   :param azi: Filter direction (degrees)
   :type azi: float
   :param xint: X interval/distance.
   :type xint: float
   :param yint: Y interval/distance.
   :type yint: float

   :returns: **dt1** -- returns directional derivative
   :rtype: float


.. py:function:: thgrad(data, xint, yint)

   Gradients.

   Compute total horizontal gradient.

   :param data: input numpy data array
   :type data: numpy array
   :param xint: X interval/distance.
   :type xint: float
   :param yint: Y interval/distance.
   :type yint: float

   :returns: **dt1** -- returns gradient.
   :rtype: float


.. py:function:: derivative_ratio(data, azi, order)

   Compute derivative ratio of image data. Based on code by Gordon Cooper.

   :param data: input numpy data array
   :type data: numpy array
   :param azi: Filter direction (degrees)
   :type azi: float
   :param order: Order of DR filter - see paper. Try 1 first.
   :type order: float

   :returns: **dr** -- returns derivative ratio
   :rtype: float


.. py:class:: Visibility2d(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Class used to gather information via a GUI, for function visibility2d.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional

   .. attribute:: wsize

      window size, must be odd

      :type: int

   .. attribute:: dh

      height of observer above surface

      :type: float


   .. py:method:: setupui()

      Set up UI.

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



.. py:function:: visibility2d(data, wsize, dh, piter=iter)

   Compute visibility as a textural measure.

   Compute vertical derivatives by calculating the visibility at different
   heights above the surface (see paper)

   :param data: input dataset - numpy MxN array
   :type data: numpy array
   :param wsize: window size, must be odd
   :type wsize: int
   :param dh: height of observer above surface
   :type dh: float
   :param piter: Progress bar iterable. The default is iter.
   :type piter: function, optional

   :returns: * **vtot** (*numpy array*) -- Total visibility.
             * **vstd** (*numpy array*) -- Visibility variation.
             * **vsum** (*numpy array*) -- Visibility vector resultant.


.. py:function:: visibilitytot(data, wsize, dh)

   Compute visibility as a textural measure.

   Compute vertical derivatives by calculating the visibility at different
   heights above the surface (see paper)

   :param data: input dataset - numpy MxN array
   :type data: numpy array
   :param wsize: window size, must be odd
   :type wsize: int
   :param dh: height of observer above surface
   :type dh: float

   :returns: * **vtot** (*numpy array*) -- Total visibility.
             * **vstd** (*numpy array*) -- Visibility variation.
             * **vsum** (*numpy array*) -- Visibility vector resultant.


.. py:function:: nextpow2(n)

   Next power of 2.

   :param n: Current value.
   :type n: float or numpy array

   :returns: **m_i** -- Output.
   :rtype: float or numpy array


.. py:class:: AGC(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Class used to gather information via a GUI, for function AGC.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional

   .. attribute:: wsize

      window size, must be odd

      :type: int


   .. py:method:: setupui()

      Set up UI.

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



.. py:function:: agc(data, wsize, atype='mean', nodata=0.0, piter=iter)

   AGC for map data, based on code by Gordon Cooper.

   :param data: Raster data.
   :type data: numpy array
   :param wsize: Window size, must be odd.
   :type wsize: int
   :param atype: AGC type - can be median, rms or mean, default is 'mean'.
   :type atype: str, optional
   :param nodata: no data value, default is 0.
   :type nodata: float, optional
   :param piter: Progress bar iterable. The default is iter.
   :type piter: function, optional

   :returns: **agcdata** -- Output AGC data
   :rtype: numpy array


