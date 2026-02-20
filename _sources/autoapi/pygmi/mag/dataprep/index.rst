pygmi.mag.dataprep
==================

.. py:module:: pygmi.mag.dataprep

.. autoapi-nested-parse::

   A set of Magnetic Data routines.



Classes
-------

.. autoapisummary::

   pygmi.mag.dataprep.ASig
   pygmi.mag.dataprep.Tilt1
   pygmi.mag.dataprep.RTP


Functions
---------

.. autoapisummary::

   pygmi.mag.dataprep.asig
   pygmi.mag.dataprep.tilt1
   pygmi.mag.dataprep.nextpow2
   pygmi.mag.dataprep.rtp
   pygmi.mag.dataprep.gradient2D


Module Contents
---------------

.. py:class:: ASig(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Calculate analytic signal via a GUI.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional


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



.. py:function:: asig(data1, showlog=print, piter=iter)

   Tilt angle calculations.

   Based on work by Gordon Cooper (School of Geosciences, University of the
                                   Witwatersrand, Johannesburg, South Africa)

   :param data1: data with matrix of double to be filtered
   :type data1: pygmi.raster.datatypes.Data
   :param showlog: Show information using a function. The default is print.
   :type showlog: function, optional
   :param piter: Progress bar iterator. The default is iter.
   :type piter: function, optional

   :returns: **asig1** -- Analytic signal
   :rtype: numpy masked array


.. py:class:: Tilt1(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Class used to gather information via a GUI, for function tilt1.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional

   .. attribute:: azi

      directional filter azimuth in degrees from East

      :type: float

   .. attribute:: smooth

      size of smoothing matrix to use - must be odd input 0 for no smoothing

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



.. py:function:: tilt1(data1, azi, s, k=2, showlog=print, piter=iter)

   Tilt angle calculations.

   Based on work by Gordon Cooper (School of Geosciences, University of the
                                   Witwatersrand, Johannesburg, South Africa)

   :param data1: data with matrix of double to be filtered
   :type data1: pygmi.raster.datatypes.Data
   :param azi: directional filter azimuth in degrees from East
   :type azi: float
   :param s: size of smoothing matrix to use - must be odd input 0 for no smoothing
   :type s: int
   :param k: Factor for EHGA filter. Must be > 0. Optional.
   :type k: int
   :param showlog: Show information using a function. The default is print.
   :type showlog: function, optional
   :param piter: Progress bar iterator. The default is iter.
   :type piter: function, optional

   :returns: * **t1** (*numpy masked array*) -- Standard tilt angle
             * **th** (*numpy masked array*) -- Hyperbolic tilt angle
             * **t2** (*numpy masked array*) -- Second order tilt angle
             * **ta** (*numpy masked array*) -- Tilt Based Directional Derivative
             * **tdx** (*numpy masked array*) -- Total Derivative
             * **tahg** (*numpy masked array*) -- Tilt Angle of the Horizontal Gradient
             * **ehga** (*numpy masked array*) -- Enhanced Horizontal Gradient Amplitude


.. py:function:: nextpow2(n)

   Next power of 2.

   Based on work by Gordon Cooper (School of Geosciences, University of the
                                   Witwatersrand, Johannesburg, South Africa).

   :param n: Current value.
   :type n: float or numpy array

   :returns: **m_i** -- Output.
   :rtype: float or numpy array


.. py:class:: RTP(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Perform Reduction to the Pole on Magnetic data.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional


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



   .. py:method:: acceptall()

      Accept option.

      Updates self.outdata, which is used as input to other modules.

      :rtype: None.



.. py:function:: rtp(data, I_deg, D_deg, Ia=20, showlog=print, piter=iter)

   Reduction to the pole.

   :param data: PyGMI raster data.
   :type data: pygmi.raster.datatypes.Data
   :param I_deg: Magnetic inclination.
   :type I_deg: float
   :param D_deg: Magnetic declination.
   :type D_deg: float
   :param Ia: Amplitude correction inclination Ia in degree. The default is 20.
   :type Ia: float
   :param showlog: Show information using a function. The default is print.
   :type showlog: function, optional
   :param piter: Progress bar iterator. The default is iter.
   :type piter: function, optional

   :returns: **dat** -- PyGMI raster data.
   :rtype: pygmi.raster.datatypes.Data


.. py:function:: gradient2D(daty, datx)

   Perform 2D gradient where spacing is inconsistent in 2D.

   :param daty: _description_
   :type daty: numpy array
   :param datx: _description_
   :type datx: numpy array

   :returns: **dx** -- output gradient array
   :rtype: numpy array


