pygmi.mag.dataprep
==================

.. py:module:: pygmi.mag.dataprep

.. autoapi-nested-parse::

   A set of Magnetic Data routines.



Classes
-------

.. autoapisummary::

   pygmi.mag.dataprep.Tilt1
   pygmi.mag.dataprep.RTP


Functions
---------

.. autoapisummary::

   pygmi.mag.dataprep.tilt1
   pygmi.mag.dataprep.nextpow2
   pygmi.mag.dataprep.vertical
   pygmi.mag.dataprep.fftprep
   pygmi.mag.dataprep.fft_getkxy
   pygmi.mag.dataprep.rtp


Module Contents
---------------

.. py:class:: Tilt1(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Class used to gather information via a GUI, for function tilt1.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional

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



.. py:function:: tilt1(data, azi, s, k=2)

   Tilt angle calculations.

   Based on work by Gordon Cooper (School of Geosciences, University of the
                                   Witwatersrand, Johannesburg, South Africa)

   :param data: matrix of double to be filtered
   :type data: numpy masked array
   :param azi: directional filter azimuth in degrees from East
   :type azi: float
   :param s: size of smoothing matrix to use - must be odd input 0 for no smoothing
   :type s: int
   :param k: Factor for EHGA filter. Must be > 0. Optional.
   :type k: int

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


.. py:function:: vertical(data, npts=None, xint=1, order=1)

   Vertical derivative.

   Based on work by Gordon Cooper (School of Geosciences, University of the
                                   Witwatersrand, Johannesburg, South Africa).

   :param data: Input data.
   :type data: numpy array
   :param npts: Number of points. The default is None.
   :type npts: int, optional
   :param xint: X interval. The default is 1.
   :type xint: float, optional
   :param order: Order of derivative. The default is 1.
   :type order: int

   :returns: **dz** -- Output data
   :rtype: numpy array


.. py:class:: RTP(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Perform Reduction to the Pole on Magnetic data.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


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



.. py:function:: fftprep(data)

   FFT Preparation.

   :param data: Input dataset.
   :type data: numpy array

   :returns: * **zfin** (*numpy array.*) -- Output prepared data.
             * **rdiff** (*int*) -- rows divided by 2.
             * **cdiff** (*int*) -- columns divided by 2.
             * **datamedian** (*float*) -- Median of data.


.. py:function:: fft_getkxy(fftmod, xdim, ydim)

   Get KX and KY.

   :param fftmod: FFT data.
   :type fftmod: numpy array
   :param xdim: cell x dimension.
   :type xdim: float
   :param ydim: cell y dimension.
   :type ydim: float

   :returns: * **KX** (*numpy array*) -- x sample frequencies.
             * **KY** (*numpy array*) -- y sample frequencies.


.. py:function:: rtp(data, I_deg, D_deg)

   Reduction to the pole.

   :param data: PyGMI raster data.
   :type data: pygmi.raster.datatypes.Data
   :param I_deg: Magnetic inclination.
   :type I_deg: float
   :param D_deg: Magnetic declination.
   :type D_deg: float

   :returns: **dat** -- PyGMI raster data.
   :rtype: pygmi.raster.datatypes.Data


