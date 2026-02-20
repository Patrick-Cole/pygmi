pygmi.raster.fft
================

.. py:module:: pygmi.raster.fft

.. autoapi-nested-parse::

   A set of FFT routines.



Functions
---------

.. autoapisummary::

   pygmi.raster.fft.fftprep
   pygmi.raster.fft.fft_getkxy
   pygmi.raster.fft.nextpow2
   pygmi.raster.fft.calculate_raps


Module Contents
---------------

.. py:function:: fftprep(data)

   FFT preparation.

   This routine pads using minimum curvature gridding.

   :param data: Input dataset.
   :type data: pygmi.raster.datatypes.Data

   :returns: * **zfin** (*numpy array.*) -- Output prepared data.
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


.. py:function:: nextpow2(n)

   Next power of 2.

   :param n: Current value.
   :type n: float or numpy array

   :returns: **m_i** -- Output.
   :rtype: float or numpy array


.. py:function:: calculate_raps(dat)

   Calculates the Radially Averaged Power Spectrum (RAPS) of a 2D dataset.

   :param dat: Input dataset.
   :type dat: pygmi.raster.datatypes.Data

   :returns: * **k** (*np.ndarray*) -- The 1D array of radial wavenumbers.
             * **raps** (*np.ndarray*) -- The 1D array of radially averaged power spectrum values.


