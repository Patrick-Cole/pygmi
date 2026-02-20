pygmi.mag.matchedfilt
=====================

.. py:module:: pygmi.mag.matchedfilt

.. autoapi-nested-parse::

   Matched filtering routine.



Classes
-------

.. autoapisummary::

   pygmi.mag.matchedfilt.MatchedFilt


Functions
---------

.. autoapisummary::

   pygmi.mag.matchedfilt.getbutter


Module Contents
---------------

.. py:class:: MatchedFilt(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Primary class for matched filtering.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional

   .. attribute:: self.mmc

      main canvas containing the image

      :type: FigureCanvas


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: settings(nodialog=False)

      Entry point into item.

      :param nodialog: Run settings without a dialog. The default is False.
      :type nodialog: bool, optional

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: calculate()

      Calculate matched filter.



   .. py:method:: fftprep()

      FFT preparation when choosing band.



   .. py:method:: saveproj()

      Save project data from class.

      :rtype: None.



.. py:function:: getbutter(lowcut, highcut, f, order=5)

   Create Butterworth bandpass filter.

   :param lowcut: Low cutoff frequencies.
   :type lowcut: list of floats
   :param highcut: High cutoff frequencies.
   :type highcut: list of floats
   :param f: List of frequencies, ending in Nyquist frequency.
   :type f: numpy array
   :param order: Order of the filter. The default is 5.
   :type order: int

   :returns: **filt** -- List of 1D Butterworth filters.
   :rtype: list


