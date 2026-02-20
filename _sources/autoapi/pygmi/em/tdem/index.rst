pygmi.em.tdem
=============

.. py:module:: pygmi.em.tdem

.. autoapi-nested-parse::

   Time Domain EM inversion, based on the SimPEG library.



Classes
-------

.. autoapisummary::

   pygmi.em.tdem.MyMplCanvas2
   pygmi.em.tdem.TDEM1D


Functions
---------

.. autoapisummary::

   pygmi.em.tdem.tonumber


Module Contents
---------------

.. py:class:: MyMplCanvas2

   Bases: :py:obj:`matplotlib.backends.backend_qtagg.FigureCanvasQTAgg`


   Matplotlib canvas widget for the actual plot.


   .. py:method:: update_line(sigma, z, times_off, zobs, zpred)

      Update the plot from data.

      :param sigma: Conductivity values.
      :type sigma: numpy array
      :param z: Depth values.
      :type z: numpy array
      :param times_off: Time.
      :type times_off: numpy array
      :param zobs: Observed dBz/dt.
      :type zobs: numpy array
      :param zpred: Predicted dBz/dt.
      :type zpred: numpy array

      :rtype: None.



   .. py:method:: disp_wave(times, wave, title)

      Display waveform.

      :param times: Times.
      :type times: numpy array
      :param wave: Waveform amplitude.
      :type wave: numpy array
      :param title: Title.
      :type title: str

      :rtype: None.



.. py:class:: TDEM1D(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   TDEM 1D inversion GUI.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional


   .. py:method:: apply()

      Invert the data.

      :rtype: None.



   .. py:method:: change_source()

      Change Source.

      :rtype: None.



   .. py:method:: disp_wave()

      Display the waveform.

      :rtype: None.



   .. py:method:: update_wave()

      Update the waveform.

      :returns: **wform** -- Waveform.
      :rtype: tdem waveform.



   .. py:method:: get_wfile(filename='')

      Get the window time filename.

      :param filename: filename (txt). The default is ''.
      :type filename: str, optional

      :rtype: None.



   .. py:method:: change_line()

      Combo to change line.

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



.. py:function:: tonumber(test, alttext=None)

   Check if something is a number or matches alttext.

   :param test: Text to test.
   :type test: str
   :param alttext: Alternate text to test. The default is None.
   :type alttext: str, optional

   :returns: Returns a lower case version of alttext, or a number.
   :rtype: str or int or float


