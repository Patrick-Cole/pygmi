pygmi.mt.dataprep
=================

.. py:module:: pygmi.mt.dataprep

.. autoapi-nested-parse::

   A set of Data Preparation routines.



Classes
-------

.. autoapisummary::

   pygmi.mt.dataprep.Metadata
   pygmi.mt.dataprep.MyMplCanvas
   pygmi.mt.dataprep.StaticShiftEDI
   pygmi.mt.dataprep.RotateEDI
   pygmi.mt.dataprep.MyMplCanvasPick
   pygmi.mt.dataprep.EditEDI
   pygmi.mt.dataprep.MySlider
   pygmi.mt.dataprep.MyMplCanvas2
   pygmi.mt.dataprep.Occam1D


Functions
---------

.. autoapisummary::

   pygmi.mt.dataprep.tonumber


Module Contents
---------------

.. py:class:: Metadata(parent=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   Edit Metadata.

   This class allows the editing of the metadata for MT data using a GUI.

   .. attribute:: banddata

      band data

      :type: dictionary

   .. attribute:: bandid

      dictionary of strings containing band names.

      :type: dictionary


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: acceptall()

      Accept option. Updates self.indata.

      :rtype: None.



   .. py:method:: rename_id()

      Rename station name.

      :rtype: None.



   .. py:method:: update_vals()

      Update the values on the interface.

      :rtype: None.



   .. py:method:: run()

      Entry point into the routine, used to run context menu item.

      :returns: True if successful, False otherwise.
      :rtype: bool.



.. py:class:: MyMplCanvas(parent=None, width=8, height=6, dpi=100)

   Bases: :py:obj:`matplotlib.backends.backend_qtagg.FigureCanvasQTAgg`


   Matplotlib canvas widget for the actual plot.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: update_line(data, ival, itype)

      Update the plot from point data.

      :param data: EDI data.
      :type data: EDI data object
      :param ival: dictionary key.
      :type ival: str
      :param itype: dictionary key.
      :type itype: str

      :rtype: None.



.. py:class:: StaticShiftEDI(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Static shift EDI data.


   .. py:method:: acceptall()

      Accept option.

      Updates self.outdata, which is used as input to other modules.

      :rtype: None.



   .. py:method:: apply()

      Apply static shift.

      :rtype: None.



   .. py:method:: reset_data()

      Reset data.

      :rtype: None.



   .. py:method:: change_band()

      Combo to change band.

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



.. py:class:: RotateEDI(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Rotate EDI data.


   .. py:method:: acceptall()

      Accept option.

      Updates self.outdata, which is used as input to other modules.

      :rtype: None.



   .. py:method:: apply()

      Apply rotation to data.

      :rtype: None.



   .. py:method:: reset_data()

      Reset data.

      :rtype: None.



   .. py:method:: change_band()

      Combo to change band.

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



.. py:class:: MyMplCanvasPick(parent=None)

   Bases: :py:obj:`matplotlib.backends.backend_qtagg.FigureCanvasQTAgg`


   Matplotlib canvas widget for the actual plot.

   This routine will also allow the picking and movement of nodes of data.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: button_press_callback(event)

      Mouse button release callback.

      :param event: event variable.
      :type event: event

      :rtype: None.



   .. py:method:: button_release_callback(event)

      Mouse button release callback.

      :param event: event variable.
      :type event: event

      :rtype: None.



   .. py:method:: motion_notify_callback(event)

      Move mouse callback.

      :param event: event variable.
      :type event: event

      :rtype: None.



   .. py:method:: onpick(event)

      Picker event.

      :param event: event variable.
      :type event: event

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: revent(width)

      Resize event.

      :param width: unused.
      :type width: event

      :rtype: None.



   .. py:method:: update_line(data, ival=None, itype=None)

      Update the plot from point data.

      :param data: EDI data.
      :type data: EDI data object
      :param ival: dictionary key.
      :type ival: str
      :param itype: dictionary key.
      :type itype: str

      :rtype: None.



.. py:class:: EditEDI(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Edit EDI Class.


   .. py:method:: acceptall()

      Accept option.

      Updates self.outdata, which is used as input to other modules.

      :rtype: None.



   .. py:method:: apply()

      Apply edited data.

      :rtype: None.



   .. py:method:: reset_data()

      Reset data.

      :rtype: None.



   .. py:method:: change_band()

      Combo to choose band.

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



.. py:class:: MySlider(parent=None)

   Bases: :py:obj:`PySide6.QtWidgets.QSlider`


   My Slider.

   Custom class which allows clicking on a horizontal slider bar with slider
   moving to click in a single step.


   .. py:method:: mousePressEvent(event)

      Mouse press event.

      :param event: event variable.
      :type event: event

      :rtype: None.



   .. py:method:: mouseMoveEvent(event)

      Jump to pointer position while moving.

      :param event: event variable.
      :type event: event

      :rtype: None.



.. py:class:: MyMplCanvas2(parent=None)

   Bases: :py:obj:`matplotlib.backends.backend_qtagg.FigureCanvasQTAgg`


   Matplotlib canvas widget for the actual plot.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: update_line(x, pdata, rdata, *, depths=None, res=None, title=None)

      Update the plot from data.

      :param x: X coordinates (period).
      :type x: numpy array
      :param pdata: Phase data.
      :type pdata: numpy array
      :param rdata: Apparent resistivity data.
      :type rdata: numpy array
      :param depths: Model depths. The default is None.
      :type depths: numpy array, optional
      :param res: Resistivities. The default is None.
      :type res: numpy array, optional
      :param title: Title text. The default is None.
      :type title: str or None, optional

      :rtype: None.



.. py:class:: Occam1D(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Occam 1D inversion.


   .. py:method:: snum()

      Change solution graph.

      :rtype: None.



   .. py:method:: acceptall()

      Accept option.

      Updates self.outdata, which is used as input to other modules.

      :rtype: None.



   .. py:method:: apply()

      Apply.

      :rtype: None.



   .. py:method:: get_occfile(filename='')

      Get Occam executable filename.

      :param filename: Occam executable filename. The default is ''.
      :type filename: str, optional

      :rtype: None.



   .. py:method:: reset_data()

      Reset data.

      :rtype: None.



   .. py:method:: change_band()

      Combo to change band.

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


