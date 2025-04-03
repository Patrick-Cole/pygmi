pygmi.misc
==========

.. py:module:: pygmi.misc

.. autoapi-nested-parse::

   Misc is a collection of routines which can be used in PyGMI in general.



Classes
-------

.. autoapisummary::

   pygmi.misc.EmittingStream
   pygmi.misc.BasicModule
   pygmi.misc.ContextModule
   pygmi.misc.QVStack2Layout
   pygmi.misc.PTime
   pygmi.misc.ProgressBar
   pygmi.misc.ProgressBarText


Functions
---------

.. autoapisummary::

   pygmi.misc.discrete_colorbar
   pygmi.misc.getinfo
   pygmi.misc.limit_memory
   pygmi.misc.textwrap2
   pygmi.misc.tick_formatter


Module Contents
---------------

.. py:class:: EmittingStream(textWritten)

   Bases: :py:obj:`PyQt6.QtCore.QObject`


   Class to intercept stdout for later use in a textbox.

   :param textwritten: Text written to stdout.
   :type textwritten: str


   .. py:method:: write(text)

      Write text.

      :param text: Text to write.
      :type text: str

      :rtype: None.



   .. py:method:: flush()

      Flush.

      :rtype: None.



   .. py:method:: fileno()

      File number.

      :returns: Returns -1.
      :rtype: int



.. py:class:: BasicModule(parent=None)

   Bases: :py:obj:`PyQt6.QtWidgets.QDialog`


   Basic Module.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional

   .. attribute:: parent

      reference to the parent routine

      :type: parent

   .. attribute:: indata

      dictionary of input datasets

      :type: dictionary

   .. attribute:: outdata

      dictionary of output datasets

      :type: dictionary

   .. attribute:: ifile

      input file, used in IO routines and to pass filename back to main.py

      :type: str

   .. attribute:: piter

      reference to a progress bar iterator.

      :type: function

   .. attribute:: pbar

      reference to a progress bar.

      :type: function

   .. attribute:: showlog

      reference to a way to view messages, normally stdout or a Qt text box.

      :type: stdout or alternative

   .. attribute:: is_import

      used to indicate whether a routine contains an import within.

      :type: bool

   .. attribute:: projdata

      Project data.

      :type: dictionary


   .. py:method:: settings(nodialog=False)

      Entry point into item.

      :param nodialog: Run settings without a dialog. The default is False.
      :type nodialog: bool, optional

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: data_init()

      Initialise Data.

      Entry point into routine. This entry point exists for
      the case  where data must be initialised before entering at the
      standard 'settings' sub module.

      :rtype: None.



   .. py:method:: loadproj(projdata)

      Load project data into class.

      :param projdata: Project data loaded from JSON project file.
      :type projdata: dictionary

      :returns: **chk** -- A check to see if settings was successfully run.
      :rtype: bool



   .. py:method:: saveproj()

      Save project data from class.

      :rtype: None.



   .. py:method:: saveobj(obj)

      Save an object to a dictionary.

      This is a convenience function for saving project information.

      :param obj: A variable to be saved.
      :type obj: variable

      :rtype: None.



.. py:class:: ContextModule(parent=None)

   Bases: :py:obj:`PyQt6.QtWidgets.QDialog`


   Context Module.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional

   .. attribute:: parent

      reference to the parent routine

      :type: parent

   .. attribute:: indata

      dictionary of input datasets

      :type: dictionary

   .. attribute:: outdata

      dictionary of output datasets

      :type: dictionary

   .. attribute:: piter

      reference to a progress bar iterator.

      :type: function

   .. attribute:: pbar

      reference to a progress bar.

      :type: function

   .. attribute:: showlog

      reference to a way to view messages, normally stdout or a Qt text box.

      :type: stdout or alternative


   .. py:method:: run()

      Run context menu item.

      :rtype: None.



.. py:class:: QVStack2Layout(parent=None)

   Bases: :py:obj:`PyQt6.QtWidgets.QGridLayout`


   QVStack2Layout custom Qt QGridLayot.

   This works like VBoxLayout, except each row takes two widgets.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: addWidget(widget1, widget2)

      Add two widgets on a row, widget can also be text.

      :param widget1: First Widget or Label on the row.
      :type widget1: str or QWidget
      :param widget2: Last Widget.
      :type widget2: QWidget

      :rtype: None.



   .. py:method:: addWidgetOld(*args, **kwargs)

      Original Add Widget.



.. py:class:: PTime

   PTime class.

   Main class in the ptimer module. Once activated, this class keeps track
   of all time since activation. Times are stored whenever its methods are
   called.

   .. attribute:: tchk

      List of times generated by the time.perf_counter routine.

      :type: list


   .. py:method:: since_first_call(msg='since first call', show=True)

      Time lapsed since first call.

      This function prints out a message and lets you know the time
      passed since the first call.

      :param msg: Optional message
      :type msg: str



   .. py:method:: since_last_call(msg='since last call', show=True)

      Time lapsed since last call.

      This function prints out a message and lets you know the time
      passed since the last call.

      :param msg: Optional message
      :type msg: str



.. py:class:: ProgressBar(parent=None)

   Bases: :py:obj:`PyQt6.QtWidgets.QProgressBar`


   Qt custom progress bar.

   Progress Bar routine which expands the QProgressBar class slightly so that
   there is a time function as well as a convenient of calling it via an
   iterable.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional

   .. attribute:: otime

      This is the original time recorded when the progress bar starts.

      :type: intr

   .. attribute:: total

      Maximum progress bar value. The default is 100.

      :type: int


   .. py:method:: iter(iterable)

      Iterate Routine.



   .. py:method:: to_max()

      Set the progress to maximum.



.. py:class:: ProgressBarText

   Text Progress bar.

   .. attribute:: otime

      This is the original time recorded when the progress bar starts.

      :type: int

   .. attribute:: total

      Maximum progress bar value. The default is 100.

      :type: int


   .. py:method:: iter(iterable)

      Iterate Routine.



   .. py:method:: printprogressbar(iteration, suffix='')

      Call in a loop to create terminal progress bar.

      Code by Alexander Veysov. (https://gist.github.com/snakers4).

      :param iteration: current iteration
      :type iteration: int
      :param suffix: Suffix string. The default is ''.
      :type suffix: str, optional

      :rtype: None.



   .. py:method:: setMaximum(val)

      Set the maximum value.



   .. py:method:: setValue(val)

      Set the progressbar value.



   .. py:method:: to_max()

      Set the progress to maximum.



.. py:function:: discrete_colorbar(axes, csp, cdat, lbls=None)

   Plot colour bar using discrete colours for a small range of values.

   :param axes: Current axes.
   :type axes: Matplotlib axes
   :param csp: Handle to Matplotlib plotting routine.
   :type csp: Plot routine
   :param cdat: Array of values.
   :type cdat: numpy array
   :param lbls:
   :type lbls: y tick labels (optional)

   :rtype: None.


.. py:function:: getinfo(txt=None, reset=False)

   Get time and memory info.

   :param txt: Descriptor used for headings. The default is None.
   :type txt: str/int/float, optional
   :param reset: Flag used to reset the time difference to zero.
   :type reset: bool

   :rtype: None.


.. py:function:: limit_memory(memory_limit)

   Limit memory in Windows.

   Based on https://stackoverflow.com/questions/54949110/limit-python-script-ram-usage-in-windows

   :param memory_limit: Memory limit in GB.
   :type memory_limit: int

   :rtype: None.


.. py:function:: textwrap2(text, width, placeholder='...', max_lines=None)

   Provide slightly different placeholder functionality to textwrap.

   Placeholders will be a part of last line, instead of replacing it.

   :param text: Text to wrap.
   :type text: str
   :param width: Maximum line length.
   :type width: int
   :param placeholder: Placeholder when lines exceed max_lines. The default is '...'.
   :type placeholder: sre, optional
   :param max_lines: Maximum number of lines. The default is None.
   :type max_lines: int, optional

   :returns: **text2** -- Output wrapped text.
   :rtype: str


.. py:function:: tick_formatter(x, pos)

   Format thousands separator in ticks for plots.

   :param x: Number to be formatted.
   :type x: float/int
   :param pos: Position of tick.
   :type pos: int

   :returns: **newx** -- Formatted coordinate.
   :rtype: str


