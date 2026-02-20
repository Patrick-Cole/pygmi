pygmi.raster.equation_editor
============================

.. py:module:: pygmi.raster.equation_editor

.. autoapi-nested-parse::

   Equation editor for raster data.



Classes
-------

.. autoapisummary::

   pygmi.raster.equation_editor.EquationEditor


Functions
---------

.. autoapisummary::

   pygmi.raster.equation_editor.eqedit
   pygmi.raster.equation_editor.eq_fix
   pygmi.raster.equation_editor.hmode
   pygmi.raster.equation_editor.mosaic
   pygmi.raster.equation_editor.mean
   pygmi.raster.equation_editor.detrend
   pygmi.raster.equation_editor.std


Module Contents
---------------

.. py:class:: EquationEditor(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Equation Editor.

   This class allows the input of equations using raster datasets as
   variables. This is commonly done in remote sensing applications, where
   there is a requirement for band ratioing etc. It uses the numexpr library.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional

   .. attribute:: equation

      string with the equation in it

      :type: str

   .. attribute:: bands

      dictionary of bands

      :type: dictionary


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: combo()

      Update combo information.

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



.. py:function:: eqedit(data, equation, dtype='auto', showlog=print, piter=iter)

   Use equations on raster data.

   :param data: List of PyGMI raster data.
   :type data: list
   :param equation: Equation to compute.
   :type equation: str
   :param dtype: The data type of the output dataset. The default is 'auto'.
   :type dtype: str, optional
   :param showlog: Show information using a function. The default is print.
   :type showlog: function, optional
   :param piter: Progress bar iterable. The default is iter.
   :type piter: function, optional

   :returns: List of PyGMI raster data.
   :rtype: list


.. py:function:: eq_fix(indata, equation, showlog=print)

   Corrects names in equation to variable names.

   :param indata: PyGMI raster dataset.
   :type indata: list of pygmi.raster.datatypes.Data.
   :param equation: Equation to fix.
   :type equation: str
   :param showlog: Show information using a function. The default is print.
   :type showlog: function, optional

   :returns: **neweq** -- Corrected equation.
   :rtype: str


.. py:function:: hmode(data)

   Use a histogram to generate a fast mode estimate.

   :param data: list of values to generate the mode from.
   :type data: list

   :returns: **mode2** -- mode value.
   :rtype: float


.. py:function:: mosaic(eq, localdict)

   Mosaics data into a single band dataset.

   :param eq: Equation with mosaic command.
   :type eq: str
   :param localdict: Dictionary of data.
   :type localdict: dictionary

   :returns: **master** -- Output array.
   :rtype: numpy array


.. py:function:: mean(eq, localdict)

   Get mean pixel value of all input bands.

   :param eq: Equation with std command.
   :type eq: str
   :param localdict: Dictionary of data.
   :type localdict: dictionary

   :returns: **findat** -- Output array.
   :rtype: numpy array


.. py:function:: detrend(eq, localdict)

   Get mean pixel value of all input bands.

   :param eq: Equation with std command.
   :type eq: str
   :param localdict: Dictionary of data.
   :type localdict: dictionary

   :returns: **findat** -- Output array.
   :rtype: numpy array


.. py:function:: std(eq, localdict)

   Get standard deviation pixel value of all input bands.

   :param eq: Equation with std command.
   :type eq: str
   :param localdict: Dictionary of data.
   :type localdict: dictionary

   :returns: **findat** -- Output array.
   :rtype: numpy array


