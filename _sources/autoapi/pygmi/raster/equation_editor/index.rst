pygmi.raster.equation_editor
============================

.. py:module:: pygmi.raster.equation_editor

.. autoapi-nested-parse::

   Equation editor.



Classes
-------

.. autoapisummary::

   pygmi.raster.equation_editor.EquationEditor


Functions
---------

.. autoapisummary::

   pygmi.raster.equation_editor.hmode


Module Contents
---------------

.. py:class:: EquationEditor(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Equation Editor.

   This class allows the input of equations using raster datasets as
   variables. This is commonly done in remote sensing applications, where
   there is a requirement for band ratioing etc. It uses the numexpr library.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional

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



   .. py:method:: eq_fix(indata)

      Corrects names in equation to variable names.

      :param indata: PyGMI raster dataset.
      :type indata: list of pygmi.raster.datatypes.Data.

      :returns: **neweq** -- Corrected equation.
      :rtype: str



   .. py:method:: mean(eq, localdict)

      Get mean pixel value of all input bands.

      :param eq: Equation with std command.
      :type eq: str
      :param localdict: Dictionary of data.
      :type localdict: dictionary

      :returns: **findat** -- Output array.
      :rtype: numpy array



   .. py:method:: std(eq, localdict)

      Get standard deviation pixel value of all input bands.

      :param eq: Equation with std command.
      :type eq: str
      :param localdict: Dictionary of data.
      :type localdict: dictionary

      :returns: **findat** -- Output array.
      :rtype: numpy array



   .. py:method:: mosaic(eq, localdict)

      Mosaics data into a single band dataset.

      :param eq: Equation with mosaic command.
      :type eq: str
      :param localdict: Dictionary of data.
      :type localdict: dictionary

      :returns: **findat** -- Output array.
      :rtype: numpy array



   .. py:method:: settings(nodialog=False)

      Entry point into item.

      :param nodialog: Run settings without a dialog. The default is False.
      :type nodialog: bool, optional

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: saveproj()

      Save project data from class.

      :rtype: None.



.. py:function:: hmode(data)

   Use a histogram to generate a fast mode estimate.

   :param data: list of values to generate the mode from.
   :type data: list

   :returns: **mode2** -- mode value.
   :rtype: float


