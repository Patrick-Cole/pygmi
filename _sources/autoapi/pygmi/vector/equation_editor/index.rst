pygmi.vector.equation_editor
============================

.. py:module:: pygmi.vector.equation_editor

.. autoapi-nested-parse::

   Equation editor for vector data.



Classes
-------

.. autoapisummary::

   pygmi.vector.equation_editor.EquationEditor


Functions
---------

.. autoapisummary::

   pygmi.vector.equation_editor.eqedit


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



.. py:function:: eqedit(data, equation, colname, showlog=print)

   Use equations on raster data.

   :param data: A GeoDataFrame containing columns of data
   :type data: GeoDataFrame
   :param equation: Equation to compute.
   :type equation: str
   :param colname: New column name.
   :type colname: str
   :param showlog: Show information using a function. The default is print.
   :type showlog: function, optional

   :returns: **outdata** -- Output GeoDataFrame containing columns of data
   :rtype: GeoDataFrame


