pygmi.menu_default
==================

.. py:module:: pygmi.menu_default

.. autoapi-nested-parse::

   Default set of menus for the main interface.

   It also includes the about box.



Classes
-------

.. autoapisummary::

   pygmi.menu_default.FileMenu
   pygmi.menu_default.HelpMenu
   pygmi.menu_default.HelpButton
   pygmi.menu_default.HelpDocs


Module Contents
---------------

.. py:class:: FileMenu(parent=None)

   Widget class to call the main interface.

   This widget class creates the raster menus to be found on the main
   interface. Normal as well as context menus are defined here.

   .. attribute:: parent

      Reference to MainWidget class found in main.py. Default is None.

      :type: pygmi.main.MainWidget, optional


.. py:class:: HelpMenu(parent=None)

   Widget class to call the main interface.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: about()

      About box for PyGMI.



   .. py:method:: webhelp()

      Help File.



.. py:class:: HelpButton(htmlfile=None, parent=None)

   Bases: :py:obj:`PyQt5.QtWidgets.QPushButton`


   Help Button.

   Convenience class to add a Help image to a pushbutton

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional
   :param htmlfile: HTML help file name.
   :type htmlfile: str


   .. py:method:: help_docs()

      Help Routine.



.. py:class:: HelpDocs(parent=None, helptxt=None)

   Bases: :py:obj:`PyQt5.QtWidgets.QDialog`


   A basic combo box application.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional
   :param helptxt: Help filename.
   :type helptxt: str

   .. attribute:: parent

      reference to the parent routine

      :type: parent

   .. attribute:: indata

      dictionary of input datasets

      :type: dictionary

   .. attribute:: outdata

      dictionary of output datasets

      :type: dictionary


