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

   Bases: :py:obj:`PyQt6.QtWidgets.QPushButton`


   Help Button.

   Convenience class to add a Help image to a pushbutton

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional
   :param htmlfile: HTML help file name.
   :type htmlfile: str


   .. py:method:: help_docs()

      Help Routine.



