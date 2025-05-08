pygmi.mt.menu
=============

.. py:module:: pygmi.mt.menu

.. autoapi-nested-parse::

   MT Menu Routines.



Classes
-------

.. autoapisummary::

   pygmi.mt.menu.MenuWidget


Module Contents
---------------

.. py:class:: MenuWidget(parent=None)

   Widget class to call the main interface.

   This widget class creates menus to be found on the main
   interface. Normal as well as context menus are defined here.

   :param parent: Reference to MainWidget class found in main.py. Default is None.
   :type parent: pygmi.main.MainWidget, optional


   .. py:method:: birrp()

      BIRRP.



   .. py:method:: export_data()

      Export data.



   .. py:method:: import_data()

      Import data.



   .. py:method:: occam1d()

      Occam 1D inversion.



   .. py:method:: rotate_data()

      Rotate data.



   .. py:method:: sshift_data()

      Calculate Static Shift.



   .. py:method:: mi_data()

      Mask and interpolate data.



   .. py:method:: metadata()

      Metadata.



   .. py:method:: show_graphs()

      Show graphs.



