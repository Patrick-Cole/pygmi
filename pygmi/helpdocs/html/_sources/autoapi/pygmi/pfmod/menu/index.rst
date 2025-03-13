pygmi.pfmod.menu
================

.. py:module:: pygmi.pfmod.menu

.. autoapi-nested-parse::

   Potential Field Modelling menus.



Classes
-------

.. autoapisummary::

   pygmi.pfmod.menu.MenuWidget


Module Contents
---------------

.. py:class:: MenuWidget(parent=None)

   Widget class to call the main interface.

   This widget class creates the modelling menus to be found on the main
   interface. Normal as well as context menus are defined here.

   :param parent: Reference to MainWidget class found in main.py. Default is None.
   :type parent: pygmi.main.MainWidget, optional


   .. py:method:: export_mod3d()

      Export 3D Model.



   .. py:method:: pfmod()

      Voxel modelling of data.



   .. py:method:: maginv()

      Voxel inversion of data.



   .. py:method:: mod3d()

      3D display of data.



   .. py:method:: stat3d()

      3D display of data.



   .. py:method:: import_mod3d()

      Import data.



   .. py:method:: merge_mod3d()

      Merge models.



