pygmi.mag.menu
==============

.. py:module:: pygmi.mag.menu

.. autoapi-nested-parse::

   Magnetic Menu Routines.



Classes
-------

.. autoapisummary::

   pygmi.mag.menu.MenuWidget


Module Contents
---------------

.. py:class:: MenuWidget(parent=None)

   Widget class to call the main interface.

   This widget class creates the raster menus to be found on the main
   interface. Normal as well as context menus are defined here.

   :param parent: Reference to MainWidget class found in main.py. Default is None.
   :type parent: pygmi.main.MainWidget, optional


   .. py:method:: depth_susc()

      Depth and Susceptibility calculations.



   .. py:method:: rtp()

      Compute RTP.



   .. py:method:: tilt()

      Compute tilt angle.



   .. py:method:: igrf()

      Compute IGRF.



