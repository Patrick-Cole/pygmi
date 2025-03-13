pygmi.clust.menu
================

.. py:module:: pygmi.clust.menu

.. autoapi-nested-parse::

   Clustering Menu Routines.



Classes
-------

.. autoapisummary::

   pygmi.clust.menu.MenuWidget


Module Contents
---------------

.. py:class:: MenuWidget(parent=None)

   Widget class to call the main interface.

   This widget class creates the clustering menus to be found on the main
   interface. Normal as well as context menus are defined here.

   :param parent: Reference to MainWidget class found in main.py. Default is None.
   :type parent: pygmi.main.MainWidget, optional


   .. py:method:: cluster_stats()

      Calculate Statistics.



   .. py:method:: cluster()

      Clustering of data.



   .. py:method:: crisp_cluster()

      Crisp Clustering of data.



   .. py:method:: fuzzy_cluster()

      Fuzzy Clustering of data.



   .. py:method:: super_class()

      Supervised Classification.



   .. py:method:: export_data()

      Export raster data.



   .. py:method:: scatter_plot()

      Scatter Plot Tool.



   .. py:method:: show_raster_data()

      Show raster data.



   .. py:method:: show_membership_data()

      Show membership data.



   .. py:method:: show_vrc_etc()

      Show vrc, xbi, obj, nce graphs.



   .. py:method:: segmentation()

      Image Segmentation.



