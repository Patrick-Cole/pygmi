pygmi.rsense.menu
=================

.. py:module:: pygmi.rsense.menu

.. autoapi-nested-parse::

   Remote Sensing Menu Routines.



Classes
-------

.. autoapisummary::

   pygmi.rsense.menu.MenuWidget


Module Contents
---------------

.. py:class:: MenuWidget(parent=None)

   Widget class to call the main interface.

   This widget class creates menus to be found on the main
   interface. Normal as well as context menus are defined here.

   :param parent: Reference to MainWidget class found in main.py. Default is None.
   :type parent: pygmi.main.MainWidget, optional


   .. py:method:: exportlist()

      Export Raster File List.



   .. py:method:: calc_change()

      Calculate change.



   .. py:method:: topo()

      Topographic correction.



   .. py:method:: sen2cor()

      Sen2Cor.



   .. py:method:: view_change()

      View Change Detection.



   .. py:method:: calc_ratios()

      Calculate Ratios.



   .. py:method:: calc_ci()

      Calculate Condition Indices.



   .. py:method:: lsat_comp()

      Calculate Landsat Composite.



   .. py:method:: mnf()

      Calculate MNF.



   .. py:method:: pca()

      Calculate PCA.



   .. py:method:: anal_spec()

      Analyse Spectra.



   .. py:method:: proc_features()

      Process Features.



   .. py:method:: import_sentinel5p()

      Import Sentinel 5P data.



   .. py:method:: import_sat()

      Import Satellite data.



   .. py:method:: batch_list()

      Import batch list.



