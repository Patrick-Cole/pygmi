pygmi.raster.menu
=================

.. py:module:: pygmi.raster.menu

.. autoapi-nested-parse::

   Raster Menu Routines.



Classes
-------

.. autoapisummary::

   pygmi.raster.menu.MenuWidget


Module Contents
---------------

.. py:class:: MenuWidget(parent=None)

   Widget class to call the main interface.

   This widget class creates the raster menus to be found on the main
   interface. Normal as well as context menus are defined here.

   :param parent: Reference to MainWidget class found in main.py. The default is None.
   :type parent: pygmi.main.MainWidget, optional


   .. py:method:: metadata()

      Metadata.



   .. py:method:: basic_stats()

      Display basic statistics.



   .. py:method:: equation_editor()

      Equation Editor.



   .. py:method:: export_data()

      Export raster data.



   .. py:method:: cut_data()

      Cut data.



   .. py:method:: clip_zoom()

      Clip to zoom.



   .. py:method:: get_prof()

      Get profile.



   .. py:method:: get_vector()

      Raster to vector.



   .. py:method:: gradients()

      Compute different gradients.



   .. py:method:: norm_data()

      Normalisation of data.



   .. py:method:: raster_interp()

      Show raster data.



   .. py:method:: cont()

      Compute Continuation.



   .. py:method:: show_ccoef()

      Show 2D correlation coefficients.



   .. py:method:: show_histogram()

      Show histogram of raster data.



   .. py:method:: show_raster_data()

      Show raster data.



   .. py:method:: show_raster_data2()

      Show raster data.



   .. py:method:: show_anaglyph()

      Show anaglyph of raster data.



   .. py:method:: show_surface()

      Show surface.



   .. py:method:: show_scatter_plot()

      Show scatter plot.



   .. py:method:: smoothing()

      Smoothing of Data.



   .. py:method:: agc()

      Compute AGC.



   .. py:method:: visibility()

      Compute visibility.



   .. py:method:: reproj()

      Reproject a dataset.



   .. py:method:: merge()

      Merge datasets.



   .. py:method:: lstack()

      Layer stack datasets.



   .. py:method:: import_data()

      Import data.



   .. py:method:: import_rgb_data()

      Import RGB data.



   .. py:method:: bandselect()

      Select bands.



