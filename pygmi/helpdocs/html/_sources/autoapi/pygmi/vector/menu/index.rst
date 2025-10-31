pygmi.vector.menu
=================

.. py:module:: pygmi.vector.menu

.. autoapi-nested-parse::

   Vector Menu Routines.



Classes
-------

.. autoapisummary::

   pygmi.vector.menu.MenuWidget


Module Contents
---------------

.. py:class:: MenuWidget(parent=None)

   Widget class to call the main interface.

   This widget class creates the vector menus to be found on the main
   interface. Normal as well as context menus are defined here.

   :param parent: Reference to MainWidget class found in main.py. The default is None.
   :type parent: pygmi.main.MainWidget, optional


   .. py:method:: colselect()

      Select bands.



   .. py:method:: grid()

      Grid datasets.



   .. py:method:: scomp()

      Structure complexity.



   .. py:method:: cut_data()

      Cut point data.



   .. py:method:: reproject()

      Reproject point data.



   .. py:method:: export_xyz()

      Export XYZ data.



   .. py:method:: export_vector()

      Export line data.



   .. py:method:: file_split()

      Text file split.



   .. py:method:: import_xyz()

      Import XYZ data.



   .. py:method:: import_vector()

      Import shape data.



   .. py:method:: metadata()

      Metadata.



   .. py:method:: plot_ccoef()

      Plot correlation coefficient data.



   .. py:method:: show_line_data()

      Show line data.



   .. py:method:: show_line_map()

      Show line map.



   .. py:method:: show_vector_data()

      Show vector data.



   .. py:method:: show_rose_diagram()

      Show rose diagram.



   .. py:method:: show_hist()

      Show histogram.



   .. py:method:: basic_stats()

      Display basic statistics.



   .. py:method:: equation_editor()

      VectorEquation Editor.



