pygmi.seis.menu
===============

.. py:module:: pygmi.seis.menu

.. autoapi-nested-parse::

   Seis Menu Routines.



Classes
-------

.. autoapisummary::

   pygmi.seis.menu.MenuWidget


Module Contents
---------------

.. py:class:: MenuWidget(parent=None)

   Widget class to call the main interface.

   This widget class creates the seismology menus to be found on the main
   interface. Normal as well as context menus are defined here.

   :param parent: Reference to MainWidget class found in main.py. The default is None.
   :type parent: pygmi.main.MainWidget, optional


   .. py:method:: export_seisan()

      Export Seisan data.



   .. py:method:: export_csv()

      Export Seisan data to csv.



   .. py:method:: sexport()

      Export Summary data.



   .. py:method:: beachball()

      Create Beachballs from Fault Plane Solutions.



   .. py:method:: import_seisan()

      Import Seismic data.



   .. py:method:: correct_desc()

      Correct Seisan descriptions.



   .. py:method:: filter_seisan()

      Filter Seisan.



   .. py:method:: import_genfps()

      Import Generic Fault Plane Solution.



   .. py:method:: delete_recs()

      Delete Records.



   .. py:method:: quarry()

      Remove quarry events.



   .. py:method:: show_QC_plots()

      Show QC plots.



   .. py:method:: show_iso_plots()

      Show QC plots.



   .. py:method:: show_TP_plots()

      Show Temporal b-value plots.



