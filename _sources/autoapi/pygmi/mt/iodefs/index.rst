pygmi.mt.iodefs
===============

.. py:module:: pygmi.mt.iodefs

.. autoapi-nested-parse::

   Import and export EDI data.



Classes
-------

.. autoapisummary::

   pygmi.mt.iodefs.ImportEDI
   pygmi.mt.iodefs.ExportEDI


Functions
---------

.. autoapisummary::

   pygmi.mt.iodefs.get_EDI


Module Contents
---------------

.. py:class:: ImportEDI(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Import Data.

   .. attribute:: ifilelist

      list of input file names.

      :type: list


   .. py:method:: settings(nodialog=False)

      Entry point into item.

      :param nodialog: Run settings without a dialog. The default is False.
      :type nodialog: bool, optional

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: saveproj()

      Save project data from class.

      :rtype: None.



.. py:function:: get_EDI(ifiles)

   EDI Import.

   :param ifiles: filenames to import
   :type ifiles: list

   :returns: **dat** -- Dataset imported
   :rtype: EDI data.


.. py:class:: ExportEDI(parent=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   Export Data.

   .. attribute:: ofile

      output file name.

      :type: str


   .. py:method:: run()

      Entry point into the routine, used to run context menu item.

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: export_edi(dat)

      Export to EDI format.

      :param dat: dataset to export
      :type dat: EDI Data

      :rtype: None.



