pygmi.vector.show_table
=======================

.. py:module:: pygmi.vector.show_table

.. autoapi-nested-parse::

   Routine which displays a table graphically with various statistics.



Classes
-------

.. autoapisummary::

   pygmi.vector.show_table.BasicStats


Module Contents
---------------

.. py:class:: BasicStats(parent=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   GUI to show a summary of basic statistics.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: run()

      Entry point into the routine, used to run context menu item.

      :rtype: None.



   .. py:method:: save()

      Save Table.

      :returns: True if successful, False otherwise.
      :rtype: bool



