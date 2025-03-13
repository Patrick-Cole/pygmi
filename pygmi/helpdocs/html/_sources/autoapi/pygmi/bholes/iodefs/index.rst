pygmi.bholes.iodefs
===================

.. py:module:: pygmi.bholes.iodefs

.. autoapi-nested-parse::

   Import borehole data, currently supports Council for Geoscience data.



Classes
-------

.. autoapisummary::

   pygmi.bholes.iodefs.ImportData


Functions
---------

.. autoapisummary::

   pygmi.bholes.iodefs.get_CGS


Module Contents
---------------

.. py:class:: ImportData(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Import borehole data.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: settings(nodialog=False)

      Entry point into item.

      :param nodialog: Run settings without a dialog. The default is False.
      :type nodialog: bool, optional

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: saveproj()

      Save project data from class.

      :rtype: None.



.. py:function:: get_CGS(lithfile, headerfile)

   Borehole Import.

   :param lithfile: Filename to import.
   :type lithfile: str
   :param headerfile: Filename to import.
   :type headerfile: str

   :returns: **dat** -- Dictionary of Pandas dataframes.
   :rtype: dictionary


