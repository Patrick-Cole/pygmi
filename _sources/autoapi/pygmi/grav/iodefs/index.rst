pygmi.grav.iodefs
=================

.. py:module:: pygmi.grav.iodefs

.. autoapi-nested-parse::

   Import Gravity Data.



Classes
-------

.. autoapisummary::

   pygmi.grav.iodefs.ImportCG5


Module Contents
---------------

.. py:class:: ImportCG5(parent)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Import CG-5 data.

   This class imports CG-5 gravimeter data with associated GPS data.

   :param parent: Reference to the parent routine.
   :type parent: parent


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: settings(nodialog=False)

      Entry point into item.

      :param nodialog: Run settings without a dialog. The default is False.
      :type nodialog: bool, optional

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: saveproj()

      Save project data from class.

      :rtype: None.



   .. py:method:: get_cg5(filename='')

      Get CG-5 filename and load data.

      :param filename: CG-5 filename submitted for testing. The default is ''.
      :type filename: str, optional

      :rtype: None.



   .. py:method:: get_gps(filename='')

      Get GPS filename and load data.

      :param filename: GPS filename (csv). The default is ''.
      :type filename: str, optional

      :rtype: None.



