pygmi.seis.utils
================

.. py:module:: pygmi.seis.utils

.. autoapi-nested-parse::

   Module for miscellaneous utilities relating to earthquake seismology.



Classes
-------

.. autoapisummary::

   pygmi.seis.utils.CorrectDescriptions


Module Contents
---------------

.. py:class:: CorrectDescriptions(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Correct SEISAN descriptions.

   This compares the descriptions found in SEISAN type 3 lines to a custom
   list.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: get_textfile(filename='')

      Get description list filename.

      :param filename: Filename submitted for testing. The default is ''.
      :type filename: str, optional

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



   .. py:method:: acceptall()

      Accept option.

      Updates self.outdata, which is used as input to other modules.

      :rtype: None.



