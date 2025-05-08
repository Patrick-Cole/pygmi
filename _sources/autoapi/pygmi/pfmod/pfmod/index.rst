pygmi.pfmod.pfmod
=================

.. py:module:: pygmi.pfmod.pfmod

.. autoapi-nested-parse::

   The main program for the potential field 3D modelling package.



Classes
-------

.. autoapisummary::

   pygmi.pfmod.pfmod.MainWidget


Module Contents
---------------

.. py:class:: MainWidget(parent=None)

   Bases: :py:obj:`PyQt6.QtWidgets.QMainWindow`


   MainWidget - Widget class to call the main interface.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: setupui()

      GUI setup.

      :rtype: None.



   .. py:method:: savemodel()

      Save model.

      :rtype: None.



   .. py:method:: help_docs()

      Help documentation.

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



   .. py:method:: data_reset()

      Reset the data.

      :rtype: None.



   .. py:method:: showtext(txt, replacelast=False)

      Show text on the text panel of the main user interface.

      :param txt: Text to display.
      :type txt: str
      :param replacelast: Whether to replace the last text written. The default is False.
      :type replacelast: bool, optional

      :rtype: None.



