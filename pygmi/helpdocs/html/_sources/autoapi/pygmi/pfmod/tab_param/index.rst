pygmi.pfmod.tab_param
=====================

.. py:module:: pygmi.pfmod.tab_param

.. autoapi-nested-parse::

   Parameter Display Tab Routines.



Classes
-------

.. autoapisummary::

   pygmi.pfmod.tab_param.MergeLith
   pygmi.pfmod.tab_param.LithNotes
   pygmi.pfmod.tab_param.ParamDisplay


Module Contents
---------------

.. py:class:: MergeLith(parent=None)

   Bases: :py:obj:`PySide6.QtWidgets.QDialog`


   Class to call up a dialog for ranged copying.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: ParamDisplay, optional


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



.. py:class:: LithNotes(parent=None)

   Bases: :py:obj:`PySide6.QtWidgets.QDialog`


   Class to call up a dialog for lithology descriptions.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.pfmod.pfmod.MainWidget, optional


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: apply_changes()

      Apply changes.

      :rtype: None.



   .. py:method:: lw_index_change()

      List box in parameter tab for definitions.

      :rtype: None.



   .. py:method:: tab_activate()

      Entry point.

      :rtype: None.



.. py:class:: ParamDisplay(parent=None)

   Bases: :py:obj:`PySide6.QtWidgets.QDialog`


   Widget class to call the main interface.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.pfmod.pfmod.MainWidget, optional


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: add_defs(deftxt='', getcol=False, lmod=None)

      Add geophysical definitions and make them editable.

      :param deftxt: Definition text. The default is ''.
      :type deftxt: str, optional
      :param getcol: Get column. The default is False.
      :type getcol: bool, optional
      :param lmod: 3D model. The default is None.
      :type lmod: LithModel, optional

      :rtype: None.



   .. py:method:: apply_lith()

      Apply lithological changes.

      :rtype: None.



   .. py:method:: apply_changes()

      Apply geophysical properties.

      :rtype: None.



   .. py:method:: change_rmi()

      Update spinboxes when rmi is changed.

      :rtype: None.



   .. py:method:: change_magnetization()

      Update spinboxes when magnetization is changed.

      :rtype: None.



   .. py:method:: change_qratio()

      Update spinboxes when qratio is changed.

      :rtype: None.



   .. py:method:: disconnect_spin()

      Disconnect spin boxes.

      :rtype: None.



   .. py:method:: connect_spin()

      Connect spin boxes.

      :rtype: None.



   .. py:method:: change_defs(item)

      Change geophysical definitions.

      :param item: Parameter definition QListWidget item.
      :type item: QListWidget item

      :rtype: None.



   .. py:method:: get_lith()

      Get parameter definitions.

      :returns: **lith** -- Lithology data.
      :rtype: GeoData



   .. py:method:: init()

      Initialize parameters.

      :rtype: None.



   .. py:method:: lw_color_change()

      Routine to allow lithologies to have their colors changed.

      :rtype: None.



   .. py:method:: lw_index_change()

      List widget in parameter tab for definitions.

      :rtype: None.



   .. py:method:: add_def()

      Add geophysical definition.

      :rtype: None.



   .. py:method:: rem_defs()

      Remove geophysical definition.

      :rtype: None.



   .. py:method:: merge_defs()

      Merge geophysical definitions.

      :rtype: None.



   .. py:method:: rename_defs()

      Rename a definition.

      :rtype: None.



   .. py:method:: set_lw_colors(lwidget, lmod=None)

      Set list widget colors.

      :param lwidget: Lithology list widget..
      :type lwidget: QListWidget
      :param lmod: 3D Model. The default is None.
      :type lmod: LithModel, optional

      :rtype: None.



   .. py:method:: tab_activate()

      Entry point.

      :rtype: None.



