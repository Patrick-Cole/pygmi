pygmi.pfmod.iodefs
==================

.. py:module:: pygmi.pfmod.iodefs

.. autoapi-nested-parse::

   Import Potential field model data.



Classes
-------

.. autoapisummary::

   pygmi.pfmod.iodefs.ImportMod3D
   pygmi.pfmod.iodefs.ExportMod3D
   pygmi.pfmod.iodefs.Exportkmz
   pygmi.pfmod.iodefs.MessageCombo


Module Contents
---------------

.. py:class:: ImportMod3D(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Import Data.

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



   .. py:method:: import_leapfrog_csv(filename)

      Import leapfrog csv block models.

      :param filename: Input filename.
      :type filename: str

      :rtype: None.



   .. py:method:: import_ascii_xyz_model(filename)

      Use to import ASCII XYZ Models of the form x,y,z,label.

      :param filename: Input filename.
      :type filename: str

      :rtype: None.



   .. py:method:: dict2lmod(indict, pre='')

      Convert a dictionary to a LithModel.

      :param indict: Imported dictionary.
      :type indict: dictionary
      :param pre: Text. The default is ''.
      :type pre: str, optional

      :rtype: None.



.. py:class:: ExportMod3D(parent=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   Export 3D model data.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: run()

      Entry point into the routine, used to run context menu item.

      :rtype: None.



   .. py:method:: savemodel()

      Save model.

      :rtype: None.



   .. py:method:: lmod2dict(outdict, pre='')

      Convert LithModel to a dictionary.

      :param outdict: Output dictionary.
      :type outdict: dictionary
      :param pre: Text. The default is ''.
      :type pre: str, optional

      :returns: **outdict** -- Output dictionary.
      :rtype: dictionary



   .. py:method:: mod3dtocsv()

      Save the 3D model in a csv file.

      :rtype: None.



   .. py:method:: mod3dtokmz()

      Save the 3D model and grids in a kmz file.

      Only the boundary of the area is in degrees. The actual coordinates
      are still in meters.

      :rtype: None.



   .. py:method:: mod3dtoshp(nodialog=False)

      Save the 3D model and grids in a shapefile file.

      Only the boundary of the area is in degrees. The actual coordinates
      are still in meters.

      :rtype: None.



.. py:class:: Exportkmz(wkt, parent=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   Export kmz dialog.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional
   :param wkt: Well Known Text (wkt) representation of the projection.
   :type wkt: str


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



.. py:class:: MessageCombo(combotext, parent=None)

   Bases: :py:obj:`PySide6.QtWidgets.QDialog`


   Message combo box.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional
   :param combotext: List of text for combo.
   :type combotext: list

   .. attribute:: parent

      Reference to the parent routine.

      :type: parent


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: acceptall()

      Accept option.

      :returns: Returns current text.
      :rtype: str



