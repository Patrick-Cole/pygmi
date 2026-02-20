pygmi.vector.vvis3d
===================

.. py:module:: pygmi.vector.vvis3d

.. autoapi-nested-parse::

   Code for the 3d voxel model display.



Classes
-------

.. autoapisummary::

   pygmi.vector.vvis3d.Mod3dDisplay


Module Contents
---------------

.. py:class:: Mod3dDisplay(parent=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   Widget class to call the main interface.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: closeEvent(QCloseEvent)

      Close event.

      :param QCloseEvent: Close event.
      :type QCloseEvent: TYPE

      :rtype: None.



   .. py:method:: save()

      Save a jpg.

      :rtype: None.



   .. py:method:: data_init()

      Initialise Data.

      Entry point into routine. This entry point exists for
      the case  where data must be initialised before entering at the
      standard 'settings' sub module.

      :rtype: None.



   .. py:method:: run()

      Entry point into the routine, used to run context menu item.

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: update_plot()

      Update 3D model.

      :rtype: None.



