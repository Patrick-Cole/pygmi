pygmi.rsense.change_viewer
==========================

.. py:module:: pygmi.rsense.change_viewer

.. autoapi-nested-parse::

   Change Detection Viewer.



Classes
-------

.. autoapisummary::

   pygmi.rsense.change_viewer.MyMplCanvas
   pygmi.rsense.change_viewer.SceneViewer


Module Contents
---------------

.. py:class:: MyMplCanvas(parent=None, width=10, height=8, dpi=100)

   Bases: :py:obj:`matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg`


   Matplotlib canvas widget for the actual plot.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional
   :param width: Width of the plot.
   :type width: float
   :param height: Height of the plot.
   :type height: float
   :param dpi: Dots per inch of the plot
   :type dpi: int


   .. py:method:: capture()

      Capture.

      :rtype: None.



   .. py:method:: compute_initial_figure(dat, dates)

      Compute initial figure.

      :param dat: PyGMI dataset.
      :type dat: pygmi.raster.datatypes.Data
      :param dates: Dates to show on title.
      :type dates: str

      :rtype: None.



   .. py:method:: update_plot(dat, dates)

      Update plot.

      :param dat: PyGMI dataset.
      :type dat: pygmi.raster.datatypes.Data
      :param dates: Dates to show on title.
      :type dates: str

      :rtype: None.



.. py:class:: SceneViewer(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Scene viewer GUI.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: settings(nodialog=False)

      Entry point into item.

      :param nodialog: Run settings without a dialog. The default is False.
      :type nodialog: bool, optional

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: updatescenelist()

      Update the scene list.

      :returns: Boolean to indicate success.
      :rtype: bool



   .. py:method:: saveproj()

      Save project data from class.

      :rtype: None.



   .. py:method:: manip_change()

      Change manipulation or bands.

      :rtype: None.



   .. py:method:: nextscene()

      Get next scene.

      :rtype: None.



   .. py:method:: prevscene()

      Get previous scene.

      :rtype: None.



   .. py:method:: newdata(indx)

      Get new dataset.

      :param indx: Current index.
      :type indx: int

      :rtype: None.



   .. py:method:: capture()

      Capture all scenes in the current view as an animation.

      :rtype: None.



