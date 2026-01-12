pygmi.pfmod.tab_prof
====================

.. py:module:: pygmi.pfmod.tab_prof

.. autoapi-nested-parse::

   Profile Display Tab Routines.



Classes
-------

.. autoapisummary::

   pygmi.pfmod.tab_prof.ProfileDisplay
   pygmi.pfmod.tab_prof.MyMplCanvas
   pygmi.pfmod.tab_prof.MySlider
   pygmi.pfmod.tab_prof.LithBound
   pygmi.pfmod.tab_prof.PlotScale
   pygmi.pfmod.tab_prof.RangedCopy
   pygmi.pfmod.tab_prof.MyToolbar
   pygmi.pfmod.tab_prof.GaugeWidget
   pygmi.pfmod.tab_prof.ImportPicture


Functions
---------

.. autoapisummary::

   pygmi.pfmod.tab_prof.gridmatch2
   pygmi.pfmod.tab_prof.rotate2d


Module Contents
---------------

.. py:class:: ProfileDisplay(parent=None)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   Widget class to call the main interface.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: cprof_add()

      Add new custom profile.

      :rtype: None.



   .. py:method:: cprof_del()

      Delete current custom profile.

      :rtype: None.



   .. py:method:: proftype_changed()

      Profile type changed.

      :rtype: None.



   .. py:method:: custom_prof_limits(curprof=None)

      Calculate custom profile limits.

      :param curprof: Current profile. The default is None.
      :type curprof: int or str, optional

      :rtype: None.



   .. py:method:: hcprofnum()

      Change a profile from a horizontal slider.

      :rtype: None.



   .. py:method:: scprofnum()

      Change a profile from a spinbox.

      :rtype: None.



   .. py:method:: borehole_import()

      Import borehole data.

      :rtype: None.



   .. py:method:: calculate_dip()

      Calculate dip.

      :rtype: None.



   .. py:method:: export_csv()

      Export profile to csv.

      :rtype: None.



   .. py:method:: lbound()

      Insert a lithological boundary.

      :rtype: None.



   .. py:method:: rcopy()

      Do a ranged copy on a profile.

      :rtype: None.



   .. py:method:: rcopy_layer(rcopy)

      Do a ranged copy on a layer.

      :param rcopy: Handle of ranged copy GUI.
      :type rcopy: RangedCopy

      :rtype: None.



   .. py:method:: rcopy_prof(rcopy)

      Ranged copy on a profile.

      :param rcopy: Handle to RangedCopy GUI.
      :type rcopy: RangedCopy

      :rtype: None.



   .. py:method:: change_defs()

      Change definitions.

      :rtype: None.



   .. py:method:: get_model()

      Get model slice.

      :rtype: None.



   .. py:method:: hprofnum()

      Change a profile from a horizontal slider.

      :rtype: None.



   .. py:method:: pic_sideview()

      Horizontal slider for picture opacity.

      Change the opacity of profile and overlain picture.


      :rtype: None.



   .. py:method:: plot_scale()

      Plot scale.

      :rtype: None.



   .. py:method:: setwidth(width)

      Set the width of the edits on the profile view.

      :param width: Edit width.
      :type width: int

      :rtype: None.



   .. py:method:: sprofnum()

      Routine to change a profile from spinbox.

      :rtype: None.



   .. py:method:: hlayer()

      Horizontal slider to change the layer.

      :rtype: None.



   .. py:method:: pic_overview()

      Horizontal slider to change picture opacity.

      :rtype: None.



   .. py:method:: pic_overview2()

      Horizontal slider to change picture opacity.

      :rtype: None.



   .. py:method:: slayer()

      Change model layer.

      :rtype: None.



   .. py:method:: calc_prof_limits(curprof=None)

      Calculate profile limits.

      :param curprof: Current profile. The default is None.
      :type curprof: int or None, optional

      :rtype: None.



   .. py:method:: prof_dir(slide=True)

      Profile direction.

      :param slide: Flag to redraw entire plot, or just update. The default is True.
      :type slide: bool, optional

      :rtype: None.



   .. py:method:: sprofdir()

      Profile direction spinbox.

      :rtype: None.



   .. py:method:: update_combo_overview(curtext=None)

      Update the overview combo.

      :param curtext: Current text in combo. Default is None.
      :type curtext: str, optional

      :rtype: None.



   .. py:method:: update_plot(slide=False)

      Update the profile on the model view.

      :param slide: Flag to redraw entire plot, or just update. The default is False.
      :type slide: bool, optional

      :rtype: None.



   .. py:method:: tab_activate()

      Entry point.

      :rtype: None.



.. py:class:: MyMplCanvas(parent=None)

   Bases: :py:obj:`matplotlib.backends.backend_qtagg.FigureCanvasQTAgg`


   Matplotlib canvas widget for the actual plot.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: button_press(event)

      Button press event.

      :param event: Event variable.
      :type event: event

      :rtype: None.



   .. py:method:: button_release(event)

      Button release event.

      :param event: Unused.
      :type event: event

      :rtype: None.



   .. py:method:: move(event)

      Mouse move event.

      :param event: Event variable.
      :type event: event

      :rtype: None.



   .. py:method:: dip(event)

      Calculate dip event.

      :param event: Event variable.
      :type event: event

      :rtype: None.



   .. py:method:: set_mdata(xdata, ydata, mdata)

      Routine to 'draw' the line on mdata.

      xdata and ydata are the cursor centre coordinates.

      :param xdata: X data.
      :type xdata: float
      :param ydata: Y data.
      :type ydata: float
      :param mdata: Model array.
      :type mdata: numpy array

      :rtype: None.



   .. py:method:: luttodat(dat)

      LUT to dat grid.

      :param dat: Input data.
      :type dat: numpy array

      :returns: **tmp** -- dat grid.
      :rtype: numpy array



   .. py:method:: on_resize(event)

      Resize event.

      Used to make sure tight_layout happens on startup.

      :param event: Unused.
      :type event: event

      :rtype: None.



   .. py:method:: init_grid(dat, dat2=None, opac=0.0)

      Initialise grid.

      :param dat: Raster dataset.
      :type dat: numpy array
      :param dat2: PyGMI raster dataset. The default is None.
      :type dat2: pygmi.raster.datatypes.Data, optional
      :param opac: Opacity between 0 and 100. The default is 0.0.
      :type opac: float, optional

      :rtype: None.



   .. py:method:: init_grid_top(dat2=None, opac=100.0)

      Initialise top grid.

      :param dat2: Combobox text. The default is None.
      :type dat2: str, optional
      :param opac: Opacity between 0 and 100. The default is 100.0.
      :type opac: float, optional

      :rtype: None.



   .. py:method:: slide_grid(dat, dat2=None, opac=None)

      Slide grid.

      :param dat: Raster data array.
      :type dat: numpy array.
      :param dat2: Raster data array. The default is None.
      :type dat2: numpy array, optional
      :param opac: Opacity between 0 and 100. The default is None.
      :type opac: float, optional

      :rtype: None.



   .. py:method:: slide_grid_top(opac=None)

      Slide top grid.

      :param opac: Opacity between 0 and 100. The default is None.
      :type opac: float, optional

      :rtype: None.



   .. py:method:: update_line()

      Update the line position.

      :rtype: None.



   .. py:method:: update_line_top()

      Update the top line position.

      :rtype: None.



   .. py:method:: init_plot(xdat, dat, extent, xdat2=None, dat2=None)

      Initialise profile line plot.

      :param xdat: X coordinates.
      :type xdat: numpy array
      :param dat: Data values.
      :type dat: numpy array
      :param extent: Extent list.
      :type extent: list
      :param xdat2: X coordinates. The default is None.
      :type xdat2: numpy array, optional
      :param dat2: Data values. The default is None.
      :type dat2: numpy array, optional

      :rtype: None.



   .. py:method:: slide_plot(xdat, dat, xdat2=None, dat2=None)

      Slide plot.

      :param xdat: X coordinates.
      :type xdat: numpy array
      :param dat: Data values.
      :type dat: numpy array
      :param xdat2: X coordinates. The default is None.
      :type xdat2: numpy array, optional
      :param dat2: Data values. The default is None.
      :type dat2: numpy array, optional

      :rtype: None.



.. py:class:: MySlider(parent=None)

   Bases: :py:obj:`PySide6.QtWidgets.QSlider`


   My Slider.

   Custom class which allows clicking on a horizontal slider bar with slider
   moving to click in a single step.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: mousePressEvent(event)

      Mouse press event.

      :param event: Event variable.
      :type event: event

      :rtype: None.



   .. py:method:: mouseMoveEvent(event)

      Mouse move event.

      Jump to pointer position while moving.

      :param event: Event variable.
      :type event: event

      :rtype: None.



.. py:class:: LithBound(lmod)

   Bases: :py:obj:`PySide6.QtWidgets.QDialog`


   Class to call up a dialog for lithological boundary.

   :param lmod: Reference to the lithology model.
   :type lmod: pygmi.pfmod.datatypes.LithModel


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: get_lith()

      Get lithology.

      :returns: * **lithlower** (*int*) -- Lower lithology index.
                * **lithupper** (*int*) -- Upper lithology index.



.. py:class:: PlotScale(parent, lmod)

   Bases: :py:obj:`PySide6.QtWidgets.QDialog`


   Class to call up a dialog for plot axis scale.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional
   :param lmod: Reference to the lithology model.
   :type lmod: pygmi.pfmod.datatypes.LithModel


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: custom()

      Set custom radiobutton when limits are changed.

      :rtype: None.



.. py:class:: RangedCopy(parent=None)

   Bases: :py:obj:`PySide6.QtWidgets.QDialog`


   Class to call up a dialog for ranged copying.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: target_update()

      Update target.

      :rtype: None.



.. py:class:: MyToolbar(parent=None)

   Bases: :py:obj:`matplotlib.backends.backend_qt.NavigationToolbar2QT`


   Custom Matplotlib toolbar.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: axis_scale()

      Axis scale.

      :rtype: None.



   .. py:method:: b_logs()

      Borehole logs.

      :rtype: None.



   .. py:method:: mag_profile()

      View magnetic profile.

      :rtype: None.



   .. py:method:: grv_profile()

      View gravity profile.

      :rtype: None.



.. py:class:: GaugeWidget(*args, **kwargs)

   Bases: :py:obj:`PySide6.QtWidgets.QDial`


   Gauge widget.


   .. py:method:: paintEvent(event)

      Paint event.

      :param event: Event variable.
      :type event: event

      :rtype: None.



.. py:class:: ImportPicture(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Import Picture dialog.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: get_filename()

      Get filename of picture.

      :rtype: None.



   .. py:method:: getcoords()

      Get coordinates.

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



.. py:function:: gridmatch2(cgrv, rgrv)

   Grid match.

   Matches the rows and columns of the second grid to the first grid.

   :param cgrv: Raster dataset.
   :type cgrv: pygmi.raster.datatypes.Data.
   :param rgrv: Raster dataset.
   :type rgrv: pygmi.raster.datatypes.Data

   :returns: Output data.
   :rtype: numpy array


.. py:function:: rotate2d(pts, cntr, ang=np.pi / 4)

   Rotate 2D.

   Rotates points(nx2) about center cntr(2) by angle ang(1) in radians.

   :param pts: Points to rotate.
   :type pts: numpy array
   :param cntr: Center of rotation.
   :type cntr: numpy array
   :param ang: Angle to rotate in radians. The default is np.pi/4.
   :type ang: float, optional

   :returns: **pts2** -- Rotated points.
   :rtype: numpy array


