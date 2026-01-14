pygmi.raster.ginterp
====================

.. py:module:: pygmi.raster.ginterp

.. autoapi-nested-parse::

   Plot Raster Data.

   This is the raster data interpretation module.  This module allows for the
   display of raster data in a variety of modes, as well as the export of that
   display to GeoTIFF format.

   Currently the following is supported
    * Pseudo Colour - data mapped to a colour map
    * Contours with solid contours
    * RGB ternary images
    * CMYK ternary images
    * Sun shaded or hill shaded images

   It can be very effectively used in conjunction with a GIS package which
   supports GeoTIFF files.



Classes
-------

.. autoapisummary::

   pygmi.raster.ginterp.MyMplCanvas
   pygmi.raster.ginterp.MySunCanvas
   pygmi.raster.ginterp.PlotInterp


Module Contents
---------------

.. py:class:: MyMplCanvas

   Bases: :py:obj:`matplotlib.backends.backend_qtagg.FigureCanvasQTAgg`


   Matplotlib canvas widget for the actual plot.

   .. attribute:: htype

      string indicating the histogram stretch to apply to the data

      :type: str

   .. attribute:: hstype

      string indicating the histogram stretch to apply to the sun data

      :type: str

   .. attribute:: cbar

      colour map to be used for pseudo colour bars

      :type: matplotlib colour map

   .. attribute:: data

      list of PyGMI raster data objects - used for colour images

      :type: list of pygmi.raster.datatypes.Data

   .. attribute:: sdata

      list of PyGMI raster data objects - used for shaded images

      :type: list of pygmi.raster.datatypes.Data

   .. attribute:: gmode

      string containing the graphics mode - Contour, Ternary, Sunshade,
      Single Colour Map.

      :type: str

   .. attribute:: argb

      list of matplotlib subplots. There are up to three.

      :type: list

   .. attribute:: hhist

      matplotlib hist associated with argb

      :type: list

   .. attribute:: hband

      list of strings containing the band names to be used.

      :type: list

   .. attribute:: htxt

      list of strings associated with hhist, denoting a raster value (where
      mouse is currently hovering over on image)

      :type: list

   .. attribute:: image

      imshow instance - this is the primary way of displaying an image.

      :type: imshow

   .. attribute:: cnt

      contour instance - used for the contour image

      :type: matplotlib contour

   .. attribute:: cntf

      contourf instance - used for the contour image

      :type: matplotlib contourf

   .. attribute:: background

      image bounding box - used in blitting

      :type: matplotlib bounding box

   .. attribute:: bbox_hist_red

      red histogram bounding box

      :type: matplotlib bounding box

   .. attribute:: bbox_hist_green

      green histogram bounding box

      :type: matplotlib bounding box

   .. attribute:: bbox_hist_blue

      blue histogram bounding box

      :type: matplotlib bounding box

   .. attribute:: axes

      axes for the plot

      :type: matplotlib axes

   .. attribute:: pinit

      calculated with aspect - used in sunshading

      :type: numpy array

   .. attribute:: qinit

      calculated with aspect - used in sunshading

      :type: numpy array

   .. attribute:: phi

      azimuth (sunshading)

      :type: float

   .. attribute:: theta

      sun elevation (sunshading)

      :type: float

   .. attribute:: cell

      between 1 and 100 - controls sunshade detail.

      :type: float

   .. attribute:: alpha

      how much incident light is reflected (0 to 1)

      :type: float

   .. attribute:: kval

      k value for CMYK mode

      :type: float


   .. py:method:: revent(event)

      Resize event.

      :param event: Resize event.
      :type event: matplotlib.backend_bases.ResizeEvent

      :rtype: None.



   .. py:method:: init_graph()

      Initialize the graph.

      :rtype: None.



   .. py:method:: move(event)

      Mouse is moving over canvas.

      :param event: Mouse event.
      :type event: matplotlib.backend_bases.MouseEvent

      :rtype: None.



   .. py:method:: update_contour()

      Update contours.

      :rtype: None.



   .. py:method:: update_graph()

      Update plot.

      :rtype: None.



   .. py:method:: update_hist_rgb(zval)

      Update the rgb histograms.

      :param zval: Data values.
      :type zval: numpy array

      :returns: **bnum** -- Bin numbers.
      :rtype: list



   .. py:method:: update_hist_single(zval=None, hno=0)

      Update the colour on a single histogram.

      :param zval: Data value.
      :type zval: float
      :param hno: Histogram number. The default is 0.
      :type hno: int, optional

      :returns: **binnum** -- Number of bins.
      :rtype: int



   .. py:method:: update_hist_text(hst, zval)

      Update the value on the histogram.

      :param hst: Histogram.
      :type hst: histogram
      :param zval: Data value.
      :type zval: float

      :rtype: None.



   .. py:method:: update_rgb()

      Update the RGB Ternary Map.

      :rtype: None.



   .. py:method:: update_single_color_map()

      Update the single colour map.

      :rtype: None.



   .. py:method:: update_shade()

      Update sun shade plot.

      :rtype: None.



   .. py:method:: update_shade_plot()

      Update shade plot for export.

      :returns: Sunshader data.
      :rtype: numpy array



.. py:class:: MySunCanvas

   Bases: :py:obj:`matplotlib.backends.backend_qtagg.FigureCanvasQTAgg`


   Canvas widget for the sunshading tool.

   .. attribute:: sun

      plot of a circle 'o' showing where the sun is

      :type: matplotlib plot instance

   .. attribute:: axes

      axes on which the sun is drawn

      :type: matplotlib axes instance


   .. py:method:: init_graph()

      Initialise graph.

      :rtype: None.



.. py:class:: PlotInterp(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   The primary GUI class for the raster data interpretation module.

   The main interface is set up from here, as well as monitoring of the mouse
   over the sunshading.

   The PlotInterp class allows for the display of raster data in a variety of
   modes, as well as the export of that display to GeoTIFF format.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional

   .. attribute:: self.mmc

      main canvas containing the image

      :type: pygmi.raster.ginterp.MyMplCanvas, FigureCanvas

   .. attribute:: self.msc

      small canvas containing the sunshading control

      :type: pygmi.raster.ginterp.MySunCanvas, FigureCanvas


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: change_allclip()

      Change all clip percentages to the current one.

      :rtype: None.



   .. py:method:: change_blue()

      Change the blue or third display band.

      :rtype: None.



   .. py:method:: change_cbar()

      Change the colour map for the colour bar.

      :rtype: None.



   .. py:method:: change_clipband()

      Change the clip percentage band.

      :rtype: None.



   .. py:method:: change_dtype()

      Change display type.

      :rtype: None.



   .. py:method:: change_green()

      Change the green or second band.

      :rtype: None.



   .. py:method:: change_htype()

      Change the histogram stretch to apply to the normal data.

      :rtype: None.



   .. py:method:: change_kval()

      Change the CMYK K value.

      :rtype: None.



   .. py:method:: change_lclip()

      Change the linear clip percentage.

      :rtype: None.



   .. py:method:: change_red()

      Change the red or first band.

      :rtype: None.



   .. py:method:: change_sun()

      Change the sunshade band.

      :rtype: None.



   .. py:method:: change_sun_checkbox()

      Use when sunshading checkbox is clicked.

      :rtype: None.



   .. py:method:: change_sunsliders()

      Change the sun shading sliders.

      :rtype: None.



   .. py:method:: data_init()

      Initialise Data.

      Entry point into routine. This entry point exists for
      the case  where data must be initialised before entering at the
      standard 'settings' sub module.

      :rtype: None.



   .. py:method:: move(event)

      Move event is used to track changes to the sunshading.

      :param event: Mouse event.
      :type event: matplotlib.backend_bases.MouseEvent

      :rtype: None.



   .. py:method:: run()

      Entry point into the routine, used to run context menu item.



   .. py:method:: save_img()

      Save image as a GeoTIFF.

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: settings(nodialog=False)

      Entry point into item.

      :param nodialog: Run settings without a dialog. The default is False.
      :type nodialog: bool, optional

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: saveproj()

      Save project data from class.

      :rtype: None.



