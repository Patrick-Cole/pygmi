pygmi.raster.anaglyph
=====================

.. py:module:: pygmi.raster.anaglyph

.. autoapi-nested-parse::

   Anaglyph routine.



Classes
-------

.. autoapisummary::

   pygmi.raster.anaglyph.MyMplCanvas
   pygmi.raster.anaglyph.PlotAnaglyph


Functions
---------

.. autoapisummary::

   pygmi.raster.anaglyph.sunshade
   pygmi.raster.anaglyph.anaglyph
   pygmi.raster.anaglyph.rot_and_clean


Module Contents
---------------

.. py:class:: MyMplCanvas(parent=None)

   Bases: :py:obj:`matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg`


   Matplotlib canvas widget for the actual plot.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: update_contours(data1, scale=7, rotang=10)

      Update the contour plot.

      :param data1: raster dataset to be used in contouring.
      :type data1: PyGMI raster data.
      :param scale: Scale. The default is 7.
      :type scale: float, optional
      :param rotang: Rotation in degrees. The default is 10.
      :type rotang: float, optional

      :rtype: None.



   .. py:method:: update_raster(data1, *, scale=7, rotang=10, atype='dubois', cmap=colormaps['jet'], shade=False)

      Update the raster plot.

      :param data1: raster dataset to be used in contouring
      :type data1: PyGMI raster Data
      :param scale: Scale. The default is 7.
      :type scale: float, optional
      :param rotang: Rotation in degrees. The default is 10.
      :type rotang: float, optional
      :param atype: Anaglyph type. The default is 'dubois'.
      :type atype: str, optional
      :param cmap: Matplotlib colormap. The default is jet.
      :type cmap: matplotlib.colors.LinearSegmentedColormap, optional
      :param shade: Option to choose sunshading. The default is False.
      :type shade: bool, optional

      :rtype: None.



   .. py:method:: update_colors(doshade=False, cmap=colormaps['jet'], atype='dubois')

      Update colors.

      :param doshade: Option to choose sunshading. The default is False.
      :type doshade: bool, optional
      :param cmap: Matplotlib colormap. The default is jet.
      :type cmap: matplotlib.colors.LinearSegmentedColormap, optional
      :param atype: Anaglyph type. The default is 'dubois'.
      :type atype: str, optional

      :rtype: None.



   .. py:method:: update_atype(atype='dubois')

      Update anaglyph type.

      :param atype: Anaglyph type. The default is 'dubois'.
      :type atype: str, optional

      :rtype: None.



.. py:class:: PlotAnaglyph(parent=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   Anaglyph GUI Graph Window.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: change_all()

      Update from all combos.

      :rtype: None.



   .. py:method:: change_colors()

      Update colour bar.

      :rtype: None.



   .. py:method:: change_atype()

      Update anaglyph type.

      :rtype: None.



   .. py:method:: change_contours()

      Update contours.

      :rtype: None.



   .. py:method:: change_image()

      Change Image, setting defaults.

      :rtype: None.



   .. py:method:: run()

      Entry point into the routine, used to run context menu item.

      :rtype: None.



.. py:function:: sunshade(data, *, azim=-np.pi / 4.0, elev=np.pi / 4.0, alpha=1, cell=100, cmap=colormaps['terrain'])

   Perform Sunshading on data.

   :param data: input MxN data to be imaged.
   :type data: numpy array
   :param azim: Sun azimuth. The default is -np.pi/4..
   :type azim: float, optional
   :param elev: Sun elevation. The default is np.pi/4..
   :type elev: float, optional
   :param alpha: how much incident light is reflected (0 to 1). The default is 1.
   :type alpha: float, optional
   :param cell: between 1 and 100 - controls sunshade detail. The default is 100.
   :type cell: float, optional
   :param cmap: Matplotlib colormap.
   :type cmap: matplotlib.colors.LinearSegmentedColormap, optional

   :returns: **colormap** -- Output colour mapped array (MxNx4).
   :rtype: numpy array


.. py:function:: anaglyph(red, blue, atype='dubois')

   Colour Anaglyph.

   :param red: Dataset for red channel.
   :type red: numpy array
   :param blue: Dataset for blue channel.
   :type blue: numpy array
   :param atype: Anaglyph type. The default is 'dubois'.
   :type atype: str, optional

   :returns: **rgb** -- Output dataset.
   :rtype: numpy array


.. py:function:: rot_and_clean(x, y, z, rotang=5, rtype='red')

   Rotate and clean rotated data for 2d view.

   :param x: X coordinates.
   :type x: numpy array
   :param y: Y coordinates.
   :type y: numpy array
   :param z: Z coordinates (or data values).
   :type z: numpy array
   :param rotang: Rotation angle. The default is 5.
   :type rotang: float, optional
   :param rtype: Rotation type. The default is 'red'.
   :type rtype: str, optional

   :returns: **zmap** -- Output data.
   :rtype: numpy array


