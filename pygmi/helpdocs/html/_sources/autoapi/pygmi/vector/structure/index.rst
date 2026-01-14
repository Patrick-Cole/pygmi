pygmi.vector.structure
======================

.. py:module:: pygmi.vector.structure

.. autoapi-nested-parse::

   Structure complexity routines.



Classes
-------

.. autoapisummary::

   pygmi.vector.structure.StructComp


Functions
---------

.. autoapisummary::

   pygmi.vector.structure.extendlines
   pygmi.vector.structure.feature_intersection_density
   pygmi.vector.structure.feature_orientation_diversity
   pygmi.vector.structure.feature_circular_stats
   pygmi.vector.structure.feature_fracdim
   pygmi.vector.structure.fractal_dimension
   pygmi.vector.structure.linesplit
   pygmi.vector.structure.segments_to_angles


Module Contents
---------------

.. py:class:: StructComp(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   GUI for structure complexity.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: method_change()

      When method is changed, this updated hidden controls.

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



.. py:function:: extendlines(gdf, length=500, piter=iter)

   Extent lines from GeoPandas dataframe.

   :param gdf: A dataframe containing LINESTRINGs.
   :type gdf: GeoDataFrame
   :param length: distance in metres to extend the line on either side.
   :type length: float
   :param piter: Progressbar iterable.
   :type piter: iter

   :returns: **gdf2** -- A dataframe containing extended LINESTRINGs.
   :rtype: GeoDataFrame


.. py:function:: feature_intersection_density(gdf, dxy, var, extend=500, piter=iter)

   Feature intersection density.

   :param gdf: GeoDataframe of linear features.
   :type gdf: GeoDataFrame
   :param dxy: Raster cell size
   :type dxy: float
   :param var: Variance.
   :type var: float
   :param extend: Distance to extend linear features.
   :type extend: float
   :param piter: Progressbar iterable.
   :type piter: iter

   :returns: * **geom2** (*GeoDataFrame*) -- New geometry with intersection points.
             * **dat** (*pygmi.raster.datatypes.Data*) -- Output raster data


.. py:function:: feature_orientation_diversity(gdf, dxy, wsize=3, piter=iter)

   Feature orientation diversity.

   :param gdf: GeoDataframe of linear features.
   :type gdf: GeoDataFrame
   :param dxy: Raster cell size
   :type dxy: float
   :param wsize: Window size (must be odd)
   :type wsize: int
   :param piter: Progressbar iterable.
   :type piter: iter

   :returns: **dat** -- Output raster data
   :rtype: pygmi.raster.datatypes.Data


.. py:function:: feature_circular_stats(gdf, dxy, wsize=3, piter=iter)

   Feature circular variance.

   :param gdf: GeoDataframe of linear features.
   :type gdf: GeoDataFrame
   :param dxy: Raster cell size
   :type dxy: float
   :param wsize: Window size (must be odd)
   :type wsize: int
   :param piter: Progressbar iterable.
   :type piter: iter

   :returns: **dat** -- Output raster data
   :rtype: pygmi.raster.datatypes.Data


.. py:function:: feature_fracdim(gdf, dxy, wsize=21, piter=iter)

   Feature fractal dimension.

   :param gdf: GeoDataframe of linear features.
   :type gdf: GeoDataFrame
   :param dxy: Raster cell size
   :type dxy: float
   :param wsize: Window size (must be odd)
   :type wsize: int
   :param piter: Progressbar iterable.
   :type piter: iter

   :returns: **dat** -- Output raster data
   :rtype: pygmi.raster.datatypes.Data


.. py:function:: fractal_dimension(warray, max_box_size=None, min_box_size=1, n_samples=20, n_offsets=0)

   Calculate the fractal dimension of a 3D numpy array.

   From: https://github.com/ChatzigeorgiouGroup/FractalDimension

   :param warray: The array to calculate the fractal dimension of.
   :type warray: np.array
   :param max_box_size: The largest box size, given as the power of 2 so that 2**max_box_size
                        gives the side length of the largest box. The default is None.
   :type max_box_size: int, optional
   :param min_box_size: The smallest box size, given as the power of 2 so that 2**min_box_size
                        gives the side length of the smallest box. The default is 1.
   :type min_box_size: int, optional
   :param n_samples: number of scales to measure over. The default is 20.
   :type n_samples: int, optional
   :param n_offsets: number of offsets to search over to find the smallest set N(s) to
                     cover all voxels>0. The default is 0.
   :type n_offsets: int, optional

   :returns: **coeffs[0]** -- Fractal dimension
   :rtype: float


.. py:function:: linesplit(curve)

   Split LineString into segments.


.. py:function:: segments_to_angles(gdf, piter=iter)

   Get line segment angles.

   :param gdf: GeoDataFrame with line segments.
   :type gdf: GeoDataFrame
   :param piter: Progressbar iterable.
   :type piter: iter

   :returns: **gdf2** -- GeoDataFrame with angles added.
   :rtype: GeoDataFrame


