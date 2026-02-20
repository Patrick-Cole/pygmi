pygmi.pfmod.mvis3d
==================

.. py:module:: pygmi.pfmod.mvis3d

.. autoapi-nested-parse::

   Code for the 3d potential field model visualisation.



Classes
-------

.. autoapisummary::

   pygmi.pfmod.mvis3d.Mod3dDisplay
   pygmi.pfmod.mvis3d.MySunCanvas


Functions
---------

.. autoapisummary::

   pygmi.pfmod.mvis3d.updatemod
   pygmi.pfmod.mvis3d.calc_norms
   pygmi.pfmod.mvis3d.normalize_v3
   pygmi.pfmod.mvis3d.MarchingCubes
   pygmi.pfmod.mvis3d.InterpolateVertices
   pygmi.pfmod.mvis3d.fancyindex
   pygmi.pfmod.mvis3d.bitget
   pygmi.pfmod.mvis3d.bitset
   pygmi.pfmod.mvis3d.sub2ind
   pygmi.pfmod.mvis3d.ind2sub
   pygmi.pfmod.mvis3d.GetTables


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



   .. py:method:: change_light()

      Change light type



   .. py:method:: save()

      Save a jpg.

      :rtype: None.



   .. py:method:: update_for_kmz()

      Update for the kmz file.

      :rtype: None.



   .. py:method:: change_defs()

      List widget routine.

      :rtype: None.



   .. py:method:: data_init()

      Initialise data.

      :rtype: None.



   .. py:method:: set_selected_liths()

      Set the selected lithologies.

      :rtype: None.



   .. py:method:: mod3d_vs()

      Vertical slider used to scale 3d view.



   .. py:method:: resetlight()

      Reset light to the current model position.

      :rtype: None.



   .. py:method:: sunclick(event)

      Sunclick event is used to track changes to the sunshading.

      :param event - matplotlib button press event: event returned by matplotlib when a button is pressed



   .. py:method:: update_color()

      Update colour only.

      :rtype: None.



   .. py:method:: run()

      Entry point into the routine, used to run context menu item.

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: update_plot()

      Update 3D model.

      :rtype: None.



   .. py:method:: update_model(issmooth=None)

      Update the 3d model.

      Faces, nodes and face normals are calculated here, from the voxel
      model.

      :param issmooth: Flag to indicate a smooth model. The default is None.
      :type issmooth: bool, optional

      :rtype: None.



   .. py:method:: update_model2()

      Update the 3d model part 2.

      :rtype: None.



.. py:class:: MySunCanvas(parent=None)

   Bases: :py:obj:`matplotlib.backends.backend_qtagg.FigureCanvasQTAgg`


   Canvas for the sunshading tool.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional

   .. attribute:: sun

      plot of a circle 'o' showing where the sun is

      :type: matplotlib plot instance

   .. attribute:: axes

      axes on which the sun is drawn

      :type: matplotlib axes instance


   .. py:method:: init_graph()

      Initialise graph.

      :rtype: None.



.. py:function:: updatemod(gdat2, cindx, cloc)

   Update model without smoothing.

   :param gdat2: Model values.
   :type gdat2: numpy array
   :param cindx: Corner index.
   :type cindx: numpy array
   :param cloc: Corner location.
   :type cloc: numpy array

   :returns: * **newcorners** (*numpy array*) -- New corner coordinates.
             * **newfaces** (*numpy array*) -- New face indices.


.. py:function:: calc_norms(faces, vtx)

   Calculate normals.

   :param faces: Array of faces.
   :type faces: numpy array
   :param vtx: Array of vertices.
   :type vtx: numpy array.

   :returns: **nrm** -- output normals.
   :rtype: numpy array


.. py:function:: normalize_v3(arr)

   Normalize a numpy array of 3 component vectors shape=(n,3).

   :param arr: Array of 3 component vectors.
   :type arr: numpy array

   :returns: **arr** -- Output array of 3 component vectors.
   :rtype: numpy array


.. py:function:: MarchingCubes(x, y, z, c, iso, *, showlog=print)

   Marching cubes.

   Use marching cubes algorithm to compute a triangulated mesh of the
   isosurface within the 3D matrix of scalar values C at isosurface value
   ISO. The 3D matrices (X,Y,Z) represent a Cartesian, axis-aligned grid
   specifying the points at which the data C is given. These coordinate
   arrays must be in the format produced by Matlab's meshgrid function.
   Output arguments F and V are the face list and vertex list
   of the resulting triangulated mesh. The orientation of the triangles is
   chosen such that the normals point from the higher values to the lower
   values. Optional arguments COLORS ans COLS can be used to produce
   interpolated mesh face colours. For usage, see Matlab's isosurface.m.
   To avoid Out of Memory errors when matrix C is large, convert matrices
   X,Y,Z and C from doubles (Matlab default) to singles (32-bit floats).

   Originally Adapted for Matlab by Peter Hammer in 2011 based on an
   Octave function written by Martin Helm <martin@mhelm.de> in 2009
   http://www.mhelm.de/octave/m/marching_cube.m

   Revised 30 September, 2011 to add code by Oliver Woodford for removing
   duplicate vertices.

   :param x: X coordinates.
   :type x: numpy array
   :param y: Y coordinates.
   :type y: numpy array
   :param z: Z coordinates.
   :type z: numpy array
   :param c: Data.
   :type c: numpy array
   :param iso: Isosurface level.
   :type iso: float
   :param showlog: Display information. The default is print.
   :type showlog: function, optional

   :returns: * **F** (*numpy array*) -- Face list.
             * **V** (*numpy array*) -- Vertex list.


.. py:function:: InterpolateVertices(isolevel, p1x, p1y, p1z, p2x, p2y, p2z, valp1, valp2)

   Interpolate vertices.

   :param isolevel: ISO level.
   :type isolevel: float
   :param p1x: p1 x coordinate.
   :type p1x: numpy array
   :param p1y: p1 y coordinate.
   :type p1y: numpy array
   :param p1z: p1 z coordinate.
   :type p1z: numpy array
   :param p2x: p2 x coordinate.
   :type p2x: numpy array
   :param p2y: p2 y coordinate.
   :type p2y: numpy array
   :param p2z: p2 z coordinate.
   :type p2z: numpy array
   :param valp1: p1 value.
   :type valp1: numpy array
   :param valp2: p2 value.
   :type valp2: numpy array

   :returns: **p** -- Interpolated vertices.
   :rtype: numpy array


.. py:function:: fancyindex(out, var1, ii, jj, kk)

   Fancy index.

   :param out: Input data.
   :type out: numpy array
   :param var1: Input data.
   :type var1: numpy array
   :param ii: i indices.
   :type ii: numpy array
   :param jj: j indices.
   :type jj: numpy array
   :param kk: k indices.
   :type kk: numpy array

   :returns: **out** -- Output data with new values.
   :rtype: numpy array


.. py:function:: bitget(byteval, idx)

   Bit get.

   :param byteval: Input value to get bit from.
   :type byteval: int
   :param idx: Position of bit to get.
   :type idx: int

   :returns: True if not 0, False otherwise.
   :rtype: bool


.. py:function:: bitset(byteval, idx)

   Bit set.

   :param byteval: Input value to get bit from.
   :type byteval: int
   :param idx: Position of bit to get.
   :type idx: int

   :returns: Output value with bit set.
   :rtype: int


.. py:function:: sub2ind(msize, row, col, layer)

   Sub to index.

   :param msize: Tuple with number of rows and columns as first two elements.
   :type msize: tuple
   :param row: Row.
   :type row: int
   :param col: Column.
   :type col: int
   :param layer: Layer.
   :type layer: numpy array

   :returns: **tmp** -- Index returned.
   :rtype: numpy array


.. py:function:: ind2sub(msize, idx)

   Index to sub.

   :param msize: Tuple with number of rows and columns as first two elements.
   :type msize: tuple
   :param idx: Array of indices.
   :type idx: numpy array

   :returns: * **row** (*int*) -- Row.
             * **col** (*int*) -- Column.
             * **layer** (*numpy array*) -- Layer.


.. py:function:: GetTables()

   Get tables.

   :returns: A list with edgetable and tritable.
   :rtype: list


