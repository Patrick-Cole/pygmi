pygmi.pfmod.grvmag3d
====================

.. py:module:: pygmi.pfmod.grvmag3d

.. autoapi-nested-parse::

   Gravity and magnetic field calculations.

   This uses the following algorithms:

   .. rubric:: References

   Singh, B., Guptasarma, D., 2001. New method for fast computation of gravity
   and magnetic anomalies from arbitrary polyhedral. Geophysics 66, 521-526.

   Blakely, R.J., 1996. Potential Theory in Gravity and Magnetic Applications,
   1st edn. Cambridge University Press, Cambridge, UK, 441 pp. 200-201



Classes
-------

.. autoapisummary::

   pygmi.pfmod.grvmag3d.GravMag
   pygmi.pfmod.grvmag3d.GeoData


Functions
---------

.. autoapisummary::

   pygmi.pfmod.grvmag3d.calc_demag
   pygmi.pfmod.grvmag3d.save_layer
   pygmi.pfmod.grvmag3d.gridmatch
   pygmi.pfmod.grvmag3d.calc_field
   pygmi.pfmod.grvmag3d.sum_fields
   pygmi.pfmod.grvmag3d.quick_model
   pygmi.pfmod.grvmag3d.dircos
   pygmi.pfmod.grvmag3d.dat_extent


Module Contents
---------------

.. py:class:: GravMag(parent=None)

   The GravMag class holds generic magnetic and gravity modelling routines.

   Routine that will calculate the final versions of the field. Other,
   related code is here as well, such as the inversion routines.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: calc_field_mag()

      Pre field-calculation routine.

      :rtype: None.



   .. py:method:: calc_field_grav()

      Pre field-calculation routine.

      :rtype: None.



   .. py:method:: calc_field_mag_changes()

      Calculate only magnetic field changes.

      :rtype: None.



   .. py:method:: calc_field_grav_changes()

      Calculate only gravity field changes.

      :rtype: None.



   .. py:method:: calc_field2(showreports=False, magcalc=False)

      Calculate magnetic and gravity field.

      :param showreports: Flag for showing reports. The default is False.
      :type showreports: bool, optional
      :param magcalc: Flag for choosing the magnetic calculation. The default is False.
      :type magcalc: bool, optional

      :rtype: None.



   .. py:method:: calc_regional()

      Calculate magnetic and gravity regional.

      Calculates a gravity and magnetic regional value based on a single
      solid lithology model. The principle is that the maximum value for a
      solid model with fixed extents and depth, using the most COMMON
      lithology, would be the MAXIMUM AVERAGE value for any model which we
      would do. Therefore the regional is simply:

          REGIONAL = OBS GRAVITY MEAN - CALC GRAVITY MAX

      This routine calculates the last term.

      :rtype: None.



   .. py:method:: test_pattern()

      Displays a test pattern of the data.

      This is an indication of the edge of model field decay. It gives an
      idea about how reliable the calculated field on the edge of the model
      is.

      :rtype: None.



   .. py:method:: update_graph(grvval, magval, modind)

      Update the graph.

      :param grvval: Array of gravity values.
      :type grvval: numpy array
      :param magval: Array of magnetic values.
      :type magval: numpy array
      :param modind: Model indices.
      :type modind: numpy array

      :rtype: None.



.. py:class:: GeoData(parent, ncols=10, nrows=10, numz=10, dxy=10.0, d_z=10.0, mht=80.0, ght=0.0)

   Data layer class.

   This class defines each geological type and calculates the field
   for one cube from the standard definitions.

   The is a class which contains the geophysical information for a single
   lithology. This includes the final calculated field for that lithology
   only.

   :param parent: Reference to the parent routine.
   :type parent: parent
   :param ncols: Number of columns in the model.
   :type ncols: int
   :param nrows: Number of rows in the model.
   :type nrows: int
   :param numz: Number of layer in the model.
   :type numz: int
   :param dxy: X and Y size of each voxel.
   :type dxy: float
   :param d_z: Layer thickness.
   :type d_z: float
   :param mht: Magnetic sensor height.
   :type mht: float
   :param ght: Gravity sensor height.
   :type ght: float


   .. py:method:: calc_origin_grav(hcor=None)

      Calculate the field values for the lithologies.

      :param hcor: Height corrections. The default is None.
      :type hcor: numpy array or None, optional

      :rtype: None.



   .. py:method:: calc_origin_mag(hcor=None, demag=False)

      Calculate the field values for the lithologies.

      :param hcor: Height corrections. The default is None.
      :type hcor: numpy array or None, optional

      :rtype: None.



   .. py:method:: rho()

      Return the density contrast.

      :returns: Density contrast.
      :rtype: float



   .. py:method:: set_xyz(ncols, nrows, numz, g_dxy, mht, ght, d_z, dxy=None, modified=True)

      Sets/updates xyz parameters.

      :param ncols: Number of columns.
      :type ncols: int
      :param nrows: Number of rows.
      :type nrows: int
      :param numz: Number of layers.
      :type numz: int
      :param g_dxy: Grid spacing in x and y direction.
      :type g_dxy: float
      :param mht: Magnetic sensor height.
      :type mht: float
      :param ght: Gravity sensor height.
      :type ght: float
      :param d_z: Model spacing in z direction.
      :type d_z: float
      :param dxy: Model spacing in x and y direction. The default is None.
      :type dxy: float, optional
      :param modified: Whether the model was modified. The default is True.
      :type modified: bool, optional

      :rtype: None.



   .. py:method:: set_xyz12()

      Set x12, y12, z12.

      This is the limits of the cubes for the model

      :rtype: None.



   .. py:method:: gboxmain(xobs, yobs, zobs, hcor)

      Gbox routine by Blakely.

      Note: xobs, yobs and zobs must be floats or there will be problems
      later.

      Subroutine GBOX computes the vertical attraction of a
      rectangular prism.  Sides of prism are parallel to x,y,z axes,
      and z axis is vertical down.

      Input parameters:
      |    Observation point is (x0,y0,z0).  The prism extends from x1
      |    to x2, from y1 to y2, and from z1 to z2 in the x, y, and z
      |    directions, respectively.  Density of prism is rho.  All
      |    distance parameters in units of m;

      Output parameters:
      |    Vertical attraction of gravity, g, in mGal/rho.
      |    Must still be multiplied by rho outside routine.
      |    Done this way for speed.

      :param xobs: Observation X coordinates.
      :type xobs: numpy array
      :param yobs: Observation Y coordinates.
      :type yobs: numpy array
      :param zobs: Observation Z coordinates.
      :type zobs: numpy array
      :param hcor: Height corrections.
      :type hcor: numpy array

      :rtype: None.



   .. py:method:: mboxmain(xobs, yobs, zobs, hcor, demag=False)

      Mbox routine by Blakely.

      Note: xobs, yobs and zobs must be floats or there will be problems
      later.

      Subroutine MBOX computes the total field anomaly of an infinitely
      extended rectangular prism.  Sides of prism are parallel to x,y,z
      axes, and z is vertical down.  Bottom of prism extends to infinity.
      Two calls to mbox can provide the anomaly of a prism with finite
      thickness; e.g.,

      |    call mbox(x0,y0,z0,x1,y1,z1,x2,y2,mi,md,fi,fd,m,theta,t1)
      |    call mbox(x0,y0,z0,x1,y1,z2,x2,y2,mi,md,fi,fd,m,theta,t2)
      |    t=t1-t2

      Requires subroutine DIRCOS.  Method from Bhattacharyya (1964).

      Input parameters:
      |    Observation point is (x0,y0,z0).  Prism extends from x1 to
      |    x2, y1 to y2, and z1 to infinity in x, y, and z directions,
      |    respectively.  Magnetization defined by inclination mi,
      |    declination md, intensity m.  Ambient field defined by
      |    inclination fi and declination fd.  X axis has declination
      |    theta. Distance units are irrelevant but must be consistent.
      |    Angles are in degrees, with inclinations positive below
      |    horizontal and declinations positive east of true north.
      |    Magnetization in A/m.

      Output parameters:
      |    Total field anomaly t, in nT.

      :param xobs: Observation X coordinates.
      :type xobs: numpy array
      :param yobs: Observation Y coordinates.
      :type yobs: numpy array
      :param zobs: Observation Z coordinates.
      :type zobs: numpy array
      :param hcor: Height corrections.
      :type hcor: numpy array

      :rtype: None.



.. py:function:: calc_demag(mvec, k, dxy, dz)

   Calculate demagnetisation correction.

   :param mvec: Body Magnetisation.
   :type mvec: numpy array
   :param k: susceptibility.
   :type k: float
   :param dxy: cell width.
   :type dxy: float
   :param dz: cell height.
   :type dz: float

   :returns: **outvec** -- Corrected magnetisation.
   :rtype: numpy array


.. py:function:: save_layer(mlist)

   Routine to save the mlayer and glayer to a file.

   :param mlist: List with 2 elements - lithology name and LithModel.
   :type mlist: list

   :returns: **outfile** -- Link to a temporary file.
   :rtype: TemporaryFile


.. py:function:: gridmatch(lmod, ctxt, rtxt)

   Match the rows and columns of the second grid to the first grid.

   :param lmod: Lithology Model.
   :type lmod: LithModel
   :param ctxt: First grid text label.
   :type ctxt: str
   :param rtxt: Second grid text label.
   :type rtxt: str

   :returns: **dat** -- Numpy array of data.
   :rtype: numpy array


.. py:function:: calc_field(lmod, pbars=None, showtext=None, parent=None, showreports=False, magcalc=False, demag=False)

   Calculate magnetic and gravity field.

   This function calculates the magnetic and gravity field. It has two
   different modes of operation, by using the magcalc switch. If magcalc=True
   then magnetic fields are calculated, otherwise only gravity is calculated.

   :param lmod: PyGMI lithological model
   :type lmod: LithModel
   :param pbars: progress bar routine if available. (internal use)
   :type pbars: module
   :param showtext: showtext routine if available. (internal use)
   :type showtext: module
   :param showreports: show extra reports
   :type showreports: bool
   :param magcalc: if True, calculates magnetic data, otherwise only gravity.
   :type magcalc: bool

   :returns: **lmod.griddata** -- dictionary of items of type Data.
   :rtype: dictionary


.. py:function:: sum_fields(k, mgval, numx, numy, modind, aaa0, aaa1, mlayers, hcorflat, mijk)

   Sum magnetic and gravity field datasets to produce final model field.

   :param k: k index.
   :type k: int
   :param mgval: Magnetic or gravity data being summed.
   :type mgval: numpy array
   :param numx: Number of x elements.
   :type numx: int
   :param numy: Number of y elements.
   :type numy: int
   :param modind: model with indices representing lithologies.
   :type modind: numpy array
   :param aaa0: x indices for offsets.
   :type aaa0: numpy array
   :param aaa1: y indices for offsets.
   :type aaa1: numpy array
   :param mlayers: Layer fields for summation.
   :type mlayers: numpy array
   :param hcorflat: Height correction.
   :type hcorflat: numpy array
   :param mijk: Current lithology index.
   :type mijk: int

   :returns: **mgval** -- Output summed data.
   :rtype: numpy array


.. py:function:: quick_model(numx=50, numy=40, numz=5, dxy=100.0, d_z=100.0, tlx=0.0, tly=0.0, tlz=0.0, mht=100.0, ght=0.0, finc=-67, fdec=-17, inputliths=None, susc=None, dens=None, minc=None, mdec=None, mstrength=None, hintn=30000.0)

   Quick model function.

   :param numx: Number of x elements. The default is 50.
   :type numx: int, optional
   :param numy: Number of y elements. The default is 40.
   :type numy: int, optional
   :param numz: number of z elements (layers). The default is 5.
   :type numz: int, optional
   :param dxy: Cell size in x and y direction. The default is 100..
   :type dxy: float, optional
   :param d_z: Layer thickness. The default is 100..
   :type d_z: float, optional
   :param tlx: Top left x coordinate. The default is 0..
   :type tlx: float, optional
   :param tly: Top left y coordinate. The default is 0..
   :type tly: float, optional
   :param tlz: Top left z coordinate. The default is 0..
   :type tlz: float, optional
   :param mht: Magnetic sensor height. The default is 100..
   :type mht: float, optional
   :param ght: Gravity sensor height. The default is 0..
   :type ght: float, optional
   :param finc: Magnetic field inclination (degrees). The default is -67.
   :type finc: float, optional
   :param fdec: Magnetic field declination (degrees). The default is -17.
   :type fdec: float, optional
   :param inputliths: List of input lithologies. The default is None.
   :type inputliths: list or None, optional
   :param susc: List of susceptibilities. The default is None.
   :type susc: list or None, optional
   :param dens: List of densities. The default is None.
   :type dens: list or None, optional
   :param minc: List of remanent inclinations (degrees). The default is None.
   :type minc: list or None, optional
   :param mdec: List of remanent declinations (degrees). The default is None.
   :type mdec: list or None, optional
   :param mstrength: List of remanent magnetisations (A/m). The default is None.
   :type mstrength: list or None, optional
   :param hintn: Magnetic field strength (nT). The default is 30000.
   :type hintn: float, optional

   :returns: **lmod** -- Output model.
   :rtype: LithModel


.. py:function:: dircos(incl, decl, azim)

   DIRCOS computes direction cosines from inclination and declination.

   :param incl: inclination in degrees positive below horizontal.
   :type incl: float
   :param decl: declination in degrees positive east of true north.
   :type decl: float
   :param azim: azimuth of x axis in degrees positive east of north.
   :type azim: float

   :returns: * **aaa** (*float*) -- First direction cosine.
             * **bbb** (*float*) -- Second direction cosine.
             * **ccc** (*float*) -- Third direction cosine.


.. py:function:: dat_extent(dat, axes)

   Get the extent of the dat variable.

   :param dat: PyGMI raster dataset.
   :type dat: pygmi.raster.datatypes.Data
   :param axes: Matplotlib axes.
   :type axes: matplotlib.axes._subplots.AxesSubplot

   :returns: * **left** (*float*) -- Left coordinate.
             * **right** (*float*) -- Right coordinate.
             * **bottom** (*float*) -- Bottom coordinate.
             * **top** (*float*) -- Top coordinate.


