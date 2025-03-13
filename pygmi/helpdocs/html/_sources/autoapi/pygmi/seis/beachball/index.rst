pygmi.seis.beachball
====================

.. py:module:: pygmi.seis.beachball

.. autoapi-nested-parse::

   Plot fault plane solutions.

   The code below is translated from bb.m written by Andy Michael and Oliver Boyd
   at http://www.ceri.memphis.edu/people/olboyd/Software/Software.html



Classes
-------

.. autoapisummary::

   pygmi.seis.beachball.MyMplCanvas
   pygmi.seis.beachball.BeachBall


Functions
---------

.. autoapisummary::

   pygmi.seis.beachball.beachball
   pygmi.seis.beachball.pol2cart
   pygmi.seis.beachball.auxplane
   pygmi.seis.beachball.strikedip
   pygmi.seis.beachball.mij2sdr
   pygmi.seis.beachball.TDL


Module Contents
---------------

.. py:class:: MyMplCanvas(parent=None)

   Bases: :py:obj:`matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg`


   Matplotlib canvas widget for the actual plot.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: init_graph()

      Initialize the graph.

      :rtype: None.



.. py:class:: BeachBall(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Create shapefiles with beachballs.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: data_init()

      Initialise Data - entry point into routine.

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: save_shp()

      Save Beachballs.

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: change_alg()

      Change algorithm.

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



.. py:function:: beachball(fm, centerx, centery, diam, isgeog, *, showlog=print)

   Beachball.

   Source code provided here are adopted from MatLab script
   `bb.m` written by Andy Michael and Oliver Boyd.

   function bb(fm, centerx, centery, diam, ta, color)
   Draws beachball diagram of earthquake double-couple focal mechanism(s).
   S1, D1, and R1, the strike, dip and rake of one of the focal planes, can
   be vectors of multiple focal mechanisms.

   :param fm: focal mechanism that is either number of mechanisms (NM) by 3
              (strike, dip, and rake) or NM x 6 (mxx, myy, mzz, mxy, mxz, myz -
              the six independent components of the moment tensor). The strike is
              of the first plane, clockwise relative to north. The dip is of the
              first plane, defined clockwise and perpendicular to strike, relative
              to horizontal such that 0 is horizontal and 90 is vertical. The rake is
              of the first focal plane solution. 90 moves the hanging wall up-dip
              (thrust), 0 moves it in the strike direction (left-lateral), -90 moves
              it down-dip (normal), and 180 moves it opposite to strike
              (right-lateral).
   :type fm: numpy array
   :param centerx: place beachball(s) at position centerx
   :type centerx: float
   :param centery: place beachball(s) at position centery
   :type centery: float
   :param diam: draw beachball with this diameter.
   :type diam: float
   :param isgeog: True if in geographic coordinates, False otherwise.
   :type isgeog: bool
   :param showlog: Routine to show text messages. The default is print.
   :type showlog: function, optional

   :returns: * **X** (*numpy array*) -- array of x coordinates for vertices
             * **Y** (*numpy array*) -- array of y coordinates for vertices
             * **xx** (*numpy array*) -- array of x coordinates for vertices
             * **yy** (*numpy array*) -- array of y coordinates for vertices


.. py:function:: pol2cart(phi, rho)

   Polar to cartesian coordinates.

   :param phi: Polar angles in radians.
   :type phi: numpy array
   :param rho: Polar r values.
   :type rho: numpy array

   :returns: * **xxx** (*numpy array*) -- X values.
             * **yyy** (*numpy array*) -- Y values.


.. py:function:: auxplane(s1, d1, r1)

   Get Strike and dip of second plane.

   Adapted from Andy Michael bothplanes.c

   :param s1: Strike 1.
   :type s1: numpy array
   :param d1: Dip 1.
   :type d1: numpy array
   :param r1: Rake 1.
   :type r1: numpy array

   :returns: * **strike** (*numpy array*) -- Strike of second plane.
             * **dip** (*numpy array*) -- Dip of second plane.
             * **rake** (*numpy array*) -- Rake of second plane.


.. py:function:: strikedip(n, e, u)

   Find strike and dip of plane given normal vector.

   Adapted from Andy Michaels stridip.c

   :param n: North coordinates for normal vector.
   :type n: numpy array
   :param e: East coordinate for normal vector.
   :type e: numpy array
   :param u: Up coordinate for normal vector.
   :type u: numpy array

   :returns: * **strike** (*numpy array*) -- Strike of plane.
             * **dip** (*numpy array*) -- Dip of plane.


.. py:function:: mij2sdr(mxx, myy, mzz, mxy, mxz, myz)

   Adapted from code, mij2d.f, created by Chen Ji.

   :param mxx - float: independent component of the moment tensor
   :param myy - float: independent component of the moment tensor
   :param mzz - float: independent component of the moment tensor
   :param mxy - float: independent component of the moment tensor
   :param mxz - float: independent component of the moment tensor
   :param myz - float: independent component of the moment tensor

   :returns: * **strike** (*float*) -- strike of first focal plane (degrees)
             * **dip** (*float*) -- dip of first focal plane (degrees)
             * **rake** (*float*) -- rake of first focal plane (degrees)


.. py:function:: TDL(AN, BN)

   TDL.

   :param AN: array comprising XN, YN, ZN
   :type AN: numpy array
   :param BN: array comprising XE, YE, ZE
   :type BN: numpy array

   :returns: * **FT** (*float*) -- relates to strike (360 - ft)
             * **FD** (*float*) -- dip
             * **FL** (*float*) -- relates to rake (180 - fl)


