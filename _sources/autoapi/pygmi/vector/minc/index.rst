pygmi.vector.minc
=================

.. py:module:: pygmi.vector.minc

.. autoapi-nested-parse::

   Minimum Curvature Gridding Routine.

   Based on the work by:

   Briggs, I. C., 1974, Machine contouring using minimum curvature, Geophysics
   vol. 39, No. 1, pp. 39-48



Functions
---------

.. autoapisummary::

   pygmi.vector.minc.minc
   pygmi.vector.minc.u_normal
   pygmi.vector.minc.u_edge
   pygmi.vector.minc.u_one_row_from_edge
   pygmi.vector.minc.u_corner
   pygmi.vector.minc.u_next_to_corner
   pygmi.vector.minc.u_edge_next_to_corner
   pygmi.vector.minc.off_grid
   pygmi.vector.minc.get_b
   pygmi.vector.minc.mcurv
   pygmi.vector.minc.morg


Module Contents
---------------

.. py:function:: minc(x, y, z, dxy, *, showlog=print, extent=None, bdist=None, maxiters=100)

   Minimum Curvature Gridding.

   :param x: 1D array with x coordinates.
   :type x: numpy array
   :param y: 1D array with y coordinates.
   :type y: numpy array
   :param z: 1D array with z coordinates.
   :type z: numpy array
   :param dxy: Cell x and y dimension.
   :type dxy: float
   :param showlog: Routine to show text messages. The default is print.
   :type showlog: function, optional
   :param extent: Extent defined as (left, right, bottom, top). The default is None.
   :type extent: list, optional
   :param bdist: Blanking distance in units of cell. The default is None.
   :type bdist: float, optional
   :param maxiters: Maximum number of iterations. The default is 100.
   :type maxiters: int, optional

   :returns: **u** -- 2D numpy array with gridding z values.
   :rtype: numpy array


.. py:function:: u_normal(u, i, j)

   Minimum curvature smoothing for normal cases.

   It is defined as:

   u[i+2, j] + u[i, j+2] + u[i-2, j] + u[i, j-2] +
   2*(u[i+1, j+1] + u[i-1, j+1] + u[i+1, j-1] + u[i-1, j-1]) -
   8*(u[i+1, j]+u[i-1, j]+u[i, j+1]+u[i, j-1]) + 20*u[i, j] = 0

   :param u: 2D grid of z values.
   :type u: numpy array
   :param i: Current row.
   :type i: int
   :param j: Current Column.
   :type j: int

   :returns: **uij** -- Smoothed value to replace in master grid.
   :rtype: float


.. py:function:: u_edge(u, i)

   Minimum curvature smoothing for edges.

   It is defined as:

   u[i-2, j] + u[i+2, j] + u[i, j+2] + u[i-1, j+1] + u[i+1, j+1] -
   4*(u[i-1, j] + u[i, j+1] + u[i+1, j]) + 7*u[i, j] = 0

   :param u: 2D grid of z values.
   :type u: numpy array
   :param i: Current row.
   :type i: int

   :returns: **uij** -- Smoothed value to replace in master grid.
   :rtype: float


.. py:function:: u_one_row_from_edge(u, i)

   Minimum curvature smoothing for one row from edge.

   It is defined as:

   u[i-2, j] + u[i+2, j] + u[i, j+2] +
   2*(u[i-1, j+1] + u[i+1, j+1]) + u[i-1, j-1]+u[i+1, j-1] -
   8*([i-1, j]+u[i, j+1]+u[i+1, j]) - 4*u[i, j-1] + 19*u[i, j] = 0

   :param u: 2D grid of z values.
   :type u: numpy array
   :param i: Current row.
   :type i: int

   :returns: **uij** -- Smoothed value to replace in master grid.
   :rtype: float


.. py:function:: u_corner(u)

   Minimum curvature smoothing for corner point.

   It is defined as:

   2*u[i, j]+u[i, j+2] + u[i+2, j] - 2*(u[i, j+1] + u[i+1, j] = 0

   :param u: 2D grid of z values.
   :type u: numpy array

   :returns: **uij** -- Smoothed value to replace in master grid.
   :rtype: float


.. py:function:: u_next_to_corner(u)

   Minimum curvature smoothing for next to corner.

   It is defined as:

   u[i, j+2] + u[i+2, j] + u[i-1, j+1] + u[i+1, j-1] + 2*u[i+1, j+1] -
   8*(u[i, j+1] + u[i+1, j]) - 4*([i, j-1]+u[i-1, j]) + 18*u[i, j] = 0

   :param u: 2D grid of z values.
   :type u: numpy array

   :returns: **uij** -- Smoothed value to replace in master grid.
   :rtype: float


.. py:function:: u_edge_next_to_corner(u)

   Minimum curvature smoothing for edge next to corner.

   It is defined as:

   u[i, j+2] + u[i+1, j+1] + u[i-1, j+1] + u[i+2, j] - 2*u[i-1, j] -
   4*(u[i+1, j] + u[i, j+1]) + 6*u[i, j] = 0

   :param u: 2D grid of z values.
   :type u: numpy array

   :returns: **uij** -- Smoothed value to replace in master grid.
   :rtype: float


.. py:function:: off_grid(u, i, j, wn, b)

   Node value calculation when data value is too far from node.

   :param u: 2D grid of z values.
   :type u: numpy array
   :param i: Current row.
   :type i: int
   :param j: Current Column.
   :type j: int
   :param wn: Data value.
   :type wn: float
   :param b: List of b values for calculation.
   :type b: list

   :returns: **uij** -- Output value.
   :rtype: float


.. py:function:: get_b(e5, n5)

   Get b values for input data.

   Calculates the b values based on the distance between the data point and
   the nearest node. Distances are expressed in units of cell.

   :param e5: x distance error.
   :type e5: float
   :param n5: y distance error.
   :type n5: float

   :returns: * **b1** (*float*) -- b1 value.
             * **b2** (*float*) -- b2 value.
             * **b3** (*float*) -- b3 value.
             * **b4** (*float*) -- b4 value.
             * **b5** (*float*) -- b5 value.


.. py:function:: mcurv(u, ufixed)

   Minimum curvature smoothing.

   This routine smooths the data between fixed data nodes.

   :param u: 2D grid of z values.
   :type u: numpy array
   :param ufixed: 2D grid of fixed node values.
   :type ufixed: numpy array

   :returns: **u** -- 2D grid of z values.
   :rtype: numpy array


.. py:function:: morg(x2, y2, z2, extent, dxy, rows, cols)

   Organise coordinates and calculate b values.

   :param x2: 1D array with x coordinates.
   :type x2: numpy array
   :param y2: 1D array with y coordinates.
   :type y2: numpy array
   :param z2: 1D array with z coordinates.
   :type z2: numpy array
   :param extent: Extent defined as (left, right, bottom, top).
   :type extent: list
   :param dxy: Cell x and y dimension.
   :type dxy: float
   :param rows: Number of rows.
   :type rows: int
   :param cols: Number of columns.
   :type cols: int

   :returns: * **coords** (*list*) -- List containing iint, jint, r and zval.
             * **b** (*list*) -- List of b values.


