Dataset Gridding
----------------
This module grids data using nearest neighbour, linear, cubic (all from SciPy) or minimum curvature (Briggs, 1974) algorithms. The input is a point or line or vector dataset, imported from the vector menu. Note that the x and y columns are defined when :doc:`importing<vector.dm.importxyzdata>` the line or point data.

The gridding options are:

1. **Gridding Method** - This can be **Nearest Neighbour**, **Linear**, **Cubic** or **Minimum Curvature**. The latter fits a polynomial surface through the points and is best suited to more continuous or smooth data such as gravity and magnetic data (potential field data sets). It should not be used for digital terrain models or radiometric data.
2. **Cell Size** - This represents the size of a square raster grid cell, in the units of the grid (normally metres).
3. **Column to Grid** - This allows you to select the Z column to grid.
4. **Null Value** – Data values that the user wants to mask.
5. **Blanking Distance** - This is a distance, in cells, from data points beyond which the grid will be masked. It is only applicable to minimum curvature gridding.

.. figure:: _images/vectorgrid.png

   Dataset Gridding options. The options on the left are available for Nearest neighbour, Linear and Cubic griding, while the options on the right apply to Minimum Curvature gridding.

References
^^^^^^^^^^
Briggs, I. C., 1974, Machine contouring using minimum curvature, Geophysics vol. 39, No. 1, pp. 39-48

