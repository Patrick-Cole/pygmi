Display/Edit Metadata
---------------------
This module allows for the display and custom editing of metadata associated with raster images. It is especially useful for defining nodata values or band names.

1. **Band Name** - Shows the current name of a raster band within the dataset. Change to a different band by using the drop-down menu.
2. **Rename Band Name** allows the user to change the name of the currently selected band.
3. The **Dataset** section shows physical information about the dataset.

  * **Top Left X-Coordinate**: Raster X coordinate in the NW corner of the dataset.
  * **Top Left Y-Coordinate**: Raster Y coordinate in the NW corner of the dataset.
  * **X-Dimension**: width of a grid cell (in the x-direction).
  * **Y-Dimension**: height of a grid cell (in the y-direction).
  * **Null/Nodata Value**: A unique value assigned to areas of the dataset with no data.
  * **Rows**: Number of rows in the dataset.
  * **Columns**: Number of columns in the dataset.
  * **Dataset minimum**: Minimum Z value in the dataset.
  * **Dataset maximum**: Maximum Z value in the dataset.
  * **Dataset mean**: Mean Z value in the dataset.
  * **Dataset units**: A label used in graphs to denote the units of the data in the currently selected band.
  * **Data type**: The data type such as float32, float64, integer etc.
  * **Acquisition Date**: Data collection date of the currently selected band 

4. **Input Projection**: Current projection assigned to the raster band. It can be reassigned, but note that this does not reproject the data.

.. figure:: _images/rastermetadata.png

   Display/Edit Metadata interface.