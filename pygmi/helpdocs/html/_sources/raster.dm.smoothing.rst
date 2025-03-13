Smoothing
---------
The **Smoothing** filters work by passing a matrix which contains the filter over your raster dataset. The filter type, filter shape and the filter size can all be specified.

The **Smoothing Filter** interface has the following components:

1. A visual representation of the smoothing filter matrix.
2. **Filter Type**: This can be either **2D median** or **2D mean**. Note that your choice of filter may affect the number of filter shape options you have.
3. **Filter shape**:

  * **Box Window** - this is a regular square matrix with constant values applied to the dataset.
  * **Disk Window** - this applies a disk of constant values to the dataset
  * **Gaussian Window** - this applies a Gaussian-shaped filter to the dataset. This option is only present for the **2D mean** filter type.
  
4. **Filter size**: The options here are dependent on the option chosen in filter shape. They are as follows:

  * **X** and **Y** for the **Box Window** - this is the number of rows and columns which will make up the box window
  * **Radius** in samples for the **Disk Window** - this creates a disk of the specified radius. The disk will have 2*radius rows and columns
  * **Standard Deviation** for the **Gaussian Window** - this creates a Gaussian box window where the values have the specified standard deviation.

.. figure:: _images/rastersmooth.png

   Smoothing filters interface.