Calculate Band Ratios
---------------------
This module allows the user to calculate ratios from a list of standard ratios found in the literature (Kalinowski and Oliver, 2004; Van der Meer et al., 2014). It takes as input either an imported remote sensing dataset, or input from a batch list. In both cases the data can either be original satellite data files in their respective formats or data that have been exported to standard raster formats. 

When clicking on the **Calculate Band Ratios** module, as dialog box appears that shows the ratios applicable to the data sensor type. If a standard raster was imported, the input sensor cannot necessarily be automatically detected and the user has to specify it using a dropdown list. The available band ratios are completely sensor dependent.

.. figure:: _images/rsenseratio.png

   Selecting the correct sensor to use for band ratio calculations when importing satellite data in standard raster format.

The **Calculate Band Ratios** interface has the following options:

1. **Sensor** - The satellite sensor type of the input dataset.
2. **Ratios** - Ratios applicable to the data sensor type. Ratios with a tick mark next to them will be calculated. To deselect a ratio click on it and the tick mark will disappear.
3. **Invert Selection** - If only a few ratios must be calculated, it may be easier to deselect them and click on Invert Selection.
 
.. figure:: _images/rsenseratio2.png

   The Band Ratios Calculation interface.

Ratios are automatically exported in **GeoTIFF** format to new subfolder named “ratio” and _ratio is added to the filename.

References
^^^^^^^^^^
 Kalinowski, A., Oliver, S., 2004, ASTER Mineral Index Processing Manual. Remote Sensing Applications, Geoscience Australia, 36 pp.

 Van der Meer, F.D., van der Werff, H.M.A., van Ruitenbeek, F.J.A., 2014. Potential of ESA’s Sentinel-2 for geological applications. Remote Sensing of Environment 148, 124-133. https://doi.org/10.1016/j.rse.2014.03.022
