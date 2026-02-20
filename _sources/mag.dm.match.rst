Matched Filtering
-----------------
Matched filtering is a technique which seeks to separate out magnetic ensembles based on features present in the frequency domain (Spector and Grant, 1970). The filter fits linear sections to the log of the magnetic power spectrum, and calculates the depth of that section based on the fit. An FFT filter is also calculated for that linear section. The output is a series of magnetic images, each representing a depth slice.

The interface has the following options:

1.	**Band to perform filtering** - This is the raster magnetic band.
2.	**Number of depth slices** - this is the number of depth slices to split the data into.
3.	**Recalculate** - option to recalculate.

The interface displays the following:

4.	**Piecewise Linear Fit** - linear fit to sections of the FFT.
5.	**FFT Filters** - the actual FFT filters used are displayed here.

.. figure:: _images/magmatch.png

   Matched Filtering interface.

References
^^^^^^^^^^
 Spector, A., & Grant, F. S. (1970). Statistical models for interpreting aeromagnetic data. Geophysics, 35, 293-3020.
 
