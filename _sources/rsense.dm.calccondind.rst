Calculate Condition Indices
---------------------------
This module allows the user to calculate condition indices such as temperature condition index (TCI), vegetation condition index (VCI) and vegetation health index (VHI) using indexes such as the enhanced vegetation index (EVI), normalised difference vegetation index (NDVI) and modified soil-adjusted vegetation index (MSAVI2) (Bento et al., 2018). It requires as input a time series of at least two but preferably more datasets over the same area, defined as a batch file list. A condition index compares the index value at a certain time with the minimum and maximum values over the whole time period.

The following options are available on the interface:

1. **Sensor** - This is the satellite sensor being used. Depending on the sensor, different indices will result.
2. **Index** - The vegetation index that will be used in the calculations of the VCI and can be EVI, NDVI or MSAVI2.
3. **Condition Indices** - A list of condition indices to calculate. VCI uses the specified vegetation index, TCI uses land surface temperature and VHI uses VCI and TCI.
4. **Invert selection** - Convenience button to invert the condition indices selection.

.. figure:: _images/rsensecind.png

   Calculate Condition Indices interface.

References
^^^^^^^^^^
 Bento, V.A., Trigo, I.F., Gouveia, C.M. and DaCamara, C.C. 2018. Contribution of Land Surface Temperature (TCI) to Vegetation Health Index: A comparative study using clear sky and all-weather climate data records. Remote Sensing, 10, https://doi.org/10.3390/rs10091324.