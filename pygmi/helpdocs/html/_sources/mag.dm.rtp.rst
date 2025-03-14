Reduction to the Pole
---------------------
This function reduces a magnetic dataset to the pole. This shifts the magnetic anomaly directly over the source resulting in a symmetric anomaly shape that is simple to interpret. However, it is only effective if the source body is not remanently magnetised.

The parameters on the interfaced are:

1. **Band to Reduce to the Pole** – The band name is directly extracted from the raster file. In the case of a multiband raster dataset the user can specify which band contains the magnetic data that must be processed.
2. **Inclination of the Magnetic Field** – Magnetic inclination at the date of the survey.
3. **Declination of the Magnetic Field** – Magnetic declination at the date of the survey.


.. figure:: _images/rtp.png

   Reduction to the Pole interface.

The resulting dataset can be exported by right-clicking on the function and selecting :doc:`Export Raster<raster.cm.export>`.