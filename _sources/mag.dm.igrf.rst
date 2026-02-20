Calculate IGRF Corrected Data
-----------------------------
This function removes the International Geomagnetic Reference Field (IGRF) from the total magnetic intensity (TMI) data (Alken et al., 2021). It requires, as input, a magnetic dataset and a digital elevation model for an area. The two datasets must be in the same projection. The IGRF correction is a translated version of the correction supplied at: http://www.ngdc.noaa.gov/IAGA/vmod/igrf.html (IGRF Code, n.d.).

The two input datasets must both be connected to the **Remove IGRF** module. Alternatively, the data can be combined in a multiband raster before removing the IGRF.

.. figure:: _images/igrf.png

   Connecting the input raster datasets to the Remove IGRF module.

The IGRF correction interface requires the following inputs:

1. **Input Projection** - Since the IGRF correction relates to latitude and longitude, the projection of the input dataset is required. Under the first dropdown list the Datum can be selected, and the projection is selected under the second dropdown list. If the data sets have projections these will be imported automatically, but the user must confirm that it is correct.
2. Survey information:

  * **Sensor clearance above ground** - the height of the magnetic sensor above the ground in metres.
  * **Date** - The magnetic survey date. For large surveys that lasted for multiple or weeks, any date during the survey can be entered since the changes in the geomagnetic field will be negligible on this time scale.

3. Input data:

  * **Digital Elevation Model** - Raster grid of the terrain in height above sea level. 
  * **Magnetic Data** - Raster grid of the magnetic data to be corrected.

.. figure:: _images/igrf2.png

   IGRF correction interface.

The result of the calculation is a four-band raster dataset with the following bands:

* *IGRF* - The total intensity of the IGRF in nT.
* *Inclination* - The inclination of the IGRF.
* *Declination* - The declination of the IGRF
* *Magnetic data* - The IGRF-corrected magnetic data. The IGRF parameters are included in the band name.

The IGRF parameters are also shown in the **Process Log** window of the main PyGMI interface 

.. figure:: _images/igrf3.png

   IGRF parameters listed in the Process Log window.

The resulting dataset can be exported by right-clicking on the function and selecting :doc:`Export Raster<raster.cm.export>`.

References
^^^^^^^^^^
 Alken, P., Thébault, E., Beggan, C.D., Amit, H., Aubert, J., Baerenzung, J., Bondar, T.N., Brown, W.J., Califf, S., Chambodut, A., Chulliat, A., Cox, G.A., Finlay, C.C., Fournier, A., Gillet, N., Grayver, A., Hammer, M.D., Holschneider, M., Huder, L., Hulot, G., Jager, T., Kloss, C., Korte, M., Kuang, W., Kuvshinov, A., Langlais, B., Léger, J.-M., Lesur, V., Livermore, P.W., Lowes, F.J., Macmillan, S., Magnes, W., Mandea, M., Marsal, S., Matzka, J., Metman, M.C., Minami, T., Morschhauser, A., Mound, J.E., Nair, M., Nakano, S., Olsen, N., Pavón-Carrasco, F.J., Petrov, V.G., Ropp, G., Rother, M., Sabaka, T.J., Sanchez, S., Saturnino, D., Schnepf, N.R., Shen, X., Stolle, C., Tangborn, A., Tøffner-Clausen, L., Toh, H., Torta, J.M., Varner, J., Vervelidou, F., Vigneron, P., Wardinski, I., Wicht, J., Woods, A., Yang, Y., Zeren, Z. and Zhou, B. 2021. International Geomagnetic Reference Field: the thirteenth generation. Earth, Planets and Space, 73, 49, https://doi.org/10.1186/s40623-020-01288-x.
 
 IGRF Code (Accessed 28 March, 2014, http://www.ngdc.noaa.gov/IAGA/vmod/igrf.html), originally written in FORTRAN, was developed using subroutines written by A. Zunde (USGS), S.R.C. Malin & D.R. Barraclough (Institute of Geological Sciences, United Kingdom). Translated into C by Craig H. Shaffer. Rewritten by David Owens and maintained by: Stefan Maus (NOAA)