Export Raster File List
-----------------------
Data imported using :doc:`Create Batch List<rsense.dm.createbatchlist>` can be exported to other raster formats, and to a ternary image if desired.

The options are:

1. **Output Directory** - The folder where the output files will be stored.
2. **Output Format** - File format to export data to. The options are **GeoTIFF**, **GeoTIFF compressed using DEFLATE**, **GeoTIFF compressed using ZSTD**, **ENVI**, **ERMapper**, **ERDAS Imagine**.
3. **Ternary Export** - Checkbox to select ternary images to be exported. If this is not selected, the data will be exported to the specified folder with the word _stack appended to the filename. The filename depends on the sensor type and the conventions are as follows:

  * ASTER - <AST>_<Acquisition date (yyyymmdd)>_<Acquisition time (hhmmss)>
  * Sentinel-2 - <Sentinel mission (S2A/S2B)>_<Tile name>_<Orbit number>_ <Acquisition date (yyyymmdd)>
  * Landsat - <Landsat sensors (LT04/LT05/LE07/L08/L09)>_<Processing level>_<WRS path and row>_<Acquisition date (yyyymmdd)>
  * EMIT - None

If **Ternary Export** is selected, the exported image files will be exported to the specified folder with the band combination appended to the filename.

4. Band selection for the ternary image.
5. **Sunshade Band** - The band used for the sun shading layer. This can be an external dataset.
6. **Sunshade Level** - The severity of sun shading. It can be **Standard** or **Heavy**.

.. figure:: _images/rsenseexport.png

   Export Raster File List interface.

