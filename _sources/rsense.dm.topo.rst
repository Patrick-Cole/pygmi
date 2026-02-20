Topographic Correction
----------------------
PyGMI uses the C correction (McDonald et al., 2000) to effect the topographic correction. The user must import a DEM and, in the case of Sentinel-2 data, the L2A Sentinel-2 dataset resulting from applying the :doc:`Sen2Cor: Sentinel-2 Atmospheric Correction<rsense.dm.sen2cor>` to L1C data and connect them to the **Topographic Correction** module.

.. figure:: _images/rsensetopo.png

   Setting up the Topographic Correction.

Double-clicking on the **Topographic Correction** module brings up a dialog box with the following options:

1. **Digital Elevation Model** - Ensure that the imported DEM file is selected here.
2. **Solar Azimuth and Solar Zenith** - If the L2A Sentinel-2 in its original format is imported using Import Satellite Data, these values will be extracted automatically from the metadata. However, if the Sentinel-2 data have been exported to a standard raster format (e.g. **GeoTIFF**) and was imported using :doc:`Import Raster Data<raster.dm.importrasterdata>`, the user will need to enter these values. It is important that the values are as accurate as possible to get the best results.

.. figure:: _images/rsensetopo2.png

   Topographic Correction interface.

The topographically corrected data can be exported to a standard raster format by right-clicking on the module and selecting :doc:`Export Raster Data<raster.cm.export>`.


References
^^^^^^^^^^
 McDonald, E.R., Wu, X., Caccetta, P.A., Campbell, N.A., 2000. Illumination correction of Landsat TM data in south east NSW. Paper Presented at: Proceedings of the Tenth Australasian Remote Sensing Conference.
