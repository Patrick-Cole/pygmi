Import Raster Data
------------------
This module imports raster data using the GDAL library, including **ENVI**, **ER Mapper**, **GeoTIFF**, **Geosoft GXF** and **Surfer**. It is possible to connect multiple imports to another module, thereby creating a 'layer stacked' dataset. Please note that satellite image imports take place in the **Remote Sensing menu**.

The full list of currently supported formats are:

* ArcGIS BIL (.bil)
* Arcinfo Binary Grid (hdr.adf)
* ASCII with .hdr header (.asc)
* ASCII XYZ (.xyz)
* ENVI (.hdr)
* ESRI ASCII (.asc)
* ERMapper (.ers)
* ERDAS Imagine (.img)
* GeoPak grid (.grd)
* Geosoft UNCOMPRESSED grid (.grd)
* Geosoft (.gxf)
* GeoTIFF (.tif .tiff)
* GMT netCDF grid (.grd)
* PCI Geomatics Database File (.pix)
* SAGA binary grid (.sdat)
* Surfer grid (.grd)