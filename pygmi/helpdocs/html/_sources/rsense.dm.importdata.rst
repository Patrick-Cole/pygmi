Import Satellite Data
---------------------
PyGMI can import data from the following remote sensing sensors:

* ASTER data – Note that this is the data as obtained from https://earthdata.nasa.gov. It can either be in **HDF** format (there will be an associated **MET** file) or imported as a **ZIP** file containing **GeoTIFF** images. ASTER visible and near infrared (VNIR), shortwave infrared (SWIR) and thermal infrared (TIR) data are provided separately and importing this dataset needs a few extra steps.
* Landsat data – Note that this is the data as obtained from https://earthexplorer.usgs.gov. It can either be in **L*.TAR.GZ** format or extracted, with a **_MTL.TXT** and **GeoTIFF** files.
* Sentinel 2 data – Note that this is the data as obtained from https://dataspace.copernicus.eu/. It can import the **ZIP** file, or if extracted the appropriate **XML** file (e.g. MTD_MSIL2A.xml).
* Hyperion L1T data – Data downloaded from https://earthexplorer.usgs.gov. It is imported as a **ZIP** file containing **GeoTIFF** images.
* WorldView Tile data (**TIL**) – The tiles will be merged into a single dataset.
* ASTER Global Emissivity data – Data downloaded from https://earthdata.nasa.gov in **H5** format.
* EMIT data – Data downloaded from https://earthdata.nasa.gov. The data are in **NC** format.

When selecting this function the user has the following options:

1. **Filename** –Select the file to be imported. 
2. **File Type** – The import function automatically detects the type of data and displays it.
3. All the bands available in the file are listed. The user can select which band/bands to import. By default the bands deemed to contain the data to be used for further processing (e.g. reflectance data) are select. To select/deselect bands simply click on a band name in the list.
4. Satellite remote sensing data are often provided in UTM coordinates for the northern hemisphere. The user can choose to convert the data from UTM to SUTM.

After the data have been imported, it can be exported to a common raster format by right-clicking on the module and selecting :doc:`Export Raster Data<raster.cm.export>`.

.. figure:: _images/rsenseimport.png

   Import Satellite Data interface.
