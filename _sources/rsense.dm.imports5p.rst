Import Sentinel 5-P
-------------------
Sentinel-5P data is imported here, but unlike other imports, this converts data to a vector data format, from where it can be :doc:`exported to a shapefile<vector.cm.exportvector>` using the context menu. It accepts **NC** data. Sentinel-5P come in large swaths and therefore the import also allows the user to cut the data according to a bounding box that is defined by the user or imported in the form of a shapefile. 

The options are:

1. **Product** - Relevant product contained in the NC file.
2. When selecting **Clip using coordinates** the user can enter the following:

  * Minimum Longitude
  * Maximum Longitude
  * Minimum Latitude
  * Maximum Latitude

3. When selecting **Clip using shapefile** the user can upload a shapefile.
4. **QA Threshold (0-100)** - The user can select the threshold above which the data will be kept. The QA values are part of the **NC** file.


.. figure:: _images/rsenseimp5p.png

   Options for importing Sentinel-5P data.