Importing ASTER data
--------------------

ASTER data are downloaded from https://earthdata.nasa.gov.

Step 1: Downloading the data
++++++++++++++++++++++++++++
For geological work the complete range of 14 bands are usually required. The recommended files to download are **AST_07XT** (crosstalk-corrected SWIR and VNIR) and **AST_05** (Surface Emissivity). Note that the SWIR sensor failed in April 2008 so only use data collected before this date.

When downloading the data be sure to select **AST_07XT** and **AST_05** scenes with exactly the same date and time of collection. PyGMI will automatically link the VNIR, SWIR and TIR datasets collected at the same time.

Data can be saved in two formats, namely **HDF** or a **ZIP** file containing the bands as **GeoTIFFs**. Both files will have an accompanying **MET** file. Both of the formats will be delivered as **ZIP** files, one for the VNIR and SWIR data, and one for the TIR data.

.. figure:: _images/tutimportaster1.png

   ASTER file formats.

Move the VNIR, SWIR and TIR files (files on the right-hand side of above figure) to a single folder (Figure 214).

.. figure:: _images/tutimportaster2.png

   VNIR, SWIR and TIR files in a single folder.

Go to :doc:`Import Satellite Data<rsense.dm.importdata>` on the **Remote Sensing menu** and select any of the files (VNIR,SWIR or TIR), it will automatically look for the other files. If you don’t see 14 bands it means one of the files was collected at a different date and time.

.. figure:: _images/tutimportaster3.png

   ASTER data import.

The data can be saved to a standard raster format by right-clicking on the **Import Satellite** module and selecting :doc:`Export Raster Data<raster.cm.export>`.

When :doc:`loading multiple scenes<rsense.dm.createbatchlist>`, all the VNIR, SWIR and TIR files must be copied into a single folder. PyGMI will link the correct files with each other based on the collection time and date.
