Create Batch List
-----------------
This option allows the user to import multiple datasets from the same sensor for use in other remote sensing modules that support batch processing. All the datasets must be in one folder.   Note that it only imports metadata, and not band data. The data can be original satellite data files in their respective formats or datasets that have been exported to standard raster format, e.g. **GeoTIFF**.

The user must select a directory as input. The options are similar to the regular :doc:`Satellite Data Import<rsense.dm.importdata>` except that the user enters the folder (directory) where the data are stored instead of an individual file.

.. figure:: _images/rsensebatch1.png

   Import Batch List interface.

The raster file list can be seen in the **Dataset Information** window when clicking on the **Create Batch List** module.

.. figure:: _images/rsensebatch2.png

   List of datasets that will be used in the batch processing.
