Principal Component Analysis
----------------------------
This module performs Principal Component Analysis (PCA) on data, with the aim of performing noise filtering. Upon completion of the calculation, a graph of explained variance is displayed, which can be used to assist in defining the optimal number of components for filtering.

The PCA module is also capable of accepting raster lists, so that batch PCA can be performed. In such a case, the user has the option to fit the PCA model to all files first, before translating the data to PCA space. This allows adjacent datasets to be mosaiced seamlessly, because the same model is applied to all input data.

The following options are available on the interface:

1. **Forward Transform Only** – This option can be selected to do a forward transformation with the specified number of components. Principal components can be used in band combinations to enhance features in the data. If the aim is to filter noise out of the data this option must not be selected. However, the user is advised to run a forward transformation once to view the **Explained Variance** graph which can be used to select the number of components that must be used in the inverse transformation.
2. **Number of components** – Number of principal components that must be kept.
3. **Fit PCA to all files** – If this is selected the PCA model will be fitted to all the files before transforming the data.

If a single file was transformed, the user can export the result by right-clicking on the PCA Transform module and selecting :doc:`Export Raster Data<raster.cm.export>`. In the case of batch processing, the results will be stored automatically as **GeoTIFF** files in a new folder called “PCA” and _PCA will be added to the filenames.


.. figure:: _images/rsensepca1.png

   Principal Component Analysis interface.

.. figure:: _images/rsensepca2.png

   Explained Variance graph that is shown after completion of an PCA calculation.