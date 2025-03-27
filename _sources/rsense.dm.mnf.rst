Minimum Noise Fraction
----------------------
This module performs Minimum Noise Fraction (MNF) on data, with the aim of performing noise filtering. Upon completion of the calculation, a graph of explained variance is displayed, which can be used to assist in defining the optimal number of components for filtering.

The following options are available on the interface:

1. **Forward Transform Only** – This option can be selected to do a forward transformation with the specified number of components. MNF components can be used in band combinations to enhance features in the data. If the aim is to filter noise out of the data this option must not be selected. However, the user is advised to run a forward transformation once to view the **Explained Variance** graph which can be used to select the number of components that must be used in the inverse transformation.
2. **Number of components** – Number of MNF components that must be kept.
3. Methods to estimate the noise in the data:

  * **Noise estimated by average horizontal and vertical shift** – This is the noise estimation as performed in ENVI.
  * **Noise estimated by diagonal shift**.
  * **Noise estimated by local quadratic surface** – Option recommended by Berman et al (2012), but slower.

If a single file was transformed, the user can export the result by right-clicking on the MNF Transform module and selecting :doc:`Export Raster Data<raster.cm.export>`. In the case of batch processing, the results will be stored automatically as **GeoTIFF** files in a new folder called “MNF” and _MNF will be added to the filenames.

.. figure:: _images/rsensemnf.png

   Minimum Noise Fraction interface.

.. figure:: _images/rsensemnf2.png

   Explained Variance graph that is shown after completion of an MNF calculation.

References
^^^^^^^^^^

 Berman, M., Phatak, A., & Traylen, A., 2012, Some invariance properties of the minimum noise fraction transform. Chemometrics and Intelligent Laboratory Systems, 117, 189-199.
 
 Ientilucci, E., 2003, Comparison and Usage of Principal Component Analysis (PCA) and Noise Adjusted Principal Component (NAPC) Analysis or Maximum Noise Fraction (MNF).
