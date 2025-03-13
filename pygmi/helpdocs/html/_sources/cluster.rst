Classification
==============

Description of Modules
----------------------
Cluster analysis and segmentation are useful machine learning or multivariate statistical analysis techniques which can perform an automatic interpretation of multi-band raster data.

A multiple band raster dataset, imported through the raster menu is input. If more than one dataset is imported, they must first be merged/stacked into a multiband dataset. It is also important to normalise the data prior to classification to ensure that the data values of the different datasets fall in the same range. If this is not done, the results will be biased towards datasets with large values, e.g. magnetic data. 

The Classification menu provides several classification/clustering tools.

.. toctree::
    :titlesonly:

    cluster.dm.clust
    cluster.dm.crisp
    cluster.dm.fuzzy
    cluster.dm.seg
    cluster.dm.super
    cluster.dm.scatter

Context Menu
------------
Context menus are available for classification modules after they have been executed. These are accessed by **right-clicking** on the green classification module. If the module has not been run it will be blue and the context menu will give you the option to select the input raster bands. **Cluster Analysis**, **Crisp Clustering** and **Supervised Classification** have the same context menu while **Fuzzy Clustering** has a slightly different context menu. The context menu for **Image Segmentation** is identical to the raster data context menu.

Classification context menus contain the raster data context menu that acts on the input raster datasets. Only the items applicable to the classification results will be discussed here.

.. figure:: _images/clustcontext.png

   Context menus for the Classification module.

.. toctree::
    :titlesonly:

    cluster.cm.stat
    cluster.cm.showclass
    cluster.cm.showgraphs
    cluster.cm.showmembership
    cluster.cm.exportclass
