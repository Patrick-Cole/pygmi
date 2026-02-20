Image Segmentation
------------------
This tool performs image segmentation, following the technique by Baatz and Schäpe (2000).

It has the following parameters:

1. **Compactness weight** - The weight assigned to compactness. Compactness indicates the degree of clustering in an area (Zhong et al., 2005). 
2. **Colour weight** - The weight assigned to colour heterogeneity. The colour heterogeneity includes all channels of a multiband image (Zhong et al., 2005).
3. **Maximum allowable cost function** - Only regions where the merge criterion is lower than this number will be combined into a segment. The merge criterion is given by (Zhong et al., 2005):

.. math::

 f = w_{col} h_{col}+(1-w_{col})[w_{com} h_{com}+(1-w_{com}) h_{sth}]

where: 

  * :math:`w_{col}` - weight assigned to colour heterogeneity.
  * :math:`h_{col}` - colour heterogeneity.
  * :math:`w_{com}` - weight assigned to circle-like compactness homogeneity.
  * :math:`h_{com}` - circle-like degree of homogeneity.
  * :math:`h_{sth}` - rectangle-like degree of homogeneity.

4. **Use K-Means to group segments** - Segments can be group by means of a K-means classifier.
5. **Number of clusters** - The number of clusters the K-means classifier must produce if that option was selected.

.. figure:: _images/clustseg.png

   Image Segmentation options.

References
^^^^^^^^^^
 Baatz, M, Schäpe, A., 2000. Multiresolution segmentation: An optimization approach for high quality multi-scale image segmentation. Proceedings of Angewandte Geographische Informationsverarbeitung XII, 12-23
 
 Zhong, C., Zhongmin, Z., Dongmei, Y. and Renxi, C. 2005. Multi-Scale Segmentation Of The High Resolution Remote Sensing Image. In: Proceedings. 2005 IEEE International Geoscience and Remote Sensing Symposium, 2005. Volume 5. IGARSS ’05., 3682.
