Imported XYZ point data display shows a single colour
-----------------------------------------------------

Problem
^^^^^^^
You have imported vector data using the :doc:`Import XYZ Data<vector.dm.importxyzdata>` module. When displaying the vector data and selecting the channel you want to view, the data points have mostly one colour (Figure 191).

.. figure:: _images/faqxyz1.png

   Imported XYZ data displays incorrectly.

Reason
^^^^^^
When importing the XYZ data, you did not specify the correct nodata value. In this example, the nodata value in the data file is -99999, but the user specified did not change the default value of 99999.
 

.. figure:: _images/faqxyz2.png

   Import XYZ Data interface with the incorrect Nodata Value specified.

Solution
^^^^^^^^
Check the nodata value in the dataset and re-import the data using this value. The data will now plot correctly.

.. figure:: _images/faqxyz3.png

   Imported XYZ data displays correctly after specifying the correct Nodata Value.
