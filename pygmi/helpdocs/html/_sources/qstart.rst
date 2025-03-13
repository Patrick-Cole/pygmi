Getting Started Guide: Example Of A Basic Workflow In PyGMI
===========================================================
This example shows how to import a raster dataset, view it, select a processing task to perform on the data, connect this module to the input data and export the result.

:doc:`Import Raster Data<raster.dm.importrasterdata>`
------------------

1. In the Raster menu click on :doc:`Import Raster Data<raster.dm.importrasterdata>`.
2. Select a dataset to import.
3. Once the dataset has been imported the module turns green.
 
.. figure:: _images/ex1.png

   Example of importing a raster dataset.

4. Right-click on the module which has appeared and select Show Raster Data (Simple). A window will pop up displaying the raster data which was imported. Close this window.

.. figure:: _images/ex2.png

   Example of the context menu that appears when right-clicking on a module.
 
.. figure:: _images/ex3.png

   Simple raster display.

Add a second module and connect it to the first module
------------------------------------------------------

5. In the Raster menu click on Smoothing. The modules can be moved apart in the interface by clicking on one and dragging it away. Click on the Line Pointer tool on the menu bar.
6. Draw a line between the :doc:`Import Raster Data<raster.dm.importrasterdata>` module and the Smoothing module.
7. Once you let go of the mouse button, a line with an arrowhead should appear.

.. figure:: _images/ex4.png

   Example of adding a new module and connecting it to another module.

8. Double-click on the Smoothing module. A dialog will pop up. You can leave the defaults for now and press OK.

The process log will now turn red indicating that an activity is happening. Once the processing is complete it will turn white again.
9. Right-click on the Smoothing module and select Show Raster Data (Simple) to see the smoothing results.
 
.. figure:: _images/ex5.png

   Example of the smoothing filter dialog box and the smoothed data.

Export the result
-----------------

10. Right-click on the Smoothing module and select :doc:`Export Raster Data<raster.cm.export>`.
11. In the dialog box that pops up, click on Output File and select the location and name of the output raster.

.. figure:: _images/ex6.png

   Example of exporting a raster dataset.
