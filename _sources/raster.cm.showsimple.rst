Show Raster Data (Simple)
-------------------------
This is convenient image display for raster data. Only one band can be viewed at a time. The band can be selected for multiband data. For more display options, use the Show Raster Data (Advanced) module.
The options on this interface are:

1. **Bands** - Select the band to be displayed.
2. **Colormap** - Select a colormap for the data display. The user can choose between **Viridis**, **Jet**, **Gray** and **Terrain**. 
3. Standard image display setting that allows the user to zoom into specific areas of the image, move the zoomed in area around, return to the full image, save the image with the colour bar, etc.
4. **Log Colour Scale** - use a logarithmic colour scale.
5.	**Coordinate Display** - show the coordinate of the image at the mouse pointer.

.. figure:: _images/rasterdisplay.png

   Show Raster (Simple) interface.

In the specialised case where a section was gridded up using the :doc:`Dataset Gridding<vector.dm.gridding>` module, the image display will be in units of distance down the section on the x-axis. Hovering over the image will give actual coordinates (as opposed to distance down the section) in the bottom right of the image.

.. figure:: _images/rasterdisplay2.png

   Section displayed on the interface.
