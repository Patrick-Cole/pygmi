Modelling tools
^^^^^^^^^^^^^^^

This section contains the tools necessary to create the actual model. Modelling in PyGMI works like a paint program, with the user selected a specific lithology and drawing on either the profile or layer view. 

.. figure:: _images/pfmodmodeltools1.png

   Modelling tools.
   
The tools are:

1. Lithology list - Use this to select the lithology you want to draw in. To erase, simply select the Background lithology and draw with that.
2. **Line Thickness** - This allows you to change the pen thickness which you are drawing with. It is in terms of voxel/cell size.
3. **Ranged Copy** - The ranged copy function allows for the copying of a lithology on the current profile or layer over a range of profiles or layers. To copy profiles, select **Side View** and **Top View** lets you copy layers.

  * **Master Profile** - The profile/layer with the lithology/lithologies that you want to copy.
  * **Range Start** - The first profile/layer of the range of profiles/layers to which the lithology/lithologies must be copied.
  * **Range End** - The last profile/layer of the range of profiles/layers to which the lithology/lithologies must be copied.
  * **Lithologies To Copy** - Select the lithologies to copy.
  * **Lithologies to Overwrite** - Select the lithologies that can be overwritten.

.. figure:: _images/pfmodmodeltools2.png

   Ranged Copy modelling tool.

4. **Add Lithological Boundary** - Adds a raster to be used as a lithological boundary. It can be used if you have, for example a raster of the surface of the bedrock to add to your model. You will be prompted to select a raster layer and then select how it will be used:

  * **Lithologies Above Layer** - You can choose not to change the current model above the boundary layer or select a lithology that must constitute the entire model above the boundary layer. 
  * **Lithologies below Layer** - Select whether to keep the model unchanged below the boundary layer, or the lithology that must be used for the entire model below the boundary layer.
  * There are two options for the units of the Z coordinates of the raster layer that will form the lithological boundary. It can either be in depth below the surface or height above sea level.

.. figure:: _images/pfmodmodeltools3.png

   Tool to add a lithological boundary to a model.

.. tip::
   
   The ranged copy function for the **Layer View** can be very useful in getting a starting model going. Draw in your basic lithology boundary based on a geological map or the geophysical data (displayed behind the model) and copy it over a range of applicable depths.