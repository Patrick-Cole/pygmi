Top toolbar
^^^^^^^^^^^

The toolbar gives options to modify the model parameters and perform calculations (Figure 125).

.. figure:: _images/pfmodtoolbar1.png
 
   Top toolbar on the Potential Field Modelling interface.

It has the following menus:

* :doc:`Model Extent Parameters<pfmod.dm.mc.mext>` – This brings up the :doc:`Model Extent Parameters<pfmod.dm.mc.mext>` dialog box where the user can adjust the dimensions and location of the model, as well as the datasets.
* :doc:`Geophysical Parameters<pfmod.dm.mc.gparam>` – Parameters for gravity and magnetic modelling are set here.
* **Lithology Notes** – A simple interface to store lithology notes. Each lithology can have a lithology code and lithology notes assigned to it.

.. figure:: _images/pfmodtoolbar2.png
 
   Lithology Notes interface.

* **Editor** – This is an option to minimise or maximise the main editor interface.
* **Regional Test** – A test of field decay to see if the dimensions of the model are large enough. In magnetic and gravity modelling, the calculated field will decay past the edges of the model. 2.5D modelling deals with this problem by adding a strike parameter where the body is automatically extended perpendicular to the profile by a constant amount. This is not the case in 3D modelling. For this reason, it is important to pad the edge of the model to ensure that the calculated field remains accurate. The regional test shows the rate of decay using one of the lithologies (selected by the user), to give an indication of whether your model is affected by this decay.

.. figure:: _images/pfmodtoolbar3.png

   Results of the regional calculation using one of the defined lithologies. The petrophysical properties of the lithology is shown.
   
* **Calculate Gravity (All)** – The gravity response of all lithologies and voxels are recalculated.
* **Calculate Magnetics (All)** – The magnetic response of all lithologies and voxels are recalculated.
* **Calculate Gravity (Changes Only)** – Only changed lithologies and voxels are used for recalculations.
* **Calculate Magnetics (Changes Only)** – Only changed lithologies and voxels are used for recalculations.
* **Apply Demagnetisation Correction** – A check box that, when selected, takes into account the demagnetisation effect of lithologies with very high magnetic susceptibilities.
* **Save Model** – Convenient option to save the model from within the interface. The model can be saved in the following formats:

  * **NPZ** – This is the format used by PyGMI, so the model must be saved in this format if you want to use it in PyGMI.
  * **SHP** – This exports each lithology to a separate shapefile.
  * **KMZ** – For use in Google Earth
  * **CSV** – Exports the model to a **CSV** file.
 