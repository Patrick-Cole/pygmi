Profile viewing options
^^^^^^^^^^^^^^^^^^^^^^^

A toolbar above the **Layer View** provides options for the profile display, and the ability to import borehole logs.

.. figure:: _images/pfmodprofview1.png

   Options to control the profile display.

The options are:

* **Field Display Limits** - This changes the scale of the Y-Axis. This can be important after performing a calculation, to see the full effect of the calculation.

  * **Scale to all maximum** – The minimum and maximum extents of the data profile are set to the minimum and maximum of the whole observed and calculated datasets.
  * **Scale to dataset maximum** – The minimum and maximum extents of the data profile are set to the minimum and maximum of the whole observed data.
  * **Scale to profile maximum** – The minimum and maximum extents of the data profile are set to the minimum and maximum of the observed data along this profile only.
  * **Scale to calculated maximum** - The minimum and maximum extents of the data profile are set to the minimum and maximum of the calculated data along this profile only.
  * **Scale to custom maximum** – The user specifies the minimum (first value) and maximum (second value) extents for the data profile.

.. figure:: _images/pfmodprofview1.png

   Profile Field Display Limits options.

* **View Magnetic Profile** – Select the show the magnetic data for the profile.
* **View Gravity Profile** – Select the show the gravity data for the profile.
* **Import Borehole Logs** - Display borehole logs imported using :doc:`Import Borehole Data<bholes.dm.import>`. The imported borehole data module must be connected to the modelling module. When **Import Borehole Logs** is selected in the **Profile viewing options** the localities of the boreholes will be shown in the :doc:`Layer View<pfmod.dm.mc.lpviews>` and the borehole lithologies are added to the relevant profiles. The lithologies in the borehole will be added to the lithologies of the model defined in :doc:`Geophysical Parameters<pfmod.dm.mc.gparam>`. The user needs to enter the petrophysical parameters for these lithologies.