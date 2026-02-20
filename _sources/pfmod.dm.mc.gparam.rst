Geophysical Parameters
^^^^^^^^^^^^^^^^^^^^^^

**Geophysical Parameters** are parameters for gravity and magnetic modelling are set here:

.. figure:: _images/pfmodgparam1.png

   Geophysical Parameters dialog box.

1. **General Properties** - Here you can set parameters relating to the data:

  * **Gravity Regional (mGal)** - A regional value can be subtracted from the Bouguer anomaly data. This simply moves the calculated gravity response up or down.
  * **Height of observation - Gravity** - The height of the gravity instrument above the ground (in metres).
  * **Height of observation - Magnetic** - The height of the magnetic sensor above the ground (in metres).
  * **Magnetic Field Intensity (nT)** - The intensity of the prevailing geomagnetic field at the data locality.
  * **Magnetic Inclination** - Inclination (in degrees) of the prevailing geomagnetic field at the data locality.
  * **Magnetic Declination** - Declination (in degrees) of the prevailing geomagnetic field at the data locality.

2. **Lithological Properties** - Lithologies to be used in the model can be added or removed here:

  * **Add New Lithological Definition** - Select this to add a new lithology. The colour picker dialog will appear and the user can select the desired colour. It will be given a generic name, e.g. “Generic 2”.
  * List of lithologies - The current list of lithologies defined in the model. The lithology colour can be changed by double-clicking on a lithology. The petrophysical properties of the selected lithology are displayed on the right-hand side.
  * **Rename Current Definition** - To change the name of a lithology, select the lithology by clicking on it and then click on **Rename Current Definition**.
  * **Remove Current Definition** - Click on this to remove the currently selected lithology.
  * **Merge Definitions** - Lithologies can be merged. A dialog box with two windows appears. In the top window the **Master Lithology** is selected. All the properties of this lithology will be retained. In the bottom window the user can select all the lithologies to merge with the **Master Lithology**. The selected lithologies will disappear from the list of lithologies.

.. figure:: _images/pfmodgparam2.png

   Dialog box for merging lithologies.

3. **Lithological Properties** - Petrophysical parameters - The petrophysical properties of the lithologies are defined here. Click on a lithology on the left to show its properties on the right. These can be changed as desired. Note that for any changes to take effect, they must be applied before clicking on a different lithology by clicking on **Apply Changes**.

The **Remanent Magnetization Intensity (nT)**, **Remanent Magnetization (A/m)** and **Q ratio** are linked. A change to one of these will result in changes to the other two. This enables the user to input the strength of remanent magnetization using the units in which their data is available.

