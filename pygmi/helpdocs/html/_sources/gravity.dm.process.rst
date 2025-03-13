Process Gravity Data
--------------------
This is a module used to process gravity data, assuming a single base station. It does so according to the North American gravity database standards, as described by Hinze et al (2005)

The input is a line dataset, imported via the **Import CG-5 Data** function. Coordinates must be in latitude and longitude, with elevation in meters.

The processing interface has the following parameters :

1. **Known Base Station Number** – This is the station number for a base station with a known absolute gravity value. 
2. **Known Base Station Absolute Gravity** – This is the known absolute gravity in mGal for the station above.
3. **Calculate local base value** – Use this option to tie in the local base station to a nearby base station with known absolute gravity. Note that the local station number must be different from the known station number. There must also be at least one local base station between successive known base station values.
4. **Background Density** – This is the background density in kg/m\ :sup:`3` .
5. **Base Station Absolute Gravity** – This is the absolute gravity value of the local base station used to correct for drift in the survey. It can be calculated by PyGMI or manually entered.
6. **Minimum Base Station Number** – All station numbers greater than this will be considered base stations.

.. figure:: _images/gravproc.png

   Gravity Processing interface.

Once the calculation has been completed a window showing the gravimeter drift will appear for quality control (QC) purposes. The **Process Log** window on the main PyGMI interface lists the calculated drift values.

.. figure:: _images/gravproc2.png

   Gravimeter drift graphs for QC.

.. figure:: _images/gravproc3.png

   Process Log window after gravity processing.

The processed data can be viewed and exported using the context menus. It consists of the following fields (all in mGal):

* *gobs_drift* – the observed drift,
* *gT* – theoretical gravity value,
* *gATM* – atmospheric effect,
* *gHC* – height correction,
* *gSB* – spherical Bouguer correction,
* *BOUGUER* – Bouguer gravity anomaly.

References
^^^^^^^^^^
 Hinze, W.J., Aiken, C., Brozena, J., Coakley, B., Dater, D., Flanagan, G., Forsberg, R., Hildenbrand, T., Kaller, G.R., Kellogg, J., Kucks, R., Li, X., Mainville, A., Moring, R., Pilkington, M., Plouff, D., Ravat, D., Roman, D., Urrutia-Fucugauchi, J., Veronneau, M., Webring, M., Winester, D. 2005. New standards for reducing gravity data: the North American gravity database. Geophysics, 70, J25-J32.
