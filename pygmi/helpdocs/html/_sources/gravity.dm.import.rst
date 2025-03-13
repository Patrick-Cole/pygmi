Import CG-5 Data
----------------
This module imports CG-5 gravimeter data from either a **TXT** or a **XYZ** file (as exported by the gravimeter). The user is also expected to input a comma delimited Global Positioning System (GPS) file with Station, Latitude, Longitude and Elevation columns.

The options on the interface are :

1. **Load CG-5 File** – Select the file that contains the recorded gravity data.
2. **Load GPS file** – Select the GPS **CSV** file.
3. **Line** – Select the column in the GPS file that contains the line number.
4. **Station** – Select the column in the GPS file that contains the station number.
5. **Longitude** – Select the column in the GPS file that contains the longitude.
6. **Latitude** – Select the column in the GPS file that contains the latitude.
7. **Ellipsoid (GPS) Elevation** – Select the column in the GPS file that contains the elevation. Elevation must be in metre.
8. **Minimum Base Station Number** – All station numbers larger than this are base stations.

.. figure:: _images/gravimp.png

   Import CG-5 Data interface.

Once the data have been imported, the Process Log window of the main PygMI interface will warn the user of any duplicate data points in the gravity dataset.

.. figure:: _images/gravimp2.png

   Process Log window after gravity data have been imported.
