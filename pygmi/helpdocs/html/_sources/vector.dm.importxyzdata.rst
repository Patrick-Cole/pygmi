Import XYZ Data
-------------------------
This module imports XYZ/point data from a file, for example a **CSV** or Excel file. The user must choose the x and y columns.

List of formats:

* Excel (.XLSX)
* Comma Delimited (.CSV)
* Geosoft XYZ (.XYZ)
* ASCII XYZ (.XYZ)
* Space Delimited (.TXT)
* Tab Delimited (.TXT)
* Intrepid Database (..DIR)

Once the file has been selected the user must specify the following parameters:

1. Select the X and Y columns. These columns must be in the coordinates specified in the **Input Projection** section.

  * **X Channel** - This is the column which contains the x-coordinates.
  * **Y Channel** - This is the column which contains the y-coordinates.

2. **Nodata Value** - This is the null or nodata value in the data file. If the nodata value in the data file is different from the value specified here, nodata values will be considered real data.
3. **Input Projection** - The datum and projection are specified here. This is especially important if the user intends to export the data to a vector (GIS) format.

.. figure:: _images/vectorimport.png

   Import XYZ Data options.
