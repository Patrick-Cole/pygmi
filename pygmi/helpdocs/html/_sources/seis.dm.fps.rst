Fault Plane Solutions
---------------------
This module allows for the display of fault plane solutions as beachball plots. The beachball plots can be exported to shapefiles. Note that two shapefiles are exported - one for the actual beachballs and one for a boundary of the beachballs.

The core code is translated from bb.m written by Andy Michael and Oliver Boyd at http://www.ceri.memphis.edu/people/olboyd/Software/Software.html

Options are:

1. **FPS Algorithm** – Can be FOCMEC (focal mechanism) or FPFIT (double-couple FPS, Reasenberg and Oppenheimer, 1985).
2. **Width scale factor** – Option to scale the “beachballs”
3. **Geographic units** or **Projected units**. – Select based on input data.
4. **Save Shapefile** – Export the beachballs to shapefiles.

.. figure:: _images/seisfps.png

   Fault Plane Solution interface.

