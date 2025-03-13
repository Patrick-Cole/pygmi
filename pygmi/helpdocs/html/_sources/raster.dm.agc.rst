Automatic Gain Control
----------------------
This enhances the low amplitude anomalies in data using **Automatic Gain Control** (AGC).

The options are:

* **Mean** - Calculates and AGC using mean values.
* **Median** - Calculates and AGC using median values.
* **RMS** - Calculates and AGC using root mean square (rms) values.
* **Window Size** - This is an odd number specifying the size of window (in units of pixels in both x and y directions) to pass over your data.

.. figure:: _images/rasteragc.png

   Automatic Gain Control options.