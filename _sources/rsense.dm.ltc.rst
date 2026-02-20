Landsat Temporal Composite
--------------------------
This module creates a temporal composite from a list of Landsat scenes. The aim is to produce a cloud free scene by substituting data from multiple scenes over areas with clouds (White et al., 2014).

The options are:

1. **Batch Directory** - The folder where all the eligible scenes can be found.
2. **Target Day** -The optimal Julian day for the scenes. It can be the mean day, or the day of the best scene. Scenes closest to this day will be given preference.

.. figure:: _images/rsenseltc.png

   Landsat Temporal Composite interface.

References
^^^^^^^^^^
 White, J.C., Wulder, M.A., Hobart, G.W., Luther, J.E., Hermosilla, T., Griffiths, P., Coops, N.C., Hall, R.J., Hostert, P., Dyk, A. and Guindon, L. 2014. Pixel-based image compositing for large-area dense time series applications and science. Canadian Journal of Remote Sensing, 40, 192-212, https://doi.org/10.1080/07038992.2014.945827.