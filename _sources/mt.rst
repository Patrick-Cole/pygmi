Magnetotellurics
================

Description of Modules
----------------------
The magnetotellurics section adds some processing capability, based on the MTpy library (https://github.com/MTgeophysics/mtpy) (Krieger and Peacock, 2014; Kirkby et al., 2019). It allows the user to import MT data and perform simple functions on the data, including 1D inversion.

.. figure:: _images/mtmenu.png

   MT menu.

.. toctree::
    :titlesonly:

    mt.dm.importedi
    mt.dm.rotateedi
    mt.dm.removesshift
    mt.dm.mask
    mt.dm.occam

Context Menu
------------
These context menus are available for modules which have MT data. Output data is only available on green modules. To access the context menu, simply right-click a green module.

.. figure:: _images/mtcontext1.png
.. figure:: _images/mtcontext2.png

   Context menus for MT functions.
.. toctree::
    :titlesonly:

    mt.cm.meta
    mt.cm.showgraphs
    mt.cm.exportedi
    mt.cm.phase

References
----------
 Kirkby, A.L., Zhang, F., Peacock, J., Hassan, R., Duan, J., 2019. The MTPy software package for magnetotelluric data analysis and visualisation. Journal of Open Source Software, 4(37), 1358. https://doi.org/10.21105/joss.01358

 Krieger, L., and Peacock, J., 2014. MTpy: A Python toolbox for magnetotellurics. Computers and Geosciences, 72, p167-175. https://doi.org/10.1016/j.cageo.2014.07.013

