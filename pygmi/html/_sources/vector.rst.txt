Vector Data Analysis
====================

Description of Modules
----------------------
The Vector Menu is a where operations on vector data reside. Here, point data or vector data are imported and displayed. If another menu requires vector data, then it may be imported through this menu.

.. toctree::
    :titlesonly:

    vector.dm.importvectordata
    vector.dm.importxyzdata
    vector.dm.colsel
    vector.dm.txtfilesplit
    vector.dm.polycut
    vector.dm.reproj
    vector.dm.gridding
    vector.dm.structcomp.rst

Context Menu
------------
All vector modules, once activated (module will be green) have a context menu which allows the display of the data. The context menu is accessed by **right clicking** on the module.

The vector context menus are available for vector data that have been imported and for modules which operate on vector data and have raster or vector output data. The menu content for the **Vector Import Module** depends on the type of data (polygon, polyline, point) that was imported. Similarly, for functions performed on vector data, the context menu will depend on the type of data produced by the function. If the output data is still in vector format the context menu will depend on the type of vector data. In the case of a raster output the context menu will contain most of the items pertaining to raster dat, and additional items applicable to the input vector data.


.. toctree::
    :titlesonly:

    vector.cm.meta
    vector.cm.stats
    vector.cm.showvector
    vector.cm.showhist
    vector.cm.showrose
    vector.cm.pltcorr
    vector.cm.showprof
    vector.cm.showmapprof
    vector.cm.exportvector
    vector.cm.exportxyz

