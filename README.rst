PyGMI
=====

.. |pythonversion| image:: https://img.shields.io/pypi/pyversions/pygmi
   :alt: PyPI - Python Version
.. |pygmiversion| image:: https://img.shields.io/pypi/v/pygmi
   :alt: PyPI - Version
.. |pygmilicence| image:: https://img.shields.io/github/license/patrick-cole/pygmi
   :alt: GitHub License


|pythonversion| |pygmiversion| |pygmilicence|

Overview
--------

PyGMI stands for Python Geoscience Modelling and Interpretation. It is a modelling and interpretation suite aimed at magnetic, gravity, remote sensing and other datasets. PyGMI has a graphical user interface, and is meant to be run as such.

PyGMI is developed at the `Council for Geoscience <http://www.geoscience.org.za>`_ (Geological Survey of South Africa).

It includes:

* Magnetic and Gravity 3D forward modelling.
* Cluster Analysis, including use of scikit-learn libraries.
* Routines for cutting, reprojecting and doing simple modifications to data.
* Convenient display of data using pseudo-color, ternary and sunshaded representation.
* MT processing and 1D inversion using MTpy.
* Gravity processing.
* Seismological functions for SEISAN data.
* Remote sensing ratios and improved imports.

It is released under the `Gnu General Public License version 3.0 <http://www.gnu.org/copyleft/gpl.html>`_

The PyGMI `Wiki <http://patrick-cole.github.io/pygmi/index.html>`_ pages, include installation and full usage! Contributors can check this `link <https://github.com/Patrick-Cole/pygmi/blob/pygmi3/CONTRIBUTING.md>`_ for ways to contribute.

The latest release version (including windows installers) can be found `here <https://github.com/Patrick-Cole/pygmi/releases>`_.

You may need to install the `Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019 <https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads>`_.

If you have any comments or queries, you can contact the author either through `GitHub <https://github.com/Patrick-Cole/pygmi>`_ or via email at pcole@geoscience.org.za


Installation
------------
The simplest installation of PyGMI is on Windows, using a pre-built installer at `64-bit <https://github.com/Patrick-Cole/pygmi/releases>`_.

If you prefer building from source, you can use PyPi or Conda.

Once built using PyPi, running pygmi can be done at the command prompt as follows:

   pygmi

If you are in python, you can run PyGMI by using the following commands:

   from pygmi.main import main

   main()

If you prefer not to install pygmi as a library, download the source code and execute the following command to run it manually:

   python quickstart.py

Requirements
^^^^^^^^^^^^
PyGMI will run on both Windows and Linux. It should be noted that the main development is done in Python 3.12 on Windows.

PyGMI should still work with Python 3.11.

PyGMI is developed and has been tested with the following libraries in order to function:

* fiona 1.9.5
* geopandas 0.14.4
* h5netcdf 1.3.0
* matplotlib 3.9.0
* mtpy 1.1.5
* natsort 8.4.0
* numexpr 2.10.1
* openpyxl 3.1.2
* psutil 6.0.0
* pyopengl 3.1.7
* pyqt5 5.15.10
* pytest 8.2.2
* rasterio 1.3.9
* rioxarray 0.15.6
* scikit-image 0.24.0
* shapelysmooth 0.2.0
* simpeg 0.21.1

PyPi - Windows
^^^^^^^^^^^^^^
Windows users can use the `WinPython <https://winpython.github.io/>`_ distribution as an alternative to Anaconda. It comes with most libraries preinstalled, so using pip should be sufficient.

Install with the following command.

   pip install pygmi

Should you wish to manually install binaries, related binaries can be obtained at the `website <https://github.com/cgohlke/geospatial-wheels/>`_ by Christoph Gohlke.

If you wish to update GDAL, you will need to download and install:

* fiona
* GDAL
* pyproj
* rasterio
* Rtree
* shapely

All these binaries should be downloaded since they have internal co-dependencies.


PyPi - Linux
^^^^^^^^^^^^
Linux normally comes with python installed, but the additional libraries will still need to be installed.

The process is as follows:

   sudo apt-get install pipx
   
   pipx ensurepath

   pipx install pygmi

Once installed, running pygmi can be done at the command prompt as follows:

   pygmi

If you get the following error: *qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.*, then you can try the following command, since this is linux issue:

   sudo apt-get install libxcb-xinerama0

Anaconda
^^^^^^^^
Anaconda users are advised not to use pip since it can break PyQt5. However, one package is installed only by pip, so a Conda environment should be created.

The process to install is as follows:

   conda create -n pygmi python=3.12

   conda activate pygmi

   conda config --add channels conda-forge

   conda config --set channel_priority flexible

   conda install pyqt

   conda install fiona

   conda install matplotlib

   conda install psutil

   conda install numexpr

   conda install rasterio

   conda install geopandas

   conda install natsort

   conda install scikit-image

   conda install pyopengl

   conda install simpeg

   conda install shapelysmooth

   conda install openpyxl

   conda install h5netcdf

   conda install rioxarray

   conda install pytest

   pip install mtpy

   conda update --all

Once this is done, download pygmi, extract (unzip) it to a directory, and run it from its root directory with the following command:

   python quickstart.py

References
----------

* Cole, P. 2012, Development of a 3D Potential Field Forward Modelling System in Python, AGU fall meeting, 3-7 December, San Francisco, USA
* Cole, P. 2013, PyGMI – The use of Python in geophysical modelling and interpretation. South African Geophysical Association, 13th Biennial Conference, Skukuza Rest Camp, Kruger National Park (7-9 October)
* Cole, P. 2014, The history and design behind the Python Geophysical Modelling and Interpretation (PyGMI) package, SciPy 2014, Austin, Texas (6-12 July)
* Cole, P. 2016, The continued evolution of the open source PyGMI project. 35th IGC, Cape Town.
