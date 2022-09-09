Overview
========

PyGMI stands for Python Geoscience Modelling and Interpretation. It is a modelling and interpretation suite aimed at magnetic, gravity, remote sensing and other datasets.

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

The PyGMI `Wiki <http://patrick-cole.github.io/pygmi/index.html>`_ pages, include installation and full usage!

The latest release version can be found `here <https://github.com/Patrick-Cole/pygmi/releases>`_.

You may need to install the `Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019 <https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads>`_.

If you have any comments or queries, you can contact the author either through `GitHub <https://github.com/Patrick-Cole/pygmi>`_ or via email at pcole@geoscience.org.za

Requirements
------------
PyGMI will run on both Windows and Linux. It should be noted that the main development is done in Python 3.10 on Windows.

PyGMI should still work with Python 3.8 and Python 3.9.

PyGMI is developed and has been tested with the following libraries in order to function:

* python 3.10.4
* discretize 0.8.0
* fiona 1.8.21
* geopandas 0.11.0
* llvmlite 0.38.1
* matplotlib 3.5.2
* mtpy 1.1.5
* natsort 8.1.0
* numba 0.55.2
* numexpr 2.8.3
* numpy 1.22.4
* pandas 1.4.3
* pillow 9.2.0
* psutil 5.9.1
* pyopengl 3.1.6
* PyQt5 5.15.7
* pytest 7.1.2
* rasterio 1.3.0
* scikit-image 0.19.3
* scikit-learn 1.1.1
* scipy 1.8.1
* shapely 1.8.2
* SimPEG 0.17.0
* sphinx 5.0.2

Installation
------------
General (Not Anaconda)
----------------------
The easiest way to install pygmi if you are working in a python environment is to use the pip command as follows:

   pip install pygmi

This will download pygmi from PyPI and install it within your python repository. Depending on your operating system, and which libraries you already have installed, you may need to follow instructions in sections below. Please note the use of pip when installing PyGMI may cause Anaconda installations to break. Anaconda users should follow the instructions below.

Alternatively, if you satisfy the requirements, you can download pygmi either from Github or PyPI, extract it and run the following command from within the extracted directory:

   python setup.py install

In either case, running pygmi can be now done at the command prompt as follows:

   pygmi

If you are in python, you can run PyGMI by using the following commands:

   import pygmi

   pygmi.main()

If you prefer not to install pygmi as a library, or if there is a problem with running it in that matter, you can simply execute the following command to run it manually:

   python quickstart.py

Windows Users
-------------
Installers are available in `64-bit <https://github.com/Patrick-Cole/pygmi/releases>`_

Alternatively, you can use the instructions above to run PyGMI with your local python installation. You may need to install some dependencies using downloaded binaries, because of compilation requirements. Therefore, if you do get an error, you can try installing precompiled binaries before installing PyGMI.

Examples of binaries you may need to get are:

* numexpr
* numba
* llvmlite
* GDAL
* discretize
* fiona

They can be obtained from the `website <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`_ by Christoph Gohlke.

Linux
-----
Linux normally comes with python installed, but the additional libraries will still need to be installed.

Typically, packages can be installed using pip. The process is as follows:

   sudo apt-get install pip

   sudo apt-get install gdal-bin

   sudo apt-get install libgdal-dev

   pip install cython

   pip install numpy

   pip install pygmi

Anaconda
--------
Anaconda users are advised not to use pip since it can break PyQt5. However, one package is installed only by pip, so a Conda environment should be created.

The process to install is as follows:

   conda create -n pygmi python=3.10

   conda activate pygmi

   conda config --add channels conda-forge

   conda config --set channel_priority flexible

   conda install pyqt

   conda install numpy

   conda install scipy

   conda install matplotlib

   conda install psutil

   conda install numexpr

   conda install pandas

   conda install rasterio

   conda install geopandas

   conda install numba

   conda install scikit-learn

   conda install scikit-image

   conda install pyopengl

   conda install natsort

   conda install simpeg

   conda install pyshp
   
   pip install mtpy

   conda update --all

Once this is done, download pygmi, extract (unzip) it to a directory, and run it from its root directory with the following command:

   python quickstart.py
