Overview
========

PyGMI stands for Python Geoscience Modelling and Interpretation. It is a modelling and interpretation suite aimed at magnetic, gravity and other datasets.

PyGMI is developed at the `Council for Geoscience <http://www.geoscience.org.za>`_ (Geological Survey of South Africa).

It includes:

* Magnetic and Gravity 3D forward modelling
* Cluster Analysis
* Routines for cutting, reprojecting and doing simple modifications to data
* Convenient display of data using pseudo-color, ternary and sunshaded representation.
* It is released under the `Gnu General Public License version 3.0 <http://www.gnu.org/copyleft/gpl.html>`_

The PyGMI `Wiki <http://patrick-cole.github.io/pygmi/index.html>`_ pages, include installation and full usage!

The latest release version can be found `here <https://github.com/Patrick-Cole/pygmi/releases>`_.

If you have any comments or queries, you can contact the author either through `GitHub <https://github.com/Patrick-Cole/pygmi>`_ or via email at pcole@geoscience.org.za

Requirements
------------
PyGMI will run on both Windows and Linux. It should be noted that the main development is done in Python 3.7 on Windows.

PyGMI is developed and has been tested with the following libraries in order to function:

* python 3.7.4
* GDAL 3.0.2
* llvmlite 0.29.0
* matplotlib 3.1.1
* numba 0.45.1
* numexpr 2.7.0
* numpy 1.16.5
* pillow 6.2.1
* pandas 0.25.1
* pyopengl 3.1.3b2
* pyqt5 5.13.1
* scipy 1.3.1
* scikit_learn 0.21.3
* scikit_image 0.16.2
* setuptools 41.0.1
* segyio 1.8.8
* geopandas 0.6.1
* pytest 5.1.2
* mtpy 1.1.3

Installation
------------
General (Not Anaconda)
----------------------
The easiest way to install pygmi if you are working in a python environment is to use the pip command as follows:

   pip install pygmi

This will download pygmi from PyPI and install it within your python repository. Please note the use of pip when installing PyGMI may cause Anaconda installations to break. Anaconda users should follow the instructions below.

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

They can be obtained from the `website <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`_ by Christoph Gohlke.

You may also need to install the `Microsoft Visual C++ 2015 Redistributable <https://www.visualstudio.com/downloads/download-visual-studio-vs#d-visual-c>`_.

Linux
-----
Linux normally comes with python installed, but the additional libraries will still need to be installed. One convenient option is to install the above libraries through `Anaconda Python <http://continuum.io/downloads>`_.

Anaconda
--------
Anaconda users are advised not to use pip since it can break PyQt5. Instead, you can install anaconda3 using the regular method, and then:

   conda update --all
   conda install numba
   conda install scipy
   conda install pyopengl
   conda install gdal
   conda install scikit-learn
   conda install pandas
   conda install matplotlib
   conda install numexpr
   conda install numpy
   conda install pillow
   conda install setuptools
   conda install segyio
   conda install geopandas
   conda install mtpy
   conda install pytest

Alternatively if you use environments you can simply use the following command:

   conda create -n pygmi2 scipy numba gdal pandas matplotlib numexpr numpy setuptools pillow pyopengl scikit-learn segyio geopandas mtpy pytest

Once this is done, download pygmi, extract it to a directory, and run it from its root directory with the following command:

   python quickstart.py

Alternatively, if you satisfy the requirements, you can run the following command from within the extracted directory:

   python setup_anaconda.py install

Running pygmi can be now done at the command prompt as follows:

   pygmi
