Overview
========

PyGMI stands for Python Geophysical Modelling and Interpretation. It is a modelling and interpretation suite aimed at magnetic, gravity and other datasets.

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
PyGMI will run on both Windows and Linux. It should be noted that the main development is done in Python 3.5 on Windows.

PyGMI is developed and has been tested with the following libraries in order to function:

* Python 3.5.1
* NumPy 1.10.4
* SciPy 0.17.0
* Matplotlib 1.5.1
* six 1.10.0 (used by Matplotlib, should be installed automatically)
* pytz 2015.7 (used by Matplotlib, should be installed automatically)
* python-dateutil 2.4.2 (used by Matplotlib, should be installed automatically)
* pyparsing 2.0.7 (used by Matplotlib, should be installed automatically)
* cycler 0.9.0 (used by Matplotlib, should be installed automatically)
* PyQt 4.11.4
* GDAL 2.0.2
* numexpr 2.4.6
* numba 0.23.1
* llvmlite 0.8.0
* PyOpenGL 3.1.1b1

Windows Users
-------------
Windows Users
-------------
You may need to install some dependencies using downloaded binaries, because of compilation requirements. Therefore, if you do get an error, you can try installing precompiled binaries before installing PyGMI. 

Examples of binaries you may need to get are:

* numexpr
* numba
* llvmlite
* GDAL

They can be obtained from the `website <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`_ by Christoph Gohlke.

You may also need to install the `Microsoft Visual C++ 2015 Redistributable <https://www.visualstudio.com/downloads/download-visual-studio-vs#d-visual-c>`_.

Linux
-----
Linux normally comes with python installed, but the additional libraries will still need to be installed. One convenient option is to install the above libraries through `Anaconda <http://continuum.io/downloads>`_.

Anaconda
--------
Anaconda does not find pyqt4 on its system even if it is there already. To install pygmi on anaconda, download the zip file manually, edit the setup.py file, and replace the install_requires switch with the following:

   install_requires=["numpy", "scipy", "matplotlib", "gdal", "numexpr", "numba", "Pillow", "PyOpenGL"],

As you can see, all we have done is removed PyQt4 from the requirements. You will need to make sure it is a part of your conda installation though. From this point the regular command will install pygmi:

   python setup.py install

Note that you can simply install Anaconda use its 'conda install' command to satisfy dependencies. For example:

    conda install gdal

    conda install krb5

Make sure that krb5 is installed, or gdal will not work.