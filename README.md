# PyGMI Readme

PyGMI stands for *Python Geophysical Modelling and Interpretation*. It is a modelling and interpretation suite aimed at magnetic, gravity and other datasets. It includes:
* Magnetic and Gravity 3D forward modelling
* Cluster Analysis
* Routines for cutting, reprojecting and doing simple modifications to data
* Convenient display of data using pseudo-color, ternary and sunshaded representation

It is released under the Gnu General Public License version 3.0

For license information see the file LICENSE.txt

## Requirements
PyGMI will run on both Windows and Linux. It should be noted that the main development is now being done in Python 3.7 on Windows.

PyGMI is developed and has been tested with the following libraries in order to function:

* python 3.5.4
* cycler 0.10.0
* GDAL 2.1.4
* llvmlite 0.19.0
* matplotlib 2.0.2
* numba 0.34.0
* numexpr 2.6.2
* numpy 1.13.1
* pillow 4.2.1
* pip 9.0.1
* pyopengl 3.1.1
* pyparsing 2.2.0
* pyqt5 5.9
* python_dateutil 2.6.1
* pytz 2017.2
* scipy 0.19.1
* scikit_learn 0.18.2
* setuptools 36.2.7
* sip 4.19.3
* six 1.10.0

## Installation
### General (Not Anaconda)
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

### Windows
I have now made available convenient installers for PyGMI, thanks to Cyrille Rossant.

Installers are available in [64-bit and 32-bit](https://github.com/Patrick-Cole/pygmi/releases)

Running the software can be achieved through the shortcut on your desktop.

You may need to install the Microsoft Visual C++ 2015 Redistributable. It can be obtained from [here](https://www.visualstudio.com/downloads/download-visual-studio-vs#d-visual-c)

### Linux
Linux normally comes with python installed, but the additional libraries will still need to be installed. One convenient option is to install the above libraries through [Anaconda](http://continuum.io/downloads>).

After installation of python, you can follow the instructions under General.

### Anaconda
Anaconda users are advised not to use pip since it can break PyQt5. Instead, you can install anaconda3 using the regular method, and then:

	conda update --all
	conda install numba=0.42.1
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

Please notice the version of numba. Older versions can cause PyGMI to crash on startup.

Alternatively if you use environments you can simply use the following command:

	conda create -n pygmi2 scipy numba=0.42.1 gdal pandas matplotlib numexpr numpy setuptools pillow pyopengl scikit-learn

Once this is done, download pygmi, extract it to a directory, and run it from its root directory with the following command:

   python quickstart.py

Alternatively, if you satisfy the requirements, you can run the following command from within the extracted directory:

	python setup_anaconda.py install

Running pygmi can be now done at the command prompt as follows:

	pygmi
