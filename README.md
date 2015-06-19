#PyGMI Readme

PyGMI stands for *Python Geophysical Modelling and Interpretation*. It is a modelling and interpretation suite aimed at magnetic, gravity and other datasets. It includes:
* Magnetic and Gravity 3D forward modelling
* Cluster Analysis
* Routines for cutting, reprojecting and doing simple modifications to data
* Convenient display of data using pseudo-color, ternary and sunshaded representation

It is released under the Gnu General Public License version 3.0

For license information see the file LICENSE.txt

##Requirements
PyGMI will run on both Windows and Linux. It should be noted that the main development is done in Python 3.4 on Windows.

PyGMI is developed and has been tested with the following libraries in order to function:

* Python 3.4.2
* NumPy 1.8.2
* SciPy 0.14.0
* Matplotlib 1.4.1
* PyQt 4.10.4
* GDAL 1.11.1
* numexpr 2.4
* numba 0.15.1

It is possible that it might work on earlier versions, especially on non-windows operating systems. Under windows, there are compiled .pyd files (python version of a dll) which require the NumPy and Python version to match. However, for actual releases, I do post windows binaries which include a standalone python distribution (that will not interfere with existing installations), so this should not be a problem.

##Installation
###General
The easiest way to install pygmi if you are worning in a python environment is to use the pip command as follows:

	pip install pygmi

This will download pygmi from PyPI and install it within your python repository. Alternatively, if you satisfy the requirements, you can download pygmi either from Github or PyPI, extract it and run the following command from within the extracted directory:

	python setup.py install

In either case, running pygmi can be now done at the command prompt as follows:

	pygmi

If you are in python, you can run PyGMI by using the following commands:

	import pygmi
	pygmi.main()


###Windows
I have now made available convenient installers for PyGMI, thanks to Cyrille Rossant. If you have already installed PyGMI using this installer, you can simply update it using the smaller pygmi_update.exe file. This smaller update file contains only the python code, if you are only interested in that.

Running the software can be achieved through the shortcut on your desktop.

###Linux
Linux normally comes with python installed, but the additional libraries will still need to be installed. One convenient option is to install the above libraries through Anaconda <http://continuum.io/downloads>.

After installation of python, you can follow the instructions under General.

###Anaconda
Anaconda does not find pyqt4 on its system even if it is there already. To install pygmi on anaconda, edit the setup.py file, and replace the install_requires switch with the following:

	install_requires=["numpy", "scipy", "matplotlib", "gdal", "numexpr", "numba"],

As you can see, all we have done is removed PyQt4 from the requirements. You will need to make sure it is a part of your conda installation though. From this point the regular command will install pygmi:

	python setup.py install

###Alternative execution

If you prefer not to install pygmi as a library, or if there is a problem with running it in that matter, you can simply execute the following command to run it manually:

	python quickstart.py
