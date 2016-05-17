#PyGMI Readme

PyGMI stands for *Python Geophysical Modelling and Interpretation*. It is a modelling and interpretation suite aimed at magnetic, gravity and other datasets. It includes:
* Magnetic and Gravity 3D forward modelling
* Cluster Analysis
* Routines for cutting, reprojecting and doing simple modifications to data
* Convenient display of data using pseudo-color, ternary and sunshaded representation

It is released under the Gnu General Public License version 3.0

For license information see the file LICENSE.txt

##Requirements
PyGMI will run on both Windows and Linux. It should be noted that the main development is now being done in Python 3.5 on Windows.

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
* llvmlite 0.8.0 (used by numba, should be installed automatically)
* PyOpenGL 3.1.1b1


It should also still work with the following or newer, if you wish to try installing Python 3.4:

* Python 3.4.3
* NumPy 1.9.2
* SciPy 0.16.0
* Matplotlib 1.4.3
* six 1.9.0 (used by Matplotlib, should be installed automatically)
* pytz 2015.4 (used by Matplotlib, should be installed automatically)
* python-dateutil 2.4.2 (used by Matplotlib, should be installed automatically)
* pyparsing 2.0.3 (used by Matplotlib, should be installed automatically)
* PyQt 4.11.4
* GDAL 1.11.3
* numexpr 2.4.3
* numba 0.20.0
* llvmlite 0.6.0 (used by numba, should be installed automatically)
* PyOpenGL 3.1.0

It is possible that it might work on earlier versions, especially on non-windows operating systems.

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
I have now made available convenient installers for PyGMI, thanks to Cyrille Rossant.

Installers are available in [64-bit and 32-bit](https://github.com/Patrick-Cole/pygmi/releases)

Running the software can be achieved through the shortcut on your desktop.

You may need to install the Microsoft Visual C++ 2015 Redistributable. It can be obtained from [here](https://www.visualstudio.com/downloads/download-visual-studio-vs#d-visual-c)

###Linux
Linux normally comes with python installed, but the additional libraries will still need to be installed. One convenient option is to install the above libraries through [Anaconda](http://continuum.io/downloads>).

After installation of python, you can follow the instructions under General.

###Anaconda
Anaconda does not find pyqt4 on its system even if it is there already. To install pygmi on anaconda, edit the setup.py file, and replace the install_requires switch with the following:

	install_requires=["numpy", "scipy", "matplotlib", "gdal", "numexpr", "numba", "Pillow", "PyOpenGL"],

As you can see, all we have done is removed PyQt4 from the requirements. You will need to make sure it is a part of your conda installation though. From this point the regular command will install pygmi:

	python setup.py install

Note that you can simply install Anaconda use its 'conda install' command to satisfy dependencies. For example:

	conda install gdal
	conda install krb5

Make sure that krb5 is installed, or gdal will not work.

###Alternative execution

If you prefer not to install pygmi as a library, or if there is a problem with running it in that matter, you can simply execute the following command to run it manually:

	python quickstart.py
