cls
set pver=2.2.2
set temp=C:\Work\Programming\pygmi
del *.rst
sphinx-apidoc -f  -F -H PyGMI -V %pver% -R %pver% -o . %temp% %temp%\pygmi\seis %temp%\setup.py %temp%\pygmi\pfmod\grvmagc.pyx

rem Note:
rem #####
rem You need to add two lines to your conf.py
rem
rem sys.path.insert(0, os.path.abspath('..'))
rem
rem 'sphinxcontrib.napoleon',