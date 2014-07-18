# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 09:44:21 2014

@author: pcole
"""

import os

os.system(r'@echo off')
os.system(r'set WINPYDIR=%~dp0..\python-3.3.5.amd64')
os.system(r'set WINPYVER=3.3.5.0')
os.system(r'set HOME=%WINPYDIR%\..\settings')
os.system(r'set PATH=%WINPYDIR%\Lib\site-packages\PyQt4;%WINPYDIR%\;%WINPYDIR%'
          r'\DLLs;%WINPYDIR%\Scripts;%WINPYDIR%\..\tools;%WINPYDIR%\..\tools'
          r'\gnuwin32\bin;%PATH%;%WINPYDIR%\..\tools\TortoiseHg')
os.system(r'start .\WinPython-64bit-3.3.5.0\python-3.3.5.amd64'
          r'\pythonw.exe main.py %*')
          