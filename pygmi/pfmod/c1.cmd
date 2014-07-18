@echo off
cls
echo Cython Phase
echo ============
copy grvmagc.pyx grvmagc_%PVER%_%BITS%.pyx
cython -a grvmagc_%PVER%_%BITS%.pyx -o grvmagc_%PVER%_%BITS%.c
