@echo off
setlocal EnableDelayedExpansion

REM ==============================
REM Usage:
REM   build.bat main.tex 1
REM   build.bat main.tex 0
REM
REM Arg1 : main .tex file
REM Arg2 : 1 = two passes (ToC, refs)
REM        0 = single pass
REM ==============================

if "%~1"=="" (
    echo ERROR: No main .tex file provided.
    echo Usage: build.bat main.tex [0^|1]
    exit /b 1
)

set MAIN=%~1
set TWOPASS=%~2
set BUILDDIR=build

if not exist "%BUILDDIR%" (
    mkdir "%BUILDDIR%"
)

echo Compiling %MAIN% ...

pdflatex --output-directory="%BUILDDIR%" "%MAIN%"

if "%TWOPASS%"=="1" (
    echo Running second pass for ToC update...
    pdflatex --output-directory="%BUILDDIR%" "%MAIN%"
)

echo Done.
endlocal