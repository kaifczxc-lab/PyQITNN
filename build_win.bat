@echo off
setlocal enabledelayedexpansion
set "VSLANG=1033"

REM ====================
REM build pyqitnn CUDA extension on Windows
REM setup.py compiles csrc/qitnn_cuda.cu directly via CUDAExtension.
REM no pre-built QITNN.lib needed.
REM ====================

set "ROOT=%~dp0"
if "%ROOT%"=="" set "ROOT=."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"

set "VSCMD="
set "VSROOT="

where cl.exe >nul 2>&1
if %ERRORLEVEL% EQU 0 goto env_ready

if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" (
    for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do (
        set "VSROOT=%%i"
    )
    if not "!VSROOT!"=="" if exist "!VSROOT!\Common7\Tools\VsDevCmd.bat" (
        set "VSCMD=!VSROOT!\Common7\Tools\VsDevCmd.bat"
    )
)

if "%VSCMD%"=="" if exist "%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" set "VSCMD=%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
if "%VSCMD%"=="" if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" set "VSCMD=%ProgramFiles%\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
if "%VSCMD%"=="" if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" set "VSCMD=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
if "%VSCMD%"=="" if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" set "VSCMD=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"

if "%VSCMD%"=="" (
    echo [pyqitnn] Visual Studio Build Tools not found. 1>&2
    exit /b 1
)

call "%VSCMD%" -arch=x64 -host_arch=x64 >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [pyqitnn] Failed to initialize Visual Studio environment. 1>&2
    exit /b 1
)

:env_ready
set "DISTUTILS_USE_SDK=1"
set "MSSdk=1"
python setup.py build_ext --inplace
exit /b %ERRORLEVEL%
