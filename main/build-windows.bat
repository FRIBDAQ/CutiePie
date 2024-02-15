REM  This batch file is a windows script to rebuild
REM  Cutiepie on windows.   
REM  Items built (all built with production):
REM     mirrorclient (C++ DLL)
REM     PyQtGui\src  (C++ dll)
REM     PyQtGui\sip   Python dynamic module with sip6.
REM     PyQtGui\standalone - program.
REM
REM Solutions may need tweaking if Python is not installed in
REM C:\Program Files and not 3.12 and if packages are not installed in
REM %APPDATA%\Roaming\Python 
REM  the latter is default but the former is not (C:\Python is I think).
REM
REM  ------------------ requirements -------------------------------
REM *  The visual C++ jsoncpp package must be installed using the vcpkg
REM  Package manager:
REM   cd \
REM   git clone https://github.com/Microsoft/vcpkg.git
REM   cd vcpkg
REM   bootstrap-vcpkg.bat
REM   vcpkg install jsoncpp
REM
REM  The build solution assumes vmpkg integrate install was not done... 
REM  this is incase you need but don't have admin on the system doing
REM the build
REM -------

REM - set up visual studio paths.
set vcbat="C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
@call %vcbat%

REM  Setup python paths as defined on my dev system.  This may need to be changed
REM  depending on how and where Python is installed.




REM Build mirrorclient-windows for production:

PUSHD mirrorclient
msbuild MirrorClient.sln /property:Configuration=Release /t:Clean;Rebuild
POPD

REM Build PyQtGui\src

PUSHD PyQtGui\src
msbuild PyQtGui-CPyConverter.sln /property:Configuration=Release /t:Clean;Rebuild
POPD

REM Build the SIP stuff:
PUSHD PyQtGui\sip
sip-build --verbose
POPD

REM build the standalon Cutie pie executable:
PUSHD PyQtGui\standalone
msbuild CutiePie.sln /property:Configuration=Release /t:Clean;Rebuild
POPD

REM Now collect all of the bits and pieces into an installer directory.
REM this will have, when we are done
REM
REM  cutiepie.bat - batch script to run stuff.
REM  bin\CutiePie.exe (from pyqtgui\standalone).
REM  Script\Bunch of ptyohn (from pyqtgui\gui)
REM  Script\CPyConverter.cp312-win_amd64.pyd from pyqtgui\sip\build\CPyConverter\build\lib.win-amd64-cpython-312
REM  Script\MirrorClient.dll,jsoncpp.dll from mirror-client\x64\release
REM  python-3.12.0-amd64.exe - python installer from here.
REM  ppackages.bat from here.

REM kill off the installer folder if it exists.
if exist installer\ (
    rmdir /s/q installer
)
mkdir installer
mkdir installer\bin
mkdir installer\Script

copy cutiepie.bat installer
copy PyQtGui\standalone\x64\Release\CutiePie.exe installer\bin
copy PyQtGui\src\x64\Release\*.dll installer\Script
copy PyQtGui\gui installer\Script
copy PyQtGui\sip\build\CPyConverter\build\lib.win-amd64-cpython-312\*.pyd installer\Script
copy mirrorclient\x64\Release\*.dll installer\Script
copy C:\vcpkg\packages\jsoncpp_x64-windows\bin\*.dll installer\Script

copy python-*.exe installer
copy ppackages.bat installer
copy CUTIEPIE-README.txt installer

REM Zip it all up  - yes windows sort of has a tar.

if exist installer.zip (
    del installer.zip
)
tar -a -c -f installer.zip installer

