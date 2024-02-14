1. unzip the file
2. Navigate intothe unzipped file's installer subdirectory.
4. Run the python installer: python-3.12.0-amd64 by double clicking it.
5. Open a command prompt
6. Navigate into the installer directory that has the python installer.
7. Add to the path the location of Python.exe  If you accepted the defaults when you 
   installed PYTHON this is done by:
   set PATH=%USERPROFILE%\appdata\local\programs\python\python312;%PATH%
8. Install python packages needed by CutiePie:
    ppackages.bat

    This will take a while.  Got get a coffee, chat with a colleague or do something else.

9. Make a new directory  C:\cutiepie
10. Copy everything in the installer subdirectory into that directory (note if you want to 
be selective, you can copy only the bin/script directories and cutiepie.bat files.  So you should have
(at least):
   C:\cutiepie\bin
   C:\cutiepie\script
   C:\cutiepie.bat
11. You will need to ensure that the directory containing PYTHON is in your path when you start
a program or command script
12. In your command terminal type regedit and accept that it must run in administrator mode.
13. In regedit navigate to 
Computer\HKEY_CURRENT_USER\Environment prepend the Python executable path to the list of 
directories in the Path variable.  Don't forget the ; separator between path elements.


