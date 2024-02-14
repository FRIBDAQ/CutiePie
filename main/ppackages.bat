REM  Note pip puts packages in a per user place:
REM 


python -mpip install --upgrade pip
python -mpip install PyQt5
python -mpip install matplotlib
python -mpip install sip
python -mpip install pkgconfig
python -mpip install scipy
python -mpip install scikit-learn scikit-build
python -mpip install opencv-python
python -mpip install PyQtWebEngine
python -mpip install flask flask-restful
python -mpip install js-d3
python -mpip install bitarray
python -mpip install uproot3
python -mpip install httplib2
python -mpip install pandas
python -mpip install requests   # for rustogramer client.
python -mpip install parse


REM   Maybe not needed but:
REM  pip list --outdated and for each listed:
REM 
REM   pip install --upgrade <pkgname>

REM  Install a virtual package to allow building jsoncpp:
REM  This is needed only on development systems.
REM cd \
REM git clone https://github.com/Microsoft/vcpkg.git
REM cd vcpkg
REM .\bootstrap-vcpkg.sh
REM .\vcpkg integrate install
REM .\vcpkg install jsoncpp







