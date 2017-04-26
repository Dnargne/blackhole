# blackhole

Python
******

pycuda
======

$ pip install -r requirements.txt
$ pip install .\boost_python-1.63-cp36-cp36m-win_amd64.whl

SET-item -path env:VS100COMNTOOLS -value "C:\Program Files (x86)\Microsoft Visual Studio 14.0\Common7\Tools"

git clone --recursive http://git.tiker.net/trees/pycuda.git
cd pycuda
python setup.py build

PUT C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\x86_amd64;C:\Python361;C:\Python361\Scripts; at the beginning of PATH env

vispy
=====
$ pip install pyglet
$ pip install pyside

$ git clone git://github.com/vispy/vispy.git
$ cd vispy
$ python setup.py develop
