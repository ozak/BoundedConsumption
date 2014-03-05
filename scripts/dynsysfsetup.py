#!/usr/bin/env python
'''
======================================================
Author:  Ömer Özak, 2013--2014 (ozak at smu.edu)
Website: http://omerozak.com
GitHub:  https://github.com/ozak/BoundedConsumption
======================================================
'''
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("dynsysf", ["dynsysf.pyx"], include_dirs=[numpy.get_include()])]
)
'''
To install Cython models
csh
setenv CC gcc
python myfile_setup.py build_ext -i
'''
 