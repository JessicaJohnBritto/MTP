# # from setuptools import setup
# # from Cython.Build import cythonize

# # setup(ext_modules=cythonize("lib_inhomogeneous_gutzwiller.pyx"))

from setuptools import setup, Extension
from Cython.Build import cythonize
import sys

# Set default compiler flags
extra_compile_args = ['-O3', '-ffast-math', '-g']

# Check if running on Windows and adjust flags
if sys.platform == "win32":
    # Windows compiler does not recognize these flags, so we'll remove them
    extra_compile_args = ['/Ox', '/fp:fast']  # Example flags for Windows

extensions = [
    Extension(
        "lib_inhomogeneous_gutzwiller",  # Replace with your module name
        ["lib_inhomogeneous_gutzwiller.pyx"],  # Replace with your .pyx file
        extra_compile_args=extra_compile_args
    )
]

setup(
    ext_modules=cythonize(extensions)
)


# '''
# program:       setup_cython.py
# author:        tc
# last-modified: Thu Aug 11 09:07:37 CEST 2016
# description:   compiles a cython module
# notes:         to be executed through
#                $ python setup_cython.py build_ext --inplace
# '''

# from __future__ import print_function

# import glob
# import os
# import sys
# from distutils.extension import Extension
# from distutils.core import setup
# from Cython.Distutils import build_ext
# from Cython.Compiler import Options


# # Set general options
# define_macros = []
# libraries = []

# Profiling = True
# print('Profiling: %i' % Profiling)
# # Profiling
# if Profiling:
#     directive_defaults = Options.get_directive_defaults()
#     directive_defaults['linetrace'] = True
#     directive_defaults['binding'] = True
#     directive_defaults['profile'] = True
#     define_macros = [('CYTHON_TRACE', '1')]

# # Loop over all pyx files
# for lib in glob.glob('*.pyx'):
#     print()
#     print('[compiling %s] start' % lib)
#     basename = lib[:-4]
#     ext_modules = [Extension(basename, [basename + '.pyx'],
#                              libraries=libraries,
#                              extra_compile_args = ['-g', '-O3', '-ffast-math'],
#                              define_macros=define_macros)]
#     # Actual compiling
#     setup(cmdclass={'build_ext': build_ext}, ext_modules=ext_modules)
#     # Change permissions of the so file
#     os.system('chmod -x %s*.so' % basename)
#     print('[compiling %s] end' % lib)
#     print()

