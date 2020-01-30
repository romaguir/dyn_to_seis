#!/usr/bin/env python                                                                     
import numpy as np
from numpy.distutils.core import setup,Extension
from setuptools import find_packages

setup(name='dyn_to_seis',
      version='0.1',
      description='Package for investigating seismic expressions of geodynamic models',
      author='Ross Maguire',
      author_email='rmaguire@umd.edu',
      url='www.github.com/romaguir/dyn_to_seis',
      packages=find_packages(),
      install_requires=['numpy','h5py','mayavi'],
      scripts = ['scripts/gen_mod_for_s40filter',
                 'scripts/filter_script',
                 'scripts/plot_report',
                 'scripts/dofilt_ESEP_new',
                 'scripts/gen_mod_for_sp12filter'],
      license='GNU')
