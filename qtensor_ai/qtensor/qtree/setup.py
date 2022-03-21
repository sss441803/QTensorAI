"""
Setup script for the Qtree library
"""

from __future__ import absolute_import
from __future__ import print_function

import setuptools

# Configure the required packages and scripts to install.
REQUIRED_PACKAGES = [
    'numpy',
    'networkx>=2.3',
    'matplotlib',
    'google-api-core<=1.17.0',
    'cirq'

]

setuptools.setup(name='qtensor-qtree',
                 version='0.1.2',
                 description='Simple quantum circuit simulator'
                 ' based on undirected graphical models',
                 url='https://github.com/Huawei-HiQ/qtree',
                 keywords='quantum_circuit quantum_algorithms',
                 author='R. Schutski, D. Lykov, D. Maslov et al. ',
                 author_email='r.schutski@skoltech.ru',
                 license='Apache',
                 packages=setuptools.find_packages(),
                 install_requires=REQUIRED_PACKAGES,
                 extras_require={
                     'tensorflow': ['tensorflow<=1.15'],
                 },
                 include_package_data=True,
                 zip_safe=False)
