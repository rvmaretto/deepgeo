import setuptools
import os
from distutils.command.bdist import bdist as _bdist
from distutils.command.sdist import sdist as _sdist

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
dist_dir = os.path.join(root_dir, 'built')

with open(os.path.join(root_dir, 'README.md'), 'r') as fh:
    long_description = fh.read()

class bdist(_bdist):
    def finalize_options(self):
        _bdist.finalize_options(self)
        self.dist_dir = dist_dir

class sdist(_sdist):
    def finalize_options(self):
        _sdist.finalize_options(self)
        self.dist_dir = dist_dir

setuptools.setup(
    name='deepgeo',
    version='0.1.0',
    author='Raian Vargas Maretto',
    author_email='rvmaretto@gmail.com',
    description='Deep Learning functionalities to the classification of Remote Sensing Images',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/rvmaretto/deepgeo',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Licence :: GPLV3.0',
        'Operating System :: OS Independent'
    ],
    cmdclass={
        'bdist': bdist,
        'sdist': sdist,
    }
)
