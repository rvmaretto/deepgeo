#!/bin/sh
cd ../src
python setup.py sdist bdist_wheel

mv -R ./build ../built
mv -R ./deepgeo.egg-info ../built
mv MANIFEST ../built
cd ../build
