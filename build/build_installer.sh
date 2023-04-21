#!/bin/sh
cd ../src
python setup.py sdist bdist_wheel

mv -f ./build ../built
mv -f ./deepgeo.egg-info ../built
mv MANIFEST ../built
cd ../build
