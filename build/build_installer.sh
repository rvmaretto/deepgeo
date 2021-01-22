#!/bin/sh
cd ../src
python setup.py sdist bdist_wheel

mv ./build ../built
mv ./deepgeo.egg-info ../built
mv MANIFEST ../built
cd ../build
