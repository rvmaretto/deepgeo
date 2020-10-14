#!/bin/sh
python ../src/setup.py bdist_wheel

mv ./build ../built
mv ./deepgeo.egg-info ../built
