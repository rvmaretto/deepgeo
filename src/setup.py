import setuptools

with open("../README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepleeo",
    version="0.0.1",
    author="Raian Vargas Maretto",
    author_email="rvmaretto@gmail.com",
    description="Deep Learning functionalities to the classification of Remote Sensing Images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rvmaretto/deepleeo",  #TODO: Verify if I can change it back to the uppercase mode, or change in the github link to keep it all lower case.
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Licence :: GPLV3.0",
        "Operating System :: OS Independent"
    ]
)