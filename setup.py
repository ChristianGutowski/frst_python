import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="frst",
    version="1.0.0",
    author="Michael Solomentsev; Ryan O'Hern",
    author_email="mys29@cornell.com",
    description="Fast Radial Symmetry Transform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=['image processing','radial transform'],
    url="https://github.com/rohern/frst_python",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ),
)