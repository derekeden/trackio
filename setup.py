################################################################################

from setuptools import find_packages, setup

################################################################################

setup(
    name="trackio",
    version="0.1.0",
    author="Derek J Eden",
    author_email="derekjeden@gmail.com",
    description="A pythonic approach to working with track/trajectory data with big datasets and small computational power in mind.",
    include_package_data=True,
    packages=find_packages(),
    install_requires=[
        "Cython",
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "tqdm",
        "more_itertools",
        "pyproj",
        "datetime",
        "rasterio",
        "geopandas",
        "gdal",
        "inpoly",
        "bs4"
    ],
    
    #dask?
    #beautifulsoup4
    #scikit learn?
    #kneed
    
    dependency_links=["https://github.com/dengwirda/inpoly-python.git"],
)

################################################################################
