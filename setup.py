################################################################################

import subprocess as sp

from setuptools import find_packages, setup

################################################################################

# install numpy before rasterio to ensure <2.0
sp.run(["pip", "install", "numpy<2.0"])

# installed prepackaged wheels
sp.run(
    [
        "pip",
        "install",
        "./trackio/supporting/GDAL-3.8.4-cp310-cp310-win_amd64.whl",
    ]
)
sp.run(
    [
        "pip",
        "install",
        "./trackio/supporting/rasterio-1.3.9-cp310-cp310-win_amd64.whl",
    ]
)
sp.run(
    ["pip", "install", "./trackio/supporting/inpoly-python-0.2.0.zip"],
    check=True,
)

# run setup
setup(
    name="trackio",
    version="0.1.0",
    author="Derek J Eden",
    author_email="derekjeden@gmail.com",
    description="A python approach to working with mass movement data with big datasets and small computational power in mind.",
    include_package_data=True,
    packages=find_packages(),
    install_requires=[
        "Cython",
        "numpy<2.0",
        "scipy==1.12.0",
        "pandas==2.2.1",
        "geopandas",
        "tqdm",
        "more_itertools",
        "pyproj",
        "beautifulsoup4",
        "scikit-learn",
        "scikit-image",
        "kneed",
        "dask==2024.2.1",
    ],
)

################################################################################
