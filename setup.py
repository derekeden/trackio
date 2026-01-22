################################################################################

from pathlib import Path

from setuptools import find_packages, setup

################################################################################

BASE_DIR = Path(__file__).resolve()

# run setup
setup(
    name="trackio",
    version="0.2.0",
    license="MIT",
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
        "matplotlib",
        "PyQt5",
        f"GDAL @ file://{BASE_DIR / 'trackio/supporting/GDAL-3.8.4-cp310-cp310-win_amd64.whl'}",
        f"rasterio @ file://{BASE_DIR / 'trackio/supporting/rasterio-1.3.9-cp310-cp310-win_amd64.whl'}",
        f"inpoly @ file://{BASE_DIR / 'trackio/supporting/inpoly-0.2.0-cp310-cp310-win_amd64.whl'}",
    ],
)

################################################################################
