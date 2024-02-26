################################################################################

from setuptools import find_packages, setup

################################################################################

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
        "numpy",
        "scipy",
        "pandas",
        "tqdm",
        "more_itertools",
        "pyproj",
        "beautifulsoup4",
        "scikit-learn",
        "kneed",
        "dask"
    ],    
)

################################################################################
