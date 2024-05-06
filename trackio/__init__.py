################################################################################

from .Dataset import Dataset

################################################################################


def read(raw_files=None, data_files=None, data_path="./data"):
    """
    Instantiates a trackio.Dataset object from either raw or processed data files.

    Args:
        raw_files (list, optional): List of raw file paths.
        data_files (list, optional): List of processed data file paths (*.points or *.tracks).
                                    If None, it detect any files in the data_path.
        data_path (str): Path for data files. Defaults to './data'.

    Returns:
        Dataset: A trackio.Dataset object initialized with the given parameters.
    """
    return Dataset(
        raw_files=raw_files, data_files=data_files, data_path=data_path
    )


def from_df(raw_df, data_path="."):
    """
    Instantiates a trackio.Dataset object from a pandas DataFrame of raw point data.

    Args:
        raw_df: pandas DataFrame of raw data.
        data_path (str): Path for data files. Defaults to './data'.

    Returns:
        Dataset: A trackio.Dataset object initialized with the given parameters.
    """
    return Dataset(raw_df=raw_df, data_path=data_path)


def from_gdf(raw_gdf, data_path="."):
    """
    Instantiates a trackio.Dataset object from a geopandas GeoDataFrame of raw data.

    Right now, trackio is limited to accepting only GeoDataFrame's with one LineString per track.
    All geometry must be LineStrings.

    Args:
        raw_gdf: geopandas GeoDataFrame of raw data.
        data_path (str): Path for data files. Defaults to './data'.

    Returns:
        Dataset: A trackio.Dataset object initialized with the given parameters.
    """
    return Dataset(raw_gdf=raw_gdf, data_path=data_path)
