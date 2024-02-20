<p align="left">
<img src="./aisio/supporting/icon.png" alt="sidepanel" width="150"  style="display: block; margin-right: auto;">
</p>

# aisio - A Python approach to working with AIS datasets

This is an open source library geared towards working with AIS data in CSV/feather format. `aisio` natively supports Marine Cadastre (https://marinecadastre.gov/data/) and many other AIS supplier formats, and has functionality to extend support to any supplier format (see step03 in `notebooks`).

If you are working with a raw AIS data feed (such as Spire), it is suggested to first use the [pyais](https://github.com/M0r13n/pyais) library to convert the raw NMEA AIS messages to a CSV/feather format. Then you can operate on the data using `aisio`.

This library grew out of necessity over the years and already contains many useful tools applicable to commerical/research applications. If you are doing typical processing of AIS data for work or research, there is a good chance it already does what you need. Just follow the workflow steps in the `notebooks` folder. If the functionality doesn't already exist, there is a good chance it can be easily integrated to leverage the `aisio` framework.

Unlike other similar libraries, `aisio` was designed for arbitrary sized datasets with arbitrary computational resources. I.e., if you have a huge amount of data and a modest machine with CPU/RAM limitations, this library will still work!!

`aisio` also has heavy operations coded in parallel to leverage powerful machines for faster analyses on large datasets.

## Table of Contents

<!--ts-->
   * [Installation](#installation)
   * [Functionality](#functionality)
   * [Examples](#examples)
<!--te-->

## Installation


This library has been tested with Python versions 3.9, and 3.10.

This library works on Window and Linux. This library requires gdal >= 3.5.1, thus creating an environment is a good idea.

If using Anaconda/Miniconda, run this from the `aisio` folder:

`conda create -n ais python=3.9`    or  `conda create -n ais python=3.10`

`conda activate ais`

`conda install -c conda-forge gdal`

`conda install -c conda-forge rasterio`

`conda install -c conda-forge geopandas`

`pip install git+https://github.com/dengwirda/inpoly-python` (might need to download Visual Studio C++ Dev Tools for Windows, the link will be in the error)

`pip install .` 

Or, just use pip if you are not using Anaconda/Miniconda.

## Functionality


The library is extensible to any AIS supplier dataset. As new data is encountered that `aisio` cannot read, QC functions can be ran to map the new data to `aisio` standards. This allows `aisio` to auto-detect this data format in the future. It also allows `aisio` to verify all the data can be read before it even is read, as with large datasets it can often take a long time to process. Nothing is worse than processing data for 24 hours, only to find it crashed on the 2nd last file because there was a random comma in a column name.

The mappings are in JSON files in the supporting folder. The standard mappings used for the AIS data columns are as follows:

```python
#Standard Raw AIS Data Column Names

['Time',        #timestamp of ais ping
 'X',           #x-coordinate of vessel at ais ping
 'Y',           #y-coordinate of vessel at ais ping
 'Draft',       #draft of vessel at ais ping
 'Name',        #name of vessel
 'MMSI',        #MMSI identifier for vessel
 'IMO',         #IMO identifier for vessel
 'AISCode',     #ais code for vessel
 'Coursing',    #coursing of vessel at  ais ping
 'Heading',     #heading of vessel at  ais ping
 'Turning Rate',#turning rate of vessel at ais ping
 'Speed',       #speed of vessel at  ais ping
 'Length',      #length of vessel (LOA)
 'Width',       #width of vessel (beam)
 'Status']      #navigational status of vessel at ping

```

`aisio` uses standard navigational status codes, as described here:

[NavStatus Codes Explained](https://help.marinetraffic.com/hc/en-us/articles/203990998-What-is-the-significance-of-the-AIS-Navigational-Status-Values-)

`aisio` also utilizes standard AIS code mappings, which are available in the support folder, and shown here:

[AIS Codes Explained](https://coast.noaa.gov/data/marinecadastre/ais/VesselTypeCodes2018.pdf)

The idea is that, regardless of how the data comes in, `aisio` will be able to map it to these standard formats and ingest it.

The library operates by using three main classes: DataSet, Vessel, and Track. The DataSet class wraps over your entire AIS dataset. This DataSet object holds a list of raw data files and has the ability to perform various QC and processing functions. 

Once the DataSet class object is processed, it will contain a series of Vessel class objects (one for each unique vessel in the DataSet), and each Vessel object will contain a series of Track class objects (one for each unique track associated with each Vessel).

Pings are grouped by vessel identifiers (usually MMSI), and then split into tracks using a spatial and temporal threshold, for example here using a 50km space and 12hr time threshold:

![AIS_TRACK](./aisio/supporting/ais_track.png)

Once the pings have been grouped and split into tracks, two metadata file are then populated and saved for easier reaccess. These metadata files contain a wealth of information on each vessel and track in the dataset, which allows you to easily query and filter data of interest.

Furthermore, a novel part of `aisio` is how tracks can be classified. Along any given track, each point will have a navigational status code as explained above. These codes tell you what the vessel was doing at that time. `aisio` includes a number of classification methods to further classify and add codes to the points. This feature of `aisio` opens up the door to classify points/tracks/vessels using any algorithm, and still leverage the `aisio` framework.

For example, you can classify tracks that enter polygons, tracks where speed thresholds are met or not, tracks that travel a pre-defined trip, where tracks are mooring, where tracks are turning, etc. The sky is the limit for classification, and you can combine all the classifications for advanced analyses!

The below diagram provides a visual example of this functionality:

![DATA](./aisio/supporting/data.png)


As the data is classified, the metadata tables are updated. This way you can use these classifications as part of filters later on to select or process the data further.

There are a number of I/O operations to convert the DataSet to GeoDataFrames, DataFrames, CSV files, GIS formats, rasters, etc. As well, there are a number of built in statistics functions to output key information about the data contained in the DataSet.

![RASTER](./aisio/supporting/raster.png)

## Examples

There are a series of Jupyter notebooks available in the `notebooks` folder.

These are broken into a series of steps. Each (or all) of these steps are the typical workflow that you would follow to obtain/process/analyze AIS data for any type of application. These notebooks cover the full range of functionality in the library.

## Coming Soon

Notebooks of simple analyses using ideas from Step1-10 notebooks.




